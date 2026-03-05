"""
compare_laplacians.py
---------------------
Wrapping script that:
1. Discovers all ArrowSpace datasets in a storage directory
2. Runs ArrowSpaceEigenAnalyzer on each
3. Compares Laplacians across datasets using spectral distance metrics
4. Produces per-dataset reports + a cross-dataset comparison report

python compare_laplacians.py --storage ../storage \
                              --output  reports/ \
                              --k 100
"""

import sys
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.sparse import issparse
from scipy.linalg import subspace_angles

# Import the analyzer from the same directory (or installed package)
sys.path.insert(0, str(Path(__file__).parent))
from graph_laplacian_analysis import ArrowSpaceEigenAnalyzer

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_SUFFIXES = [
    "-gl-matrix.parquet",
    "-lambdas.parquet",
]

OPTIONAL_SUFFIXES = [
    "-raw_input.parquet",
    "-clustered-dm.parquet",
    "-laplacian-input.parquet",
]


def discover_datasets(storage_dir: str) -> List[str]:
    """
    Scan storage_dir for complete ArrowSpace dataset groups.
    A dataset is identified by its gl-matrix Parquet file.
    The dataset name is everything before '-gl-matrix.parquet'.

    Returns:
        List of dataset name strings (e.g., ['dataset_e20274', ...])
    """
    storage_path = Path(storage_dir)
    if not storage_path.exists():
        raise FileNotFoundError(f"Storage directory not found: {storage_path}")

    # Find all gl-matrix files — they define a dataset
    gl_files = sorted(storage_path.glob("*-gl-matrix.parquet"))
    if not gl_files:
        raise ValueError(
            f"No ArrowSpace datasets found in {storage_path}. "
            "Expected files matching *-gl-matrix.parquet"
        )

    datasets = []
    for gl_file in gl_files:
        dataset_name = gl_file.name.replace("-gl-matrix.parquet", "")

        # Validate that all required files are present
        missing = [
            s for s in REQUIRED_SUFFIXES
            if not (storage_path / f"{dataset_name}{s}").exists()
        ]
        if missing:
            print(f"  ⚠ Skipping {dataset_name}: missing {missing}")
            continue

        # Report optional files
        optional_present = [
            s for s in OPTIONAL_SUFFIXES
            if (storage_path / f"{dataset_name}{s}").exists()
        ]

        datasets.append(dataset_name)
        print(
            f"  ✓ Found dataset: {dataset_name}  "
            f"({len(optional_present)}/{len(OPTIONAL_SUFFIXES)} optional files)"
        )

    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# 2. PER-DATASET ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_single_analysis(
    storage_dir: str,
    dataset_name: str,
    output_dir: Path,
    k_eigenpairs: Optional[int] = None,
) -> Dict:
    """
    Load one dataset, run full eigenanalysis, save per-dataset plots/CSVs.

    Returns the summary dict from generate_full_report() augmented with the
    eigenvalues and eigenvectors arrays for downstream comparison.
    """
    ds_output = output_dir / dataset_name
    ds_output.mkdir(parents=True, exist_ok=True)

    analyzer = ArrowSpaceEigenAnalyzer(storage_dir, dataset_name)
    analyzer.load_all_artifacts()

    # Extract Laplacian & compute eigenpairs so we can reuse them in comparison
    L = analyzer.extract_laplacian_matrix(as_sparse=True)
    eigenvalues, eigenvectors = analyzer.compute_eigen_decomposition(
        L, k=k_eigenpairs
    )

    # Store on the analyzer so generate_full_report can skip recomputation
    analyzer._eigenvalues = eigenvalues
    analyzer._eigenvectors = eigenvectors
    analyzer._L = L

    # Patch generate_full_report to use pre-computed values
    summary = _generate_report_from_precomputed(
        analyzer, eigenvalues, eigenvectors, L, ds_output
    )
    summary["dataset_name"] = dataset_name

    # Attach arrays to summary dict (numpy arrays — NOT serialised to JSON)
    summary["_eigenvalues"] = eigenvalues
    summary["_eigenvectors"] = eigenvectors

    return summary


def _generate_report_from_precomputed(
    analyzer: ArrowSpaceEigenAnalyzer,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    L,
    output_dir: Path,
) -> Dict:
    """Run all analysis steps using already-computed eigendecomposition."""
    topology = analyzer.characterize_topology(eigenvalues)
    stats = analyzer.analyze_eigenvalue_distribution(eigenvalues)
    curvature = analyzer.compute_curvature_measures(eigenvalues, eigenvectors)
    subspaces = analyzer.analyze_subspace_characteristics(eigenvectors, eigenvalues)

    print(f"  Topology: {topology.upper()}")

    analyzer.plot_eigenvalue_spectrum(eigenvalues, output_dir / "eigenvalue_spectrum.png")
    analyzer.plot_spectral_embedding(
        eigenvectors, eigenvalues, save_path=output_dir / "spectral_embedding.png"
    )
    analyzer.plot_curvature_distribution(
        eigenvalues, output_dir / "curvature_distribution.png"
    )

    # Numerical CSV
    results_df = pd.DataFrame(
        {
            "index": np.arange(len(eigenvalues)),
            "eigenvalue": eigenvalues,
            "spectral_gap": np.concatenate([np.diff(eigenvalues), [0]]),
        }
    )
    results_df.to_csv(output_dir / "eigenanalysis_results.csv", index=False)

    summary = {
        "dataset": analyzer.dataset_name,
        "laplacian_shape": list(L.shape),
        "laplacian_nnz": int(L.nnz),
        "topology_type": topology,
        "statistics": stats,
        "curvature_measures": curvature,
        "subspaces": subspaces,
    }

    with open(output_dir / "eigenanalysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 3. CROSS-DATASET COMPARISON METRICS
# ─────────────────────────────────────────────────────────────────────────────

def spectral_distance(ev_a: np.ndarray, ev_b: np.ndarray) -> float:
    """
    L2 distance between sorted eigenvalue vectors (truncated to the shorter).
    Measures how similar the spectral profiles are.
    """
    n = min(len(ev_a), len(ev_b))
    return float(np.linalg.norm(ev_a[:n] - ev_b[:n]))


def normalised_spectral_distance(ev_a: np.ndarray, ev_b: np.ndarray) -> float:
    """
    Spectral distance normalised by the sum of both norms — bounded in [0, 1].
    """
    n = min(len(ev_a), len(ev_b))
    diff = np.linalg.norm(ev_a[:n] - ev_b[:n])
    denom = np.linalg.norm(ev_a[:n]) + np.linalg.norm(ev_b[:n])
    return float(diff / denom) if denom > 1e-12 else 0.0


def subspace_similarity(evec_a: np.ndarray, evec_b: np.ndarray, k: int = 10) -> float:
    """
    Mean cosine of principal angles between the span of the first k eigenvectors.
    1.0 = identical subspaces, 0.0 = orthogonal subspaces.
    Requires same number of rows (same F), but handles different k.
    """
    if evec_a.shape[0] != evec_b.shape[0]:
        return float("nan")
    k_a = min(k, evec_a.shape[1])
    k_b = min(k, evec_b.shape[1])
    k_use = min(k_a, k_b)
    if k_use < 1:
        return float("nan")
    angles = subspace_angles(evec_a[:, :k_use], evec_b[:, :k_use])
    return float(np.mean(np.cos(angles)))


def eigenvalue_correlation(ev_a: np.ndarray, ev_b: np.ndarray) -> float:
    """Pearson correlation between sorted eigenvalue sequences (truncated to shorter)."""
    n = min(len(ev_a), len(ev_b))
    if n < 2:
        return float("nan")
    corr = np.corrcoef(ev_a[:n], ev_b[:n])[0, 1]
    return float(corr)


def build_comparison_matrix(
    summaries: List[Dict],
    metric_fn,
    array_key: str = "_eigenvalues",
) -> pd.DataFrame:
    """
    Build an NxN symmetric matrix of pairwise metric values.

    Args:
        summaries:  List of per-dataset summary dicts (must contain array_key)
        metric_fn:  Function(array_a, array_b) -> float
        array_key:  Key in summary dict that holds the numpy array to compare
    """
    names = [s["dataset_name"] for s in summaries]
    n = len(names)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.0 if "distance" in metric_fn.__name__ else 1.0
            elif j < i:
                matrix[i, j] = matrix[j, i]  # symmetry
            else:
                matrix[i, j] = metric_fn(
                    summaries[i][array_key], summaries[j][array_key]
                )

    return pd.DataFrame(matrix, index=names, columns=names)


# ─────────────────────────────────────────────────────────────────────────────
# 4. COMPARISON VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_eigenvalue_overlay(summaries: List[Dict], save_path: Path):
    """Overlay eigenvalue spectra for all datasets in one plot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    palette = sns.color_palette("husl", len(summaries))

    for idx, s in enumerate(summaries):
        ev = s["_eigenvalues"]
        color = palette[idx]
        label = s["dataset_name"]

        # Linear scale
        axes[0].plot(ev, "-", linewidth=1.5, color=color, label=label, alpha=0.85)

        # Log scale (positive eigenvalues only)
        ev_pos = ev[ev > 1e-10]
        if len(ev_pos):
            axes[1].semilogy(ev_pos, "-", linewidth=1.5, color=color, alpha=0.85)

    for ax, title in zip(axes, ["Eigenvalue Spectra (Linear)", "Eigenvalue Spectra (Log)"]):
        ax.set_xlabel("Index k")
        ax.set_ylabel("Eigenvalue λₖ")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved eigenvalue overlay: {save_path}")
    plt.close()


def plot_spectral_gap_comparison(summaries: List[Dict], save_path: Path):
    """Bar chart of max spectral gap and its position per dataset."""
    names = [s["dataset_name"] for s in summaries]
    max_gaps = [s["statistics"]["spectral_gaps"]["max"] for s in summaries]
    gap_positions = [s["statistics"]["spectral_gaps"]["max_gap_index"] for s in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(names, max_gaps, color=sns.color_palette("husl", len(names)))
    axes[0].set_xlabel("Max Spectral Gap (λₖ₊₁ − λₖ)")
    axes[0].set_title("Max Spectral Gap per Dataset")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(names, gap_positions, color=sns.color_palette("mako", len(names)))
    axes[1].set_xlabel("Index k of Max Gap")
    axes[1].set_title("Position of Max Spectral Gap")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved spectral gap comparison: {save_path}")
    plt.close()


def plot_topology_scorecard(summaries: List[Dict], save_path: Path):
    """
    Heatmap-style scorecard comparing key scalar metrics across all datasets.
    """
    metrics = {
        "Mean λ":        lambda s: s["statistics"]["mean"],
        "Std λ":         lambda s: s["statistics"]["std"],
        "Max gap":       lambda s: s["statistics"]["spectral_gaps"]["max"],
        "Eff. dim":      lambda s: s["statistics"]["effective_dimension"],
        "VN entropy":    lambda s: s["statistics"]["von_neumann_entropy"],
        "Mean curv.":    lambda s: s["curvature_measures"]["mean_curvature"],
        "Curv. conc.":   lambda s: s["curvature_measures"]["curvature_concentration"],
        "Near-zero λ":   lambda s: float(s["statistics"]["near_zero_eigenvalues"]),
    }

    names = [s["dataset_name"] for s in summaries]
    data = {}
    for col, fn in metrics.items():
        try:
            data[col] = [fn(s) for s in summaries]
        except Exception:
            data[col] = [float("nan")] * len(summaries)

    df = pd.DataFrame(data, index=names)

    # Z-score normalise each column for colour scale
    df_norm = (df - df.mean()) / (df.std() + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(names) * 0.8)))

    sns.heatmap(
        df_norm,
        ax=axes[0],
        cmap="RdYlGn",
        center=0,
        annot=False,
        linewidths=0.4,
        cbar_kws={"label": "Z-score"},
    )
    axes[0].set_title("Spectral Metrics Heatmap (Z-score)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

    sns.heatmap(
        df,
        ax=axes[1],
        cmap="viridis",
        annot=True,
        fmt=".3g",
        linewidths=0.4,
        cbar_kws={"label": "Raw value"},
    )
    axes[1].set_title("Spectral Metrics (Raw Values)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved topology scorecard: {save_path}")
    plt.close()

    return df


def plot_pairwise_distance_heatmap(
    distance_df: pd.DataFrame,
    title: str,
    save_path: Path,
):
    """Annotated heatmap of pairwise spectral distances / similarities."""
    fig, ax = plt.subplots(figsize=(max(6, len(distance_df) * 1.2),
                                    max(5, len(distance_df) * 1.0)))
    sns.heatmap(
        distance_df,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Distance"},
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved heatmap: {save_path}")
    plt.close()


def plot_lambda_distribution_boxplots(summaries: List[Dict], save_path: Path):
    """Side-by-side box plots of lambda (eigenvalue) distributions."""
    fig, ax = plt.subplots(figsize=(max(8, len(summaries) * 2), 6))
    palette = sns.color_palette("husl", len(summaries))

    data_list = [s["_eigenvalues"] for s in summaries]
    labels = [s["dataset_name"] for s in summaries]

    bplots = ax.boxplot(
        data_list,
        labels=labels,
        patch_artist=True,
        notch=False,
        vert=True,
    )
    for patch, color in zip(bplots["boxes"], palette):
        patch.set_facecolor(color)

    ax.set_ylabel("Eigenvalue λ")
    ax.set_title("Eigenvalue Distribution per Dataset")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved lambda boxplots: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. CROSS-DATASET SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_comparison_report(
    summaries: List[Dict],
    scorecard_df: pd.DataFrame,
    norm_dist_df: pd.DataFrame,
    subspace_sim_df: pd.DataFrame,
    output_dir: Path,
):
    """Write a Markdown summary report comparing all datasets."""
    report_path = output_dir / "comparison_report.md"

    topology_map = {s["dataset_name"]: s["topology_type"] for s in summaries}

    with open(report_path, "w") as f:
        f.write("# ArrowSpace Laplacian Comparison Report\n\n")
        f.write(f"Datasets analysed: {len(summaries)}\n\n")

        # Topology table
        f.write("## Topology Classification\n\n")
        f.write("| Dataset | Topology | Eff. Dim | Max Gap | VN Entropy |\n")
        f.write("|---------|----------|----------|---------|------------|\n")
        for s in summaries:
            st = s["statistics"]
            f.write(
                f"| {s['dataset_name']} "
                f"| {s['topology_type']} "
                f"| {st['effective_dimension']:.2f} "
                f"| {st['spectral_gaps']['max']:.4f} "
                f"| {st['von_neumann_entropy']:.4f} |\n"
            )

        # Most similar / most different pairs
        f.write("\n## Pairwise Normalised Spectral Distance\n\n")
        f.write(norm_dist_df.to_markdown())
        f.write("\n")

        f.write("\n## Subspace Similarity (first 10 eigenvectors)\n\n")
        f.write(subspace_sim_df.to_markdown())
        f.write("\n")

        # Scalar scorecard
        f.write("\n## Scalar Metrics Scorecard\n\n")
        f.write(scorecard_df.to_markdown(floatfmt=".4f"))
        f.write("\n")

    print(f"✓ Saved comparison report: {report_path}")
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def compare_all(
    storage_dir: str = "storage",
    output_dir: str = "comparison_output",
    k_eigenpairs: Optional[int] = None,
):
    """
    Discover all ArrowSpace datasets in storage_dir, run eigenanalysis on each,
    then produce cross-dataset comparison plots and a summary report.

    Args:
        storage_dir:  Directory containing all dataset Parquet files.
        output_dir:   Root directory for all output artefacts.
        k_eigenpairs: Compute only this many eigenpairs per dataset
                      (None = full decomposition; set e.g. 100 for large matrices).
    """
    out = Path(__file__).parent / Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    storage_dir = Path(__file__).parent.parent / storage_dir

    print("=" * 70)
    print("ArrowSpace Multi-Dataset Laplacian Comparison")
    print("=" * 70)

    # ── Discover datasets ────────────────────────────────────────────────────
    print(f"\nScanning storage directory: {storage_dir}")
    datasets = discover_datasets(storage_dir)
    if not datasets:
        print("No valid datasets found. Exiting.")
        return

    print(f"\nFound {len(datasets)} dataset(s): {datasets}\n")

    # ── Per-dataset analysis ─────────────────────────────────────────────────
    summaries = []
    for ds in datasets:
        print("-" * 70)
        print(f"Analysing: {ds}")
        print("-" * 70)
        try:
            summary = run_single_analysis(storage_dir, ds, out, k_eigenpairs)
            summaries.append(summary)
        except Exception as exc:
            print(f"  ✗ Failed to analyse {ds}: {exc}")

    if len(summaries) < 1:
        print("No datasets analysed successfully. Exiting.")
        return

    # ── Cross-dataset comparison ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Cross-dataset Comparison")
    print("=" * 70)

    compare_out = out / "_comparison"
    compare_out.mkdir(exist_ok=True)

    # Eigenvalue overlay
    plot_eigenvalue_overlay(summaries, compare_out / "eigenvalue_overlay.png")

    # Spectral gap comparison
    plot_spectral_gap_comparison(
        summaries, compare_out / "spectral_gap_comparison.png"
    )

    # Lambda boxplots
    plot_lambda_distribution_boxplots(
        summaries, compare_out / "lambda_boxplots.png"
    )

    # Scorecard heatmap
    scorecard_df = plot_topology_scorecard(
        summaries, compare_out / "topology_scorecard.png"
    )

    # Pairwise distance matrices (only if >1 dataset)
    if len(summaries) > 1:
        norm_dist_df = build_comparison_matrix(
            summaries, normalised_spectral_distance, "_eigenvalues"
        )
        plot_pairwise_distance_heatmap(
            norm_dist_df,
            "Normalised Spectral Distance (lower = more similar)",
            compare_out / "normalised_spectral_distance.png",
        )
        norm_dist_df.to_csv(compare_out / "normalised_spectral_distance.csv")

        # Subspace similarity requires same F; skip if shapes differ
        shapes = [s["laplacian_shape"][0] for s in summaries]
        if len(set(shapes)) == 1:
            subspace_sim_df = build_comparison_matrix(
                summaries,
                lambda a, b: subspace_similarity(a, b, k=10),
                "_eigenvectors",
            )
            plot_pairwise_distance_heatmap(
                subspace_sim_df,
                "Subspace Similarity — first 10 eigenvectors (higher = more similar)",
                compare_out / "subspace_similarity.png",
            )
            subspace_sim_df.to_csv(compare_out / "subspace_similarity.csv")
        else:
            print(
                f"  ⚠ Skipping subspace similarity: datasets have different F "
                f"({shapes}). Only eigenvalue-based metrics computed."
            )
            subspace_sim_df = pd.DataFrame()

        corr_df = build_comparison_matrix(
            summaries, eigenvalue_correlation, "_eigenvalues"
        )
        corr_df.to_csv(compare_out / "eigenvalue_correlation.csv")
        print(f"✓ Saved eigenvalue correlation matrix")
    else:
        norm_dist_df = pd.DataFrame()
        subspace_sim_df = pd.DataFrame()

    # ── Global summary CSV ───────────────────────────────────────────────────
    scorecard_df.to_csv(compare_out / "scorecard.csv")

    # ── Markdown report ──────────────────────────────────────────────────────
    write_comparison_report(
        summaries, scorecard_df, norm_dist_df, subspace_sim_df, compare_out
    )

    print("\n" + "=" * 70)
    print(f"Done! All outputs written to: {out}")
    print("=" * 70)

    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare ArrowSpace Laplacians across all datasets in a storage directory."
    )
    parser.add_argument(
        "--storage",
        default="storage",
        help="Path to the ArrowSpace storage directory (default: ./storage)",
    )
    parser.add_argument(
        "--output",
        default="comparison_output",
        help="Root output directory for reports and plots (default: ./comparison_output)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of eigenpairs to compute per dataset (default: all). "
             "Set e.g. --k 100 for large matrices.",
    )
    args = parser.parse_args()

    compare_all(
        storage_dir=args.storage,
        output_dir=args.output,
        k_eigenpairs=args.k,
    )
