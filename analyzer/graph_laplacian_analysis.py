import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import seaborn as sns
from scipy.sparse import csr_matrix, coo_matrix, issparse
from scipy.sparse.linalg import eigsh
from scipy.stats import gaussian_kde

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ArrowSpaceEigenAnalyzer:
    """
    Comprehensive analyzer for ArrowSpace Laplacian eigenvectors and eigenvalues.
    Implements analysis as described in the query focusing on topology-eigenvalue relationships.
    """
    
    def __init__(self, storage_dir: str, dataset_name: str):
        """
        Initialize analyzer with ArrowSpace storage directory.
        
        Args:
            storage_dir: Path to storage/ directory
            dataset_name: Dataset identifier (e.g., 'dataset_e20274')
        """
        self.storage_dir = Path(storage_dir)
        self.dataset_name = dataset_name
        self.metadata = {}
        self.data = {}
        
    def load_metadata(self, file_suffix: str) -> Dict:
        """Load JSON metadata for a given artifact."""
        meta_path = self.storage_dir / f"{self.dataset_name}{file_suffix}_metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                return json.load(f)
        return {}
    
    def load_parquet(self, file_suffix: str) -> pd.DataFrame:
        """Load parquet file for a given artifact."""
        parquet_path = self.storage_dir / f"{self.dataset_name}{file_suffix}.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    def load_all_artifacts(self):
        """Load all ArrowSpace artifacts from storage."""
        print(f"Loading ArrowSpace artifacts for dataset: {self.dataset_name}")
        
        # Load metadata for all artifacts
        artifacts = [
            'raw_input',
            'clustered-dm', 
            'gl-matrix',
            'lambdas',
            'laplacian-input'
        ]
        
        for artifact in artifacts:
            try:
                meta = self.load_metadata(f'-{artifact}')
                self.metadata[artifact] = meta
                print(f"✓ Loaded metadata: {artifact}")
            except Exception as e:
                print(f"✗ Could not load metadata for {artifact}: {e}")
        
        # Load data arrays
        try:
            # Raw input vectors (NxF)
            self.data['raw_input'] = self.load_parquet('-raw_input')
            print(f"✓ Loaded raw_input: {self.data['raw_input'].shape}")
            
            # Clustered centroids (CxF) 
            self.data['centroids'] = self.load_parquet('-clustered-dm')
            print(f"✓ Loaded centroids: {self.data['centroids'].shape}")
            
            # Graph Laplacian (FxF) - feature space in CSR format
            self.data['laplacian'] = self.load_parquet('-gl-matrix')
            print(f"✓ Loaded Laplacian (CSR): {self.data['laplacian'].shape}")
            
            # Lambda scores (N,)
            self.data['lambdas'] = self.load_parquet('-lambdas')
            print(f"✓ Loaded lambdas: {self.data['lambdas'].shape}")
            
            # Laplacian input (input to Laplacian computation)
            self.data['laplacian_input'] = self.load_parquet('-laplacian-input')
            print(f"✓ Loaded laplacian_input: {self.data['laplacian_input'].shape}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def extract_laplacian_matrix(self, as_sparse: bool = True):
        """
        Extract Laplacian from ArrowSpace CSR Parquet format.
        
        ArrowSpace stores sparse matrices in COO (Coordinate) format with schema:
        - name_id: identifier
        - n_rows: number of rows
        - n_cols: number of columns  
        - nnz: number of non-zero entries
        - row: row indices
        - col: column indices
        - value: matrix values
        
        Args:
            as_sparse: If True, return scipy.sparse.csr_matrix; else dense np.ndarray
            
        Returns:
            Laplacian matrix (FxF)
        """
        df = self.data['laplacian']
        
        print(f"\nExtracting Laplacian from CSR format...")
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Extract dimensions from first row (all rows should have same metadata)
        n_rows = int(df['n_rows'].iloc[0])
        n_cols = int(df['n_cols'].iloc[0])
        nnz = int(df['nnz'].iloc[0])
        
        print(f"  Matrix dimensions: {n_rows} × {n_cols}")
        print(f"  Non-zero entries: {nnz:,} ({100 * nnz / (n_rows * n_cols):.2f}% dense)")
        
        # Extract triplets (row, col, value)
        row_indices = df['row'].values.astype(np.int32)
        col_indices = df['col'].values.astype(np.int32)
        values = df['value'].values.astype(np.float64)
        
        print(f"  Actual entries loaded: {len(values):,}")
        
        # Validate data
        assert len(row_indices) == len(col_indices) == len(values), \
            "Row, col, and value arrays must have same length"
        assert len(values) == nnz, \
            f"Expected {nnz} entries but got {len(values)}"
        
        # Build sparse matrix in COO format, then convert to CSR
        L_coo = coo_matrix((values, (row_indices, col_indices)), 
                          shape=(n_rows, n_cols))
        L_csr = L_coo.tocsr()
        
        print(f"✓ Built sparse Laplacian: {L_csr.shape}, nnz={L_csr.nnz:,}")
        
        # Check symmetry (Laplacian should be symmetric)
        symmetry_error = np.abs(L_csr - L_csr.T).max()
        print(f"  Symmetry error: {symmetry_error:.2e}")
        
        if as_sparse:
            return L_csr
        else:
            print(f"  Converting to dense array ({n_rows}×{n_cols} = {n_rows*n_cols:,} elements)...")
            L_dense = L_csr.toarray()
            return L_dense
    
    def compute_eigen_decomposition(self, L, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigendecomposition of Laplacian.
        
        Args:
            L: Laplacian matrix (FxF), can be sparse or dense
            k: Number of eigenpairs to compute (None = all)
            
        Returns:
            eigenvalues: (F,) array sorted ascending
            eigenvectors: (F, F) or (F, k) array, columns are eigenvectors
        """
        F = L.shape[0]
        print(f"\nComputing eigendecomposition of {L.shape} Laplacian...")
        
        # Convert to dense if needed for symmetrization
        if issparse(L):
            print(f"  Using sparse eigendecomposition")
            # For sparse matrices, use specialized eigensolver
            
            # Determine number of eigenvalues to compute
            if k is None:
                # For full decomposition, convert to dense
                print(f"  Converting to dense for full eigendecomposition...")
                L_dense = L.toarray()
                L_sym = (L_dense + L_dense.T) / 2
                eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
            else:
                # Use sparse solver for partial decomposition
                k_actual = min(k, F - 2)  # eigsh requires k < n-1
                print(f"  Computing {k_actual} smallest eigenvalues...")
                
                # Symmetrize sparse matrix
                L_sym = (L + L.T) / 2
                
                try:
                    eigenvalues, eigenvectors = eigsh(L_sym, k=k_actual, which='SM')
                except Exception as e:
                    print(f"  Sparse solver failed: {e}")
                    print(f"  Falling back to dense solver...")
                    L_dense = L.toarray()
                    L_sym = (L_dense + L_dense.T) / 2
                    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
                    eigenvalues = eigenvalues[:k_actual]
                    eigenvectors = eigenvectors[:, :k_actual]
        else:
            # Dense matrix
            L_sym = (L + L.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
            
            if k is not None:
                eigenvalues = eigenvalues[:k]
                eigenvectors = eigenvectors[:, :k]
        
        # Sort by eigenvalue magnitude (ascending)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        print(f"✓ Computed {len(eigenvalues)} eigenpairs")
        print(f"  Eigenvalue range: [{eigenvalues[0]:.6f}, {eigenvalues[-1]:.6f}]")
        if len(eigenvalues) > 1:
            print(f"  Spectral gap (λ₁ - λ₀): {eigenvalues[1] - eigenvalues[0]:.6f}")
        
        return eigenvalues, eigenvectors
    
    def analyze_eigenvalue_distribution(self, eigenvalues: np.ndarray) -> Dict:
        """
        Analyze eigenvalue distribution to characterize manifold structure.
        
        Returns statistics about spectral gaps, clustering, etc.
        """
        stats = {
            'n_eigenvalues': len(eigenvalues),
            'min': float(eigenvalues[0]),
            'max': float(eigenvalues[-1]),
            'mean': float(np.mean(eigenvalues)),
            'median': float(np.median(eigenvalues)),
            'std': float(np.std(eigenvalues)),
        }
        
        # Spectral gaps (differences between consecutive eigenvalues)
        gaps = np.diff(eigenvalues)
        stats['spectral_gaps'] = {
            'mean': float(np.mean(gaps)),
            'max': float(np.max(gaps)),
            'max_gap_index': int(np.argmax(gaps)),
            'top_5_gaps': [float(x) for x in np.sort(gaps)[-5:][::-1]]
        }
        
        # Multiplicity analysis (near-zero eigenvalues)
        zero_threshold = 1e-10
        stats['near_zero_eigenvalues'] = int(np.sum(np.abs(eigenvalues) < zero_threshold))
        
        # Effective dimensionality (von Neumann entropy)
        eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues_pos) > 0:
            p = eigenvalues_pos / np.sum(eigenvalues_pos)
            entropy = -np.sum(p * np.log(p + 1e-10))
            stats['von_neumann_entropy'] = float(entropy)
            stats['effective_dimension'] = float(np.exp(entropy))
        else:
            stats['von_neumann_entropy'] = 0.0
            stats['effective_dimension'] = 0.0
        
        return stats
    
    def characterize_topology(self, eigenvalues: np.ndarray) -> str:
        """
        Characterize topology based on eigenvalue distribution.
        
        Returns:
            topology_type: 'clustered', 'smooth_manifold', or 'noisy_uniform'
        """
        if len(eigenvalues) < 2:
            return 'insufficient_data'
        
        gaps = np.diff(eigenvalues)
        max_gap = np.max(gaps)
        mean_gap = np.mean(gaps)
        
        # Normalized gap measure
        gap_ratio = max_gap / (mean_gap + 1e-10)
        
        # Check for large spectral gap (indicates clustering)
        if gap_ratio > 3.0 and np.argmax(gaps) < len(gaps) // 3:
            return 'clustered'
        
        # Check for smooth spectrum (manifold)
        gap_variance = np.var(gaps)
        if gap_variance < mean_gap:
            return 'smooth_manifold'
        
        # Otherwise likely noisy/uniform
        return 'noisy_uniform'
    
    def compute_curvature_measures(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> Dict:
        """
        Compute curvature-related measures from spectral data.
        
        The eigenvalue distribution reflects manifold curvature:
        - Small eigenvalues → low curvature (flat regions)
        - Large eigenvalues → high curvature (curved regions)
        """
        measures = {}
        
        # Mean curvature (average eigenvalue excluding smallest)
        if len(eigenvalues) > 1:
            measures['mean_curvature'] = float(np.mean(eigenvalues[1:]))
        else:
            measures['mean_curvature'] = 0.0
        
        # Gaussian curvature estimate (product of smallest non-zero eigenvalues)
        nonzero_eigs = eigenvalues[eigenvalues > 1e-10]
        if len(nonzero_eigs) >= 2:
            measures['gaussian_curvature_estimate'] = float(nonzero_eigs[0] * nonzero_eigs[1])
        else:
            measures['gaussian_curvature_estimate'] = 0.0
        
        # Curvature concentration (ratio of max to mean eigenvalue)
        if measures['mean_curvature'] > 1e-10:
            measures['curvature_concentration'] = float(eigenvalues[-1] / measures['mean_curvature'])
        else:
            measures['curvature_concentration'] = 0.0
        
        return measures
    
    def plot_eigenvalue_spectrum(self, eigenvalues: np.ndarray, save_path: str = 'eigenvalue_spectrum.png'):
        """Plot eigenvalue spectrum with gap visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Eigenvalue plot
        ax = axes[0, 0]
        ax.plot(eigenvalues, 'o-', markersize=3, linewidth=1)
        ax.set_xlabel('Index k')
        ax.set_ylabel('Eigenvalue λₖ')
        ax.set_title('Eigenvalue Spectrum')
        ax.grid(True, alpha=0.3)
        
        # 2. Log-scale eigenvalue plot
        ax = axes[0, 1]
        eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues_pos) > 0:
            ax.semilogy(eigenvalues_pos, 'o-', markersize=3, linewidth=1)
            ax.set_xlabel('Index k')
            ax.set_ylabel('log(λₖ)')
            ax.set_title('Eigenvalue Spectrum (Log Scale)')
            ax.grid(True, alpha=0.3)
        
        # 3. Spectral gaps
        ax = axes[1, 0]
        if len(eigenvalues) > 1:
            gaps = np.diff(eigenvalues)
            ax.plot(gaps, 'o-', markersize=3, linewidth=1, color='red')
            ax.axhline(np.mean(gaps), color='black', linestyle='--', label=f'Mean gap: {np.mean(gaps):.6f}')
            max_gap_idx = np.argmax(gaps)
            ax.axvline(max_gap_idx, color='green', linestyle='--', label=f'Max gap at k={max_gap_idx}')
            ax.set_xlabel('Gap index')
            ax.set_ylabel('Gap (λₖ₊₁ - λₖ)')
            ax.set_title('Spectral Gaps')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Cumulative eigenvalue distribution
        ax = axes[1, 1]
        cumsum = np.cumsum(eigenvalues)
        cumsum_norm = cumsum / cumsum[-1] if cumsum[-1] > 0 else cumsum
        ax.plot(cumsum_norm, linewidth=2)
        ax.axhline(0.9, color='red', linestyle='--', label='90% threshold')
        idx_90 = np.searchsorted(cumsum_norm, 0.9)
        ax.axvline(idx_90, color='green', linestyle='--', label=f'k₉₀ = {idx_90}')
        ax.set_xlabel('Number of eigenvalues k')
        ax.set_ylabel('Cumulative proportion')
        ax.set_title('Cumulative Eigenvalue Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved eigenvalue spectrum plot: {save_path}")
        plt.close()
    
    def plot_spectral_embedding(self, eigenvectors: np.ndarray, eigenvalues: np.ndarray,
                                centroids: np.ndarray = None, save_path: str = 'spectral_embedding.png'):
        """
        Plot 2D/3D spectral embedding using first non-trivial eigenvectors.
        
        Eigenvector coordinates encode geodesic distances on the manifold.
        """
        if eigenvectors.shape[1] < 3:
            print("⚠ Need at least 3 eigenvectors for embedding visualization")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Use eigenvectors 1 and 2 (skip 0th which is constant for connected graphs)
        v1 = eigenvectors[:, 1]
        v2 = eigenvectors[:, 2]
        
        # Number of features (rows in eigenvector matrix)
        n_features = len(v1)
        
        # Color by feature index
        colors = np.arange(n_features)
        
        # 2D embedding
        ax = axes[0]
        scatter = ax.scatter(v1, v2, c=colors, cmap='viridis', s=50, alpha=0.7)
        ax.set_xlabel('v₁ (Eigenvector 1)')
        ax.set_ylabel('v₂ (Eigenvector 2)')
        ax.set_title('Spectral Embedding (First 2 Eigenvectors)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Feature Index')
        
        # Eigenvalue-weighted scatter
        ax = axes[1]
        
        # Compute feature importance as the L2 norm across the first few eigenvectors
        # This represents how much each feature contributes to the spectral embedding
        n_components = min(10, eigenvectors.shape[1])
        feature_importance = np.linalg.norm(eigenvectors[:, :n_components], axis=1)
        
        # Normalize to [20, 220] range for marker sizes
        if feature_importance.max() > feature_importance.min():
            sizes = 20 + 200 * (feature_importance - feature_importance.min()) / \
                    (feature_importance.max() - feature_importance.min())
        else:
            sizes = 50 * np.ones(n_features)
        
        scatter = ax.scatter(v1, v2, c=colors, s=sizes, cmap='plasma', alpha=0.6)
        ax.set_xlabel('v₁ (Eigenvector 1)')
        ax.set_ylabel('v₂ (Eigenvector 2)')  
        ax.set_title('Spectral Embedding (Size = Feature Importance)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Feature Index')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved spectral embedding plot: {save_path}")
        plt.close()
    
    def plot_curvature_distribution(self, eigenvalues: np.ndarray, 
                                   save_path: str = 'curvature_distribution.png'):
        """
        Visualize curvature distribution across the manifold.
        
        Eigenvalues encode local curvature: λ ∝ curvature in diffusion geometry.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Eigenvalue histogram (curvature distribution)
        ax = axes[0, 0]
        ax.hist(eigenvalues, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Eigenvalue λ (curvature proxy)')
        ax.set_ylabel('Frequency')
        ax.set_title('Curvature Distribution (Eigenvalue Histogram)')
        ax.grid(True, alpha=0.3)
        
        # 2. Kernel Density Estimate
        ax = axes[0, 1]
        if len(eigenvalues) > 1:
            kde = gaussian_kde(eigenvalues)
            x_range = np.linspace(eigenvalues.min(), eigenvalues.max(), 500)
            ax.plot(x_range, kde(x_range), linewidth=2, color='darkred')
            ax.fill_between(x_range, kde(x_range), alpha=0.3, color='red')
            ax.set_xlabel('Eigenvalue λ')
            ax.set_ylabel('Density')
            ax.set_title('Curvature Density (KDE)')
            ax.grid(True, alpha=0.3)
        
        # 3. Quantile plot
        ax = axes[1, 0]
        quantiles = np.percentile(eigenvalues, np.linspace(0, 100, 20))
        ax.plot(np.linspace(0, 100, 20), quantiles, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Quantile (%)')
        ax.set_ylabel('Eigenvalue λ')
        ax.set_title('Eigenvalue Quantiles')
        ax.grid(True, alpha=0.3)
        
        # 4. Box plot with outliers
        ax = axes[1, 1]
        ax.boxplot(eigenvalues, vert=True, patch_artist=True,
                  boxprops=dict(facecolor='lightblue'),
                  medianprops=dict(color='red', linewidth=2))
        ax.set_ylabel('Eigenvalue λ')
        ax.set_title('Eigenvalue Distribution (Box Plot)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved curvature distribution plot: {save_path}")
        plt.close()
    
    def analyze_subspace_characteristics(self, eigenvectors: np.ndarray, 
                                        eigenvalues: np.ndarray, 
                                        n_subspaces: int = 5) -> Dict:
        """
        Characterize spectral subspaces by eigenvalue ranges.
        
        Different eigenvalue ranges correspond to different frequency scales:
        - Low eigenvalues: smooth, global structure
        - High eigenvalues: rough, local structure
        """
        F = len(eigenvalues)
        subspace_size = F // n_subspaces
        
        subspaces = {}
        for i in range(n_subspaces):
            start = i * subspace_size
            end = (i + 1) * subspace_size if i < n_subspaces - 1 else F
            
            if start >= len(eigenvalues):
                break
            
            eigs_sub = eigenvalues[start:end]
            vecs_sub = eigenvectors[:, start:end]
            
            subspaces[f'subspace_{i}'] = {
                'range': (int(start), int(end)),
                'eigenvalue_range': (float(eigs_sub.min()), float(eigs_sub.max())),
                'mean_eigenvalue': float(np.mean(eigs_sub)),
                'n_eigenvectors': int(vecs_sub.shape[1]),
                'frequency_scale': 'low' if i == 0 else ('high' if i == n_subspaces-1 else 'medium')
            }
        
        return subspaces
    
    def generate_full_report(self, output_dir: str = '.', k_eigenpairs: int = None):
        """
        Generate complete analysis report with all plots and statistics.
        
        Args:
            output_dir: Directory to save outputs
            k_eigenpairs: Number of eigenpairs to compute (None = all, recommended for large matrices)
        """
        output_dir = Path(__file__).parent / Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("ArrowSpace Eigenanalysis Report")
        print("="*70)
        
        # Extract Laplacian (keep as sparse)
        L = self.extract_laplacian_matrix(as_sparse=True)
        print(f"\nLaplacian shape: {L.shape}")
        print(f"Laplacian sparsity: {100 * (1 - L.nnz / (L.shape[0] * L.shape[1])):.2f}%")
        
        # Eigendecomposition
        eigenvalues, eigenvectors = self.compute_eigen_decomposition(L, k=k_eigenpairs)
        
        # Topology characterization
        topology = self.characterize_topology(eigenvalues)
        print(f"\nTopology type: {topology.upper()}")
        
        # Eigenvalue statistics
        print("\n" + "-"*70)
        print("Eigenvalue Statistics")
        print("-"*70)
        stats = self.analyze_eigenvalue_distribution(eigenvalues)
        for key, value in stats.items():
            if key != 'spectral_gaps':
                print(f"  {key}: {value}")
        print("\n  Spectral gaps:")
        for key, value in stats['spectral_gaps'].items():
            print(f"    {key}: {value}")
        
        # Curvature measures
        print("\n" + "-"*70)
        print("Curvature Measures")
        print("-"*70)
        curvature = self.compute_curvature_measures(eigenvalues, eigenvectors)
        for key, value in curvature.items():
            print(f"  {key}: {value:.6f}")
        
        # Subspace analysis
        print("\n" + "-"*70)
        print("Spectral Subspace Characterization")
        print("-"*70)
        subspaces = self.analyze_subspace_characteristics(eigenvectors, eigenvalues)
        for name, props in subspaces.items():
            print(f"\n  {name}:")
            for key, value in props.items():
                print(f"    {key}: {value}")
        
        # Generate plots
        print("\n" + "-"*70)
        print("Generating Visualizations")
        print("-"*70)
        self.plot_eigenvalue_spectrum(eigenvalues, output_dir / 'eigenvalue_spectrum.png')
        self.plot_spectral_embedding(eigenvectors, eigenvalues, save_path=output_dir / 'spectral_embedding.png')
        self.plot_curvature_distribution(eigenvalues, output_dir / 'curvature_distribution.png')
        
        # Save numerical results to CSV
        results_df = pd.DataFrame({
            'index': np.arange(len(eigenvalues)),
            'eigenvalue': eigenvalues,
            'spectral_gap': np.concatenate([np.diff(eigenvalues), [0]])
        })
        results_path = output_dir / 'eigenanalysis_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"✓ Saved numerical results: {results_path}")
        
        # Save summary report
        summary = {
            'dataset': self.dataset_name,
            'laplacian_shape': list(L.shape),
            'laplacian_nnz': int(L.nnz),
            'topology_type': topology,
            'statistics': stats,
            'curvature_measures': curvature,
            'subspaces': subspaces
        }
        
        summary_path = output_dir / 'eigenanalysis_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary report: {summary_path}")
        
        print("\n" + "="*70)
        print("Analysis Complete!")
        print("="*70)
        
        return summary


# Example usage (commented out - will be activated when user provides files)
# analyzer = ArrowSpaceEigenAnalyzer('storage', 'dataset_e20274')
# analyzer.load_all_artifacts()
# summary = analyzer.generate_full_report(output_dir='eigenanalysis_output')

print("✓ ArrowSpaceEigenAnalyzer class defined successfully")
print("\nTo use:")
print("  analyzer = ArrowSpaceEigenAnalyzer('storage', 'dataset_e20274')")
print("  analyzer.load_all_artifacts()")
print("  summary = analyzer.generate_full_report(output_dir='output')")

analyzer = ArrowSpaceEigenAnalyzer('storage', 'dataset_e20274')
analyzer.load_all_artifacts()
summary = analyzer.generate_full_report(output_dir='output')