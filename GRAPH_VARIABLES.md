# Variables involved in the creation of the arrowspace

ArrowSpace’s λτ-graph is controlled by four parameters: eps (distance cutoff), k (max neighbors), p (kernel sharpness), and sigma (scale in the weight function); in this codebase, distances are built from rectified cosine, weights are 1/(1+(distance/σ)^p), and defaults favor a sparse, stable graph with eps≈1e-3, k≈6, p=2.0, and σ defaulting to eps, so tuning these changes sparsity, stability, and spectral contrast directly. The pipeline pre-normalizes item vectors to unit norm before graph building, so magnitude information in embeddings is intentionally suppressed; this improves cosine comparability but can flatten spectral variation if important cues live in vector norms rather than angular differences, which aligns with the observation about normalization “flattening” spectral information.[^1][^2][^3]

### Parameter roles

- eps: Maximum allowed cosine distance per edge; distance is defined as d(i,j)=1−max(0,cos(x_i,x_j)), so edges are kept only if d≤eps prior to capping by k, making eps the primary sparsity/coverage control for candidate neighbors.[^2]
- k: Per-node cap on neighbors (k-NN via a smallest-distance heap); after thresholding by eps, only the closest up to k are retained, symmetrized, and converted to a Laplacian, so larger k densifies the graph and increases compute and memory costs.[^2]
- p: Exponent in the edge weight kernel w=1/(1+(d/σ)^p), controlling how sharply weights decay with distance; higher p makes the kernel more “hard-edged,” lower p makes it smoother and more tolerant to distance.[^2]
- sigma: Scale parameter in the weight; default is σ=eps (with a small floor), so by default the soft-decay “knee” aligns to the distance threshold, stabilizing weights near the cutoff and keeping behavior predictable across datasets.[^1][^2]


### Defaults and feasible ranges

- Defaults: eps≈1e-3, k≈6, p=2.0, sigma=None→σ:=eps; these are intended to keep the λ-graph connected but sparse and make the kernel near-quadratic around small gaps to avoid overpowering cosine similarities downstream.[^1]
- Feasible eps: Since d∈ due to max(0,cos), eps∈ is meaningful; values ≥1 connect almost all pairs with nonnegative cosine, while values ≪1 produce very sparse graphs that can fragment components if too small.[^4][^2]
- Feasible k: Any positive integer up to N−1; typical 3–10 balances spectral stability with cost, while very large k quickly increases density of L and the cost to build both item-wise and the later F×F feature Laplacian in GraphFactory::build_spectral_laplacian.[^3][^1]
- Feasible p: Positive values; p≈1 (linear-ish) yields gentle decay, p≈2 (default) gives quadratic decay, and p≫2 approaches a near-binary weighting within ε, increasing sensitivity to small distance changes around σ.[^2]
- Feasible σ: Positive scale; default σ=eps ensures edges near the cutoff retain moderate weights, whereas σ≪eps strongly down-weights anything but very tight neighbors and σ≫eps flattens weight variation within the allowed radius.[^2]


### Effects by parameter

- eps small: Very sparse candidate set; stabilizes against spurious edges but risks disconnected subgraphs and underestimation of degrees, which can distort Rayleigh energies and synthetic indices.[^2]
- eps large: Many candidates pass threshold; after k-capping the graph still densifies due to symmetrization and higher degrees, increasing Laplacian density and Rayleigh compute cost later in the pipeline.[^3][^2]
- k small: Enforces sparsity even if eps is large; good for speed and memory but can suppress important mid-range relationships if embeddings are noisy or anisotropic.[^2]
- k large: Improves connectivity and can smooth spectral estimates but increases O(N·k) edges, making Laplacian building and especially the F×F feature Laplacian step heavier in time and memory.[^3][^2]
- p small (≈1): Smooth weight curve tolerant to modest distance variation, more robust under quantization or small angular differences, but less contrastive in separating close vs. moderately close pairs.[^2]
- p large (≥3): Contrastive and selective near σ, helpful when small angular differences matter, but can make weights brittle to tiny distance changes and amplify measurement noise.[^2]
- σ relative to eps: With σ=eps (default), the soft-decay aligns with the selection radius; σ<eps turns the kernel sharper inside the allowed radius, σ>eps makes weights flatter so the eps cutoff dominates edge selection.[^2]


### Notes on embedding scale and precision

- Pre-normalization: Items are normalized to unit norm before computing cosine distances, so magnitude information is intentionally dropped and only angles drive edge formation, which can “flatten” spectral information if norms encode relevant variability in the domain.[^2]
- eps vs. precision: Because distance is 1−cos_+, embeddings with coarse precision (e.g., ≈1e−2) can produce quantized cosines; choosing eps on the same order (e.g., 1e−2 to 1e−1) admits near-duplicates that would otherwise be excluded by quantization effects, matching the empirical guidance noted.[^2]
- Negative cosine handling: The rectification max(0,cos) maps negative cosine to zero similarity (distance 1), so eps cannot “rescue” anti-correlated pairs; this focuses the graph on nonnegative angular relationships by design.[^2]


### Practical tuning recipes

- Start point: eps≈1e−3, k≈6, p=2.0, σ:=eps, as set by the builder defaults for general-purpose sparse but connected graphs that don’t overpower cosine in downstream scoring.[^1]
- Low-precision embeddings: Increase eps toward the embedding quantization scale and keep p≈2 to avoid over-penalizing small distance steps caused by rounding; σ:=eps keeps the knee aligned to the selection radius.[^1][^2]
- Heavy graphs or large F: Reduce k (and/or eps) to bound Laplacian density before the F×F spectral step in GraphFactory::build_spectral_laplacian to keep FXF construction tractable on high-dimensional data.[^3]
- Need stronger selectivity: Increase p (e.g., to 3) and/or reduce σ below eps to sharpen the kernel inside the selection radius without changing the candidate set defined by eps and k.[^2]
- Preserve magnitude cues: The current pipeline normalizes to unit norm; if norms encode meaning in a domain, consider removing or gating normalization in the Laplacian builder to reintroduce scale, acknowledging this deviates from the present implementation.[^2]


### Downstream context

- After the Laplacian is built, ArrowSpace uses it to compute spectral signals and synthetic indices, while query-time retrieval blends semantic (cosine) similarity with a spectral proximity term, so graph sparsity/weights set by eps, k, p, σ directly shape the spectral half of the scoring.[^4][^3]