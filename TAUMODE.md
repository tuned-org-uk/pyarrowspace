# current computation of the synthetic tau index

* how the index is computed
* how the rayleigh energy is incorporated
* what is the role of e_raw and e_bounded, g_raw and g_clamped
* what is the role of cosine similarity

The synthetic index for a query vector x is computed as $S(x)=\tau\,E'(x)+(1-\tau)\,G(x)$, where $E'(x)=\frac{E(x)}{E(x)+\tau}$, $E(x)=\frac{x^\top L x}{x^\top x}$, and $G(x)$ is a clamped dispersion built from edgewise Dirichlet energy shares on the Laplacian corresponding to the ArrowSpace signals matrix. Cosine similarity does not appear in this per-vector synthetic index computation; it is used downstream at query time as the semantic term blended with lambda proximity in ArrowSpace search utilities.[^11][^12][^13]

### Index computation

- Given a feature-space Laplacian $L\in\mathbb{R}^{F\times F}$ (stored in ArrowSpace as signals), the synthetic index for an item vector $x\in\mathbb{R}^F$ is defined as \$ S(x)=\tau\,E'(x)+(1-\tau)\,G(x), \$ with $\tau>0$ supplied externally in this function and reused both as the bounding scale and as the mixing weight in the convex combination.[^12][^11]
- The implementation ensures $x$ is nonzero (panics on all-zeros) and computes each term deterministically from $x$ and $L$ without invoking cosine similarity in this path.[^11]


### Rayleigh energy incorporation

- The raw Rayleigh energy is $E(x)=\frac{x^\top L x}{x^\top x}$, which measures the Dirichlet energy (spectral roughness) of $x$ over the graph induced by $L$ and is nonnegative for Laplacians; this is computed as e_raw in code via a dense matrix Rayleigh quotient routine.[^12][^11]
- A bounded transform $E'(x)=\frac{E(x)}{E(x)+\tau}$ compresses the scale of the raw energy into $(0,1)$ to stabilize heavy tails and make values comparable across items; this is computed as e_bounded in code using the provided $\tau$.[^11]


### Roles of e_raw, e_bounded, g_raw, g_clamped

- e_raw is the unbounded Rayleigh energy $E(x)=\frac{x^\top L x}{x^\top x}$ and carries the primary spectral smoothness/roughness signal from the graph spectrum.[^12][^11]
- e_bounded is the normalized energy $E'(x)=\frac{e_{raw}}{e_{raw}+\tau}$, mapping the positive real line to $(0,1)$ to prevent domination by large energies when mixing with dispersion and to keep the index bounded.[^11]
- g_raw is the dispersion term computed from edgewise Dirichlet shares, using the identity $x^\top L x=\sum_{(i,j)}w_{ij}(x_i-x_j)^2$ with $w_{ij}=-L_{ij}\ge 0$, and is implemented as a Gini-like concentration $\sum s_{ij}^2$ where $s_{ij}=\frac{w_{ij}(x_i-x_j)^2}{\sum_{u,v}w_{uv}(x_u-x_v)^2}$ if the denominator is nonzero.[^11]
- g_clamped is simply $\operatorname{clip}(g_{raw},0,1)$ to ensure the dispersion contribution stays within a stable numeric range before blending into $S(x)$.[^11]


### Role of cosine similarity

- Cosine similarity is not used inside this synthetic index computation; the function builds $S(x)$ only from the Rayleigh energy with respect to the feature Laplacian and the dispersion of edgewise energies, then blends them with $\tau$ as $S(x)=\tau\,E'(x)+(1-\tau)\,G(x)$.[^11]
- Cosine similarity is used later during retrieval by ArrowSpace’s lambda-aware search, which combines semantic similarity with spectral proximity as \$ score(a,b)=\alpha\,\cos(a,b)+\beta\,\frac{1}{1+|\lambda_a-\lambda_b|} \$, where $\lambda$ can be the stored synthetic index forming the spectral term; this separates concerns between constructing $S$ and using it in search [^13].
<span style="display:none">[^1][^10][^2][^3][^4][^5][^6][^7][^8][^9]</span>

```
<div style="text-align: center">⁂</div>
```

[^1]: https://crates.io/crates/arrowspace

[^2]: https://www.youtube.com/watch?v=EqgT69iZ_As

[^3]: https://aws.amazon.com/blogs/compute/optimizing-aws-lambda-extensions-in-c-and-rust/

[^4]: https://fourtheorem.com/wp-content/uploads/2024/06/fourTheorem-Rust-Lambda.pdf

[^5]: https://www.youtube.com/watch?v=Mdh_2PXe9i8

[^6]: https://gist.github.com/kvark/f067ba974446f7c5ce5bd544fe370186

[^7]: https://www.awholenother.com/2022/11/27/aws-lambda-functions-in-rust-the-easy-way.html

[^8]: https://www.buoyantdata.com/blog/2025-04-22-rust-is-good-for-the-climate.html

[^9]: https://users.rust-lang.org/t/using-rust-in-aws-lambda/66218

[^10]: https://serverlessland.com/repos/lambda-extension-optimization