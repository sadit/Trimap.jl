# Trimap.jl

[![CI](https://github.com/sadit/Trimap.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/sadit/Trimap.jl/actions/workflows/CI.yml)
[![Docs (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://sadit.github.io/Trimap.jl/stable/)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://sadit.github.io/Trimap.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Trimap.jl** is a fast, flexible, and differentiable implementation of the [TriMAP](https://github.com/eamid/trimap) dimensionality reduction technique in Julia, powered by [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl) and [Lux.jl](https://github.com/LuxDL/Lux.jl).

---

## 📖 Documentation & Tutorials

Read the full documentation and interactive tutorials at:
👉 **[https://sadit.github.io/Trimap.jl/](https://sadit.github.io/Trimap.jl/)**

Tutorials include:
1. **[Tabular Data - Fisher's Iris](https://sadit.github.io/Trimap.jl/dev/tutorial_iris/)**: Exact nearest neighbor search and PCA initialization.
2. **[Non-linear Manifolds - Two Moons & Spirals](https://sadit.github.io/Trimap.jl/dev/tutorial_moons_spirals/)**: Approximate $k$-NN search with `SearchGraph` (`MinRecall`) and landmark negative sampling with `fft` + cluster weighting (`sample_probs`).
3. **[General Metric Spaces - Prime Factorization](https://sadit.github.io/Trimap.jl/dev/tutorial_prime_factors/)**: Embedding non-vector arithmetic sets with `Dist.Sets.Jaccard()` and out-of-sample projection with `ParametricTrimap`.

---

## 🚀 Key Features

- **Decoupled Search & Optimization**: Receives precomputed $k$-NN matrices (`knns::AbstractMatrix{UInt32}` and `dists::AbstractMatrix{Float32}`) from any search index in [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl) (e.g., exact `ExhaustiveSearch` or high-throughput `SearchGraph`).
- **Support for Arbitrary Metric Spaces**: Embed vectors, strings, sets, graphs, or histograms into 2D/3D Euclidean coordinates.
- **Representative Negative Sampling**: Landmark selection using Farthest First Traversal (`fft`) and weighted cluster probabilities (`sample_probs` computed from `res.nn`).
- **Parametric TriMAP**: Deep neural network embedding model powered by [Lux.jl](https://github.com/LuxDL/Lux.jl), allowing instant $O(1)$ projection of unseen queries (`predict`).
- **Differentiable & GPU-ready**: Pure Julia implementation using `Optimisers.jl` and `Zygote.jl`.

---

## 📦 Installation

Install `Trimap.jl` via the Julia package manager:

```julia
using Pkg
Pkg.add("Trimap")
```

Or install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/sadit/Trimap.jl.git")
```

---

## ⚡ Quick Start

### 1. Non-Parametric TriMAP (`Trimap`)

```julia
using SimilaritySearch, Trimap

# 1. Dataset (dim × n)
X = randn(Float32, 10, 1000)
db = MatrixDatabase(X)

# 2. Compute k-nearest neighbors
index = ExhaustiveSearch(Dist.L2(), db)
ctx = GenericContext()
k = 25 # inliers + margin outliers
knns, dists = allknn(index, ctx, k)

# 3. Fit TriMAP (PCA initialization is automatic)
model = fit(Trimap, X, knns, dists; maxoutdim=2, n_epochs=400)

# 4. Access 2D embedding coordinates
Y = model.embedding # (2, 1000) Matrix{Float32}
```

### 2. Landmark Negative Sampling with `fft`

```julia
# Select diverse landmark centers using Farthest First Traversal
res_fft = fft(Dist.L2(), db, 50)
sample_centers = res_fft.centers

# Compute cluster size weights from res_fft.nn
counts = Dict{UInt32, Float32}()
for c in res_fft.nn
    counts[c] = get(counts, c, 0f0) + 1f0
end
sample_probs = Float32[get(counts, c, 0f0) / length(res_fft.nn) for c in sample_centers]

# Fit with weighted negative landmarks
model = fit(Trimap, X, knns, dists; sample=sample_centers, sample_probs=sample_probs, maxoutdim=2)
```

### 3. Parametric TriMAP (`ParametricTrimap`)

```julia
# Train neural network (Lux) to learn the embedding mapping
pmodel = fit(
    ParametricTrimap,
    X,
    knns,
    dists;
    maxoutdim=2,
    hidden_dims=(128, 64),
    n_epochs=400
)

# Project new unseen data points in O(1) time
X_new = randn(Float32, 10, 100)
Y_new = predict(pmodel, X_new) # (2, 100) Matrix{Float32}
```

---

## 🔗 Related Packages

- **[SimSearchManifoldLearning.jl](https://github.com/sadit/SimSearchManifoldLearning.jl)**: Companion package in the [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl) ecosystem. It implements UMAP natively for similarity search graphs and serves as a bridge to seamlessly integrate `SimilaritySearch.jl` indices with the diverse manifold learning algorithms in [ManifoldLearning.jl](https://github.com/wildart/ManifoldLearning.jl) (such as Isomap, LLE, Laplacian Eigenmaps, and Hessian LLE).
- **[UMAP.jl](https://github.com/dillondaudert/UMAP.jl)**: A Julia implementation of the Uniform Manifold Approximation and Projection (UMAP) algorithm for dimension reduction.
- **[ManifoldLearning.jl](https://github.com/wildart/ManifoldLearning.jl)**: A Julia package for non-linear dimensionality reduction algorithms.
- **[SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl)**: High-performance similarity search and approximate nearest neighbor index structures in Julia.

---

## 📚 Reference

- Amid, E., & Warmuth, M. K. (2019). *TriMAP: Large-scale Dimensionality Reduction Using Triplets*. arXiv preprint [arXiv:1910.00204](https://arxiv.org/abs/1910.00204).
- Tellez, E. S., et al. [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl): Efficient similarity search and metric index structures in Julia.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

