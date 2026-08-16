# Trimap.jl

*Fast and flexible TriMAP dimensionality reduction in Julia, powered by SimilaritySearch.jl and Lux.jl.*

---

## Overview

[TriMAP](https://github.com/eamid/trimap) (Large-scale Dimensionality Reduction Using Triplets) is a state-of-the-art manifold learning and dimensionality reduction technique. It preserves both **local neighborhood structure** (inliers vs margin outliers) and **global topological structure** (inliers vs random negative samples) through triplet constraints.

`Trimap.jl` provides:
- **Decoupled Metric Search & Optimization**: Receives precomputed $k$-NN matrices (`knns::AbstractMatrix{UInt32}` and `dists::AbstractMatrix{Float32}`) from any search index in [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl), allowing exact search (`ExhaustiveSearch`) or high-throughput approximate search (`SearchGraph`).
- **Support for Arbitrary Metric Spaces**: Embed non-vector data (graphs, strings, sets, histograms, audio signatures) as easily as Euclidean vectors.
- **Representative Negative Sampling**: Landmark selection using Farthest First Traversal (`fft`) and weighted cluster probabilities (`sample_probs` computed from `.nn`).
- **Parametric TriMAP**: Deep neural network embedding model powered by [Lux.jl](https://github.com/LuxDL/Lux.jl), enabling $O(1)$ out-of-sample projection of unseen queries.
- **Pure Julia & GPU/Differentiable**: Powered by `Optimisers.jl` and `Zygote.jl`.

---

## Installation

`Trimap.jl` can be installed using the Julia package manager:

```julia
using Pkg
Pkg.add("Trimap")
```

Or from GitHub:

```julia
Pkg.add(url="https://github.com/sadit/Trimap.jl.git")
```

---

## Quick Start

```julia
using SimilaritySearch, Trimap

# 1. High-dimensional data
X = randn(Float32, 10, 1000) # 10 dimensions, 1000 points
db = MatrixDatabase(X)

# 2. Compute k-nearest neighbors
index = ExhaustiveSearch(Dist.L2(), db)
ctx = GenericContext()
k = 25 # inliers + outliers + margin
knns, dists = allknn(index, ctx, k)

# 3. Fit non-parametric TriMAP (PCA initialization is default)
model = fit(Trimap, X, knns, dists; maxoutdim=2)

# model.embedding is a (2, 1000) Float32 matrix
```

---

## Related Packages

- **[SimSearchManifoldLearning.jl](https://github.com/sadit/SimSearchManifoldLearning.jl)**: Companion package in the [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl) ecosystem focused on scalable manifold learning, non-linear dimensionality reduction, and metric embeddings. It provides complementary algorithms (such as LapMaps, Laplacian Eigenmaps, and UMAP-compatible projections) natively designed to operate on similarity search graphs and arbitrary metric structures.
- **[UMAP.jl](https://github.com/dillondaudert/UMAP.jl)**: A Julia implementation of the Uniform Manifold Approximation and Projection (UMAP) algorithm for dimension reduction.
- **[SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl)**: High-performance similarity search and approximate nearest neighbor index structures in Julia.

---

## API Reference

```@autodocs
Modules = [Trimap]
```
