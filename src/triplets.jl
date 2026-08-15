# This file is a part of Trimap.jl

"""
    generate_triplets(knns::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32};
                      n_inliers=15,
                      n_outliers=5,
                      n_random=5,
                      weight_adj=0.1,
                      rng=Random.default_rng()) -> (triplets_i, triplets_j, triplets_k, weights)

Generates TriMAP triplet constraints `(i, j, k)` where item `j` is closer to `i` than item `k` in the original space.

# Arguments
- `knns`: `(k × n)` matrix of nearest neighbor IDs of type `UInt32` (e.g., from `allknn` or `searchbatch`).
- `dists`: `(k × n)` matrix of corresponding distances of type `Float32`.

# Keyword Arguments
- `n_inliers`: Number of nearest neighbors treated as inliers (local structure). Default: `15`.
- `n_outliers`: Number of further neighbors treated as margin outliers. Default: `5`.
- `n_random`: Number of random negative samples per inlier (global structure). Default: `5`.
- `weight_adj`: Weight adjustment parameter controlling the influence of distance gaps. Default: `0.1`.
- `rng`: Random number generator for negative sampling. Default: `Random.default_rng()`.

# Returns
- `(triplets_i, triplets_j, triplets_k, weights)`: 1-based integer indices (`Int32`) and float weights (`Float32`) for triplet loss.

# Examples

## Exact search with `ExhaustiveSearch`
```julia
using SimilaritySearch, Trimap

# Create dataset and exact search index
X = randn(Float32, 10, 500)
db = MatrixDatabase(X)
index = ExhaustiveSearch(Dist.L2(), db)

# Compute exact all-kNN (e.g., k = 25)
k = 25
ctx = GenericContext()
knns, dists = allknn(index, ctx, k)

# Generate triplets
triplets_i, triplets_j, triplets_k, weights = generate_triplets(knns, dists; n_inliers=10, n_outliers=5, n_random=5)
```

## Approximate search with `SearchGraph` (recall < 1)
```julia
using SimilaritySearch, Trimap

# Create dataset and search graph index
X = randn(Float32, 10, 500)
db = MatrixDatabase(X)
G = SearchGraph(Dist.L2(), db)
ctx = SearchGraphContext()
index!(G, ctx)

# Optimize index for fast approximate search with target recall < 1.0
optimize_index!(G, ctx, MinRecall(0.85))

# Compute approximate all-kNN
k = 25
knns, dists = allknn(G, ctx, k)

# Generate triplets
triplets_i, triplets_j, triplets_k, weights = generate_triplets(knns, dists; n_inliers=10, n_outliers=5, n_random=5)
```
"""
function generate_triplets(
    knns::AbstractMatrix{UInt32},
    dists::AbstractMatrix{Float32};
    n_inliers::Integer=15,
    n_outliers::Integer=5,
    n_random::Integer=5,
    weight_adj::Real=0.1,
    rng::Random.AbstractRNG=Random.default_rng()
)
    size(knns) == size(dists) || throw(DimensionMismatch("knns and dists must have identical dimensions; got $(size(knns)) and $(size(dists))"))
    k_knn, n = size(knns)

    triplets_i = Int32[]
    triplets_j = Int32[]
    triplets_k = Int32[]
    weights = Float32[]

    # Estimate capacity
    n_triplets_est = n * (n_inliers * n_outliers + n_inliers * n_random)
    sizehint!(triplets_i, n_triplets_est)
    sizehint!(triplets_j, n_triplets_est)
    sizehint!(triplets_k, n_triplets_est)
    sizehint!(weights, n_triplets_est)

    for i in 1:n
        # Filter out self and invalid identifiers
        neighbors = Int32[]
        distances = Float32[]
        for row in 1:k_knn
            nid = knns[row, i]
            ndist = dists[row, i]
            if nid != i && nid > 0 && nid <= n
                push!(neighbors, Int32(nid))
                push!(distances, Float32(ndist))
            end
        end

        num_neighbors = length(neighbors)
        n_in = min(n_inliers, num_neighbors)
        n_out = min(n_outliers, max(0, num_neighbors - n_in))

        inliers = view(neighbors, 1:n_in)
        outliers = view(neighbors, (n_in + 1):(n_in + n_out))

        # Weight scale based on distance to nearest neighbor
        scale = length(distances) >= 1 ? max(1f-5, distances[1]) : 1.0f0

        # Type 1: Inliers vs Outliers (local neighborhood boundary)
        for (idx_j, j) in enumerate(inliers)
            d_ij = distances[idx_j]
            w_ij = exp(-Float32(d_ij) / (scale + 1f-5))
            for (idx_k_rel, k) in enumerate(outliers)
                idx_k = n_in + idx_k_rel
                d_ik = distances[idx_k]
                if d_ik > d_ij
                    # Weight adjusted by distance gap
                    w = w_ij * exp(-Float32(weight_adj * (d_ik - d_ij) / (scale + 1f-5)))
                    push!(triplets_i, Int32(i))
                    push!(triplets_j, Int32(j))
                    push!(triplets_k, Int32(k))
                    push!(weights, Float32(w))
                end
            end
        end

        # Type 2: Inliers vs Random negative samples (global structure)
        for (idx_j, j) in enumerate(inliers)
            d_ij = distances[idx_j]
            w_ij = exp(-Float32(d_ij) / (scale + 1f-5))
            for _ in 1:n_random
                k = rand(rng, 1:n)
                if k != i && k != j
                    push!(triplets_i, Int32(i))
                    push!(triplets_j, Int32(j))
                    push!(triplets_k, Int32(k))
                    push!(weights, Float32(w_ij))
                end
            end
        end
    end

    # Normalize weights so mean weight is 1.0
    if !isempty(weights)
        m_w = mean(weights)
        if m_w > 0
            weights ./= m_w
        end
    end

    triplets_i, triplets_j, triplets_k, weights
end
