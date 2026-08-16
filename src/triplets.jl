# This file is a part of Trimap.jl

"""
    generate_triplets(knns::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32};
                      sample=nothing,
                      sample_probs=nothing,
                      n_inliers=nothing,
                      n_outliers=nothing,
                      n_random=nothing,
                      weight_adj=0.1,
                      rng=Random.default_rng()) -> (triplets_i, triplets_j, triplets_k, weights)

Generates TriMAP triplet constraints `(i, j, k)` where item `j` is closer to `i` than item `k` in the original space.

# Arguments
- `knns`: `(k × n)` matrix of nearest neighbor IDs of type `UInt32` (e.g., from `allknn` or `searchbatch`).
- `dists`: `(k × n)` matrix of corresponding distances of type `Float32`.

# Keyword Arguments
- `sample`: Optional subset of sample points / landmarks from which negative samples are drawn (e.g., `Vector{UInt32}` from `fft(dist, db, k).centers` or `AbstractMatrix{Float32}`). If `nothing` (default), negative samples are drawn from `1:n`.
- `sample_probs`: Optional vector of sampling probabilities or weights corresponding to each element in `sample` (or `1:n`). If `nothing` (default), uniform sampling is used.
- `n_inliers`: Number of nearest neighbors treated as inliers (local structure). Default: `round(Int, 3k/4)`.
- `n_outliers`: Number of further neighbors treated as margin outliers. Default: `round(Int, k/4)`.
- `n_random`: Number of random negative samples per inlier (global structure). Default: equal to `n_outliers`.
- `weight_adj`: Weight adjustment parameter controlling the influence of distance gaps. Default: `0.1`.
- `rng`: Random number generator for negative sampling. Default: `Random.default_rng()`.

# Returns
- `(triplets_i, triplets_j, triplets_k, weights)`: 1-based integer indices (`Int32`) and float weights (`Float32`) for triplet loss.

# Examples

## Exact search with `ExhaustiveSearch` and `fft` negative sampling with relative cluster probabilities
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

# Extract diverse landmarks using Farthest First Traversal (fft)
res_fft = fft(Dist.L2(), db, 50)
sample_centers = res_fft.centers

# Count how many points choose each center in res_fft.nn to compute relative probabilities
counts = Dict{UInt32, Float32}()
for c in res_fft.nn
    counts[c] = get(counts, c, 0f0) + 1f0
end
sample_probs = Float32[get(counts, c, 0f0) / length(res_fft.nn) for c in sample_centers]

# Generate triplets using the weighted fft sample
triplets_i, triplets_j, triplets_k, weights = generate_triplets(
    knns, dists;
    sample=sample_centers,
    sample_probs=sample_probs,
    n_inliers=10,
    n_outliers=5,
    n_random=5
)
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
    sample=nothing,
    sample_probs=nothing,
    n_inliers::Union{Nothing, Integer}=nothing,
    n_outliers::Union{Nothing, Integer}=nothing,
    n_random::Union{Nothing, Integer}=nothing,
    weight_adj::Real=0.1,
    rng::Random.AbstractRNG=Random.default_rng()
)
    size(knns) == size(dists) || throw(DimensionMismatch("knns and dists must have identical dimensions; got $(size(knns)) and $(size(dists))"))
    k_knn, n = size(knns)

    n_inliers_val = n_inliers === nothing ? max(1, round(Int, 3 * k_knn / 4)) : Int(n_inliers)
    n_outliers_val = n_outliers === nothing ? max(1, round(Int, k_knn / 4)) : Int(n_outliers)
    n_random_val = n_random === nothing ? n_outliers_val : Int(n_random)

    # Determine sample source for random negative sampling (indices or range)
    sample = if sample === nothing
        1:n
    else
        sample
    end

    cum_probs = if sample_probs !== nothing
        length(sample_probs) == length(sample) || throw(DimensionMismatch("sample_probs must have length $(length(sample)); got $(length(sample_probs))"))
        cp = cumsum(Float64.(sample_probs))
        cp[end] > 0 || throw(ArgumentError("Sum of sample_probs must be positive"))
        cp
    else
        nothing
    end

    triplets_i = Int32[]
    triplets_j = Int32[]
    triplets_k = Int32[]
    weights = Float32[]

    # Estimate capacity
    n_triplets_est = n * (n_inliers_val * n_outliers_val + n_inliers_val * n_random_val)
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
            if nid != i && nid > 0 && nid <= n && ndist < typemax(Float32)
                push!(neighbors, Int32(nid))
                push!(distances, Float32(ndist))
            end
        end

        num_neighbors = length(neighbors)
        n_in = min(n_inliers_val, num_neighbors)
        n_out = min(n_outliers_val, max(0, num_neighbors - n_in))

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
            for _ in 1:n_random_val
                k = if cum_probs === nothing
                    rand(rng, sample)
                else
                    r = rand(rng, Float64) * cum_probs[end]
                    idx = searchsortedfirst(cum_probs, r)
                    idx = clamp(idx, 1, length(sample))
                    sample[idx]
                end
                if k != i && k != j && k >= 1 && k <= n
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
