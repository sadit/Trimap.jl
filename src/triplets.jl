# This file is a part of Trimap.jl

"""
    generate_triplets(index_or_data;
                      n_inliers=15,
                      n_outliers=5,
                      n_random=5,
                      weight_adj=0.1,
                      dist=Dist.L2(),
                      searchctx=nothing,
                      rng=Random.default_rng()) -> (triplets_i, triplets_j, triplets_k, weights)

Generates TriMAP triplet constraints (i, j, k) where item j is closer to i than item k in the original space.
"""
function generate_triplets(
    index_or_data;
    n_inliers::Integer=15,
    n_outliers::Integer=5,
    n_random::Integer=5,
    weight_adj::Real=0.1,
    dist=Dist.L2(),
    searchctx=nothing,
    rng::Random.AbstractRNG=Random.default_rng()
)
    # Ensure index
    index = if index_or_data isa AbstractSearchIndex
        index_or_data
    elseif index_or_data isa AbstractDatabase
        ExhaustiveSearch(dist, index_or_data)
    elseif index_or_data isa AbstractMatrix
        ExhaustiveSearch(dist, MatrixDatabase(Float32.(index_or_data)))
    else
        ExhaustiveSearch(dist, VectorDatabase(index_or_data))
    end

    n = length(index)
    k_knn = min(n, n_inliers + n_outliers + 1)

    # 1. kNN graph for nearest neighbors (inliers) and margin outliers
    ctx = if searchctx !== nothing
        searchctx
    elseif index isa SearchGraph
        SearchGraphContext()
    else
        GenericContext()
    end

    knns, dists = allknn(index, ctx, k_knn)
    # knns is (k_knn, n) of UInt32, dists is (k_knn, n) of Float32

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
        # Filter out self
        neighbors = Int32[]
        distances = Float32[]
        for row in 1:k_knn
            nid = knns[row, i]
            ndist = dists[row, i]
            if nid != i
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
