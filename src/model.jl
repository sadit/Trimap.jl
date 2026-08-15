# This file is a part of Trimap.jl

"""
    Trimap

Non-parametric TriMAP embedding model. Holds the learned embedding coordinates `embedding`.
"""
struct Trimap{M<:AbstractMatrix{Float32}}
    embedding::M
end

const TrimapModel = Trimap

"""
    ParametricTrimap

Parametric TriMAP model backed by a neural network `model` (Lux).
Allows O(1) projection of out-of-sample points via `predict(m, X_new)`.
"""
struct ParametricTrimap{L<:LuxCore.AbstractLuxLayer, P, S}
    model::L
    ps::P
    st::S
end

"""
    pca_init(X::AbstractMatrix{Float32}, out_dim::Integer; scale::Real=0.01f0) -> Matrix{Float32}

Computes PCA-based initialization of size `(out_dim, n)` from high-dimensional data `X` of size `(dim, n)`, scaled by `scale`.
"""
function pca_init(X::AbstractMatrix{Float32}, out_dim::Integer; scale::Real=0.01f0)
    # X is (dim, n)
    d, n = size(X)
    X_mean = mean(X, dims=2)
    Xc = X .- X_mean
    # SVD of centered data
    F = svd(Xc)
    # Principal components projection:
    k = min(out_dim, size(F.Vt, 1))
    Y = zeros(Float32, out_dim, n)
    Y[1:k, :] .= F.Vt[1:k, :]
    Y .* Float32(scale)
end

"""
    fit(::Type{Trimap}, knns::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32};
        out_dim=2,
        maxoutdim=out_dim,
        Y_init=nothing,
        sample=nothing,
        sample_probs=nothing,
        n_inliers=15,
        n_outliers=5,
        n_random=5,
        weight_adj=0.1,
        max_iters=400,
        n_epochs=max_iters,
        opt=nothing,
        learning_rate=0.1,
        rng=Random.default_rng(),
        verbose=false) -> Trimap

Fits non-parametric TriMAP dimensionality reduction given precomputed nearest neighbor IDs (`knns::AbstractMatrix{UInt32}`) and distances (`dists::AbstractMatrix{Float32}`).

# Arguments
- `knns`: `(k × n)` matrix of nearest neighbor IDs of type `UInt32` (e.g. from `allknn` or `searchbatch`).
- `dists`: `(k × n)` matrix of nearest neighbor distances of type `Float32`.

# Keyword Arguments
- `out_dim`: Target embedding dimension (default: `2`).
- `maxoutdim`: Alias for `out_dim`.
- `Y_init`: Optional initial embedding matrix of size `(out_dim, n)` (e.g. `Matrix{Float32}`). If `nothing`, initialized with Gaussian noise `randn * 0.01`. You may pass `pca_init(X, out_dim)`.
- `sample`: Optional subset of sample points / landmarks from which negative samples are drawn (e.g. `Vector{UInt32}` from `fft(dist, db, k).centers` or `AbstractMatrix{Float32}`). If `nothing` (default), negative samples are drawn from `1:n`.
- `sample_probs`: Optional vector of sampling probabilities or weights corresponding to each element in `sample` (or `1:n`). If `nothing` (default), uniform sampling is used.
- `n_inliers`: Number of nearest neighbors treated as inliers (local structure). Default: `15`.
- `n_outliers`: Number of further neighbors treated as margin outliers. Default: `5`.
- `n_random`: Number of random negative samples per inlier (global structure). Default: `5`.
- `weight_adj`: Weight adjustment parameter controlling the influence of distance gaps. Default: `0.1`.
- `max_iters` / `n_epochs`: Optimization iterations. Default: `400`.
- `opt`: Optimizer instance from `Optimisers.jl` (default: `AdamW(learning_rate)`).
- `learning_rate`: Learning rate if `opt` is not specified. Default: `0.1`.
- `rng`: Random number generator. Default: `Random.default_rng()`.
- `verbose`: Whether to log iteration loss. Default: `false`.

# Examples

## Exact search with `ExhaustiveSearch` and `fft` negative sampling with cluster probabilities
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

# Extract a diverse sample for negative sampling using Farthest First Traversal (fft)
res_fft = fft(Dist.L2(), db, 50)
sample_centers = res_fft.centers

# Compute relative frequencies/probabilities from res_fft.nn
counts = Dict{UInt32, Float32}()
for c in res_fft.nn
    counts[c] = get(counts, c, 0f0) + 1f0
end
sample_probs = Float32[get(counts, c, 0f0) / length(res_fft.nn) for c in sample_centers]

# Fit non-parametric TriMAP with weighted fft negative sampling
model = fit(Trimap, knns, dists; sample=sample_centers, sample_probs=sample_probs, out_dim=2)
# model.embedding is of size (2, 500)
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

# Fit non-parametric TriMAP (optionally using PCA initialization from X)
model = fit(Trimap, knns, dists; out_dim=2, Y_init=pca_init(X, 2))
```
"""
function fit(
    ::Type{Trimap},
    knns::AbstractMatrix{UInt32},
    dists::AbstractMatrix{Float32};
    out_dim::Integer=2,
    maxoutdim::Integer=out_dim,
    Y_init::Union{Nothing, AbstractMatrix{<:Real}}=nothing,
    sample=nothing,
    sample_probs=nothing,
    n_inliers::Integer=15,
    n_outliers::Integer=5,
    n_random::Integer=5,
    weight_adj::Real=0.1,
    max_iters::Integer=400,
    n_epochs::Integer=max_iters,
    opt=nothing,
    learning_rate::Real=0.1,
    rng::Random.AbstractRNG=Random.default_rng(),
    verbose::Bool=false
)
    final_out_dim = maxoutdim != 2 ? maxoutdim : out_dim
    final_iters = n_epochs != 400 ? n_epochs : max_iters
    optimizer = if opt !== nothing
        opt
    else
        Optimisers.AdamW(Float32(learning_rate))
    end

    size(knns) == size(dists) || throw(DimensionMismatch("knns and dists must have identical dimensions; got $(size(knns)) and $(size(dists))"))
    n = size(knns, 2)

    i, j, k, w = generate_triplets(
        knns,
        dists;
        sample=sample,
        sample_probs=sample_probs,
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        n_random=n_random,
        weight_adj=weight_adj,
        rng=rng
    )

    # Initial embedding
    local Y_start::Matrix{Float32}
    if Y_init !== nothing
        size(Y_init) == (final_out_dim, n) || throw(DimensionMismatch("Y_init must have size ($(final_out_dim), $(n)); got $(size(Y_init))"))
        Y_start = Float32.(Y_init)
    else
        Y_start = randn(rng, Float32, final_out_dim, n) .* 0.01f0
    end

    Y = optimize_embedding_triplets(
        Y_start, i, j, k, w;
        max_iters=final_iters, opt=optimizer, verbose=verbose
    )

    Trimap(Y)
end

# Module-level convenience dispatch: fit(Trimap, ...)
function fit(m::Module, knns::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32}; kwargs...)
    if nameof(m) === :Trimap
        fit(Trimap, knns, dists; kwargs...)
    else
        throw(MethodError(fit, (m, knns, dists)))
    end
end

"""
    fit(::Type{ParametricTrimap}, X::AbstractMatrix, knns::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32};
        model=nothing,
        out_dim=2,
        maxoutdim=out_dim,
        hidden_dims=(128, 64),
        sample=nothing,
        sample_probs=nothing,
        n_inliers=15,
        n_outliers=5,
        n_random=5,
        weight_adj=0.1,
        max_iters=200,
        n_epochs=max_iters,
        opt=nothing,
        learning_rate=0.001,
        rng=Random.default_rng(),
        verbose=false) -> ParametricTrimap

Fits a Parametric TriMAP model using a neural network (`Lux`), training on high-dimensional features `X` and triplet constraints generated from nearest neighbor IDs (`knns::AbstractMatrix{UInt32}`) and distances (`dists::AbstractMatrix{Float32}`).

# Arguments
- `X`: `(dim × n)` input data matrix.
- `knns`: `(k × n)` matrix of nearest neighbor IDs of type `UInt32`.
- `dists`: `(k × n)` matrix of nearest neighbor distances of type `Float32`.

# Keyword Arguments
- `model`: Custom `Lux` layer/network mapping `dim -> out_dim`. If `nothing`, a multilayer perceptron with `hidden_dims` and ReLU activations is constructed.
- `out_dim`: Target embedding dimension (default: `2`).
- `maxoutdim`: Alias for `out_dim`.
- `hidden_dims`: Tuple of hidden layer dimensions for default MLP (default: `(128, 64)`).
- `sample`: Optional subset of sample points / landmarks from which negative samples are drawn (e.g. `Vector{UInt32}` from `fft(dist, db, k).centers` or `AbstractMatrix{Float32}`). If `nothing` (default), negative samples are uniformly drawn from `1:n`.
- `sample_probs`: Optional vector of sampling probabilities or weights corresponding to each element in `sample` (or `1:n`). If `nothing` (default), uniform sampling is used.
- `n_inliers`: Number of inlier neighbors. Default: `15`.
- `n_outliers`: Number of outlier neighbors. Default: `5`.
- `n_random`: Number of random negative samples. Default: `5`.
- `weight_adj`: Weight adjustment parameter. Default: `0.1`.
- `max_iters` / `n_epochs`: Optimization epochs. Default: `200`.
- `opt`: Optimizer instance from `Optimisers.jl` (default: `AdamW(learning_rate)`).
- `learning_rate`: Learning rate if `opt` is not specified. Default: `0.001`.
- `rng`: Random number generator. Default: `Random.default_rng()`.
- `verbose`: Whether to log iteration loss. Default: `false`.

# Examples

## Exact search with `ExhaustiveSearch` and `fft` negative sampling with cluster probabilities
```julia
using SimilaritySearch, Trimap

# Create dataset and exact search index
X = randn(Float32, 10, 500)
db = MatrixDatabase(X)
index = ExhaustiveSearch(Dist.L2(), db)

# Compute exact all-kNN
k = 25
ctx = GenericContext()
knns, dists = allknn(index, ctx, k)

# Extract a diverse sample for negative sampling using Farthest First Traversal (fft)
res_fft = fft(Dist.L2(), db, 50)
sample_centers = res_fft.centers

# Compute relative frequencies/probabilities from res_fft.nn
counts = Dict{UInt32, Float32}()
for c in res_fft.nn
    counts[c] = get(counts, c, 0f0) + 1f0
end
sample_probs = Float32[get(counts, c, 0f0) / length(res_fft.nn) for c in sample_centers]

# Fit Parametric TriMAP with weighted fft negative sampling
pmodel = fit(ParametricTrimap, X, knns, dists; sample=sample_centers, sample_probs=sample_probs, out_dim=2, hidden_dims=(64, 32))

# Predict embeddings for unseen points
X_new = randn(Float32, 10, 50)
Y_new = predict(pmodel, X_new)
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

# Fit Parametric TriMAP
pmodel = fit(ParametricTrimap, X, knns, dists; out_dim=2)
```
"""
function fit(
    ::Type{ParametricTrimap},
    X::AbstractMatrix,
    knns::AbstractMatrix{UInt32},
    dists::AbstractMatrix{Float32};
    model::Union{Nothing, LuxCore.AbstractLuxLayer}=nothing,
    out_dim::Integer=2,
    maxoutdim::Integer=out_dim,
    hidden_dims::Tuple=(128, 64),
    sample=nothing,
    sample_probs=nothing,
    n_inliers::Integer=15,
    n_outliers::Integer=5,
    n_random::Integer=5,
    weight_adj::Real=0.1,
    max_iters::Integer=200,
    n_epochs::Integer=max_iters,
    opt=nothing,
    learning_rate::Real=0.001,
    rng::Random.AbstractRNG=Random.default_rng(),
    verbose::Bool=false
)
    final_out_dim = maxoutdim != 2 ? maxoutdim : out_dim
    final_iters = n_epochs != 200 ? n_epochs : max_iters
    optimizer = if opt !== nothing
        opt
    else
        Optimisers.AdamW(Float32(learning_rate))
    end

    X32 = Float32.(X)
    in_dim, n = size(X32)

    size(knns) == size(dists) || throw(DimensionMismatch("knns and dists must have identical dimensions; got $(size(knns)) and $(size(dists))"))
    size(knns, 2) == n || throw(DimensionMismatch("Number of columns in knns ($(size(knns, 2))) must match number of columns in X ($n)"))

    # Build default MLP if none provided
    net = if model === nothing
        layers = []
        prev_dim = in_dim
        for h in hidden_dims
            push!(layers, Lux.Dense(prev_dim => h, Lux.relu))
            prev_dim = h
        end
        push!(layers, Lux.Dense(prev_dim => final_out_dim))
        Lux.Chain(layers...)
    else
        model
    end

    i, j, k, w = generate_triplets(
        knns,
        dists;
        sample=sample,
        sample_probs=sample_probs,
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        n_random=n_random,
        weight_adj=weight_adj,
        rng=rng
    )

    ps, st = train_parametric_trimap(
        net, X32, i, j, k, w;
        max_iters=final_iters, opt=optimizer, rng=rng, verbose=verbose
    )

    ParametricTrimap(net, ps, st)
end

# Module-level convenience dispatch: fit(Trimap, X, knns, dists; ...)
function fit(m::Module, X::AbstractMatrix, knns::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32}; kwargs...)
    if nameof(m) === :Trimap
        fit(ParametricTrimap, X, knns, dists; kwargs...)
    else
        throw(MethodError(fit, (m, X, knns, dists)))
    end
end

"""
    predict(m::ParametricTrimap, X::AbstractMatrix) -> Matrix{Float32}

Maps unseen high-dimensional data points to embedding coordinates using the trained neural network.
"""
function predict(m::ParametricTrimap, X::AbstractMatrix)
    X32 = Float32.(X)
    Y, _ = Lux.apply(m.model, X32, m.ps, m.st)
    Y
end
