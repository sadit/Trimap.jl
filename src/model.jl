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
    # Principal component scores: Xc = U*S*Vt, so U'*Xc = S*Vt (top-k rows)
    k = min(out_dim, length(F.S))
    Y = zeros(Float32, out_dim, n)
    Y[1:k, :] .= F.S[1:k] .* F.Vt[1:k, :]
    Y .* Float32(scale)
end

"""
    fit(::Type{Trimap}, X::AbstractMatrix{Float32}, knns::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32};
        maxoutdim=2,
        n_epochs=400,
        Y_init=nothing,
        sample=nothing,
        sample_probs=nothing,
        n_inliers=nothing,
        n_outliers=nothing,
        n_random=nothing,
        weight_adj=0.1,
        opt=nothing,
        learning_rate=0.1,
        rng=Random.default_rng(),
        verbose=false) -> Trimap

Fits non-parametric TriMAP dimensionality reduction given high-dimensional features `X`, precomputed nearest neighbor IDs (`knns::AbstractMatrix{UInt32}`), and distances (`dists::AbstractMatrix{Float32}`).

# Arguments
- `X`: `(dim × n)` input data matrix of type `Float32`.
- `knns`: `(k × n)` matrix of nearest neighbor IDs of type `UInt32` (e.g. from `allknn` or `searchbatch`).
- `dists`: `(k × n)` matrix of nearest neighbor distances of type `Float32`.

# Keyword Arguments
- `maxoutdim`: Target embedding dimension (default: `2`).
- `n_epochs`: Optimization iterations / epochs (default: `400`).
- `Y_init`: Initial embedding matrix of size `(maxoutdim, n)`. If `nothing` (default), initialized via `pca_init(X, maxoutdim)`. Pass `:random` for Gaussian noise.
- `sample`: Optional subset of sample points / landmarks from which negative samples are drawn (e.g. `Vector{UInt32}` from `fft(dist, db, k).centers`). If `nothing` (default), negative samples are drawn from `1:n`.
- `sample_probs`: Optional vector of sampling probabilities or weights corresponding to each element in `sample` (or `1:n`). If `nothing` (default), uniform sampling is used.
- `n_inliers`: Number of nearest neighbors treated as inliers (local structure). Default: `round(Int, 3k/4)`.
- `n_outliers`: Number of further neighbors treated as margin outliers. Default: `round(Int, k/4)`.
- `n_random`: Number of random negative samples per inlier (global structure). Default: equal to `n_outliers`.
- `weight_adj`: Weight adjustment parameter controlling the influence of distance gaps. Default: `0.1`.
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

# Fit non-parametric TriMAP with weighted fft negative sampling (PCA initialization is automatic)
model = fit(Trimap, X, knns, dists; sample=sample_centers, sample_probs=sample_probs, maxoutdim=2)
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

# Fit non-parametric TriMAP
model = fit(Trimap, X, knns, dists; maxoutdim=2)
```
"""
function fit(
    ::Type{Trimap},
    X::AbstractMatrix{Float32},
    knns::AbstractMatrix{UInt32},
    dists::AbstractMatrix{Float32};
    maxoutdim::Integer=2,
    n_epochs::Integer=400,
    Y_init::Union{Nothing, Symbol, AbstractMatrix{<:Real}}=nothing,
    sample=nothing,
    sample_probs=nothing,
    n_inliers::Union{Nothing, Integer}=nothing,
    n_outliers::Union{Nothing, Integer}=nothing,
    n_random::Union{Nothing, Integer}=nothing,
    weight_adj::Real=0.1,
    opt=nothing,
    learning_rate::Real=0.1,
    rng::Random.AbstractRNG=Random.default_rng(),
    verbose::Bool=false
)
    optimizer = if opt !== nothing
        opt
    else
        Optimisers.AdamW(Float32(learning_rate))
    end

    size(knns) == size(dists) || throw(DimensionMismatch("knns and dists must have identical dimensions; got $(size(knns)) and $(size(dists))"))
    n = size(knns, 2)
    size(X, 2) == n || throw(DimensionMismatch("Number of columns in X ($(size(X, 2))) must match number of columns in knns ($n)"))

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

    # Initial embedding: default to PCA initialization on X
    local Y_start::Matrix{Float32}
    if Y_init isa AbstractMatrix
        size(Y_init) == (maxoutdim, n) || throw(DimensionMismatch("Y_init must have size ($(maxoutdim), $(n)); got $(size(Y_init))"))
        Y_start = Float32.(Y_init)
    elseif Y_init === :random
        Y_start = randn(rng, Float32, maxoutdim, n) .* 0.01f0
    else
        Y_start = pca_init(X, maxoutdim)
    end

    Y = optimize_embedding_triplets(
        Y_start, i, j, k, w;
        max_iters=n_epochs, opt=optimizer, verbose=verbose
    )

    Trimap(Y)
end

# Module-level convenience dispatch: fit(Trimap, X, knns, dists; ...)
function fit(m::Module, X::AbstractMatrix{Float32}, knns::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32}; kwargs...)
    if nameof(m) === :Trimap
        fit(Trimap, X, knns, dists; kwargs...)
    else
        throw(MethodError(fit, (m, X, knns, dists)))
    end
end

"""
    fit(::Type{ParametricTrimap}, X::AbstractMatrix{Float32}, knns::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32};
        model=nothing,
        maxoutdim=2,
        n_epochs=400,
        hidden_dims=(128, 64),
        sample=nothing,
        sample_probs=nothing,
        n_inliers=nothing,
        n_outliers=nothing,
        n_random=nothing,
        weight_adj=0.1,
        opt=nothing,
        learning_rate=0.1,
        rng=Random.default_rng(),
        verbose=false) -> ParametricTrimap

Fits a Parametric TriMAP model using a neural network (`Lux`), training on high-dimensional features `X` and triplet constraints generated from nearest neighbor IDs (`knns::AbstractMatrix{UInt32}`) and distances (`dists::AbstractMatrix{Float32}`).

# Arguments
- `X`: `(dim × n)` input data matrix of type `Float32`.
- `knns`: `(k × n)` matrix of nearest neighbor IDs of type `UInt32`.
- `dists`: `(k × n)` matrix of nearest neighbor distances of type `Float32`.

# Keyword Arguments
- `model`: Custom `Lux` layer/network mapping `dim -> maxoutdim`. If `nothing`, a multilayer perceptron with `hidden_dims` and ReLU activations is constructed.
- `maxoutdim`: Target embedding dimension (default: `2`).
- `n_epochs`: Optimization epochs (default: `400`).
- `hidden_dims`: Tuple of hidden layer dimensions for default MLP (default: `(128, 64)`).
- `sample`: Optional subset of sample points / landmarks from which negative samples are drawn (e.g. `Vector{UInt32}` from `fft(dist, db, k).centers`). If `nothing` (default), negative samples are uniformly drawn from `1:n`.
- `sample_probs`: Optional vector of sampling probabilities or weights corresponding to each element in `sample` (or `1:n`). If `nothing` (default), uniform sampling is used.
- `n_inliers`: Number of inlier neighbors. Default: `round(Int, 3k/4)`.
- `n_outliers`: Number of outlier neighbors. Default: `round(Int, k/4)`.
- `n_random`: Number of random negative samples. Default: equal to `n_outliers`.
- `weight_adj`: Weight adjustment parameter. Default: `0.1`.
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
pmodel = fit(ParametricTrimap, X, knns, dists; sample=sample_centers, sample_probs=sample_probs, maxoutdim=2, hidden_dims=(64, 32))

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
pmodel = fit(ParametricTrimap, X, knns, dists; maxoutdim=2)
```
"""
function fit(
    ::Type{ParametricTrimap},
    X::AbstractMatrix{Float32},
    knns::AbstractMatrix{UInt32},
    dists::AbstractMatrix{Float32};
    model::Union{Nothing, LuxCore.AbstractLuxLayer}=nothing,
    maxoutdim::Integer=2,
    n_epochs::Integer=400,
    hidden_dims::Tuple=(128, 64),
    sample=nothing,
    sample_probs=nothing,
    n_inliers::Union{Nothing, Integer}=nothing,
    n_outliers::Union{Nothing, Integer}=nothing,
    n_random::Union{Nothing, Integer}=nothing,
    weight_adj::Real=0.1,
    opt=nothing,
    learning_rate::Real=0.1,
    rng::Random.AbstractRNG=Random.default_rng(),
    verbose::Bool=false
)
    optimizer = if opt !== nothing
        opt
    else
        Optimisers.AdamW(Float32(learning_rate))
    end

    in_dim, n = size(X)

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
        push!(layers, Lux.Dense(prev_dim => maxoutdim))
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
        net, X, i, j, k, w;
        max_iters=n_epochs, opt=optimizer, rng=rng, verbose=verbose
    )

    ParametricTrimap(net, ps, st)
end

"""
    predict(m::ParametricTrimap, X::AbstractMatrix{Float32}) -> Matrix{Float32}

Maps unseen high-dimensional data points to embedding coordinates using the trained neural network.
"""
function predict(m::ParametricTrimap, X::AbstractMatrix{Float32})
    Y, _ = Lux.apply(m.model, X, m.ps, m.st)
    Y
end
