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
    pca_init(X::AbstractMatrix{Float32}, out_dim::Int; scale=0.01f0)

Computes PCA-based initialization scaled by `scale`.
"""
function pca_init(X::AbstractMatrix{Float32}, out_dim::Int; scale::Float32=0.01f0)
    # X is (dim, n)
    d, n = size(X)
    X_mean = mean(X, dims=2)
    Xc = X .- X_mean
    # SVD of covariance or centered data
    F = svd(Xc)
    # U is (d, min(d, n)), Vt is (min(d, n), n)
    # Principal components projection:
    k = min(out_dim, size(F.Vt, 1))
    Y = zeros(Float32, out_dim, n)
    # F.Vt[1:k, :] is (k, n)
    Y[1:k, :] .= F.Vt[1:k, :]
    Y .* scale
end

"""
    fit(::Type{Trimap}, index_or_data;
        out_dim=2,
        maxoutdim=out_dim,
        n_inliers=15,
        n_outliers=5,
        n_random=5,
        weight_adj=0.1,
        max_iters=400,
        n_epochs=max_iters,
        opt=nothing,
        learning_rate=0.1,
        dist=Dist.L2(),
        searchctx=nothing,
        verbose=false) -> Trimap

Fits non-parametric TriMAP dimensionality reduction.
"""
function fit(
    ::Type{Trimap},
    index_or_data;
    out_dim::Integer=2,
    maxoutdim::Integer=out_dim,
    n_inliers::Integer=15,
    n_outliers::Integer=5,
    n_random::Integer=5,
    weight_adj::Real=0.1,
    max_iters::Integer=400,
    n_epochs::Integer=max_iters,
    opt=nothing,
    learning_rate::Real=0.1,
    dist=Dist.L2(),
    searchctx=nothing,
    verbose::Bool=false
)
    final_out_dim = maxoutdim != 2 ? maxoutdim : out_dim
    final_iters = n_epochs != 400 ? n_epochs : max_iters
    optimizer = if opt !== nothing
        opt
    else
        Optimisers.AdamW(Float32(learning_rate))
    end

    i, j, k, w = generate_triplets(
        index_or_data;
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        n_random=n_random,
        weight_adj=weight_adj,
        dist=dist,
        searchctx=searchctx
    )

    n = if index_or_data isa AbstractMatrix
        size(index_or_data, 2)
    elseif index_or_data isa MatrixDatabase
        size(index_or_data.matrix, 2)
    else
        length(index_or_data)
    end

    # Initial embedding
    local Y_init
    if index_or_data isa AbstractMatrix
        X = Float32.(index_or_data)
        Y_init = pca_init(X, final_out_dim)
    elseif index_or_data isa MatrixDatabase
        X = Float32.(index_or_data.matrix)
        Y_init = pca_init(X, final_out_dim)
    elseif index_or_data isa AbstractSearchIndex && database(index_or_data) isa MatrixDatabase
        X = Float32.(database(index_or_data).matrix)
        Y_init = pca_init(X, final_out_dim)
    else
        Y_init = randn(Float32, final_out_dim, n) .* 0.01f0
    end

    Y = optimize_embedding_triplets(
        Y_init, i, j, k, w;
        max_iters=final_iters, opt=optimizer, verbose=verbose
    )

    Trimap(Y)
end

# Module-level convenience dispatch: fit(Trimap, ...)
function fit(m::Module, index_or_data; kwargs...)
    if nameof(m) === :Trimap
        fit(Trimap, index_or_data; kwargs...)
    else
        throw(MethodError(fit, (m, index_or_data)))
    end
end

"""
    fit(::Type{ParametricTrimap}, X::AbstractMatrix;
        model=nothing,
        out_dim=2,
        maxoutdim=out_dim,
        hidden_dims=(128, 64),
        n_inliers=15,
        n_outliers=5,
        n_random=5,
        weight_adj=0.1,
        max_iters=200,
        n_epochs=max_iters,
        opt=nothing,
        learning_rate=0.001,
        dist=Dist.L2(),
        searchctx=nothing,
        rng=Random.default_rng(),
        verbose=false) -> ParametricTrimap

Fits a Parametric TriMAP model using a neural network (Lux).
"""
function fit(
    ::Type{ParametricTrimap},
    X::AbstractMatrix;
    model::Union{Nothing, LuxCore.AbstractLuxLayer}=nothing,
    out_dim::Integer=2,
    maxoutdim::Integer=out_dim,
    hidden_dims::Tuple=(128, 64),
    n_inliers::Integer=15,
    n_outliers::Integer=5,
    n_random::Integer=5,
    weight_adj::Real=0.1,
    max_iters::Integer=200,
    n_epochs::Integer=max_iters,
    opt=nothing,
    learning_rate::Real=0.001,
    dist=Dist.L2(),
    searchctx=nothing,
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
    in_dim = size(X32, 1)

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
        X32;
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        n_random=n_random,
        weight_adj=weight_adj,
        dist=dist,
        searchctx=searchctx
    )

    ps, st = train_parametric_trimap(
        net, X32, i, j, k, w;
        max_iters=final_iters, opt=optimizer, rng=rng, verbose=verbose
    )

    ParametricTrimap(net, ps, st)
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
