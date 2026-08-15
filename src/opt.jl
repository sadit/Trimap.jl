# This file is a part of Trimap.jl

"""
    optimize_embedding_triplets(Y_init::AbstractMatrix{Float32}, triplets_i, triplets_j, triplets_k, weights;
                                max_iters=400, opt=Optimisers.AdamW(0.1f0), verbose=false)

Optimizes the low-dimensional embedding `Y` given precomputed triplets and weights.
"""
function optimize_embedding_triplets(
    Y_init::AbstractMatrix{Float32},
    triplets_i::AbstractVector{<:Integer},
    triplets_j::AbstractVector{<:Integer},
    triplets_k::AbstractVector{<:Integer},
    weights::AbstractVector{Float32};
    max_iters::Integer=400,
    opt=Optimisers.AdamW(0.1f0),
    verbose::Bool=false
)
    Y = copy(Y_init)
    opt_state = Optimisers.setup(opt, Y)

    for iter in 1:max_iters
        loss, grads = Zygote.withgradient(Y) do y
            trimap_loss(y, triplets_i, triplets_j, triplets_k, weights)
        end

        if verbose && (iter % 50 == 0 || iter == 1 || iter == max_iters)
            @info "TriMAP iteration $(iter)/$(max_iters): loss = $(loss)"
        end

        opt_state, Y = Optimisers.update!(opt_state, Y, grads[1])
    end

    Y
end

"""
    train_parametric_trimap(model::LuxCore.AbstractLuxLayer, X::AbstractMatrix{Float32},
                            triplets_i, triplets_j, triplets_k, weights;
                            max_iters=200, opt=Optimisers.AdamW(0.001f0),
                            rng=Random.default_rng(), verbose=false)

Trains a parametric neural network `model` (Lux) to map high-dimensional data `X` to low-dimensional embedding.
"""
function train_parametric_trimap(
    model::LuxCore.AbstractLuxLayer,
    X::AbstractMatrix{Float32},
    triplets_i::AbstractVector{<:Integer},
    triplets_j::AbstractVector{<:Integer},
    triplets_k::AbstractVector{<:Integer},
    weights::AbstractVector{Float32};
    max_iters::Integer=200,
    opt=Optimisers.AdamW(0.001f0),
    rng::Random.AbstractRNG=Random.default_rng(),
    verbose::Bool=false
)
    ps, st = Lux.setup(rng, model)
    opt_state = Optimisers.setup(opt, ps)

    for iter in 1:max_iters
        loss, grads = Zygote.withgradient(ps) do p
            Y, _ = Lux.apply(model, X, p, st)
            trimap_loss(Y, triplets_i, triplets_j, triplets_k, weights)
        end

        if verbose && (iter % 25 == 0 || iter == 1 || iter == max_iters)
            @info "Parametric TriMAP iteration $(iter)/$(max_iters): loss = $(loss)"
        end

        opt_state, ps = Optimisers.update!(opt_state, ps, grads[1])
    end

    st_test = Lux.testmode(st)
    ps, st_test
end
