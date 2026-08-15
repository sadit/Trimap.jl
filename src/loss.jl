# Differentiable Triplet Loss for TriMAP

"""
    trimap_loss(Y::AbstractMatrix,
                triplets_i::AbstractVector{<:Integer},
                triplets_j::AbstractVector{<:Integer},
                triplets_k::AbstractVector{<:Integer},
                weights::AbstractVector{<:Real})

Computes the TriMAP triplet loss over embedding `Y` (where columns represent data points).
Uses the Student-t kernel:
- ``d_{ij} = 1 + \\|y_i - y_j\\|^2``
- ``d_{ik} = 1 + \\|y_i - y_k\\|^2``
- ``\\mathcal{L} = \\sum_{(i,j,k)} w_{ijk} \\frac{d_{ij}}{d_{ij} + d_{ik}}``
"""
function trimap_loss(Y::AbstractMatrix,
                     triplets_i::AbstractVector{<:Integer},
                     triplets_j::AbstractVector{<:Integer},
                     triplets_k::AbstractVector{<:Integer},
                     weights::AbstractVector{<:Real})
    Yi = Y[:, triplets_i]
    Yj = Y[:, triplets_j]
    Yk = Y[:, triplets_k]

    dij = 1f0 .+ sum(abs2.(Yi .- Yj), dims=1)
    dik = 1f0 .+ sum(abs2.(Yi .- Yk), dims=1)

    loss_vec = dij ./ (dij .+ dik)
    sum(vec(weights) .* vec(loss_vec))
end
