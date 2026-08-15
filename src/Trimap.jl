# This file is a part of Trimap.jl

module Trimap

using SimilaritySearch
using Distances
using LinearAlgebra
using Statistics
using Random
using Lux
using LuxCore
using Optimisers
using Zygote

export
    # Model structs
    Trimap,
    TrimapModel,
    ParametricTrimap,
    # High-level API
    fit,
    predict,
    # Triplet extraction & loss
    generate_triplets,
    trimap_loss,
    # Initialization & optimization
    pca_init,
    optimize_embedding_triplets,
    train_parametric_trimap

include("loss.jl")
include("triplets.jl")
include("opt.jl")
include("model.jl")

end # module Trimap
