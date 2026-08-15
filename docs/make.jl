# This file is a part of Trimap.jl

using Documenter
using Trimap
using SimilaritySearch
using PlotlyLight
using Lux
using Primes
using Random
using Statistics

DocMeta.setdocmeta!(Trimap, :DocTestSetup, :(using Trimap); recursive=true)

makedocs(
    sitename = "Trimap.jl",
    modules = [Trimap],
    authors = "Eric S. Tellez",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://sadit.github.io/Trimap.jl/stable/",
        assets = String[],
        size_threshold = 500 * 1024,
        size_threshold_warn = 300 * 1024
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Iris Dataset" => "tutorial_iris.md",
            "Two Moons and Spirals" => "tutorial_moons_spirals.md",
            "Prime Factorization" => "tutorial_prime_factors.md",
        ],
    ]
)

deploydocs(
    repo = "github.com/sadit/Trimap.jl.git",
    devbranch = "master"
)
