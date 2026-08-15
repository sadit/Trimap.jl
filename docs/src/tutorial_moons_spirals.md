# Tutorial 2: Non-linear Manifolds - Two Moons and Spirals

In this tutorial, we evaluate TriMAP on synthetic non-linear manifolds: the classic **Two Moons** dataset and **Intertwined Spirals**. We showcase:
1. Fast approximate nearest neighbor search using `SimilaritySearch.SearchGraph` with `MinRecall`.
2. Representative negative sampling using **Farthest First Traversal (`fft`)** and cluster size weighting (`sample_probs` from `res.nn`).
3. Interactive visualization with **PlotlyLight.jl**.

---

## 1. Two Moons with `SearchGraph` (Approximate Search)

First, we generate two interlocking half-moon manifolds with noise:

```@example moons
using SimilaritySearch, Trimap
using PlotlyLight, Random, Statistics

Random.seed!(42)

function make_moons(n_samples=600, noise=0.08f0)
    n_half = div(n_samples, 2)
    theta1 = range(0, π, length=n_half)
    theta2 = range(0, π, length=n_half)
    
    x1 = cos.(theta1) .+ noise .* randn(Float32, n_half)
    y1 = sin.(theta1) .+ noise .* randn(Float32, n_half)
    
    x2 = 1.0f0 .- cos.(theta2) .+ noise .* randn(Float32, n_half)
    y2 = 1.0f0 .- sin.(theta2) .- 0.5f0 .+ noise .* randn(Float32, n_half)
    
    X = hcat(vcat(x1', y1'), vcat(x2', y2'))
    labels = vcat(fill("Moon A", n_half), fill("Moon B", n_half))
    X, labels
end

X_moons, labels_moons = make_moons(800, 0.06f0)
n_moons = size(X_moons, 2)
println("Moons dataset: ", size(X_moons))
```

### Approximate $k$-NN with SearchGraph

Instead of exhaustive search, we construct an approximate `SearchGraph` and optimize its search parameters for a target recall (e.g. 90%):

```@example moons
db_moons = MatrixDatabase(X_moons)
G = SearchGraph(Dist.L2(), db_moons)
ctx_g = SearchGraphContext()
index!(G, ctx_g)

# Optimize for 90% recall
optimize_index!(G, ctx_g, MinRecall(0.90))

k = 25
knns_moons, dists_moons = allknn(G, ctx_g, k)
```

### TriMAP Embedding

```@example moons
model_moons = fit(
    Trimap,
    knns_moons,
    dists_moons;
    out_dim=2,
    n_inliers=15,
    n_outliers=5,
    n_random=5,
    max_iters=300
)

Y_moons = model_moons.embedding

traces_moons = [
    Config(
        x = Y_moons[1, labels_moons .== "Moon A"],
        y = Y_moons[2, labels_moons .== "Moon A"],
        mode = "markers",
        name = "Moon A",
        marker = Config(size=5, color="#1f77b4", opacity=0.85)
    ),
    Config(
        x = Y_moons[1, labels_moons .== "Moon B"],
        y = Y_moons[2, labels_moons .== "Moon B"],
        mode = "markers",
        name = "Moon B",
        marker = Config(size=5, color="#ff7f0e", opacity=0.85)
    )
]

Plot(traces_moons, Config(title="TriMAP 2D Projection of Two Moons (SearchGraph k-NN)"))
```

---

## 2. Intertwined Spirals with `fft` Weighted Negative Sampling

Complex spiral structures are notoriously challenging for dimensionality reduction methods because standard random negative sampling can mistakenly pull distant spiral arms together across gaps.

By choosing negative samples from **Farthest First Traversal (`fft`)** landmarks and weighting them by their Voronoi basin sizes (`sample_probs` from `res.nn`), TriMAP achieves superior global unwrapping.

### Generating 3 Intertwined Spirals

```@example spirals
using SimilaritySearch, Trimap
using PlotlyLight, Random, Statistics

Random.seed!(42)

function make_spirals(n_points_per_arm=300, noise=0.04f0)
    arms = []
    labels = []
    
    for arm in 1:3
        theta = range(0.5, 4.0 * π, length=n_points_per_arm) .+ (arm - 1) * (2π / 3)
        r = range(0.2, 2.0, length=n_points_per_arm)
        
        x = r .* cos.(theta) .+ noise .* randn(Float32, n_points_per_arm)
        y = r .* sin.(theta) .+ noise .* randn(Float32, n_points_per_arm)
        # Add a subtle 3rd dimension variation
        z = 0.5f0 .* sin.(theta) .+ noise .* randn(Float32, n_points_per_arm)
        
        push!(arms, vcat(x', y', z'))
        push!(labels, fill("Spiral $(arm)", n_points_per_arm))
    end
    
    X = hcat(arms...)
    labels_all = vcat(labels...)
    X, labels_all
end

X_spirals, labels_spirals = make_spirals(350, 0.03f0)
n_spirals = size(X_spirals, 2)
println("Spirals dataset: ", size(X_spirals))
```

### Computing $k$-NN and Farthest First Traversal (`fft`)

```@example spirals
db_spirals = MatrixDatabase(X_spirals)
index_spirals = ExhaustiveSearch(Dist.L2(), db_spirals)
ctx = GenericContext()
knns_spirals, dists_spirals = allknn(index_spirals, ctx, 30)

# 1. Farthest First Traversal to select 60 landmarks
res_fft = fft(Dist.L2(), db_spirals, 60)
sample_centers = res_fft.centers

# 2. Compute relative probabilities from res_fft.nn cluster assignments
counts = Dict{UInt32, Float32}()
for c in res_fft.nn
    counts[c] = get(counts, c, 0f0) + 1f0
end
sample_probs = Float32[get(counts, c, 0f0) / length(res_fft.nn) for c in sample_centers]

println("Number of FFT landmarks: ", length(sample_centers))
println("Sum of probabilities: ", sum(sample_probs))
```

### TriMAP Embedding with Weighted Negative Sampling

```@example spirals
model_spirals = fit(
    Trimap,
    knns_spirals,
    dists_spirals;
    sample=sample_centers,
    sample_probs=sample_probs,
    out_dim=2,
    n_inliers=15,
    n_outliers=5,
    n_random=6,
    weight_adj=0.15,
    max_iters=400,
    learning_rate=0.1
)

Y_spirals = model_spirals.embedding

colors = Dict("Spiral 1" => "#636EFA", "Spiral 2" => "#EF553B", "Spiral 3" => "#00CC96")
traces_spirals = Config[]

for arm_name in ["Spiral 1", "Spiral 2", "Spiral 3"]
    idx = findall(==(arm_name), labels_spirals)
    push!(traces_spirals, Config(
        x = Y_spirals[1, idx],
        y = Y_spirals[2, idx],
        mode = "markers",
        name = arm_name,
        marker = Config(size=4, color=colors[arm_name], opacity=0.85)
    ))
end

layout = Config(
    title = "TriMAP Embedding of 3 Intertwined Spirals (with FFT Weighted Sampling)",
    xaxis = Config(title = "TriMAP 1", zeroline=false),
    yaxis = Config(title = "TriMAP 2", zeroline=false),
    hovermode = "closest"
)

Plot(traces_spirals, layout)
```

The combination of dense local inlier triplets and representative, probability-weighted landmark negative triplets cleanly resolves the 3 spirals into continuous, non-overlapping trajectories in 2D.
