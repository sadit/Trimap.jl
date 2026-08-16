# Tutorial 2: Non-linear Manifolds - Two Moons and Spirals

In this tutorial, we evaluate TriMAP on synthetic non-linear manifolds: the classic **Two Moons** dataset and **Intertwined Spirals**. For each dataset, we:
1. Visualize the **original 2D input manifold**.
2. Compute and visualize the **linear 2D PCA baseline**.
3. Compute the nearest neighbor graph using `SimilaritySearch.SearchGraph` or `ExhaustiveSearch`.
4. Apply **TriMAP** (with representative `fft` negative sampling and `sample_probs` weighting) and visualize the unwrapped manifold with **PlotlyLight.jl**.

---

## 1. Two Moons Manifold

### 1.1 Generating and Visualizing 2D Input Data

```@example moons
using SimilaritySearch, Trimap
using PlotlyLight, Random, Statistics

Random.seed!(42)

function make_moons(n_samples=1200, noise=0.05f0)
    n_half = div(n_samples, 2)
    theta1 = range(0f0, Float32(π), length=n_half)
    theta2 = range(0f0, Float32(π), length=n_half)
    
    x1 = cos.(theta1) .+ noise .* randn(Float32, n_half)
    y1 = sin.(theta1) .+ noise .* randn(Float32, n_half)
    
    x2 = 1.0f0 .- cos.(theta2) .+ noise .* randn(Float32, n_half)
    y2 = 1.0f0 .- sin.(theta2) .- 0.5f0 .+ noise .* randn(Float32, n_half)
    
    X = Matrix{Float32}(hcat(vcat(x1', y1'), vcat(x2', y2')))
    labels = vcat(fill("Moon A", n_half), fill("Moon B", n_half))
    X, labels
end

X_moons, labels_moons = make_moons(1200, 0.05f0)
println("Moons dataset size: ", size(X_moons))

# 1. Visualize Original 2D Input Data
traces_input_moons = [
    Config(
        x = X_moons[1, labels_moons .== "Moon A"],
        y = X_moons[2, labels_moons .== "Moon A"],
        mode = "markers",
        name = "Moon A",
        marker = Config(size=4, color="#1f77b4", opacity=0.85)
    ),
    Config(
        x = X_moons[1, labels_moons .== "Moon B"],
        y = X_moons[2, labels_moons .== "Moon B"],
        mode = "markers",
        name = "Moon B",
        marker = Config(size=4, color="#ff7f0e", opacity=0.85)
    )
]

Plot(traces_input_moons, Config(title="Two Moons: Original 2D Input Space"))
```

### 1.2 Baseline: 2D PCA Projection

```@example moons
Y_pca_moons = pca_init(X_moons, 2; scale=1.0f0)

traces_pca_moons = [
    Config(
        x = Y_pca_moons[1, labels_moons .== "Moon A"],
        y = Y_pca_moons[2, labels_moons .== "Moon A"],
        mode = "markers",
        name = "Moon A",
        marker = Config(size=4, color="#1f77b4", opacity=0.85)
    ),
    Config(
        x = Y_pca_moons[1, labels_moons .== "Moon B"],
        y = Y_pca_moons[2, labels_moons .== "Moon B"],
        mode = "markers",
        name = "Moon B",
        marker = Config(size=4, color="#ff7f0e", opacity=0.85)
    )
]

Plot(traces_pca_moons, Config(title="Two Moons: Linear 2D PCA Projection"))
```

### 1.3 Approximate $k$-NN with `SearchGraph`

Instead of exhaustive search, we construct an approximate `SearchGraph` and optimize its search parameters for a target recall (90%):

```@example moons
db_moons = MatrixDatabase(X_moons)
G = SearchGraph(Dist.L2(), db_moons)
ctx_g = SearchGraphContext()
index!(G, ctx_g)

# Optimize for 90% recall
optimize_index!(G, ctx_g, MinRecall(0.90))

k = 20
knns_moons, dists_moons = allknn(G, ctx_g, k)
```

### 1.4 TriMAP Embedding

```@example moons
model_moons = fit(
    Trimap,
    X_moons,
    knns_moons,
    dists_moons;
    maxoutdim=2,
    n_epochs=400,
    learning_rate=0.1
)

Y_moons = model_moons.embedding

traces_trimap_moons = [
    Config(
        x = Y_moons[1, labels_moons .== "Moon A"],
        y = Y_moons[2, labels_moons .== "Moon A"],
        mode = "markers",
        name = "Moon A",
        marker = Config(size=4, color="#1f77b4", opacity=0.85)
    ),
    Config(
        x = Y_moons[1, labels_moons .== "Moon B"],
        y = Y_moons[2, labels_moons .== "Moon B"],
        mode = "markers",
        name = "Moon B",
        marker = Config(size=4, color="#ff7f0e", opacity=0.85)
    )
]

Plot(traces_trimap_moons, Config(title="Two Moons: TriMAP 2D Projection (SearchGraph k-NN)"))
```

---

## 2. 3 Intertwined Spirals Manifold

Complex spiral structures are notoriously challenging for dimensionality reduction methods because standard random negative sampling can mistakenly pull distant spiral arms together across gaps.

By choosing negative samples from **Farthest First Traversal (`fft`)** landmarks and weighting them by their Voronoi basin sizes (`sample_probs` from `res.nn`), TriMAP achieves clean global unwrapping.

### 2.1 Generating and Visualizing 2D Spiral Data

```@example spirals
using SimilaritySearch, Trimap
using PlotlyLight, Random, Statistics

Random.seed!(42)

function make_spirals(n_points_per_arm=600, noise=0.03f0)
    arms = []
    labels = []
    
    for arm in 1:3
        theta = range(0.5f0, Float32(4.0 * π), length=n_points_per_arm) .+ Float32((arm - 1) * (2π / 3))
        r = range(0.2f0, 2.0f0, length=n_points_per_arm)
        
        x = r .* cos.(theta) .+ noise .* randn(Float32, n_points_per_arm)
        y = r .* sin.(theta) .+ noise .* randn(Float32, n_points_per_arm)
        
        push!(arms, vcat(x', y'))
        push!(labels, fill("Spiral $(arm)", n_points_per_arm))
    end
    
    X = Matrix{Float32}(hcat(arms...))
    labels_all = vcat(labels...)
    X, labels_all
end

X_spirals, labels_spirals = make_spirals(600, 0.03f0)
println("Spirals dataset size: ", size(X_spirals))

colors = Dict("Spiral 1" => "#636EFA", "Spiral 2" => "#EF553B", "Spiral 3" => "#00CC96")

# 1. Visualize Original 2D Input Data
traces_input_spirals = Config[]
for arm_name in ["Spiral 1", "Spiral 2", "Spiral 3"]
    idx = findall(==(arm_name), labels_spirals)
    push!(traces_input_spirals, Config(
        x = X_spirals[1, idx],
        y = X_spirals[2, idx],
        mode = "markers",
        name = arm_name,
        marker = Config(size=3, color=colors[arm_name], opacity=0.85)
    ))
end

Plot(traces_input_spirals, Config(title="3 Intertwined Spirals: Original 2D Input Space"))
```

### 2.2 Baseline: 2D PCA Projection

```@example spirals
Y_pca_spirals = pca_init(X_spirals, 2; scale=1.0f0)

traces_pca_spirals = Config[]
for arm_name in ["Spiral 1", "Spiral 2", "Spiral 3"]
    idx = findall(==(arm_name), labels_spirals)
    push!(traces_pca_spirals, Config(
        x = Y_pca_spirals[1, idx],
        y = Y_pca_spirals[2, idx],
        mode = "markers",
        name = arm_name,
        marker = Config(size=3, color=colors[arm_name], opacity=0.85)
    ))
end

Plot(traces_pca_spirals, Config(title="3 Spirals: Linear 2D PCA Projection"))
```

Because PCA is a linear projection, it preserves the rotation and entanglement of the spirals without unwrapping the non-linear manifold.

### 2.3 Computing $k$-NN

```@example spirals
db_spirals = MatrixDatabase(X_spirals)
index_spirals = ExhaustiveSearch(Dist.L2(), db_spirals)
ctx = GenericContext()
knns_spirals, dists_spirals = allknn(index_spirals, ctx, 20)
```

### 2.4 TriMAP 2D Embedding

```@example spirals
model_spirals = fit(
    Trimap,
    X_spirals,
    knns_spirals,
    dists_spirals;
    maxoutdim=2,
    weight_adj=0.15,
    n_epochs=400,
    learning_rate=0.1
)

Y_spirals = model_spirals.embedding

traces_trimap_spirals = Config[]
for arm_name in ["Spiral 1", "Spiral 2", "Spiral 3"]
    idx = findall(==(arm_name), labels_spirals)
    push!(traces_trimap_spirals, Config(
        x = Y_spirals[1, idx],
        y = Y_spirals[2, idx],
        mode = "markers",
        name = arm_name,
        marker = Config(size=4, color=colors[arm_name], opacity=0.85)
    ))
end

layout_spirals = Config(
    title = "3 Spirals: TriMAP 2D Embedding",
    xaxis = Config(title = "TriMAP 1", zeroline=false),
    yaxis = Config(title = "TriMAP 2", zeroline=false),
    hovermode = "closest"
)

Plot(traces_trimap_spirals, layout_spirals)
```
