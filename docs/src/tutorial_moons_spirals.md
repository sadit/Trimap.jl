# Tutorial 2: Non-linear Manifolds - Two Moons and Spirals

In this tutorial, we evaluate TriMAP on synthetic non-linear manifolds: the classic **Two Moons** dataset and **Intertwined Spirals**. Both start life as a simple 2D curve, but a 2D curve *is already* its own best 2D representation — projecting it back to 2D with PCA would trivially "solve" the problem without testing anything. So for each dataset, we:

1. Generate the **original 2D curve** (for intuition) and then **embed it non-linearly into a higher-dimensional ambient space** (random sinusoidal features + a random rotation + noise). This is the actual input to the pipeline below, and is the point where linear PCA genuinely has something to fail at.
2. Compute and visualize the **linear 2D PCA baseline** on that high-dimensional embedding.
3. Compute the nearest neighbor graph using `SimilaritySearch.SearchGraph` or `ExhaustiveSearch`.
4. Apply **TriMAP** and visualize the recovered 2D manifold with **PlotlyLight.jl**.

---

## 1. Two Moons Manifold

### 1.1 Generating and Visualizing the Original 2D Manifold

```@example moons
using SimilaritySearch, Trimap
using PlotlyLight, Random, Statistics, LinearAlgebra

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

X_moons_2d, labels_moons = make_moons(1200, 0.05f0)
println("Moons dataset size (ground-truth 2D curve): ", size(X_moons_2d))

# Visualize the ground-truth 2D shape we are trying to recover
traces_input_moons = [
    Config(
        x = X_moons_2d[1, labels_moons .== "Moon A"],
        y = X_moons_2d[2, labels_moons .== "Moon A"],
        mode = "markers",
        name = "Moon A",
        marker = Config(size=4, color="#1f77b4", opacity=0.85)
    ),
    Config(
        x = X_moons_2d[1, labels_moons .== "Moon B"],
        y = X_moons_2d[2, labels_moons .== "Moon B"],
        mode = "markers",
        name = "Moon B",
        marker = Config(size=4, color="#ff7f0e", opacity=0.85)
    )
]

Plot(traces_input_moons, Config(title="Two Moons: Ground-Truth 2D Manifold"))
```

### 1.2 Embedding into a Higher-Dimensional Ambient Space

We lift the 2D curve into a 50-dimensional ambient space with a random non-linear (sinusoidal) feature map, then mix all 50 coordinates with a random rotation and add a little noise. This is the same trick used to build classic "swiss roll"-style benchmarks: the manifold is still intrinsically 2D and its *local* neighborhood structure is preserved (nearby points on the curve stay close in the 50D space), but the *global* embedding is no longer a simple linear subspace, so linear PCA can no longer trivially recover it.

```@example moons
function embed_high_dim(X2d::AbstractMatrix{Float32}, D::Integer; noise=0.02f0, rng=Random.default_rng())
    d, n = size(X2d)
    A = randn(rng, Float32, D - d, d)              # random frequencies
    b = Float32(2π) .* rand(rng, Float32, D - d)    # random phases
    extra = sin.(A * X2d .+ b)                      # (D-d, n) non-linear features of (x, y)
    X_stack = vcat(X2d, extra)                      # (D, n)
    Qmat = Matrix(qr(randn(rng, Float32, D, D)).Q)  # random rotation, mixes all D coordinates
    Qmat * X_stack .+ noise .* randn(rng, Float32, D, n)
end

Random.seed!(7)
X_moons = embed_high_dim(X_moons_2d, 50; noise=0.02f0)
println("Moons dataset size (50D ambient embedding): ", size(X_moons))
```

### 1.3 Baseline: 2D PCA Projection

```@example moons
Xc = X_moons .- mean(X_moons, dims=2)
var_explained = svd(Xc).S .^ 2 ./ sum(svd(Xc).S .^ 2)
println("Variance explained by top 2 PCs: ", round(sum(var_explained[1:2]) * 100, digits=1), "%")
```

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

Plot(traces_pca_moons, Config(title="Two Moons: Linear 2D PCA Projection of the 50D Embedding"))
```

With most of the variance spread outside the top 2 components, the PCA projection visibly tangles the two moons — a real linear-projection failure, unlike projecting the original 2D curve onto itself.

### 1.4 Approximate $k$-NN with `SearchGraph`

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

### 1.5 TriMAP Embedding

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

TriMAP with its default settings recovers two clean, well-separated crescents from the 50D embedding — the two moons are simple, well-separated curves, so the default inlier/outlier margin triplets work well here.

---

## 2. 3 Intertwined Spirals Manifold

Complex spiral structures are notoriously challenging for dimensionality reduction methods. As with the moons above, we first embed the 2D spiral non-linearly into a higher-dimensional ambient space so PCA has a real unwrapping problem to fail at.

### 2.1 Generating and Visualizing the Original 2D Spiral Curve

```@example spirals
using SimilaritySearch, Trimap
using PlotlyLight, Random, Statistics, LinearAlgebra

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

X_spirals_2d, labels_spirals = make_spirals(600, 0.03f0)
println("Spirals dataset size (ground-truth 2D curve): ", size(X_spirals_2d))

colors = Dict("Spiral 1" => "#636EFA", "Spiral 2" => "#EF553B", "Spiral 3" => "#00CC96")

traces_input_spirals = Config[]
for arm_name in ["Spiral 1", "Spiral 2", "Spiral 3"]
    idx = findall(==(arm_name), labels_spirals)
    push!(traces_input_spirals, Config(
        x = X_spirals_2d[1, idx],
        y = X_spirals_2d[2, idx],
        mode = "markers",
        name = arm_name,
        marker = Config(size=3, color=colors[arm_name], opacity=0.85)
    ))
end

Plot(traces_input_spirals, Config(title="3 Intertwined Spirals: Ground-Truth 2D Manifold"))
```

### 2.2 Embedding into a Higher-Dimensional Ambient Space

```@example spirals
function embed_high_dim(X2d::AbstractMatrix{Float32}, D::Integer; noise=0.02f0, rng=Random.default_rng())
    d, n = size(X2d)
    A = randn(rng, Float32, D - d, d)
    b = Float32(2π) .* rand(rng, Float32, D - d)
    extra = sin.(A * X2d .+ b)
    X_stack = vcat(X2d, extra)
    Qmat = Matrix(qr(randn(rng, Float32, D, D)).Q)
    Qmat * X_stack .+ noise .* randn(rng, Float32, D, n)
end

Random.seed!(7)
X_spirals = embed_high_dim(X_spirals_2d, 50; noise=0.02f0)
println("Spirals dataset size (50D ambient embedding): ", size(X_spirals))
```

### 2.3 Baseline: 2D PCA Projection

```@example spirals
Xc = X_spirals .- mean(X_spirals, dims=2)
var_explained = svd(Xc).S .^ 2 ./ sum(svd(Xc).S .^ 2)
println("Variance explained by top 2 PCs: ", round(sum(var_explained[1:2]) * 100, digits=1), "%")
```

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

Plot(traces_pca_spirals, Config(title="3 Spirals: Linear 2D PCA Projection of the 50D Embedding"))
```

With less than half the variance captured by 2 components, linear PCA scrambles the three arms into an unrecognizable blob.

### 2.4 Computing $k$-NN

```@example spirals
db_spirals = MatrixDatabase(X_spirals)
index_spirals = ExhaustiveSearch(Dist.L2(), db_spirals)
ctx = GenericContext()
knns_spirals, dists_spirals = allknn(index_spirals, ctx, 20)
```

### 2.5 TriMAP with Default Settings

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
    title = "3 Spirals: TriMAP 2D Embedding (default settings)",
    xaxis = Config(title = "TriMAP 1", zeroline=false),
    yaxis = Config(title = "TriMAP 2", zeroline=false),
    hovermode = "closest"
)

Plot(traces_trimap_spirals, layout_spirals)
```

This is noticeably better than PCA, but still tangled in places — each spiral arm is a single continuous curve with no real cluster boundary between "near" and "far" neighbors along it, so the default inlier/outlier margin triplets (tuned for genuinely clustered data) fight against the smoothness of the curve.

### 2.6 Tuning for Continuous Manifolds: Fewer Margin Outliers + `fft` Negative Sampling

Two changes help specifically because this manifold is one continuous curve per arm, not a set of clusters:

- **`n_outliers=0`**: drop the inlier/outlier margin triplets entirely and rely only on inliers plus random negatives (`n_random`). Without a real cluster boundary, the margin triplets mostly inject noise; removing them stops that fight.
- **`fft`/`sample_probs` landmark negative sampling**: pick a diverse landmark set with Farthest First Traversal (`fft`) and weight each by its Voronoi basin size (`res_fft.nn` counts), instead of drawing negatives uniformly from all `n` points.

```@example spirals
res_fft = fft(Dist.L2(), db_spirals, 60)
sample_centers = res_fft.centers

counts = Dict{UInt32, Float32}()
for c in res_fft.nn
    counts[c] = get(counts, c, 0f0) + 1f0
end
sample_probs = Float32[get(counts, c, 0f0) / length(res_fft.nn) for c in sample_centers]

model_spirals_tuned = fit(
    Trimap,
    X_spirals,
    knns_spirals,
    dists_spirals;
    sample=sample_centers,
    sample_probs=sample_probs,
    n_outliers=0,
    n_random=7,
    maxoutdim=2,
    weight_adj=0.15,
    n_epochs=400,
    learning_rate=0.1
)

Y_spirals_tuned = model_spirals_tuned.embedding

traces_trimap_spirals_tuned = Config[]
for arm_name in ["Spiral 1", "Spiral 2", "Spiral 3"]
    idx = findall(==(arm_name), labels_spirals)
    push!(traces_trimap_spirals_tuned, Config(
        x = Y_spirals_tuned[1, idx],
        y = Y_spirals_tuned[2, idx],
        mode = "markers",
        name = arm_name,
        marker = Config(size=4, color=colors[arm_name], opacity=0.85)
    ))
end

layout_spirals_tuned = Config(
    title = "3 Spirals: TriMAP 2D Embedding (n_outliers=0 + fft landmark negatives)",
    xaxis = Config(title = "TriMAP 1", zeroline=false),
    yaxis = Config(title = "TriMAP 2", zeroline=false),
    hovermode = "closest"
)

Plot(traces_trimap_spirals_tuned, layout_spirals_tuned)
```

This tuned setting unwinds the three arms far more cleanly than both the PCA baseline and the default TriMAP run. The general lesson: `n_outliers`/margin triplets are designed for data with genuine cluster boundaries (like [FashionMNIST](tutorial_fashion_mnist.md) or [Iris](tutorial_iris.md)); for data that is intrinsically one continuous curve per class, lowering or zeroing `n_outliers` and leaning on `n_random` (optionally with `fft`/`sample_probs` landmarks) is worth trying.
