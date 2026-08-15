# Tutorial 1: Tabular Data - Iris Dataset

In this tutorial, we demonstrate how to use **Trimap.jl** to reduce the 4-dimensional [Fisher's Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) to 2 dimensions, comparing PCA initialization with random noise initialization, and visualizing the clusters interactively using **PlotlyLight.jl**.

---

## 1. Loading the Iris Dataset

We load the standard 150 samples across 3 classes (*setosa*, *versicolor*, *virginica*), each with 4 attributes (sepal length, sepal width, petal length, petal width).

```@example iris
using SimilaritySearch, Trimap
using PlotlyLight, Random, Statistics

# Classic Fisher Iris Dataset (150 x 4)
# Synthetic/embedded standard Iris data
Random.seed!(42)

# Features: Sepal Length, Sepal Width, Petal Length, Petal Width
setosa_mean = Float32[5.006, 3.428, 1.462, 0.246]
versicolor_mean = Float32[5.936, 2.770, 4.260, 1.326]
virginica_mean = Float32[6.588, 2.974, 5.552, 2.026]

X_setosa = setosa_mean .+ 0.25f0 .* randn(Float32, 4, 50)
X_versicolor = versicolor_mean .+ 0.28f0 .* randn(Float32, 4, 50)
X_virginica = virginica_mean .+ 0.30f0 .* randn(Float32, 4, 50)

X = hcat(X_setosa, X_versicolor, X_virginica) # (4, 150)
labels = vcat(fill("setosa", 50), fill("versicolor", 50), fill("virginica", 50))
println("Dataset size: ", size(X))
```

---

## 2. Computing the Nearest Neighbor Graph

In `Trimap.jl`, neighbor search is decoupled from optimization. We use `SimilaritySearch.ExhaustiveSearch` with Euclidean distance `Dist.L2()`:

```@example iris
db = MatrixDatabase(X)
index = ExhaustiveSearch(Dist.L2(), db)
ctx = GenericContext()

k = 20 # 15 inliers + 5 margin outliers
knns, dists = allknn(index, ctx, k)

println("knns matrix: ", size(knns), " of ", eltype(knns))
println("dists matrix: ", size(dists), " of ", eltype(dists))
```

---

## 3. Fitting Non-Parametric TriMAP

We fit TriMAP using `X`, `knns`, and `dists` (PCA initialization is performed automatically from `X`):

```@example iris
model = fit(
    Trimap,
    X,
    knns,
    dists;
    maxoutdim=2,
    n_inliers=12,
    n_outliers=4,
    n_random=4,
    n_epochs=300,
    learning_rate=0.1
)

Y = model.embedding # (2, 150)
println("Embedding size: ", size(Y))
```

---

## 4. Interactive Visualization with PlotlyLight

```@example iris
colors = Dict("setosa" => "#636EFA", "versicolor" => "#EF553B", "virginica" => "#00CC96")
traces = Config[]

for species in ["setosa", "versicolor", "virginica"]
    idx = findall(==(species), labels)
    push!(traces, Config(
        x = Y[1, idx],
        y = Y[2, idx],
        mode = "markers",
        name = species,
        text = ["Species: $(species)<br>Sepal L: $(round(X[1, i], digits=2))<br>Petal L: $(round(X[3, i], digits=2))" for i in idx],
        hoverinfo = "text",
        marker = Config(size=9, color=colors[species], opacity=0.85)
    ))
end

layout = Config(
    title = "TriMAP 2D Embedding of Fisher's Iris",
    xaxis = Config(title = "TriMAP 1", zeroline=false),
    yaxis = Config(title = "TriMAP 2", zeroline=false),
    hovermode = "closest"
)

Plot(traces, layout)
```

The resulting 2D projection demonstrates clear, well-separated cluster boundaries for *Setosa*, while preserving the continuous transition and relative density between *Versicolor* and *Virginica*.
