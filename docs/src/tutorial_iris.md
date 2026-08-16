# Tutorial 1: Tabular Data - Iris Dataset

In this tutorial, we demonstrate how to use **Trimap.jl** to reduce the 4-dimensional [Fisher's Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) to 2 dimensions, comparing PCA initialization with random noise initialization, and visualizing the clusters interactively using **PlotlyLight.jl**.

---

## 1. Loading the Iris Dataset

We load the standard 150 samples across 3 classes (*setosa*, *versicolor*, *virginica*), each with 4 attributes (sepal length, sepal width, petal length, petal width).

```@example iris
using SimilaritySearch, Trimap
using MLDatasets, DataFrames
using PlotlyLight, Random, Statistics

# Load Fisher Iris Dataset from MLDatasets.jl
dataset = Iris()

# Features matrix: (4 × 150) Float32 matrix
X = Matrix{Float32}(transpose(Matrix(dataset.features)))
labels = String.(dataset.targets.class)
unique_species = unique(labels)

println("Dataset size: ", size(X))
println("Classes: ", unique_species)
```

---

## 2. Baseline: 2D PCA Projection

Before applying non-linear dimensionality reduction, we compute a standard linear 2D Principal Component Analysis (PCA) projection using `pca_init`:

```@example iris
Y_pca = pca_init(X, 2; scale=1.0f0)

colors = Dict("Iris-setosa" => "#636EFA", "Iris-versicolor" => "#EF553B", "Iris-virginica" => "#00CC96")
traces_pca = Config[]

for species in unique_species
    idx = findall(==(species), labels)
    push!(traces_pca, Config(
        x = Y_pca[1, idx],
        y = Y_pca[2, idx],
        mode = "markers",
        name = species,
        text = ["Species: $(species)<br>Sepal L: $(round(X[1, i], digits=2))<br>Petal L: $(round(X[3, i], digits=2))" for i in idx],
        hoverinfo = "text",
        marker = Config(size=9, color=colors[species], opacity=0.85)
    ))
end

layout_pca = Config(
    title = "Iris Dataset: Linear 2D PCA Projection",
    xaxis = Config(title = "PC 1", zeroline=false),
    yaxis = Config(title = "PC 2", zeroline=false),
    hovermode = "closest"
)

Plot(traces_pca, layout_pca)
```

While linear PCA separates *Setosa*, *Versicolor* and *Virginica* remain heavily overlapping along the linear subspace.

---

## 3. Computing the Nearest Neighbor Graph

In `Trimap.jl`, neighbor search is decoupled from optimization. We use `SimilaritySearch.ExhaustiveSearch` with Euclidean distance `Dist.L2()`:

```@example iris
db = MatrixDatabase(X)
index = ExhaustiveSearch(Dist.L2(), db)
ctx = GenericContext()

k = 20
knns, dists = allknn(index, ctx, k)

println("knns matrix: ", size(knns), " of ", eltype(knns))
println("dists matrix: ", size(dists), " of ", eltype(dists))
```

---

## 4. Fitting Non-Parametric TriMAP

We fit TriMAP using `X`, `knns`, and `dists` (PCA initialization is performed automatically from `X`):

```@example iris
model = fit(
    Trimap,
    X,
    knns,
    dists;
    maxoutdim=2,
    n_epochs=400,
    learning_rate=0.1
)

Y = model.embedding # (2, 150)
println("Embedding size: ", size(Y))
```

---

## 5. TriMAP 2D Embedding Visualization with PlotlyLight

```@example iris
traces = Config[]

for species in unique_species
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
