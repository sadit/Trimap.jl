# Tutorial 4: Real-World Images - FashionMNIST

In this tutorial we apply **Trimap.jl** to the classic [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset: 28×28 grayscale images of clothing items across 10 categories. We load a subset via `MLDatasets.jl`, reduce the raw pixel space with PCA, build an approximate nearest-neighbor graph with `SimilaritySearch.SearchGraph`, and embed with TriMAP using `fft`/`sample_probs` landmark negative sampling — a setting that pays off much more here than on the small synthetic spirals, since FashionMNIST has genuine cluster structure and a much larger `n`.

---

## 1. Loading a FashionMNIST Subset

The full test split has 10,000 images; we take a balanced subset (200 per class = 2,000 images) so the tutorial runs quickly.

```@example fmnist
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"  # allow the dataset download without an interactive prompt

using SimilaritySearch, Trimap
using MLDatasets, PlotlyLight, Random, Statistics

Random.seed!(42)

fmnist = FashionMNIST(:test)
class_names = fmnist.metadata["class_names"]

n_per_class = 200
idxs = Int[]
for c in 0:9
    class_idxs = findall(==(c), fmnist.targets)
    append!(idxs, class_idxs[randperm(length(class_idxs))[1:n_per_class]])
end
shuffle!(idxs)

images = fmnist.features[:, :, idxs]           # (28, 28, n)
labels = fmnist.targets[idxs]                  # 0-based class ids
n = length(idxs)

X = Float32.(reshape(images, 28 * 28, n))       # (784, n) flattened pixel vectors
println("FashionMNIST subset size: ", size(X), "  classes: ", class_names)
```

---

## 2. Baseline: 2D PCA Projection

```@example fmnist
Y_pca = pca_init(X, 2; scale=1.0f0)

colors = ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A","#19D3F3","#FF6692","#B6E880","#FF97FF","#FECB52"]
traces_pca = Config[]
for c in 0:9
    idx = findall(==(c), labels)
    push!(traces_pca, Config(
        x = Y_pca[1, idx],
        y = Y_pca[2, idx],
        mode = "markers",
        name = class_names[c + 1],
        marker = Config(size=4, color=colors[c + 1], opacity=0.8)
    ))
end

Plot(traces_pca, Config(title="FashionMNIST: Linear 2D PCA Projection"))
```

Linear PCA on raw pixels gives some coarse grouping (e.g. footwear vs. clothing), but categories such as *Shirt*, *Pullover*, *Coat* and *T-shirt/top* overlap heavily.

---

## 3. Preprocessing and Approximate $k$-NN

784-dimensional raw pixels make exact nearest-neighbor search unnecessarily expensive. We first project to 50 PCA dimensions (a standard preprocessing step before manifold learning), then build an approximate `SearchGraph` index:

```@example fmnist
X_pca50 = pca_init(X, 50; scale=1.0f0)

db = MatrixDatabase(X_pca50)
G = SearchGraph(Dist.L2(), db)
ctx = SearchGraphContext()
index!(G, ctx)

optimize_index!(G, ctx, MinRecall(0.9))

k = 25
knns, dists = allknn(G, ctx, k)
```

---

## 4. TriMAP Embedding with `fft`/`sample_probs` Negative Sampling

With 2,000 points, drawing negative samples uniformly from all of them is wasteful and noisier than necessary. Instead we pick a diverse landmark set with Farthest First Traversal (`fft`) and weight each landmark by its Voronoi basin size:

```@example fmnist
res_fft = fft(Dist.L2(), db, 120)
sample_centers = res_fft.centers

counts = Dict{UInt32, Float32}()
for c in res_fft.nn
    counts[c] = get(counts, c, 0f0) + 1f0
end
sample_probs = Float32[get(counts, c, 0f0) / length(res_fft.nn) for c in sample_centers]

model = fit(
    Trimap,
    X_pca50,
    knns,
    dists;
    sample=sample_centers,
    sample_probs=sample_probs,
    maxoutdim=2,
    n_epochs=400,
    learning_rate=0.1
)

Y = model.embedding
println("Embedding size: ", size(Y))
```

---

## 5. TriMAP 2D Embedding Visualization

```@example fmnist
traces = Config[]
for c in 0:9
    idx = findall(==(c), labels)
    push!(traces, Config(
        x = Y[1, idx],
        y = Y[2, idx],
        mode = "markers",
        name = class_names[c + 1],
        marker = Config(size=4, color=colors[c + 1], opacity=0.8)
    ))
end

layout = Config(
    title = "FashionMNIST: TriMAP 2D Embedding (784 → 50 PCA → 2D)",
    xaxis = Config(title = "TriMAP 1", zeroline=false),
    yaxis = Config(title = "TriMAP 2", zeroline=false),
    hovermode = "closest"
)

Plot(traces, layout)
```

Footwear (*Sandal*, *Sneaker*, *Ankle boot*), *Trouser* and *Bag* typically form tight, well-separated groups, since they look quite different from one another in raw pixel space. The upper-body garments (*T-shirt/top*, *Pullover*, *Coat*, *Shirt*, *Dress*) remain partially intermixed — that overlap is a genuine property of raw-pixel similarity between visually similar garments, not an artifact of the embedding; a learned feature space (e.g. via [`ParametricTrimap`](@ref)) or more epochs can sharpen it further.
