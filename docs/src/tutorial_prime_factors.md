# Tutorial 3: Non-Vector Metric Spaces - Prime Factorization

One of the most powerful aspects of combining **SimilaritySearch.jl** with **Trimap.jl** is that TriMAP can embed **any metric space**, not just Euclidean vector spaces.

In this tutorial (inspired by the [SimilaritySearch.jl prime factors tutorial](https://github.com/sadit/SimilaritySearch.jl)), we represent integers by their **prime factor sets**, measure their arithmetic similarity using the **Jaccard distance**, and project the resulting discrete metric space into a continuous 2D manifold. We also demonstrate **Parametric TriMAP** using Lux.jl to project unseen integers.

---

## 1. Prime Factorization as a Metric Space

Every integer $n \ge 2$ has a unique prime factorization. For example:
- $12 = 2^2 \times 3 \implies \text{factors}(12) = \{2, 3\}$
- $18 = 2 \times 3^2 \implies \text{factors}(18) = \{2, 3\}$
- $35 = 5 \times 7 \implies \text{factors}(35) = \{5, 7\}$

Two integers are similar if they share a high proportion of prime factors. We quantify this using the **Jaccard distance**:
$$d(A, B) = 1 - \frac{|A \cap B|}{|A \cup B|}$$

```@example primes
using SimilaritySearch, Trimap
using Primes, PlotlyLight, Lux, Random, Statistics

Random.seed!(42)

# Consider integers from 2 to 3000
N_max = 3000
numbers = collect(2:N_max)
n_items = length(numbers)

# Extract unique prime factors for each number
prime_sets = [unique(factor(Vector, num)) for num in numbers]
println("First 5 numbers and their prime factors:")
for i in 1:5
    println("  $(numbers[i]) => $(prime_sets[i])")
end

# Build a binary prime-indicator feature matrix (dim = number of primes <= N_max)
all_primes = primes(N_max)
prime_to_idx = Dict(p => i for (i, p) in enumerate(all_primes))
n_features = length(all_primes)

function featurize(factor_set)
    v = zeros(Float32, n_features)
    for p in factor_set
        if haskey(prime_to_idx, p)
            v[prime_to_idx[p]] = 1.0f0
        end
    end
    v
end

X_primes = hcat([featurize(fs) for fs in prime_sets]...) # (n_features, n_items)
println("Prime indicator matrix: ", size(X_primes))
```

---

## 2. Indexing with `SearchGraph` and `Dist.Sets.Jaccard`

We wrap the factor sets in a `VectorDatabase` and index them using `SimilaritySearch.SearchGraph` with `Dist.Sets.Jaccard()`:

```@example primes
db_primes = VectorDatabase(prime_sets)
G = SearchGraph(Dist.Sets.Jaccard(), db_primes)
ctx = SearchGraphContext()
index!(G, ctx)

# Optimize for fast 90% recall search
optimize_index!(G, ctx, MinRecall(0.90))

k = 20
knns_primes, dists_primes = allknn(G, ctx, k)
```

---

## 3. FFT Landmark Selection and Probability Weighting

We apply Farthest First Traversal (`fft`) to select 100 well-distributed landmark numbers across the prime factorization metric space, and weight each landmark by the size of its Voronoi cluster in `res_fft.nn`:

```@example primes
res_fft = fft(Dist.Sets.Jaccard(), db_primes, 100)
sample_centers = res_fft.centers

# Compute relative frequencies/probabilities
counts = Dict{UInt32, Float32}()
for c in res_fft.nn
    counts[c] = get(counts, c, 0f0) + 1f0
end
sample_probs = Float32[get(counts, c, 0f0) / length(res_fft.nn) for c in sample_centers]

println("Total landmarks: ", length(sample_centers))
```

---

## 4. Fitting Non-Parametric TriMAP

```@example primes
model_primes = fit(
    Trimap,
    X_primes,
    knns_primes,
    dists_primes;
    maxoutdim=2,
    n_epochs=400,
    learning_rate=0.1
)

Y_primes = model_primes.embedding
```

---

## 5. Interactive Visualization with PlotlyLight

We categorize numbers by their dominant arithmetic characteristics (powers of 2, multiples of 3, 5, 7, and prime numbers) and display interactive tooltips showing the exact factorization when hovering over any point:

```@example primes
# Categorize for visualization
function categorize(num, factors)
    if isprime(num)
        return "Prime Numbers"
    elseif 2 in factors && length(factors) == 1
        return "Powers of 2"
    elseif 2 in factors
        return "Even Numbers (multiples of 2)"
    elseif 3 in factors
        return "Multiples of 3"
    elseif 5 in factors
        return "Multiples of 5"
    else
        return "Other Composites"
    end
end

categories = [categorize(numbers[i], prime_sets[i]) for i in 1:n_items]
unique_cats = ["Prime Numbers", "Powers of 2", "Even Numbers (multiples of 2)", "Multiples of 3", "Multiples of 5", "Other Composites"]
palette = Dict(
    "Prime Numbers" => "#EF553B",
    "Powers of 2" => "#636EFA",
    "Even Numbers (multiples of 2)" => "#AB63FA",
    "Multiples of 3" => "#00CC96",
    "Multiples of 5" => "#FFA15A",
    "Other Composites" => "#7F7F7F"
)

traces_primes = Config[]
for cat in unique_cats
    idx = findall(==(cat), categories)
    push!(traces_primes, Config(
        x = Y_primes[1, idx],
        y = Y_primes[2, idx],
        mode = "markers",
        name = cat,
        text = ["n = $(numbers[i])<br>Factors: $(prime_sets[i])" for i in idx],
        hoverinfo = "text",
        marker = Config(size=5, color=palette[cat], opacity=0.85)
    ))
end

layout_primes = Config(
    title = "TriMAP 2D Embedding of Integers under Jaccard Prime Metric",
    xaxis = Config(title = "TriMAP 1", zeroline=false),
    yaxis = Config(title = "TriMAP 2", zeroline=false),
    hovermode = "closest"
)

Plot(traces_primes, layout_primes)
```

---

## 6. Parametric TriMAP: Out-of-Sample Prediction

What if we want to project **new unseen integers** (e.g. $n \in \{3001, \dots, 3150\}$) without re-optimizing the entire dataset?

We use `ParametricTrimap` backed by a `Lux` neural network trained on the multi-hot feature vectors `X_primes`:

```@example primes
# Fit Parametric TriMAP
pmodel = fit(
    ParametricTrimap,
    X_primes,
    knns_primes,
    dists_primes;
    maxoutdim=2,
    hidden_dims=(128, 64),
    n_epochs=400,
    learning_rate=0.1
)

# Unseen test integers
test_numbers = collect(3001:3150)
test_prime_sets = [unique(factor(Vector, num)) for num in test_numbers]
X_test = hcat([featurize(fs) for fs in test_prime_sets]...)

# Instant O(1) projection
Y_test = predict(pmodel, X_test)
println("Projected $(size(Y_test, 2)) unseen test integers into 2D coordinates!")
```

The parametric model successfully maps new unseen numbers into the appropriate arithmetic geometric regions of the manifold in constant time $O(1)$.
