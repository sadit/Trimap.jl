# This file is a part of Trimap.jl

using Test
using Trimap
using SimilaritySearch
using Lux
using Zygote
using Random
using LinearAlgebra

@testset "Trimap.jl Tests" begin
    Random.seed!(42)
    dim = 8
    n = 100
    X = randn(Float32, dim, n)
    db = MatrixDatabase(X)

    # 1. Exact k-NN with ExhaustiveSearch
    ex_index = ExhaustiveSearch(Dist.L2(), db)
    k = 25
    ctx_ex = GenericContext()
    knns_exact, dists_exact = allknn(ex_index, ctx_ex, k)

    # 2. Approximate k-NN with SearchGraph (recall < 1)
    G = SearchGraph(Dist.L2(), db)
    ctx = SearchGraphContext()
    index!(G, ctx)
    optimize_index!(G, ctx, MinRecall(0.85))
    knns_approx, dists_approx = allknn(G, ctx, k)

    @testset "Triplet generation" begin
        # Default proportional generation (2/3 k and 1/3 k)
        i0, j0, k0, w0 = generate_triplets(knns_exact, dists_exact)
        @test length(i0) == length(j0) == length(k0) == length(w0)
        @test length(i0) > 0

        # Direct matrices from ExhaustiveSearch with custom parameters
        i, j, k_trip, w = generate_triplets(knns_exact, dists_exact; n_inliers=10, n_outliers=5, n_random=5)
        @test length(i) == length(j) == length(k_trip) == length(w)
        @test length(i) > 0
        @test all(1 .<= i .<= n)
        @test all(1 .<= j .<= n)
        @test all(1 .<= k_trip .<= n)
        @test all(w .>= 0.0f0)

        # Direct matrices from SearchGraph with recall < 1
        i2, j2, k2, w2 = generate_triplets(knns_approx, dists_approx; n_inliers=10, n_outliers=5, n_random=5)
        @test length(i2) == length(j2) == length(k2) == length(w2)
        @test length(i2) > 0
        @test all(1 .<= i2 .<= n)
        @test all(1 .<= j2 .<= n)
        @test all(1 .<= k2 .<= n)
        @test all(w2 .>= 0.0f0)

        # Dimension mismatch error check
        @test_throws DimensionMismatch generate_triplets(knns_exact[1:10, :], dists_exact)

        # Type strictness check (user is responsible for passing UInt32 and Float32)
        @test_throws MethodError generate_triplets(Int64.(knns_exact), dists_exact)
        @test_throws MethodError generate_triplets(knns_exact, Float64.(dists_exact))
    end

    @testset "PCA Initialization" begin
        Y_pca = pca_init(X, 2)
        @test size(Y_pca) == (2, n)
        @test all(isfinite, Y_pca)
    end

    @testset "Loss and Gradients" begin
        out_dim = 2
        Y = randn(Float32, out_dim, n) .* 0.01f0
        i, j, k_trip, w = generate_triplets(knns_exact, dists_exact; n_inliers=5, n_outliers=2, n_random=2)

        l = trimap_loss(Y, i, j, k_trip, w)
        @test isfinite(l)
        @test l > 0

        # Gradient w.r.t Y
        grads = Zygote.gradient(y -> trimap_loss(y, i, j, k_trip, w), Y)
        @test size(grads[1]) == size(Y)
        @test all(isfinite, grads[1])
    end

    @testset "Non-parametric TriMAP fit" begin
        # Using exact search k-NN with default PCA initialization
        model_exact = fit(Trimap, X, knns_exact, dists_exact; maxoutdim=2, n_inliers=10, n_outliers=5, n_random=5, n_epochs=20, learning_rate=0.1)
        @test model_exact isa TrimapModel
        @test size(model_exact.embedding) == (2, n)
        @test all(isfinite, model_exact.embedding)

        # Using approximate search k-NN with recall < 1 and random init
        model_approx = fit(Trimap, X, knns_approx, dists_approx; maxoutdim=2, Y_init=:random, n_inliers=10, n_outliers=5, n_random=5, n_epochs=20)
        @test model_approx isa TrimapModel
        @test size(model_approx.embedding) == (2, n)
        @test all(isfinite, model_approx.embedding)

        # Using explicit matrix Y_init
        Y_init = pca_init(X, 2)
        model_custom_init = fit(Trimap, X, knns_exact, dists_exact; maxoutdim=2, Y_init=Y_init, n_epochs=10)
        @test size(model_custom_init.embedding) == (2, n)

        # Using fft negative sampling
        fft_sample = fft(Dist.L2(), db, 20)
        model_fft = fit(Trimap, X, knns_exact, dists_exact; sample=fft_sample.centers, maxoutdim=2, n_inliers=10, n_outliers=5, n_random=5, n_epochs=20)
        @test model_fft isa TrimapModel
        @test size(model_fft.embedding) == (2, n)
        @test all(isfinite, model_fft.embedding)

        # Using index range as sample
        model_range_sample = fit(Trimap, X, knns_exact, dists_exact; sample=1:50, maxoutdim=2, n_inliers=10, n_outliers=5, n_random=5, n_epochs=20)
        @test model_range_sample isa TrimapModel
        @test size(model_range_sample.embedding) == (2, n)

        # Using fft negative sampling with cluster probabilities from .nn
        counts = Dict{UInt32, Float32}()
        for c in fft_sample.nn
            counts[c] = get(counts, c, 0f0) + 1f0
        end
        sample_probs = Float32[get(counts, c, 0f0) / length(fft_sample.nn) for c in fft_sample.centers]
        model_fft_weighted = fit(Trimap, X, knns_exact, dists_exact; sample=fft_sample.centers, sample_probs=sample_probs, maxoutdim=2, n_inliers=10, n_outliers=5, n_random=5, n_epochs=20)
        @test model_fft_weighted isa TrimapModel
        @test size(model_fft_weighted.embedding) == (2, n)
        @test all(isfinite, model_fft_weighted.embedding)

        # Dimension mismatch for sample_probs
        @test_throws DimensionMismatch fit(Trimap, X, knns_exact, dists_exact; sample=fft_sample.centers, sample_probs=sample_probs[1:5])
    end

    @testset "Parametric TriMAP fit and predict" begin
        # Default MLP with exact k-NN and fft negative sampling with sample_probs
        fft_sample = fft(Dist.L2(), db, 20)
        counts = Dict{UInt32, Float32}()
        for c in fft_sample.nn
            counts[c] = get(counts, c, 0f0) + 1f0
        end
        sample_probs = Float32[get(counts, c, 0f0) / length(fft_sample.nn) for c in fft_sample.centers]

        pm = fit(ParametricTrimap, X, knns_exact, dists_exact; sample=fft_sample.centers, sample_probs=sample_probs, maxoutdim=2, hidden_dims=(32,), n_inliers=10, n_outliers=5, n_random=5, n_epochs=20)
        @test pm isa ParametricTrimap
        
        # In-sample prediction
        Y_train = predict(pm, X)
        @test size(Y_train) == (2, n)
        @test all(isfinite, Y_train)

        # Out-of-sample prediction (unseen points)
        X_test = randn(Float32, dim, 15)
        Y_test = predict(pm, X_test)
        @test size(Y_test) == (2, 15)
        @test all(isfinite, Y_test)

        # Custom Lux Layer with approximate k-NN (recall < 1)
        custom_net = Lux.Chain(Lux.Dense(dim => 16, Lux.tanh), Lux.Dense(16 => 3))
        pm_custom = fit(ParametricTrimap, X, knns_approx, dists_approx; model=custom_net, maxoutdim=3, n_epochs=15)
        @test pm_custom isa ParametricTrimap
        Y_custom = predict(pm_custom, X_test)
        @test size(Y_custom) == (3, 15)
    end
end
