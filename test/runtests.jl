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

    @testset "Triplet generation" begin
        # Direct matrix
        i, j, k, w = generate_triplets(X; n_inliers=10, n_outliers=5, n_random=5)
        @test length(i) == length(j) == length(k) == length(w)
        @test length(i) > 0
        @test all(1 .<= i .<= n)
        @test all(1 .<= j .<= n)
        @test all(1 .<= k .<= n)
        @test all(w .>= 0.0f0)

        # Using MatrixDatabase
        db = MatrixDatabase(X)
        i2, j2, k2, w2 = generate_triplets(db; n_inliers=10, n_outliers=5, n_random=5)
        @test length(i2) == length(j2) == length(k2) == length(w2)
        @test length(i2) > 0
        @test all(1 .<= i2 .<= n)

        # Using SearchGraph
        G = SearchGraph(Dist.L2(), db)
        ctx = SearchGraphContext()
        index!(G, ctx)
        i3, j3, k3, w3 = generate_triplets(G; n_inliers=10, n_outliers=5, n_random=5, searchctx=ctx)
        @test length(i3) == length(j3) == length(k3) == length(w3)
        @test length(i3) > 0
        @test all(1 .<= i3 .<= n)
    end

    @testset "Loss and Gradients" begin
        out_dim = 2
        Y = randn(Float32, out_dim, n) .* 0.01f0
        i, j, k, w = generate_triplets(X; n_inliers=5, n_outliers=2, n_random=2)

        l = trimap_loss(Y, i, j, k, w)
        @test isfinite(l)
        @test l > 0

        # Gradient w.r.t Y
        grads = Zygote.gradient(y -> trimap_loss(y, i, j, k, w), Y)
        @test size(grads[1]) == size(Y)
        @test all(isfinite, grads[1])
    end

    @testset "Non-parametric TriMAP fit and predict" begin
        model = fit(Trimap, X; out_dim=2, n_inliers=10, n_outliers=5, n_random=5, max_iters=20, learning_rate=0.1)
        @test model isa TrimapModel
        @test size(model.embedding) == (2, n)
        @test all(isfinite, model.embedding)

        # Test with SearchGraph index input
        db = MatrixDatabase(X)
        G = SearchGraph(Dist.L2(), db)
        ctx = SearchGraphContext()
        index!(G, ctx)
        model_g = fit(Trimap, G; maxoutdim=2, n_inliers=10, n_outliers=5, n_random=5, n_epochs=20, searchctx=ctx)
        @test model_g isa TrimapModel
        @test size(model_g.embedding) == (2, n)
    end

    @testset "Parametric TriMAP fit and predict" begin
        # Default MLP
        pm = fit(ParametricTrimap, X; out_dim=2, hidden_dims=(32,), n_inliers=10, n_outliers=5, n_random=5, max_iters=20)
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

        # Custom Lux Layer
        custom_net = Lux.Chain(Lux.Dense(dim => 16, Lux.tanh), Lux.Dense(16 => 3))
        pm_custom = fit(ParametricTrimap, X; model=custom_net, max_iters=15)
        @test pm_custom isa ParametricTrimap
        Y_custom = predict(pm_custom, X_test)
        @test size(Y_custom) == (3, 15)
    end
end
