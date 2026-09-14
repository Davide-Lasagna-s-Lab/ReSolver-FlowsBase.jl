@testset "CUDA dot product                                                    " begin
    # construct modes
    M = 5
    Ψ = ntuple(n -> zeros(ComplexF64, M, Ny, (Nx >> 1) + 1, Nz, Nt), 1)

    # construct projected fields
    a = ProjectedField(g, randn(M, (Nx >> 1) + 1, Nz, Nt), Ψ); ad = CUDA.cu(a)
    b = ProjectedField(g, randn(M, (Nx >> 1) + 1, Nz, Nt), Ψ); bd = CUDA.cu(b)

    @testset "methods are correct" begin
        # initialise dot product methods
        method_twostage = CUDAExt.DotTwoStage(ad)
        method_atomic   = CUDAExt.DotAtomic(ad)
        method_shared   = CUDAExt.DotShared(ad)

        # test result
        res_host = dot(a, b)
        @test abs(res_host - dot(ad, bd, method_twostage)) < 8e-4
        @test abs(res_host - dot(ad, bd, method_atomic))   < 8e-4
        @test abs(res_host - dot(ad, bd, method_shared))   < 8e-4
    end

    # second dummy field for testin auto-tuning
    g2 = MockChannelGrid(16, 3, 1, 1)
    ad2 = CUDA.cu(ProjectedField(g2, randn(2, (3 >> 1) + 1, 1, 1), Ψ))

    @testset "auto-tuning" begin
        @test isempty(CUDAExt.DOT_METHODS)
        CUDAExt.dot_method(ad)
        @test length(CUDAExt.DOT_METHODS) == 1
        reset_dot_cache!()
        @test isempty(CUDAExt.DOT_METHODS)
        CUDAExt.dot_method(ad)
        CUDAExt.dot_method(ad2)
        @test length(CUDAExt.DOT_METHODS) == 2
        reset_dot_cache!(ad2)
        @test length(CUDAExt.DOT_METHODS) == 1
        CUDAExt.dot_method(ad)
        @test length(CUDAExt.DOT_METHODS) == 1
        CUDAExt.dot_method(ad2)
        reset_dot_cache!()
        @test isempty(CUDAExt.DOT_METHODS)
    end
end
