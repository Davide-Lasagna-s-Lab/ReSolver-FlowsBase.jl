@testset "CUDA galerkin methods                                               " begin
    # construct modes
    M = 5
    Ψ = ntuple(n -> zeros(ComplexF64, M, Ny, (Nx >> 1) + 1, Nz, Nt), 1)

    for nt in 1:Nt, nz in 1:Nz, nx in 2:(Nx >> 1) + 1
        Ψ[1][:, :, nx, nz, nt] .= (Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(ComplexF64, Ny, M)).Q[:, 1:M])'
    end
    for nz in 2:(Nz >> 1) + 1, nt in 2:Nt
        Ψ[1][:, :, 1,     nz,       nt]   .= (Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(ComplexF64, Ny, M)).Q[:, 1:M])'
        Ψ[1][:, :, 1, end-nz+2, end-nt+2] .= conj.(Ψ[1][:, :, 1, nz, nt])
    end
    for nz in 2:Nz
        Ψ[1][:, :, 1,     nz,   1] .= (Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(ComplexF64, Ny, M)).Q[:, 1:M])'
        Ψ[1][:, :, 1, end-nz+2, 1] .= conj.(Ψ[1][:, :, 1, nz, 1])
    end
    for nt in 2:Nt
        Ψ[1][:, :, 1, 1,     nt]   .= (Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(ComplexF64, Ny, M)).Q[:, 1:M])'
        Ψ[1][:, :, 1, 1, end-nt+2] .= conj.(Ψ[1][:, :, 1, 1, nt])
    end
    Ψ[1][:, :, 1, 1, 1] .= (Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(Float64, Ny, M)).Q[:, 1:M])'

    # construct fields
    u = VectorField(g, N=1)
    a = ProjectedField(g, Ψ); a .= randn(ComplexF64, M, (Nx >> 1) + 1, Nz, Nt)
    ReSolverFlowsBase.normalise_mean!(parent(ReSolverFlowsBase.apply_symmetry!(a)), (2, 3, 4))
    ud = CUDA.cu(u)
    ad = CUDA.cu(a)

    # construct all galerkin methods
    project_methods = [
                       CUDAExt.ProjectLoop(),
                       CUDAExt.ProjectShared(ad, ud),
                    #    CUDAExt.ProjectSharedTiled(ad, ud), # ! not implemented yet
                    #    CUDAExt.ProjectSharedTree(ad, ud),  # ! not implemented yet
                       ]
    expand_methods  = [
                       CUDAExt.ExpandModal(over_vector=true),
                       CUDAExt.ExpandModal(over_vector=false),
                       ]

    @testset "projection-expansion are compatible" for pmethod in project_methods, emethod in expand_methods
        out_d = project!(zero(ad), expand!(ud, ad, emethod), pmethod)
        @test norm(Array(parent(ad - out_d))) < 1e-5
    end

    # second dummy field for testin auto-tuning
    g2 = MockChannelGrid(16, 3, 1, 1)
    ud2 = CUDA.cu(VectorField(g2, N=1))
    ad2 = CUDA.cu(ProjectedField(g2, Ψ))

    @testset "projection autotuning" begin
        @test isempty(CUDAExt.PROJECT_METHODS)
        CUDAExt.project_method(ad, ud)
        @test length(CUDAExt.PROJECT_METHODS) == 1
        reset_project_cache!()
        @test isempty(CUDAExt.PROJECT_METHODS)
        CUDAExt.project_method(ad, ud)
        CUDAExt.project_method(ad2, ud2)
        @test length(CUDAExt.PROJECT_METHODS) == 2
        reset_project_cache!(ad2, ud2)
        @test length(CUDAExt.PROJECT_METHODS) == 1
        CUDAExt.project_method(ad, ud)
        @test length(CUDAExt.PROJECT_METHODS) == 1
        CUDAExt.project_method(ad2, ud2)
        reset_project_cache!()
        @test isempty(CUDAExt.PROJECT_METHODS)
    end

    @testset "expansion autotuning" begin
        @test isempty(CUDAExt.EXPAND_METHODS)
        CUDAExt.expand_method(ud, ad)
        @test length(CUDAExt.EXPAND_METHODS) == 1
        reset_expand_cache!()
        @test isempty(CUDAExt.EXPAND_METHODS)
        CUDAExt.expand_method(ud, ad)
        CUDAExt.expand_method(ud2, ad2)
        @test length(CUDAExt.EXPAND_METHODS) == 2
        reset_expand_cache!(ud2, ad2)
        @test length(CUDAExt.EXPAND_METHODS) == 1
        CUDAExt.expand_method(ud, ad)
        @test length(CUDAExt.EXPAND_METHODS) == 1
        CUDAExt.expand_method(ud2, ad2)
        reset_expand_cache!()
        @test isempty(CUDAExt.EXPAND_METHODS)
    end
end
