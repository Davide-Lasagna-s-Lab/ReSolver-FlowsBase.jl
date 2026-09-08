@testset "CUDA fields                                                         " begin
    # use this grid over all these tests
    gd = CUDA.cu(g)

    @testset "FTField" begin
        u  = FTField(g)
        ud = FTField(gd)

        @test ud isa FTField{<:CUDAExt.GPUGrid, <:CuArray{ComplexF32}}
        @test size(ud) == (16, 8, 15, 15)
        @test typeof(CUDA.cu(u)) == typeof(ud)
    end

    @testset "Field" begin
        u   = Field(g)
        ud  = Field(gd; dealias=false)
        udd = Field(gd; dealias=true)

        @test ud  isa Field{<:CUDAExt.GPUGrid, <:CuArray{Float32}}
        @test udd isa Field{<:CUDAExt.GPUGrid, <:CuArray{Float32}}
        @test size(ud)  == (16, 15, 15, 15)
        @test size(udd) == (16, 23, 23, 23)
        @test typeof(CUDA.cu(u)) == typeof(ud) == typeof(udd)
    end

    @testset "VectorFied" begin
        N   = rand(1:4)
        û   = VectorField(g,  FTField; N=N)
        u   = VectorField(g,    Field; N=N)
        ûd  = VectorField(gd, FTField; N=N)
        ud  = VectorField(gd,   Field; N=N, dealias=false)
        udd = VectorField(gd,   Field; N=N, dealias=true)

        @test ûd  isa VectorField{N, <:FTField{<:CUDAExt.GPUGrid, <:CuArray{ComplexF32}}}
        @test ud  isa VectorField{N,   <:Field{<:CUDAExt.GPUGrid, <:CuArray{Float32}}}
        @test udd isa VectorField{N,   <:Field{<:CUDAExt.GPUGrid, <:CuArray{Float32}}}
        @test typeof(CUDA.cu(û)) == typeof(ûd)
        @test typeof(CUDA.cu(u)) == typeof(ud) == typeof(udd)
    end

    @testset "ProjectedField" begin
        M = rand(1:Ny)
        N = rand(1:4)
        modes = ntuple(_ -> zeros(ComplexF64, M, Ny, (Nx >> 1) + 1, Nz, Nt), N)
        a  = ReSolverFlowsBase.ProjectedField(g,          modes)
        ad = ReSolverFlowsBase.ProjectedField(gd, map(comp -> CUDA.cu(comp), modes))

        @test ad isa ReSolverFlowsBase.ProjectedField{<:CUDAExt.GPUGrid, <:NTuple{N, <:CuArray{ComplexF32}}, <:CuArray{ComplexF32}}
        @test size(ad) == (M, 8, 15, 15)
        @test typeof(CUDA.cu(a)) == typeof(ad)
    end
end
