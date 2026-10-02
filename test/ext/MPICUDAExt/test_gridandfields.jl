using Test

import MPI
import FDGrids
import CUDA
import Adapt
import HaloArrays

using ReSolverFlowsBase

const MPICUDAExt = Base.get_extension(ReSolverFlowsBase, :MPICUDAExt)

include("../../mock_channel_grid.jl")

# setup MPI environment
MPI.Initialized() || MPI.Init()
comm   = MPI.COMM_WORLD
nranks = MPI.Comm_size(comm)
rank   = MPI.Comm_rank(comm)

# Pick a wall-normal size that divides cleanly across any number of ranks
# used in CI (we test with 1 and 4 in `runtests.jl`).
const Ny = 16; const Nx = 7; const Nz = 9; const Nt = 5
const NHALO = 1

# construct grid
gp = MockChannelGrid(Ny, Nx, Nz, Nt)
g  = distributed(gp, comm; decomposed_physical_dims=(:x2,),
                           nprocesses              =(nranks,),
                           nhalo                   =(NHALO,))

Ny_local = Ny ÷ nranks
y_offset = rank * Ny_local

@testset "Decomposed CUDA grid                                                " begin
    gd = @test_nowarn CUDA.cu(g)
    @test gd isa MPICUDAExt.DecomposedGPUGrid{Float32, 4, (2, 1, 3, 4), (2, 3, 4), (1,), (1, 0, 0, 0), (Ny_local, 7, 9, 5)}

    gd_adapt = @test_nowarn Adapt.adapt(CUDA.KernelAdaptor(), gd)
    @test gd_adapt isa MPICUDAExt.DecomposedDeviceGrid{Float32, 4, (2, 1, 3, 4), (2, 3, 4), (1,), (1, 0, 0, 0), (Ny_local, 7, 9, 5)}
    @test gd_adapt.parent.parent.D₁ isa FDGrids.DiffMatrix{Float32, 3, true, <:CUDA.CuDeviceArray}
    @test gd_adapt.parent.parent.D₂ isa FDGrids.DiffMatrix{Float32, 3, true, <:CUDA.CuDeviceArray}
    @test isbits(gd_adapt)
end

@testset "Decomosed fields                                                    " begin
    # construct device grid
    gd = CUDA.cu(g)

    @testset "FTField" begin
        ud = FTField(gd)
        @test ud isa FTField{<:MPICUDAExt.DecomposedGPUGrid{Float32}, <:HaloArrays.HaloArray{ComplexF32}}
        @test size(ud) == (Ny_local, (Nx >> 1) + 1, Nz, Nt)
        @test typeof(CUDA.cu(FTField(g))) == typeof(ud)
        @test isbits(Adapt.adapt(CUDA.KernelAdaptor(), ud))
    end

    @testset "Field" begin
        ud = Field(gd)
        @test ud isa Field{<:MPICUDAExt.DecomposedGPUGrid{Float32}, <:HaloArrays.HaloArray{Float32}}
        @test size(ud) == (Ny_local, Nx, Nz, Nt)
        @test typeof(CUDA.cu(Field(g))) == typeof(ud)
        @test isbits(Adapt.adapt(CUDA.KernelAdaptor(), ud))
    end
end
