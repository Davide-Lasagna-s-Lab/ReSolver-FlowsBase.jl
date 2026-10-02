using Test

import MPI
import FDGrids
import CUDA

using ReSolverFlowsBase

include("../../mock_channel_grid.jl")

# setup MPI environment
MPI.Initialized() || MPI.Init()
comm   = MPI.COMM_WORLD
nranks = MPI.Comm_size(comm)
rank   = MPI.Comm_rank(comm)

function build_seeded_field(g, dealias)
    u = ReSolverFlowsBase.Field(g; dealias=dealias)
    y, x, z, t = ReSolverFlowsBase.points(g; dealias=dealias)
    parent(u) .= @. sin(y) * cos(x) + 0.3 * sin(2*z) * cos(t)
    return u
end

# Pick a wall-normal size that divides cleanly across any number of ranks
# used in CI (we test with 1 and 4 in `runtests.jl`).
const Ny = 16; const Nx = 7; const Nz = 9; const Nt = 5
const NHALO = 1

# construct grid
gp = MockChannelGrid(Ny, Nx, Nz, Nt, α=1, β=1)
g  = CUDA.cu(distributed(gp, comm; decomposed_physical_dims=(:x2,),
                                   nprocesses              =(nranks,),
                                   nhalo                   =(NHALO,)))

Ny_local = Ny ÷ nranks
y_offset = rank * Ny_local

@testset "Decomposed cuFFT                                                    " for dealias in [false, true]
    @testset "construction" begin
        p = @test_nowarn FFTPlans(g; dealias=dealias)
        szs = dealias ? ReSolverFlowsBase.get_padded_size((Nx, Nz, Nt), (1, 2, 3)) : (Nx, Nz, Nt)
        @test p.norm == Float32(1/prod(szs))
        @test p.cache isa CUDA.CuArray
        @test size(p.cache) == (Ny_local + 2*NHALO, ntuple(i -> i==1 ? (szs[i] >> 1) + 1 : szs[i], 3)...)
        @test p.backend == CUDA.cuFFT
    end

    @testset "execution" begin
        p = FFTPlans(g; dealias=dealias)

        u    = build_seeded_field(g, dealias)
        uhat = ReSolverFlowsBase.FTField(g)
        v    = ReSolverFlowsBase.Field(g; dealias=dealias)

        @test parent(parent(p(v, p(uhat, u)))) ≈ parent(parent(u))
    end
end
