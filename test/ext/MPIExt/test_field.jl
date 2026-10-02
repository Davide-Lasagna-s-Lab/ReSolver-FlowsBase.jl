# Tests for the physical-space field constructor in
# `MPIExt/src/field.jl`.
#
# Verifies that `Field(g)` and `Field(g, func)` on a decomposed grid:
#   - use the per-rank local size (`local_size`)
#   - allocate `HaloArrays.HaloArray` storage
#   - evaluate `func` only at this rank's owned collocation points

using Test

import HaloArrays
import MPI

using ReSolverFlowsBase,
      FDGrids

MPI.Initialized() || MPI.Init()

include("../../mock_channel_grid.jl")

nranks = MPI.Comm_size(MPI.COMM_WORLD)
rank   = MPI.Comm_rank(MPI.COMM_WORLD)

const Ny = 16; const Nx = 7; const Nz = 9; const Nt = 5
const NHALO = 1

base_comm = MPI.Comm_dup(MPI.COMM_WORLD)
g = distributed(MockChannelGrid(Ny, Nx, Nz, Nt), base_comm;
                decomposed_physical_dims=(:x2,), nprocesses=(nranks,), nhalo=(NHALO,))

Ny_local = Ny ÷ nranks

@testset "Field allocation: shape + storage                                   " begin
    u = ReSolverFlowsBase.Field(g)
    @test parent(u) isa HaloArrays.HaloArray
    @test size(u) == size(g)
    @test size(u, 1) == Ny_local
    @test all(parent(u) .== 0)
end

@testset "Field allocation with dealias=true grows FFT dimensions             " begin
    u  = ReSolverFlowsBase.Field(g; dealias=false)
    ud = ReSolverFlowsBase.Field(g; dealias=true)
    @test size(ud, 1) == size(u, 1)            # wall-normal unchanged
    @test size(ud, 2) >= size(u, 2)            # x grew (3/2-rule)
    @test size(ud, 3) >= size(u, 3)            # z grew
    @test size(ud, 4) >= size(u, 4)            # t grew
end

@testset "Field(g, func) evaluates func only at this rank's coords            " begin
    f(y, x, z, t) = y + 10 * x  # depends on local coords
    u = ReSolverFlowsBase.Field(g, f)

    y, x, z, t = ReSolverFlowsBase.points(g)
    expected = @. f(y, x, z, t)
    # `parent(u)` exposes only the interior on HaloArray storage.
    @test parent(u) ≈ expected
end

MPI.free(base_comm)
