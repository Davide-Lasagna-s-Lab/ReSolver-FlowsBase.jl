# Tests for the spectral-field constructor in
# `MPIExt/src/ftfield.jl`.
#
# Verifies that `FTField(g)` on a decomposed grid:
#   - allocates `HaloArrays.HaloArray` storage
#   - sizes the interior to `ReSolverFlowsBase.transform_size(g)`

using Test

import HaloArrays
import MPI

using ReSolverFlowsBase,
      FDGrids

const MPIExt = Base.get_extension(ReSolverFlowsBase, :MPIExt)

MPI.Initialized() || MPI.Init()

include("../../mock_channel_grid.jl")

nranks = MPI.Comm_size(MPI.COMM_WORLD)
rank   = MPI.Comm_rank(MPI.COMM_WORLD)

const Ny = 16; const Nx = 7; const Nz = 9; const Nt = 5
const NHALO = 1

base_comm = MPI.Comm_dup(MPI.COMM_WORLD)
g_parent = MockChannelGrid(Ny, Nx, Nz, Nt)
g_halo   = distributed(g_parent, base_comm;
                        decomposed_physical_dims=(:x2,), nprocesses=(nranks,), nhalo=(NHALO,))

@testset "FTField on decomposed grid is HaloArray-backed                      " begin
    uhat = ReSolverFlowsBase.FTField(g_halo)
    @test parent(uhat) isa HaloArrays.HaloArray
    @test size(uhat) == ReSolverFlowsBase.transform_size(g_halo)
    @test HaloArrays.nhalo(parent(uhat)) == MPIExt.nhalo(g_halo)
    @test all(parent(uhat) .== 0)
end

@testset "FTField via convert preserves storage policy                        " begin
    uhat = ReSolverFlowsBase.FTField(g_halo)
    uhat32 = similar(uhat, ComplexF32)
    @test parent(uhat32) isa HaloArrays.HaloArray
end

MPI.free(base_comm)
