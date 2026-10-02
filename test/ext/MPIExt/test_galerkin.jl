# Tests for the `project!` Galerkin override in
# `MPIExt/src/galerkin.jl`.
#
# The override extends `ReSolverFlowsBase.project!` with an `MPI.Allreduce!` over
# the grid communicator so each rank's partial weighted inner product is
# summed into the global modal coefficients. We verify that the override
# runs end-to-end and produces identical coefficients on every rank
# (the Allreduce step is what guarantees rank-consistency).

using Test

import LinearAlgebra
import MPI
import Random

using ReSolverFlowsBase,
      FDGrids

const MPIExt = Base.get_extension(ReSolverFlowsBase, :MPIExt)

MPI.Initialized() || MPI.Init()

include("../../mock_channel_grid.jl")

nranks = MPI.Comm_size(MPI.COMM_WORLD)
rank   = MPI.Comm_rank(MPI.COMM_WORLD)

const Ny = 16; const Nx = 5; const Nz = 5; const Nt = 5
const NHALO = 1
const NMODES = 2
const NCOMP = 2

base_comm = MPI.Comm_dup(MPI.COMM_WORLD)
g = distributed(MockChannelGrid(Ny, Nx, Nz, Nt), base_comm;
                decomposed_physical_dims=(:x2,), nprocesses=(nranks,), nhalo=(NHALO,))

Ny_local = Ny ÷ nranks

# `ProjectedField(grid, modes)` expects `modes` to be a tuple of arrays
# of shape `(Nm, inh_axes..., fft_axes...)`. For the channel layout that
# is `(Nm, Ny_local, (Nx÷2)+1, Nz, Nt)` per component. We use a per-rank
# deterministic seed so each rank fills its own slab; ProjectedField
# storage is a plain Array (no HaloArray), so its parent is broadcastable
# directly through MPI.
Random.seed!(0xc0ffee + rank)
mode_shape = (NMODES, Ny_local, (Nx >> 1) + 1, Nz, Nt)
modes_components = ntuple(_ -> randn(ComplexF64, mode_shape...), NCOMP)
basis = ReSolverFlowsBase.ProjectedField(g, modes_components)

# Velocity field with rank-shared random parent data. Use a flat Vector
# view for the Bcast so HaloArray-backed storage stays out of MPI's way.
u = ReSolverFlowsBase.VectorField(g, ReSolverFlowsBase.FTField; N=NCOMP)
for n in 1:NCOMP
    flat = Vector{ComplexF64}(vec(randn(ComplexF64, size(parent(u[n]))...)))
    MPI.Bcast!(flat, 0, MPIExt.comm(g))
    parent(u[n]) .= reshape(flat, size(parent(u[n])))
end

@testset "project! coefficients are rank-consistent                           " begin
    a_local = ReSolverFlowsBase.project!(similar(basis), u)

    # Pull rank 0's coefficients and compare to each rank's local copy.
    a_root = Vector{ComplexF64}(vec(parent(a_local)))
    MPI.Bcast!(a_root, 0, MPIExt.comm(g))
    @test vec(parent(a_local)) ≈ a_root
end

MPI.free(base_comm)
