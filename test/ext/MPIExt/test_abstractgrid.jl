# Tests for the decomposed-grid interface defined in
# `MPIExt/src/abstractgrid.jl`.
#
# Runs under any rank count; uses a `MockChannelGrid` parent wrapped via
# `MPIExt.distributed` along the wall-normal `:x2` direction. Every
# user-facing accessor exported by abstractgrid.jl gets at least one
# assertion.

using Test

import LinearAlgebra
import MPI
import FDGrids
import HaloArrays

using ReSolverFlowsBase

const MPIExt = Base.get_extension(ReSolverFlowsBase, :MPIExt)

MPI.Initialized() || MPI.Init()

include("../../mock_channel_grid.jl")

# -- common setup -----------------------------------------------------------
nranks = MPI.Comm_size(MPI.COMM_WORLD)
rank   = MPI.Comm_rank(MPI.COMM_WORLD)

# Pick a wall-normal size that divides cleanly across any number of ranks
# used in CI (we test with 1 and 4 in `runtests.jl`).
const Ny = 16; const Nx = 7; const Nz = 9; const Nt = 5
const NHALO = 1

base_comm = MPI.Comm_dup(MPI.COMM_WORLD)

g_parent = MockChannelGrid(Ny, Nx, Nz, Nt)
g        = distributed(g_parent, base_comm;
                        decomposed_physical_dims=(:x2,), nprocesses=(nranks,), nhalo=(NHALO,))

Ny_local = Ny ÷ nranks
y_offset = rank * Ny_local

@testset "DecomposedGrid alias                                                " begin
    @test g isa MPIExt.DecomposedGrid
end

@testset "comm / comm_size / comm_rank                                        " begin
    # `distributed(...)` builds a full-Cart comm (one direction per
    # storage dim) for HaloArrays. `comm(g)` returns that comm.
    g_comm = MPIExt.comm(g)
    @test MPI.Cartdim_get(g_comm) == length(size(g))
    @test MPIExt.comm_size(g) == nranks
    @test MPIExt.comm_rank(g) == rank
end

@testset "decomposition_storage_dims / _physical_dims                         " begin
    @test MPIExt.decomposition_storage_dims(g) == (1,)
    @test MPIExt.decomposition_physical_dims(g) == (:x2,)
    @test MPIExt.ndecomposed_dims(g) == 1
end

@testset "size / global_size                                                  " begin
    @test size(g) == (Ny_local, Nx, Nz, Nt)
    @test MPIExt.global_size(g) == (Ny, Nx, Nz, Nt)
    @test size(g, 1) == Ny_local
end

@testset "nhalo                                                               " begin
    # Halos live on the decomposed (wall-normal) storage axis only.
    @test MPIExt.nhalo(g) == (NHALO, 0, 0, 0)
    @test MPIExt.nhalo(g, :x2) == NHALO
end

@testset "global_first_index                                                  " begin
    @test MPIExt.global_first_index(g, :x2) == y_offset + 1
    # Non-decomposed coordinates always start at global index 1.
    @test MPIExt.global_first_index(g, :x1) == 1
    @test MPIExt.global_first_index(g, :x3) == 1
    @test MPIExt.global_first_index(g, :t) == 1
end

@testset "local_interior_range / local_boundary_ranges                        " begin
    interior = MPIExt.local_interior_range(g, :x2)
    lower, upper = MPIExt.local_boundary_ranges(g, :x2)

    # On every rank, the three ranges (interior, lower, upper) form a
    # disjoint cover of 1:Ny_local. Empty bands are 1:0.
    all_rows = Set(1:Ny_local)
    covered  = Set(interior) ∪ Set(lower) ∪ Set(upper)
    @test covered == all_rows

    # Interior excludes halo-adjacent rows only where there is a neighbour.
    has_lower_neighbor = rank > 0
    has_upper_neighbor = rank < nranks - 1
    @test first(interior) == (has_lower_neighbor ? NHALO + 1 : 1)
    @test last(interior)  == (has_upper_neighbor ? Ny_local - NHALO : Ny_local)
    @test lower == (has_lower_neighbor ? (1:NHALO) : (1:0))
    @test upper == (has_upper_neighbor ? ((Ny_local - NHALO + 1):Ny_local) : (1:0))

    # Non-decomposed direction: interior is the whole axis, boundaries are empty.
    @test MPIExt.local_interior_range(g, :x1) == 1:Nx
    @test MPIExt.local_boundary_ranges(g, :x1) == (1:0, 1:0)
end

@testset "derivative_matrix forwards to parent                                " begin
    y_sd = ReSolverFlowsBase.storage_dim(g, :x2)
    x_sd = ReSolverFlowsBase.storage_dim(g, :x1)

    D1 = ReSolverFlowsBase.derivative_matrix(g, y_sd, Val(1), Direct())
    D2 = ReSolverFlowsBase.derivative_matrix(g, y_sd, Val(2), Direct())
    @test D1 === g_parent.D₁
    @test D2 === g_parent.D₂
end

@testset "local_size names the per-rank interior size                         " begin
    @test MPIExt.local_size(g) == size(g)
end

MPI.free(base_comm)
