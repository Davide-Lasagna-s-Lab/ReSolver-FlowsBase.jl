# Tests for HaloArray-backed field storage and halo request setup.
#
# Covers:
#   - HaloArray-backed Field / FTField allocation
#   - init_requests! on Field, FTField, VectorField
#   - wait_requests! on MPI MultiRequest batches and nested tuples

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
g        = distributed(g_parent, base_comm;
                        decomposed_physical_dims=(:x2,), nprocesses=(nranks,), nhalo=(NHALO,))

@testset "Field allocation picks HaloArray storage when nhalo > 0             " begin
    u = ReSolverFlowsBase.Field(g)
    @test parent(u) isa HaloArrays.HaloArray
    @test HaloArrays.nhalo(parent(u)) == MPIExt.nhalo(g)
    @test all(parent(u) .== 0)
end

@testset "FTField allocation picks HaloArray storage when nhalo > 0           " begin
    uhat = ReSolverFlowsBase.FTField(g)
    @test parent(uhat) isa HaloArrays.HaloArray
    @test HaloArrays.nhalo(parent(uhat)) == MPIExt.nhalo(g)
    @test all(parent(uhat) .== 0)
end

@testset "VectorField allocation propagates halo policy                       " begin
    q = ReSolverFlowsBase.VectorField(g, ReSolverFlowsBase.FTField; N=3)
    @test length(q) == 3
    @test all(n -> parent(q[n]) isa HaloArrays.HaloArray, 1:3)
end

@testset "init_requests! on Field / FTField                                   " begin
    u    = ReSolverFlowsBase.Field(g)
    uhat = ReSolverFlowsBase.FTField(g)

    reqs_u    = MPIExt.init_requests!(u)
    reqs_uhat = MPIExt.init_requests!(uhat)

    @test MPIExt.wait_requests!(reqs_u) === nothing
    @test MPIExt.wait_requests!(reqs_uhat) === nothing
end

@testset "init_requests! on VectorField returns nested request tuple          " begin
    q = ReSolverFlowsBase.VectorField(g, ReSolverFlowsBase.FTField; N=2)
    reqs = MPIExt.init_requests!(q)
    @test reqs isa Tuple && length(reqs) == 2
    @test MPIExt.wait_requests!(reqs) === nothing
end

if nranks > 1
    @testset "halo requests propagates owned values to neighbouring ranks         " begin
        # Tag each rank's owned interior with its rank number; the lower and
        # upper halo cells should contain the neighbours' values after exchange.
        u = ReSolverFlowsBase.Field(g)
        a = parent(u)
        fill!(parent(a), 0)  # zero the dense storage, including halo cells
        a .= Float64(rank + 1)

        reqs = MPIExt.init_requests!(u)
        MPIExt.wait_requests!(reqs)

        # `HaloArray` scalar indexing accepts halo indices such as 0 and n+1, but
        # `view` follows Julia's ordinary array bounds.  Inspect whole halo planes
        # through the dense parent storage, shifting by the halo width.
        P = parent(a)
        h = HaloArrays.nhalo(a)[1]
        if rank > 0
            # Lower halo received from rank - 1.
            @test all(view(P, 1:h, :, :, :) .== Float64(rank))
        end
        if rank < nranks - 1
            # Upper halo received from rank + 1.
            @test all(view(P, (h+size(g, 1)+1):(2*h+size(g, 1)), :, :, :) .== Float64(rank + 2))
        end
    end
end

MPI.free(base_comm)
