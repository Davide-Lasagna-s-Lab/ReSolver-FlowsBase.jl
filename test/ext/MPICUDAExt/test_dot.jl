using Test

import MPI
import FDGrids
import CUDA

using ReSolverFlowsBase

CUDAExt = Base.get_extension(ReSolverFlowsBase, :CUDAExt)

include("../../mock_channel_grid.jl")

# setup MPI environment
MPI.Initialized() || MPI.Init()
comm   = MPI.COMM_WORLD
rank   = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# grid constants
const Ny = 32; const Nx = 7; const Nz = 9; const Nt = 5
const NCOMP = 2
const M = 5
const NHALO = 2

# construct grid
gp = distributed(MockChannelGrid(Ny, Nx, Nz, Nt), comm; decomposed_physical_dims=(:y,),
                                                        nprocesses              =(nranks,),
                                                        nhalo                   =(NHALO,))

# initialise fields to re-use
Ψ = ntuple(_->zeros(M, Ny, (Nx >> 1) + 1, Nz, Nt), NCOMP)
a = ProjectedField(gp, Ψ); a .= randn(ComplexF64, M, (Nx >> 1) + 1, Nz, Nt)
b = ProjectedField(gp, Ψ); b .= randn(ComplexF64, M, (Nx >> 1) + 1, Nz, Nt)
ad = CUDA.cu(a); bd = CUDA.cu(b)

# intialise dot product methods
dot_methods = [CUDAExt.DotTwoStage(ad),
               CUDAExt.DotAtomic(ad),
               CUDAExt.DotShared(ad)]

@testset "projected field dot product correctness                             " for method in dot_methods
    @test abs(dot(ad, bd, method) - dot(a, b)) < 1e-4
end
