using Test

using LinearAlgebra

import MPI
import FDGrids
import CUDA

using ReSolverFlowsBase

include("../../mock_channel_grid.jl")

# initialise MPI environment
MPI.Initialized() || MPI.Init()
comm   = MPI.COMM_WORLD
rank   = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# grid construction parameters
const Ny = 16; const Nx = 5; const Nz = 5; const Nt = 5
const Ny_local = Ny ÷ nranks
const NHALO = 1
const M     = 2
const NCOMP = 3

# construct grid
gb = MockChannelGrid(Ny, Nx, Nz, Nt)
gp = distributed(gb, comm; decomposed_physical_dims=(:y,),
                           nprocesses              =(nranks,),
                           nhalo                   =(NHALO,))
g = CUDA.cu(gp)

# construct fields to re-use
u = VectorField(gp; N=NCOMP); v = VectorField(gp; N=NCOMP)
for n in 1:NCOMP
    u[n] .= randn(ComplexF64, Ny_local, (Nx >> 1) + 1, Nz, Nt)
    v[n] .= randn(ComplexF64, Ny_local, (Nx >> 1) + 1, Nz, Nt)
end
ud = CUDA.cu(u); vd = CUDA.cu(v)

@testset "Decomposed CUDA operators                                           " begin
    @testset "Cartesian primitive 3d equations" begin
        Re = 50
        op_nl_h = @test_nowarn CartesianPrimitive3DNSE( gp, Re; flags=FFTW.ESTIMATE)
        op_nl_d = @test_nowarn CartesianPrimitive3DNSE( g,  Re; flags=FFTW.ESTIMATE)
        op_ln_h = @test_nowarn CartesianPrimitive3DLNSE(gp, Re; flags=FFTW.ESTIMATE, mode=Forward())
        op_ln_d = @test_nowarn CartesianPrimitive3DLNSE(g,  Re; flags=FFTW.ESTIMATE, mode=Forward())
        # ! can't do this test becuase MockChannelGrid doesn't pre-compute adjoint DiffMatrices
        # op_ad_h = @test_nowarn CartesianPrimitive3DLNSE(gp, Re; flags=FFTW.ESTIMATE, mode=AdjointDiscrete())
        # op_ad_d = @test_nowarn CartesianPrimitive3DLNSE(g,  Re; flags=FFTW.ESTIMATE, mode=AdjointDiscrete())

        out1_h = op_nl_h(0, u,      similar(u) )
        out1_d = op_nl_d(0, ud,     similar(ud))
        out2_h = op_ln_h(0, u,  v,  similar(u) )
        out2_d = op_ln_d(0, ud, vd, similar(ud))
        # out3_h = op_ad_h(0, u,  v,  similar(u) )
        # out3_d = op_ad_d(0, ud, vd, similar(ud))
        CUDA.allowscalar() do
            @test out1_d ≈ out1_h
            @test out2_d ≈ out2_h
            # @test out3_d ≈ out3_h
        end
    end
end
