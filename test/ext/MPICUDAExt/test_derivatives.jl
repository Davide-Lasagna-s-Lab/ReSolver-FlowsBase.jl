using Test

import MPI
import FDGrids
import CUDA

using ReSolverFlowsBase

include("../../mock_channel_grid.jl")

# setup MPI environment
MPI.Initialized() || MPI.Init()
comm   = MPI.COMM_WORLD
rank   = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Single-mode analytic test field. Each FFT direction has exactly one
# non-trivial wavenumber so the spectral derivative is exact for the
# chosen resolution. The wall-normal factor `(1 - y^2)` is a degree-2
# polynomial, exactly differentiable by the 5-point FD stencil.
const α_test = 2π
const β_test = 5.8
const γ_test = 1.0   # temporal wavenumber

u_fun(y, x, z, t)      =  (1 - y^2) * cos(α_test*x) * cos(β_test*z) * cos(γ_test*t)
dudx_fun(y, x, z, t)   = -(1 - y^2) * α_test * sin(α_test*x) * cos(β_test*z) * cos(γ_test*t)
d2udx2_fun(y, x, z, t) = -(1 - y^2) * α_test^2 * cos(α_test*x) * cos(β_test*z) * cos(γ_test*t)
dudy_fun(y, x, z, t)   = -2y * cos(α_test*x) * cos(β_test*z) * cos(γ_test*t)
d2udy2_fun(y, x, z, t) = -2 * cos(α_test*x) * cos(β_test*z) * cos(γ_test*t)
dudz_fun(y, x, z, t)   = -(1 - y^2) * β_test * cos(α_test*x) * sin(β_test*z) * cos(γ_test*t)
d2udz2_fun(y, x, z, t) = -(1 - y^2) * β_test^2 * cos(α_test*x) * cos(β_test*z) * cos(γ_test*t)
dudt_fun(y, x, z, t)   = -(1 - y^2) * γ_test * cos(α_test*x) * cos(β_test*z) * sin(γ_test*t)
lapl_fun(y, x, z, t)   = d2udx2_fun(y, x, z, t) +
                         d2udy2_fun(y, x, z, t) +
                         d2udz2_fun(y, x, z, t)

const Ny = 32; const Nx = 7; const Nz = 9; const Nt = 5
const NHALO = 2  # 5-point stencil -> half-width 2

gp = distributed(MockChannelGrid(Ny, Nx, Nz, Nt; stencil_width=5, α=α_test, β=β_test), comm; decomposed_physical_dims=(:y,),
                                                                                             nprocesses              =(nranks,),
                                                                                             nhalo                   =(NHALO,))
g = CUDA.cu(gp)

@testset "Decomposed CUDA derivative                                          " begin
    # construct the fields
    u = CUDA.cu(FFT(ReSolverFlowsBase.Field(gp, u_fun)))

    @testset "Spectral derivative directions" begin
        @test Array(parent(parent(ReSolverFlowsBase.ddx!(similar(u), u)))) ≈ parent(parent(FFT(Field(gp, dudx_fun))))
        @test Array(parent(parent(ReSolverFlowsBase.ddz!(similar(u), u)))) ≈ parent(parent(FFT(Field(gp, dudz_fun))))
        @test Array(parent(parent(ReSolverFlowsBase.ddt!(similar(u), u)))) ≈ parent(parent(FFT(Field(gp, dudt_fun))))
    end
    @testset "Non-spectral derivatives" begin
        @test Array(parent(parent(ReSolverFlowsBase.ddy!(similar(u), u)))) ≈ parent(parent(FFT(Field(gp, dudy_fun))))
    end
    @testset "Laplacian operator" begin
        @test Array(parent(parent(ReSolverFlowsBase.laplacian!(similar(u), u)))) ≈ parent(parent(FFT(Field(gp, lapl_fun))))
    end
end
