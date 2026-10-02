# Tests for the Val{STORAGE_DIM}-dispatched derivative API in
# `MPIExt/src/derivatives.jl`.
#
# Covers:
#   - `interior_dd!` / `boundary_dd!` cover the local dimension exactly
#   - staged `init_ddy!` / `complete_ddy!` matches the global derivative
#   - spectral directions (`:x1`, `:x3`, `:s`) bypass the FD path
#   - vector-field derivatives consume per-component halo request tuples
#   - `interior_laplacian!` / `boundary_laplacian!` reproduce the
#     analytic Laplacian after halo exchange
#   - the 2-arg `ReSolverFlowsBase.laplacian!` auto-exchange form agrees

using Test

import FDGrids
import LinearAlgebra
import MPI

using ReSolverFlowsBase,
      FDGrids

const MPIExt = Base.get_extension(ReSolverFlowsBase, :MPIExt)

MPI.Initialized() || MPI.Init()

include("../../mock_channel_grid.jl")

nranks = MPI.Comm_size(MPI.COMM_WORLD)
rank   = MPI.Comm_rank(MPI.COMM_WORLD)

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

base_comm = MPI.Comm_dup(MPI.COMM_WORLD)
# Wavenumber scales chosen so that x has period 1/2 (k_x = 2π gives spectral
# coefficients at integer wavenumbers) and z has period 2π/5.8 to match
# `cos(4π*x)` and `cos(5.8*z)` factors in the analytic test field.
g_parent = MockChannelGrid(Ny, Nx, Nz, Nt; stencil_width=5, α=α_test, β=β_test)
g = distributed(g_parent, base_comm;
                decomposed_physical_dims=(:x2,), nprocesses=(nranks,), nhalo=(NHALO,))

# Pre-build FFT plans and reusable physical/spectral fields.
plans  = ReSolverFlowsBase.FFTPlans(g; dealias=false, flags=ReSolverFlowsBase.FFTW.ESTIMATE)
u_phys = ReSolverFlowsBase.Field(g, u_fun)
u      = ReSolverFlowsBase.FTField(g); plans(u, u_phys)

@testset "staged ddx2! matches analytic du/dy                                  " begin
    out = ReSolverFlowsBase.FTField(g)
    ReSolverFlowsBase.ddx2!(out, u)

    expected = ReSolverFlowsBase.FTField(g)
    plans(expected, ReSolverFlowsBase.Field(g, dudy_fun))

    @test parent(out) ≈ parent(expected) rtol=1e-6
end

@testset "staged VectorField derivative uses per-component halo requests      " begin
    q        = ReSolverFlowsBase.VectorField(g, ReSolverFlowsBase.FTField; N=2)
    expected = ReSolverFlowsBase.VectorField(g, ReSolverFlowsBase.FTField; N=2)
    out      = ReSolverFlowsBase.VectorField(g, ReSolverFlowsBase.FTField; N=2)

    plans(q[1], ReSolverFlowsBase.Field(g, u_fun))
    plans(q[2], ReSolverFlowsBase.Field(g, (y, x, z, t) -> 2u_fun(y, x, z, t)))
    plans(expected[1], ReSolverFlowsBase.Field(g, dudy_fun))
    plans(expected[2], ReSolverFlowsBase.Field(g, (y, x, z, t) -> 2dudy_fun(y, x, z, t)))

    ReSolverFlowsBase.ddx2!(out, q)

    @test parent(out[1]) ≈ parent(expected[1]) rtol=1e-6
    @test parent(out[2]) ≈ parent(expected[2]) rtol=1e-6
end

sd(sym) = ReSolverFlowsBase.physical_to_storage_dim(ReSolverFlowsBase.grid(u), Val(sym))

@testset "interior_dd! + wait + boundary_dd! covers the FD direction          " begin
    out = ReSolverFlowsBase.FTField(g)
    requests = MPIExt.init_requests!(u)
    MPIExt.interior_dd!(out, u, sd(:x2))
    MPIExt.wait_requests!(requests)
    MPIExt.boundary_dd!(out, u, sd(:x2))

    expected = ReSolverFlowsBase.FTField(g)
    plans(expected, ReSolverFlowsBase.Field(g, dudy_fun))
    @test parent(out) ≈ parent(expected) rtol=1e-6
end

@testset "dd! along an FFT direction is spectral (no halo needed)             " begin
    out = ReSolverFlowsBase.FTField(g)
    ReSolverFlowsBase.dd!(out, u, sd(:x1))
    expected = ReSolverFlowsBase.FTField(g)
    plans(expected, ReSolverFlowsBase.Field(g, dudx_fun))
    @test parent(out) ≈ parent(expected) rtol=1e-6

    out_z = ReSolverFlowsBase.FTField(g)
    ReSolverFlowsBase.dd!(out_z, u, sd(:x3))
    expected_z = ReSolverFlowsBase.FTField(g)
    plans(expected_z, ReSolverFlowsBase.Field(g, dudz_fun))
    @test parent(out_z) ≈ parent(expected_z) rtol=1e-6

    out_t = ReSolverFlowsBase.FTField(g)
    ReSolverFlowsBase.dd!(out_t, u, sd(:s))
    expected_t = ReSolverFlowsBase.FTField(g)
    plans(expected_t, ReSolverFlowsBase.Field(g, dudt_fun))
    @test parent(out_t) ≈ parent(expected_t) rtol=1e-6
end

@testset "interior_laplacian! + boundary_laplacian! matches analytic Laplacian" begin
    out = ReSolverFlowsBase.FTField(g)
    requests = MPIExt.init_requests!(u)
    MPIExt.interior_laplacian!(out, u)
    MPIExt.wait_requests!(requests)
    MPIExt.boundary_laplacian!(out, u)

    expected = ReSolverFlowsBase.FTField(g)
    plans(expected, ReSolverFlowsBase.Field(g, lapl_fun))
    @test parent(out) ≈ parent(expected) rtol=1e-6
end

@testset "ReSolverFlowsBase.laplacian!(out, u) agrees with the explicit form  " begin
    out_a = ReSolverFlowsBase.FTField(g)
    out_b = ReSolverFlowsBase.FTField(g)

    requests = MPIExt.init_requests!(u)
    MPIExt.interior_laplacian!(out_a, u)
    MPIExt.wait_requests!(requests)
    MPIExt.boundary_laplacian!(out_a, u)

    # 2-arg overload — auto-exchange.
    ReSolverFlowsBase.laplacian!(out_b, u)

    @test parent(out_a) ≈ parent(out_b)
end

MPI.free(base_comm)
