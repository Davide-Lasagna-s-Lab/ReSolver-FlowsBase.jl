# Benchmark: @generated loop kernels (better-spectral-loops) vs
#            CartesianIndices loop kernels (cartesian-indices).
#
# Grid: TripleGrid(Ny, Nx, Nz) — one inhomogeneous wall-normal dimension
# and two homogeneous FFT dimensions, matching ChannelFlow's layout.
# Sizes are chosen to be representative of a mid-resolution channel run.
#
# Run from the repo root:
#   julia --project=. benchmark/bench_galerkin.jl

using BenchmarkTools
using LinearAlgebra
using ReSolverFlowsBase

include("../test/fake.jl")
include("../test/test_grids.jl")

# ------------------------------------------------------------------ #
# @generated kernels (verbatim from better-spectral-loops)           #
# ------------------------------------------------------------------ #

@generated function _project_generated!(pa,
                                        u::VectorField{N},
                                        a::ProjectedField,
                                        w,
                                        g::AbstractGrid{<:Any, D, AXES, FFT_DIMS_ORDER}
                                        ) where {N, D, AXES, FFT_DIMS_ORDER}
    Nhom     = length(FFT_DIMS_ORDER)
    inh_dims = [d for d in 1:D if d ∉ FFT_DIMS_ORDER]
    syms     = [Symbol("_i", d) for d in 1:D]

    pa_ref = Expr(:ref, :pa,      :m, (syms[d] for d in FFT_DIMS_ORDER)...)
    pu_ref = Expr(:ref, :pu,      (syms[d] for d in 1:D)...)
    w_ref  = Expr(:ref, :w,       (syms[d] for d in inh_dims)...)
    mn_ref = Expr(:ref, :modes_n,
                  :m,
                  (syms[d] for d in inh_dims)...,
                  (syms[d] for d in FFT_DIMS_ORDER)...)

    body = Expr(:for, Expr(:(=), :m, :(1:Nm)),
                :($pa_ref += conj($mn_ref) * wuj))
    body = Expr(:block, :(wuj = $w_ref * $pu_ref), body)
    for d in inh_dims
        body = Expr(:for, Expr(:(=), syms[d], :(1:Base.size(g, $d))), body)
    end
    for k in 1:Nhom
        d   = FFT_DIMS_ORDER[k]
        rng = k == 1 ? :(1:(Base.size(g, $d) >> 1) + 1) : :(1:Base.size(g, $d))
        body = Expr(:for, Expr(:(=), syms[d], rng), body)
    end
    body = Expr(:block, :(pu = parent(u[n])), :(modes_n = modes(a)[n]), body)
    body = Expr(:for, Expr(:(=), :n, :(1:$N)), body)
    return quote
        Nm = Base.size(pa, 1)
        @inbounds $body
    end
end

@generated function _expand_generated!(u::VectorField{N},
                                       pa,
                                       a::ProjectedField,
                                       g::AbstractGrid{<:Any, D, AXES, FFT_DIMS_ORDER}
                                       ) where {N, D, AXES, FFT_DIMS_ORDER}
    Nhom     = length(FFT_DIMS_ORDER)
    inh_dims = [d for d in 1:D if d ∉ FFT_DIMS_ORDER]
    syms     = [Symbol("_i", d) for d in 1:D]

    pa_ref = Expr(:ref, :pa,      :m, (syms[d] for d in FFT_DIMS_ORDER)...)
    pu_ref = Expr(:ref, :pu,      (syms[d] for d in 1:D)...)
    mn_ref = Expr(:ref, :modes_n,
                  :m,
                  (syms[d] for d in inh_dims)...,
                  (syms[d] for d in FFT_DIMS_ORDER)...)

    body = Expr(:for, Expr(:(=), :m, :(1:Nm)), :(acc += $mn_ref * $pa_ref))
    body = Expr(:block, :(acc = zero(eltype(pa))), body, :($pu_ref = acc))
    for d in inh_dims
        body = Expr(:for, Expr(:(=), syms[d], :(1:Base.size(g, $d))), body)
    end
    for k in 1:Nhom
        d   = FFT_DIMS_ORDER[k]
        rng = k == 1 ? :(1:(Base.size(g, $d) >> 1) + 1) : :(1:Base.size(g, $d))
        body = Expr(:for, Expr(:(=), syms[d], rng), body)
    end
    body = Expr(:block, :(pu = parent(u[n])), :(modes_n = modes(a)[n]), body)
    body = Expr(:for, Expr(:(=), :n, :(1:$N)), body)
    return quote
        Nm = Base.size(pa, 1)
        @inbounds $body
    end
end

function project_generated!(a::ProjectedField{G},
                             u::VectorField{N, <:FTField{G}},
                             ) where {N, T, G<:AbstractGrid{T}}
    a .= zero(Complex{T})
    _project_generated!(parent(a), u, a, weights(grid(u)), grid(u))
    return a
end

function expand_generated!(u::VectorField{N, <:FTField{G}},
                            a::ProjectedField{G},
                            ) where {N, T, G<:AbstractGrid{T}}
    _expand_generated!(u, parent(a), a, grid(u))
    return u
end

# ------------------------------------------------------------------ #
# CartesianIndices kernels (current cartesian-indices branch)        #
# ------------------------------------------------------------------ #

project_cartesian!(a, u) = project!(a, u, LoopGalerkin())
expand_cartesian!(u, a)  = expand!(u, a, LoopGalerkin())

# ------------------------------------------------------------------ #
# Setup: ChannelFlow-like grid                                       #
# ------------------------------------------------------------------ #

const Ny = 97      # wall-normal (inhomogeneous, Chebyshev-like)
const Nx = 128     # streamwise  (rfft)
const Nz = 96      # spanwise    (signed FFT)
const Nm = 10      # number of basis modes
const Nv = 3       # velocity components (u, v, w)

g = TripleGrid(Ny, Nx, Nz)

# Random complex modes array: size (Nm, Ny, Nx÷2+1, Nz)
make_modes() = [randn(ComplexF64, Nm, Ny, (Nx >> 1) + 1, Nz) for _ in 1:Nv]

# Allocate fields
a  = ProjectedField(g, make_modes())
u  = VectorField(g, FTField)

# Fill u with random spectral data
for n in 1:Nv
    parent(u[n]) .= randn(ComplexF64, size(parent(u[n])))
end

# Warm-up (trigger compilation)
project_generated!(a, u)
project_cartesian!(a, u)
expand_generated!(u, a)
expand_cartesian!(u, a)

# ------------------------------------------------------------------ #
# Benchmarks                                                         #
# ------------------------------------------------------------------ #

println("Grid: Ny=$Ny, Nx=$Nx, Nz=$Nz, Nm=$Nm, Nv=$Nv")
println()

println("=== project! ===")
bg = @benchmark project_generated!($a, $u)  samples=200 evals=1
bc = @benchmark project_cartesian!($a, $u)  samples=200 evals=1
println("  @generated : ", BenchmarkTools.prettytime(median(bg).time),
        "  (allocs: ", allocs(median(bg)), ")")
println("  CartesianIndices: ", BenchmarkTools.prettytime(median(bc).time),
        "  (allocs: ", allocs(median(bc)), ")")
println("  ratio (generated/cartesian): ", round(median(bg).time / median(bc).time, digits=2))

println()
println("=== expand! ===")
bg2 = @benchmark expand_generated!($u, $a)  samples=200 evals=1
bc2 = @benchmark expand_cartesian!($u, $a)  samples=200 evals=1
println("  @generated : ", BenchmarkTools.prettytime(median(bg2).time),
        "  (allocs: ", allocs(median(bg2)), ")")
println("  CartesianIndices: ", BenchmarkTools.prettytime(median(bc2).time),
        "  (allocs: ", allocs(median(bc2)), ")")
println("  ratio (generated/cartesian): ", round(median(bg2).time / median(bc2).time, digits=2))

println()
println("=== GemmGalerkin (reference) ===")
bge = @benchmark project!($a, $u, GemmGalerkin())  samples=200 evals=1
bge2 = @benchmark expand!($u, $a, GemmGalerkin())   samples=200 evals=1
println("  GemmGalerkin project! : ", BenchmarkTools.prettytime(median(bge).time),
        "  (allocs: ", allocs(median(bge)), ")")
println("  GemmGalerkin expand!  : ", BenchmarkTools.prettytime(median(bge2).time),
        "  (allocs: ", allocs(median(bge2)), ")")

# ------------------------------------------------------------------ #
# Flipped grid: FFT dimensions come FIRST in storage                 #
# ------------------------------------------------------------------ #
# Layout: array dim 1 = rfft (x), array dim 2 = signed FFT (z),
#         array dim 3 = inhomogeneous (y)  [same physical setup,
#         different storage order].
#
# For this grid FFT_DIMS_ORDER = (1, 2), so:
#
#   FTField shape : ((Nx>>1)+1, Nz, Ny)          ← FFT dims first
#   modes shape   : (Ny, Nm, (Nx>>1)+1, Nz)      ← inh, mode, FFT (fixed convention)
#   pa shape      : (Nm, (Nx>>1)+1, Nz)           ← mode, FFT
#
# @generated kernel: correct — generates pa[m, _i1, _i2] where _i1/_i2
#   are the loop variables for grid dims 1/2 (wavenumber values).
#
# CartesianIndices kernel: INCORRECT — homogeneous_axes(g, a) returns
#   axes(a,1) = 1:Nm and axes(a,2) = 1:(Nx>>1)+1, so Ih iterates over
#   (mode_idx, rfft_idx) instead of (rfft_idx, z_idx).  The z-dimension
#   is never touched and pa is indexed at wrong positions.

struct FlippedChannelGrid <: AbstractGrid{Float64, 3, (1, 3, 2, nothing), (1, 2)}
    Nx :: Int
    Nz :: Int
    Ny :: Int
    ws :: Vector{Float64}
end

function FlippedChannelGrid(Nx, Nz, Ny)
    FlippedChannelGrid(Nx, Nz, Ny, ones(Ny))
end

# size(g) returns the storage-order sizes: (Nx, Nz, Ny)
Base.size(g::FlippedChannelGrid) = (g.Nx, g.Nz, g.Ny)
ReSolverFlowsBase.weights(g::FlippedChannelGrid)             = g.ws
ReSolverFlowsBase.wavenumber_scale(::FlippedChannelGrid, ::Int) = 1.0

gf = FlippedChannelGrid(Nx, Nz, Ny)

# modes: (Nm, Ny, (Nx>>1)+1, Nz) = (Nm, inh_sz..., kH_sz...)
make_modes_f() = [randn(ComplexF64, Nm, Ny, (Nx >> 1) + 1, Nz) for _ in 1:Nv]

af = ProjectedField(gf, make_modes_f())
uf = VectorField(gf, FTField)
for n in 1:Nv
    parent(uf[n]) .= randn(ComplexF64, size(parent(uf[n])))
end

# Correctness check: compare @generated result against a reference computed
# from the explicit formula using a single small wavenumber.
println()
println("=== Layout diagnostics (FFT dims first) ===")
println("  FTField uf[1] shape   : ", size(parent(uf[1])))   # should be (65, 96, 97) = (rfft, z, y)
println("  ProjectedField af shape: ", size(parent(af)))      # should be (Nm, 65, 96) = (mode, rfft, z)
println("  homogeneous_axes(gf, af)       = ", ReSolverFlowsBase.homogeneous_axes(gf, af))
println("  inhomogeneous_axes(gf, uf[1]) = ", ReSolverFlowsBase.inhomogeneous_axes(gf, uf[1]))

println()
println("=== Correctness check (FFT dims first) ===")

# Reference: compute pa[1, 1, 1] manually from the definition
#   pa[1, k1, k2] = sum_n sum_j w[j] * conj(modes_n[1, j, k1, k2]) * pu[k1, k2, j]
ref_val = let s = zero(ComplexF64)
    for n in 1:Nv
        mn = modes(af)[n]          # shape (Nm, Ny, (Nx>>1)+1, Nz)
        pu_f = parent(uf[n])       # shape ((Nx>>1)+1, Nz, Ny)
        for j in 1:Ny
            s += conj(mn[1, j, 1, 1]) * pu_f[1, 1, j]
        end
    end
    s
end
# (weights are all 1)

af_gen = deepcopy(af); project_generated!(af_gen, uf)
af_ci  = deepcopy(af); project_cartesian!(af_ci,  uf)

pa_gen = parent(af_gen)
pa_ci  = parent(af_ci)

# Check at the reference element (m=1, rfft=1, z=1)
err_gen = abs(pa_gen[1, 1, 1] - ref_val)
err_ci  = abs(pa_ci[1,  1, 1] - ref_val)

# Full-array norm difference between the two results
full_diff = norm(pa_gen .- pa_ci)

println("  reference pa[1,1,1]               = ", round(ref_val, digits=6))
println("  @generated pa[1,1,1]              = ", round(pa_gen[1,1,1], digits=6),
        "  error = ", round(err_gen, sigdigits=3))
println("  CartesianIndices pa[1,1,1]        = ", round(pa_ci[1,1,1],  digits=6),
        "  error = ", round(err_ci,  sigdigits=3))
println("  ||pa_generated - pa_cartesian||   = ", round(full_diff, sigdigits=4),
        full_diff < 1e-10 ? "  (identical)" : "  ← DIFFER (correctness bug)")
println("  @generated correct?         ", err_gen < 1e-10 ? "YES ✓" : "NO ✗")
println("  CartesianIndices correct?   ", err_ci  < 1e-10 ? "YES ✓" : "NO ✗  ← layout bug")

println()
println("=== project!/expand! FFT-dims-first (only @generated is correct) ===")
project_generated!(af_gen, uf); expand_generated!(uf, af_gen)  # warm-up

bgf  = @benchmark project_generated!($af_gen, $uf) samples=200 evals=1
bgf2 = @benchmark expand_generated!($uf, $af_gen)  samples=200 evals=1
println("  @generated project! : ", BenchmarkTools.prettytime(median(bgf).time),
        "  (allocs: ", allocs(median(bgf)), ")")
println("  @generated expand!  : ", BenchmarkTools.prettytime(median(bgf2).time),
        "  (allocs: ", allocs(median(bgf2)), ")")
println()
println("  (TripleGrid @generated for comparison)")
println("  @generated project! : ", BenchmarkTools.prettytime(median(bg).time))
println("  @generated expand!  : ", BenchmarkTools.prettytime(median(bg2).time))
