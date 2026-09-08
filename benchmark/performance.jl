# Performance benchmarks for ReSolverFlowsBase hot paths on a 4-D channel-flow layout.
#
# Sections:
#   1. dot(u, v)       — FTField and ProjectedField, CartesianIndices vs split loop
#   2. shift!(u, s)    — FTField, ReSolverFlowsBase vs equivalent hand-written double loop
#   3. normdiff+shift  — overhead of the shift relative to plain normdiff
#
# Grid layout (mirrors ChannelGrid from ReSolver-ChannelFlow.jl):
#   physical axes  (x, y, z, t)  →  storage dims  (2, 1, 3, 4)
#   storage order: (y, x_half, z, t) = (Ny, Nx_half, Nz, Nt)
#   FFT dims: (2=rfft, 3=signed FFT, 4=signed FFT)
#
# Run: julia --project=benchmark benchmark/performance.jl

using LinearAlgebra
using Printf
using BenchmarkTools
using ReSolverFlowsBase

# ──────────────────────────────────────────────────────────────────────────────
# Mock grid — minimal concrete subtype carrying only the type parameters and
# the grid size.  No FFTW plans, no differentiation matrices.
# ──────────────────────────────────────────────────────────────────────────────
#
# CHANNEL_AXES      = (2, 1, 3, 4): physical dim k lives in storage dim AXES[k]
# CHANNEL_FFT_ORDER = (2, 3, 4):    rfft on dim 2, signed FFTs on dims 3 and 4
#
const BENCH_AXES      = (2, 1, 3, 4)
const BENCH_FFT_ORDER = (2, 3, 4)

struct BenchGrid{S} <: ReSolverFlowsBase.AbstractGrid{Float64, 4, BENCH_AXES, BENCH_FFT_ORDER}
    ws :: Vector{Float64}   # wall-normal (inhomogeneous) quadrature weights
end

Base.size(g::BenchGrid{S}) where {S} = ReSolverFlowsBase.to_storage_order(S, g)
ReSolverFlowsBase.weights(g::BenchGrid) = g.ws
ReSolverFlowsBase.wavenumber_scale(::BenchGrid, ::Int) = 1.0
ReSolverFlowsBase.points(g::BenchGrid; dealias=false) = ntuple(d -> ones(size(g, d)), 4)

# Minimal ddx! stub (not exercised in dot benchmark)
ReSolverFlowsBase.ddx!(out, u, ::Val; kwargs...) = (out .= 0; out)
ReSolverFlowsBase._inhomogeneous_laplacian!(out, u; kwargs...) = (out .= 0; out)

# ──────────────────────────────────────────────────────────────────────────────
# Grid sizes  (representative channel-flow run: 63³ × 33)
# ──────────────────────────────────────────────────────────────────────────────
#   physical: Nx=63 Ny=33 Nz=63 Nt=63
#   stored:   S = (Nx, Ny, Nz, Nt) in physical-coord order
#   FTField parent size: (Ny, Nx_half, Nz, Nt)  where Nx_half = (Nx>>1)+1 = 32
const Nx = 63
const Ny = 33
const Nz = 63
const Nt = 63
const Nx_half = (Nx >> 1) + 1  # 32

ws = rand(Ny) .+ 0.1   # strictly positive weights (arbitrary, only timing matters)
g  = BenchGrid{(Nx, Ny, Nz, Nt)}(ws)

u = FTField(g)
v = FTField(g)
parent(u) .= randn(ComplexF64, size(parent(u)))
parent(v) .= randn(ComplexF64, size(parent(v)))

@assert size(parent(u)) == (Ny, Nx_half, Nz, Nt)

# ──────────────────────────────────────────────────────────────────────────────
# Approach A: CartesianIndices  (current better-spectral-loops)
# ──────────────────────────────────────────────────────────────────────────────
function dot_cartesian(u, v, ws, g)
    s = 0.0
    @inbounds for I in CartesianIndices(u)
        s += ReSolverFlowsBase.one_or_two(I, g) *
             ws[ReSolverFlowsBase.inhomogeneous_indices(I, g)...] *
             real(conj(u[I]) * v[I])
    end
    return s / 2
end

# ──────────────────────────────────────────────────────────────────────────────
# Approach B: hand-written loop nest matching the channel storage layout.
# Storage: (Ny=dim1, Nx_half=dim2, Nz=dim3, Nt=dim4).
# dim1 (j) is stride-1, so it goes innermost.
# The rfft branch is lifted outside the j-loop to avoid per-element branching.
# Loop order (outermost → innermost): Nt, Nz  →  rfft split  →  j (stride-1)
# ──────────────────────────────────────────────────────────────────────────────
function dot_handwritten(u, v, ws)
    s = 0.0
    # storage: (Ny=dim1, Nx_half=dim2, Nz=dim3, Nt=dim4)
    for it in 1:Nt, iz in 1:Nz               # slowest-varying dims outermost
        for j in 1:Ny                         # ix=1 (rfft zero, weight 1): stride-1
            @inbounds s += ws[j] * real(conj(u[j, 1, iz, it]) * v[j, 1, iz, it])
        end
        for ix in 2:Nx_half                   # rfft positive wavenumbers, weight 2
            for j in 1:Ny                     # innermost: stride-1
                @inbounds s += 2*ws[j] * real(conj(u[j, ix, iz, it]) * v[j, ix, iz, it])
            end
        end
    end
    return s / 2
end

# ──────────────────────────────────────────────────────────────────────────────
# Approach C: ReSolverFlowsBase.dot  (whatever is on the current branch)
# ──────────────────────────────────────────────────────────────────────────────
dot_ReSolverFlowsBase(u, v) = dot(u, v)

# ──────────────────────────────────────────────────────────────────────────────
# Correctness
# ──────────────────────────────────────────────────────────────────────────────
ref = dot_cartesian(parent(u), parent(v), ws, g)
let d = abs(dot_handwritten(parent(u), parent(v), ws) - ref)
    @assert d < 1e-6 "hand-written disagrees: |Δ| = $d"
end
let d = abs(dot_ReSolverFlowsBase(u, v) - ref)
    @assert d < 1e-6 "ReSolverFlowsBase.dot disagrees: |Δ| = $d"
end
println("Correctness: ok")

# ──────────────────────────────────────────────────────────────────────────────
# ProjectedField benchmark
# ──────────────────────────────────────────────────────────────────────────────
const M = 10   # number of projection modes

# modes tensor: (Ny, M, Nx_half, Nz, Nt) — shape required by ProjectedField
modes = ntuple(_ -> randn(ComplexF64, Ny, M, Nx_half, Nz, Nt), 2)
# ProjectedField data: (M, Nx_half, Nz, Nt)
a_data = randn(ComplexF64, M, Nx_half, Nz, Nt)
b_data = randn(ComplexF64, M, Nx_half, Nz, Nt)
pa = ProjectedField(g, a_data, modes)
pb = ProjectedField(g, b_data, modes)

dot_proj_ReSolverFlowsBase(a, b) = dot(a, b)   # ReSolverFlowsBase.dot for ProjectedField

function dot_proj_handwritten(a, b)
    s = 0.0
    ad, bd = parent(a), parent(b)
    # ProjectedField parent shape: (M, Nx_half, Nz, Nt) — rfft dim is index 2
    for it in 1:Nt, iz in 1:Nz
        for m in 1:M
            @inbounds s += real(conj(ad[m, 1, iz, it]) * bd[m, 1, iz, it])
        end
        for ix in 2:Nx_half, m in 1:M
            @inbounds s += 2 * real(conj(ad[m, ix, iz, it]) * bd[m, ix, iz, it])
        end
    end
    return s / 2
end

let d = abs(dot_proj_ReSolverFlowsBase(pa, pb) - dot_proj_handwritten(pa, pb))
    @assert d < 1e-6 "ProjectedField dot disagrees: |Δ| = $d"
end
println("ProjectedField correctness: ok")

# ──────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ──────────────────────────────────────────────────────────────────────────────
BenchmarkTools.DEFAULT_PARAMETERS.samples = 50

pu = parent(u); pv = parent(v)   # plain arrays for the raw-loop variants

println("\n── FTField dot  (Nx=$Nx, Ny=$Ny, Nz=$Nz, Nt=$Nt) ─────────────────────")
t_cart = @belapsed dot_cartesian($pu, $pv, $ws, $g)
t_hand = @belapsed dot_handwritten($pu, $pv, $ws)
t_nse  = @belapsed dot_ReSolverFlowsBase($u, $v)
@printf "  A  CartesianIndices (raw loop)  : %7.3f ms\n"  t_cart * 1e3
@printf "  B  hand-written split loop      : %7.3f ms\n"  t_hand * 1e3
@printf "  C  ReSolverFlowsBase.dot (FTField)        : %7.3f ms\n"  t_nse  * 1e3
@printf "  ratio C/A : %.3f\n"  (t_nse / t_cart)
@printf "  ratio C/B : %.3f\n"  (t_nse / t_hand)

println("\n── ProjectedField dot  (M=$M, Nx=$Nx, Nz=$Nz, Nt=$Nt) ─────────────────")
t_proj_hand = @belapsed dot_proj_handwritten($pa, $pb)
t_proj_nse  = @belapsed dot_proj_ReSolverFlowsBase($pa, $pb)
@printf "  A  hand-written split loop          : %7.3f ms\n"  t_proj_hand * 1e3
@printf "  B  ReSolverFlowsBase.dot (ProjectedField)     : %7.3f ms\n"  t_proj_nse  * 1e3
@printf "  ratio B/A : %.3f\n"  (t_proj_nse / t_proj_hand)

# ──────────────────────────────────────────────────────────────────────────────
# shift! and normdiff-with-shift benchmarks
# ──────────────────────────────────────────────────────────────────────────────
const SHIFTS = (0.13, -0.21, 0.07)   # one per homogeneous dim (x, z, t)

u2 = copy(u); v2 = copy(v)           # fresh copies so shift! doesn't accumulate
tmp_ft = zero(u)

# hand-written reference: explicit 4-D loop nest matching the storage layout
# (Ny=dim1, Nx_half=dim2, Nz=dim3, Nt=dim4).  Wavenumbers derived inline:
#   rfft (dim2):       k_x = ix - 1         (always ≥ 0)
#   signed FFT (dim3): k_z = iz-1 if iz-1 ≤ Nz÷2, else iz-1-Nz
#   signed FFT (dim4): k_t = it-1 if it-1 ≤ Nt÷2, else it-1-Nt
# Loop order (outermost→innermost): Nt, Nz, Nx_half, Ny — Ny is stride-1.
function shift_handwritten!(u, shifts)
    pu = parent(u)
    sx, sz, st = shifts
    for it in 1:Nt
        kt = it - 1 <= Nt >> 1 ? it - 1 : it - 1 - Nt
        for iz in 1:Nz
            kz = iz - 1 <= Nz >> 1 ? iz - 1 : iz - 1 - Nz
            for ix in 1:Nx_half
                kx = ix - 1
                phase = cis(kx * sx + kz * sz + kt * st)
                for j in 1:Ny                  # innermost: stride-1
                    @inbounds pu[j, ix, iz, it] *= phase
                end
            end
        end
    end
    return u
end

println("\n── shift!(FTField)  (Nx=$Nx, Ny=$Ny, Nz=$Nz, Nt=$Nt) ─────────────────")
t_shift_nse  = @belapsed shift!($u2, $SHIFTS)
t_shift_hand = @belapsed shift_handwritten!($u2, $SHIFTS)
@printf "  A  hand-written double loop         : %7.3f ms\n"  t_shift_hand * 1e3
@printf "  B  ReSolverFlowsBase.shift! (FTField)         : %7.3f ms\n"  t_shift_nse  * 1e3
@printf "  ratio B/A : %.3f\n"  (t_shift_nse / t_shift_hand)

println("\n── normdiff with shift  (Nx=$Nx, Ny=$Ny, Nz=$Nz, Nt=$Nt) ─────────────")
t_normdiff_shift = @belapsed normdiff($u, $v, $SHIFTS, $tmp_ft)
t_normdiff_plain = @belapsed normdiff($u, $v)
@printf "  normdiff (no shift)                 : %7.3f ms\n"  t_normdiff_plain * 1e3
@printf "  normdiff (with shift, pre-alloc tmp): %7.3f ms\n"  t_normdiff_shift * 1e3
@printf "  overhead of shift : %.3f×\n"  (t_normdiff_shift / t_normdiff_plain)
