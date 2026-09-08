# Barebone benchmark comparing two loop strategies for ProjectedField dot product.
#
# Array layout: (modes, nx, nz, nt)
#   nx  — rfft dimension: stores indices 1:(Nx>>1)+1 (wavenumbers 0..Nx/2 only)
#   nz  — signed FFT dimension: all of 1:Nz
#   nt  — signed FFT dimension: all of 1:Nt
#
# Inner product with rfft Hermitian symmetry:
#   • nx-index 1  (wavenumber 0): Hermitian weight 1 — stored once, counted once
#   • nx-index >1 (wavenumber >0): Hermitian weight 2 — stored once, represents ±nx
#
# The divide-by-2 at the end corrects for the signed-FFT pairs (nz,nt)/(-nz,-nt)
# both appearing in storage.
#
# Run: julia perf/benchmark_projected_dot.jl

using LinearAlgebra

# ──────────────────────────────────────────────────────────────────────────────
# Parameters (representative channel-flow sizes)
# ──────────────────────────────────────────────────────────────────────────────
const M       = 10            # projection modes
const Nx      = 32            # streamwise (rfft)
const Nz      = 63            # spanwise   (signed FFT, odd)
const Nt      = 63            # temporal   (signed FFT, odd)
const Nx_half = (Nx >> 1) + 1 # rfft storage width

a = randn(ComplexF64, M, Nx_half, Nz, Nt)
b = randn(ComplexF64, M, Nx_half, Nz, Nt)

# ──────────────────────────────────────────────────────────────────────────────
# Approach A: two-pass, macro style
# Split the rfft dimension into two separate loops so no conditional appears
# inside the hot path.  This mirrors the @loop_nznt-based implementation.
# ──────────────────────────────────────────────────────────────────────────────
function dot_twopass(a, b)
    s = 0.0
    # nx = 0 plane (rfft index 1), Hermitian weight 1
    for _nt in 1:Nt, _nz in 1:Nz, m in 1:M
        @inbounds s += real(conj(a[m, 1, _nz, _nt]) * b[m, 1, _nz, _nt])
    end
    # nx > 0 planes (rfft indices 2:Nx_half), Hermitian weight 2
    for _nt in 1:Nt, _nz in 1:Nz, _nx in 2:Nx_half, m in 1:M
        @inbounds s += 2 * real(conj(a[m, _nx, _nz, _nt]) * b[m, _nx, _nz, _nt])
    end
    return s / 2
end

# ──────────────────────────────────────────────────────────────────────────────
# Approach B: single-pass with conditional weight
# Mirrors the for_each_wavenumber-based implementation in ReSolverFlowsBase.
# ──────────────────────────────────────────────────────────────────────────────
function dot_conditional(a, b)
    s = 0.0
    for _nt in 1:Nt, _nz in 1:Nz, _nx in 1:Nx_half, m in 1:M
        w = _nx == 1 ? 1 : 2
        @inbounds s += w * real(conj(a[m, _nx, _nz, _nt]) * b[m, _nx, _nz, _nt])
    end
    return s / 2
end

# ──────────────────────────────────────────────────────────────────────────────
# Correctness check
# ──────────────────────────────────────────────────────────────────────────────
let diff = abs(dot_twopass(a, b) - dot_conditional(a, b))
    @assert diff < 1e-10 "Results disagree: diff = $diff"
    println("Correctness: ok (|Δ| = $(diff))")
end

# ──────────────────────────────────────────────────────────────────────────────
# Timing
# ──────────────────────────────────────────────────────────────────────────────
function benchmark(f, a, b; warmup=10, reps=200)
    for _ in 1:warmup; f(a, b); end
    t = @elapsed for _ in 1:reps; f(a, b); end
    return t / reps
end

t_a = benchmark(dot_twopass,    a, b)
t_b = benchmark(dot_conditional, a, b)

println("\nResults (M=$M, Nx=$Nx, Nz=$Nz, Nt=$Nt):")
println("  A  two-pass, no conditional : $(round(t_a*1e6, digits=2)) μs")
println("  B  single-pass, conditional : $(round(t_b*1e6, digits=2)) μs")
println("  ratio B/A                   : $(round(t_b/t_a, digits=3))")
