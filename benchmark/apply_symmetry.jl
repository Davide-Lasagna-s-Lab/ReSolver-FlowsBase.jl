# Benchmark: apply_symmetry! — specific (FTField / ProjectedField) vs generic array.
#
# Grid: 4-D QuadGrid with one inhomogeneous wall-normal dimension and three
# homogeneous FFT dimensions (FFT_DIMS_ORDER = (2, 3, 4)):
#
#   array dim 1: inhomogeneous (y), length Ny
#   array dim 2: rfft           (x), length Nx  — FFT_DIMS_ORDER[1]
#   array dim 3: signed FFT     (z), length Nz  — FFT_DIMS_ORDER[2]
#   array dim 4: signed FFT     (t), length Nt  — FFT_DIMS_ORDER[3]
#
# Three implementations are compared at each grid size:
#
#   generic(FTField)          — apply_symmetry!(parent(u), Val(FFT_DIMS_ORDER))
#   specific(FTField)         — apply_symmetry!(u::FTField)
#   generic(ProjectedField)   — apply_symmetry!(parent(a), Val((2,3,4)))
#   specific(ProjectedField)  — apply_symmetry!(a::ProjectedField)
#
# The FTField-specific method uses homogeneous_axes/inhomogeneous_axes and
# combine_indices to navigate grid storage; the generic method iterates over
# the raw parent array using the FFT_DIMS_ORDER positions directly.  For
# ProjectedField the parent always has kH at axes 2..Nhom+1, so the generic
# call uses Val((2,3,4)) regardless of the grid's FFT_DIMS_ORDER.
#
# Run from the repo root:
#   julia --project=benchmark benchmark/apply_symmetry.jl

using BenchmarkTools
using LinearAlgebra
using Printf
using ReSolverFlowsBase
const apply_symmetry! = ReSolverFlowsBase.apply_symmetry!

include("../test/fake.jl")
include("../test/test_grids.jl")

# ------------------------------------------------------------------ #
# 4-D grid                                                           #
# ------------------------------------------------------------------ #

struct QuadGrid <: AbstractGrid{Float64, 4, (2, 1, 3, 4), (2, 3, 4)}
    Ny :: Int
    Nx :: Int
    Nz :: Int
    Nt :: Int
    ws :: Vector{Float64}
end

QuadGrid(Ny, Nx, Nz, Nt) = QuadGrid(Ny, Nx, Nz, Nt, ones(Ny))

Base.size(g::QuadGrid)                      = (g.Ny, g.Nx, g.Nz, g.Nt)
ReSolverFlowsBase.weights(g::QuadGrid)               = g.ws
ReSolverFlowsBase.wavenumber_scale(::QuadGrid, ::Int) = 1.0

# ------------------------------------------------------------------ #
# Grid sizes                                                          #
# ------------------------------------------------------------------ #
#
# Three regimes representative of small / medium / large 4-D fields.
# Total FTField element counts are kept comparable to the 3-D cases in
# bench_ddx.jl so timing is in the same range.

sizes = [
    (Ny=4,  Nx=8,  Nz=6,  Nt=4,  label="small  (4×8×6×4)"),
    (Ny=16, Nx=32, Nz=24, Nt=16, label="medium (16×32×24×16)"),
    (Ny=32, Nx=64, Nz=48, Nt=24, label="large  (32×64×48×24)"),
]

const Nm = 8   # number of modes for ProjectedField benchmarks

# ------------------------------------------------------------------ #
# Correctness check                                                   #
# ------------------------------------------------------------------ #

let g = QuadGrid(4, 8, 6, 4)
    u = FTField(g)
    parent(u) .= randn(ComplexF64, size(parent(u)))
    u_copy = FTField(g, copy(parent(u)))

    # generic and specific must agree on the result
    apply_symmetry!(u)
    apply_symmetry!(parent(u_copy), fft_dims(g))
    err = maximum(abs, parent(u) .- parent(u_copy))
    @printf("FTField correctness:          specific vs generic  err=%.2e  %s\n",
            err, err < 1e-14 ? "✓" : "✗")

    Ny, Nx, Nz, Nt = 4, 8, 6, 4
    ms = ntuple(_ -> randn(ComplexF64, Nm, Ny, (Nx>>1)+1, Nz, Nt), 1)
    a  = ProjectedField(g, ms);  parent(a) .= randn(ComplexF64, size(parent(a)))
    a2 = ProjectedField(g, ms);  parent(a2) .= copy(parent(a))
    dims_pa = ntuple(k -> k+1, length(fft_dims(g)))
    apply_symmetry!(a)
    apply_symmetry!(parent(a2), dims_pa)
    err2 = maximum(abs, parent(a) .- parent(a2))
    @printf("ProjectedField correctness:   specific vs generic  err=%.2e  %s\n",
            err2, err2 < 1e-14 ? "✓" : "✗")
end
println()

# ------------------------------------------------------------------ #
# FTField benchmarks                                                  #
# ------------------------------------------------------------------ #

println("=== FTField apply_symmetry! ===")
println()
@printf("  %-24s  %s\n", "", "specific       generic(parent)  ratio")
for s in sizes
    g  = QuadGrid(s.Ny, s.Nx, s.Nz, s.Nt)
    u  = FTField(g)
    parent(u) .= randn(ComplexF64, size(parent(u)))
    fft_d = fft_dims(g)

    apply_symmetry!(u)
    apply_symmetry!(parent(u), fft_d)

    b_sp  = @benchmark apply_symmetry!($u)                       samples=500 evals=5
    b_gen = @benchmark apply_symmetry!($(parent(u)), $(fft_d))   samples=500 evals=5

    t_sp  = median(b_sp).time  / 1e3
    t_gen = median(b_gen).time / 1e3
    @printf("  %-24s  %7.2f µs (%d alloc)   %7.2f µs (%d alloc)   ×%.2f\n",
            s.label,
            t_sp,  allocs(median(b_sp)),
            t_gen, allocs(median(b_gen)),
            t_gen / t_sp)
end

# ------------------------------------------------------------------ #
# ProjectedField benchmarks                                           #
# ------------------------------------------------------------------ #

println()
println("=== ProjectedField apply_symmetry!  (Nm=$Nm) ===")
println()
@printf("  %-24s  %s\n", "", "specific       generic(parent)  ratio")
for s in sizes
    g       = QuadGrid(s.Ny, s.Nx, s.Nz, s.Nt)
    ms      = ntuple(_ -> randn(ComplexF64, Nm, s.Ny, (s.Nx>>1)+1, s.Nz, s.Nt), 1)
    a       = ProjectedField(g, ms)
    parent(a) .= randn(ComplexF64, size(parent(a)))
    dims_pa = ntuple(k -> k+1, length(fft_dims(g)))   # (2, 3, 4) for QuadGrid

    apply_symmetry!(a)
    apply_symmetry!(parent(a), dims_pa)

    b_sp  = @benchmark apply_symmetry!($a)                         samples=500 evals=5
    b_gen = @benchmark apply_symmetry!($(parent(a)), $(dims_pa))   samples=500 evals=5

    t_sp  = median(b_sp).time  / 1e3
    t_gen = median(b_gen).time / 1e3
    @printf("  %-24s  %7.2f µs (%d alloc)   %7.2f µs (%d alloc)   ×%.2f\n",
            s.label,
            t_sp,  allocs(median(b_sp)),
            t_gen, allocs(median(b_gen)),
            t_gen / t_sp)
end
