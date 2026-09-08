# Benchmark: @generated ddx! vs CartesianIndices vs broadcast for spectral
# differentiation.
#
# Grid: TripleGrid(Ny, Nx, Nz) — one inhomogeneous wall-normal dimension
# and two homogeneous FFT dimensions (FFT_DIMS_ORDER = (2, 3)):
#
#   array dim 1  : inhomogeneous (y), stored first
#   array dim 2  : rfft (x), non-negative wavenumbers 0…(Nx÷2)
#   array dim 3  : signed FFT (z), wavenumbers 0…(Nz÷2), -(Nz÷2)+1…-1
#
# Three ddx! implementations are compared:
#
#   ddx_generated!  — the former @generated loop nest (fully unrolled at compile time)
#   ddx_cartesian!  — the current production ddx! (CartesianIndices, split blocks)
#   ddx_broadcast!  — pre-generate a wavenumber NamedTuple (one pre-reshaped
#                     array per FFT dim, keyed :dim2/:dim3/…); the kernel is a
#                     single .= broadcast multiply — zero allocations in the hot path
#
# Benchmarks are run for:
#   - 3-D TripleGrid: dim 2 (rfft/x) and dim 3 (signed FFT/z)
#   - 4-D QuadGrid:   dim 2 (rfft/x), dim 3 (signed FFT/z), dim 4 (signed FFT/t)
#   - small / medium / large array sizes per dimensionality
#
# Run from the repo root:
#   julia --project=benchmark benchmark/bench_ddx.jl

using BenchmarkTools
using LinearAlgebra
using Printf
using ReSolverFlowsBase

include("../test/fake.jl")
include("../test/test_grids.jl")

# ------------------------------------------------------------------ #
# Implementation 1: @generated (former production code)              #
# ------------------------------------------------------------------ #
#
# Verbatim copy of the @generated ddx! that was the production implementation
# before being replaced by the CartesianIndices version.  Kept here so the
# benchmark can measure whether the explicit loop-unrolling buys anything over
# the CartesianIndices approach.
#
# The generated code emits a fully unrolled loop nest for each (DIM, grid)
# specialisation: dim 1 is always innermost (column-major), the rfft block is
# a single loop, and signed-FFT blocks are split into pos/neg sub-loops.

@generated function ddx_generated!(out::FTField{G}, u::FTField{G}, ::Val{DIM};
                                    adjoint::Bool=false) where {
        DIM, T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}

    (isnothing(DIM) || isnothing(AXES[DIM])) && return :(return out)
    DIM ∉ FFT_DIMS_ORDER && return :(throw(ReSolverFlowsBase.NotImplementedError(grid(u), Val($DIM))))

    syms  = [Symbol("_i", d) for d in 1:D]
    n_sym = Symbol("_n", DIM)

    assign = quote
        @inbounds parent(out)[$(syms...)] =
            _ddx_sign * $n_sym * _ddx_scale * parent(u)[$(syms...)]
    end

    function cache_ordered_loop(body, ranges, wavenumbers)
        for d in 1:D
            sym = syms[d]
            rng = ranges[d]
            body = if d == DIM
                :(for $sym in $rng; $n_sym = $(wavenumbers[d]); $body; end)
            else
                :(for $sym in $rng; $body; end)
            end
        end
        return body
    end

    ranges = [:(1:Base.size(u, $d)) for d in 1:D]
    wnums  = Any[:nothing for _ in 1:D]

    if DIM == FFT_DIMS_ORDER[1]
        wnums[DIM] = :($(syms[DIM]) - 1)
        body = cache_ordered_loop(assign, ranges, wnums)
    else
        pos_ranges = copy(ranges); neg_ranges = copy(ranges)
        pos_ranges[DIM] = :(1:(Base.size(u, $DIM) >> 1) + 1)
        neg_ranges[DIM] = :((Base.size(u, $DIM) >> 1) + 2:Base.size(u, $DIM))
        wnums[DIM] = :($(syms[DIM]) - 1)
        pos_body = cache_ordered_loop(assign, pos_ranges, wnums)
        wnums[DIM] = :($(syms[DIM]) - Base.size(u, $DIM) - 1)
        neg_body = cache_ordered_loop(assign, neg_ranges, wnums)
        body = :($pos_body; $neg_body)
    end

    return quote
        _ddx_scale = wavenumber_scale(grid(u), $DIM)
        _ddx_sign  = adjoint ? -1im : 1im
        $body
        return out
    end
end

# ------------------------------------------------------------------ #
# Implementation 2: CartesianIndices with split blocks               #
# ------------------------------------------------------------------ #
#
# For the rfft dimension all stored wavenumbers are non-negative, so a
# single loop suffices.  For signed FFT dimensions the index range is split
# into a "positive" block [1:(N÷2)+1] and a "negative" block [(N÷2)+2:N]
# so that the signed wavenumber can be computed as a simple offset inside
# each block without an in-loop branch.
#
# DIM and FFT_DIMS_ORDER are compile-time constants, so the if-branch and
# the ntuple construction are specialised away by Julia.

function ddx_cartesian!(out::FTField{G}, u::FTField{G}, ::Val{DIM};
                        adjoint::Bool=false
    ) where {DIM, T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}

    (isnothing(DIM) || isnothing(AXES[DIM])) && return out
    DIM ∉ FFT_DIMS_ORDER && throw(ReSolverFlowsBase.NotImplementedError(grid(u), Val(DIM)))

    scale = wavenumber_scale(grid(u), DIM)
    coeff = adjoint ? -im * T(scale) : im * T(scale)
    Nd    = size(u, DIM)
    pu    = parent(u)
    pout  = parent(out)

    if DIM == FFT_DIMS_ORDER[1]
        # rfft dim: wavenumber n = i - 1 (always ≥ 0), single pass
        @inbounds for I in CartesianIndices(pu)
            pout[I] = (coeff * (I[DIM] - 1)) * pu[I]
        end
    else
        # Signed FFT dim: split into non-negative and negative wavenumber blocks
        pos_ranges = ntuple(d -> d == DIM ? (1:(Nd >> 1) + 1)     : Base.OneTo(size(pu, d)), Val(D))
        neg_ranges = ntuple(d -> d == DIM ? ((Nd >> 1) + 2:Nd)    : Base.OneTo(size(pu, d)), Val(D))
        @inbounds for I in CartesianIndices(pos_ranges)
            pout[I] = (coeff * (I[DIM] - 1)) * pu[I]
        end
        @inbounds for I in CartesianIndices(neg_ranges)
            pout[I] = (coeff * (I[DIM] - Nd - 1)) * pu[I]
        end
    end
    return out
end

# ------------------------------------------------------------------ #
# Implementation 3: broadcast with pre-generated wavenumber vectors  #
# ------------------------------------------------------------------ #
#
# `make_wavenumber_dict` builds a `Dict{Int, Vector{Complex{T}}}` keyed by
# storage-dimension index, one entry per FFT dimension:
#
#   rfft dim   : [0, 1, …, Nd-1]                       × coeff
#   signed FFT : [0, 1, …, Nd÷2, -(Nd÷2)+1, …, -1]   × coeff
#
# `Nd` is taken from `transform_size(g)` so the rfft dimension gets its
# half-spectrum length `(Nx÷2)+1`.  Construct once per (grid, adjoint) pair
# before the timed region.
#
# Inside `ddx_broadcast!` the kernel looks up the vector by DIM, builds the
# broadcast shape `(1,…,Nd,…,1)` as a compile-time-structured ntuple (the
# position of Nd is known at compile time from Val{DIM}), and fuses the
# multiply into a single in-place broadcast — zero heap allocations.

function make_wavenumber_dict(
        g::G; adjoint::Bool=false
    ) where {T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}
    wns = Dict{Int, Vector{Complex{T}}}()
    tsz = transform_size(g)
    for (k, dim) in enumerate(FFT_DIMS_ORDER)
        scale = wavenumber_scale(g, dim)
        coeff = adjoint ? -im * T(scale) : im * T(scale)
        Nd    = tsz[dim]
        if k == 1
            wns[dim] = coeff .* collect(0:Nd-1)
        else
            half = Nd >> 1
            wn   = Vector{Complex{T}}(undef, Nd)
            for i in 1:half+1;   wn[i] = coeff * (i - 1)      end
            for i in half+2:Nd;  wn[i] = coeff * (i - Nd - 1) end
            wns[dim] = wn
        end
    end
    return wns
end

function ddx_broadcast!(out::FTField{G}, u::FTField{G}, ::Val{DIM},
                        wns::Dict{Int, Vector{Complex{T}}}) where {
        DIM, T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}

    (isnothing(DIM) || isnothing(AXES[DIM])) && return out
    DIM ∉ FFT_DIMS_ORDER && throw(ReSolverFlowsBase.NotImplementedError(grid(u), Val(DIM)))

    wn = wns[DIM]
    # Shape has Nd at position DIM and 1 everywhere else.
    # Val(D) lets Julia unroll the ntuple at compile time; the only runtime
    # value is length(wn), so the shape is structurally constant.
    shape = ntuple(d -> d == DIM ? length(wn) : 1, Val(D))
    parent(out) .= reshape(wn, shape) .* parent(u)
    return out
end

# ------------------------------------------------------------------ #
# Correctness check                                                   #
# ------------------------------------------------------------------ #

function check_correctness(g, dim_val, label)
    u   = FTField(g)
    parent(u) .= randn(ComplexF64, size(parent(u)))

    out_gen  = FTField(g)
    out_ci   = FTField(g)
    out_bc   = FTField(g)

    wns = make_wavenumber_dict(g)

    # First call triggers compilation; compare results for correctness.
    ddx_generated!(out_gen, u, dim_val)
    ddx_cartesian!(out_ci,  u, dim_val)
    ddx_broadcast!(out_bc,  u, dim_val, wns)

    err_ci = maximum(abs, parent(out_ci) .- parent(out_gen))
    err_bc = maximum(abs, parent(out_bc) .- parent(out_gen))

    ok_ci = err_ci < 1e-12
    ok_bc = err_bc < 1e-12
    @printf("  %-30s  cartesian: %s (err=%.2e)  broadcast: %s (err=%.2e)\n",
            label,
            ok_ci ? "✓" : "✗", err_ci,
            ok_bc ? "✓" : "✗", err_bc)

    # Second call is post-compilation: @allocated reflects steady-state heap use.
    alloc_gen = @allocated ddx_generated!(out_gen, u, dim_val)
    alloc_ci  = @allocated ddx_cartesian!(out_ci,  u, dim_val)
    alloc_bc  = @allocated ddx_broadcast!(out_bc,  u, dim_val, wns)
    @printf("  %-30s  alloc:  generated=%4d B  cartesian=%4d B  broadcast=%4d B\n",
            "", alloc_gen, alloc_ci, alloc_bc)
end

# ------------------------------------------------------------------ #
# 4-D grid for extra correctness coverage                            #
# ------------------------------------------------------------------ #
#
# Layout: AXES = (2, 1, 3, 4), FFT_DIMS_ORDER = (2, 3, 4)
#
#   array dim 1: inhomogeneous (y), length Ny
#   array dim 2: rfft           (x), length Nx  — FFT_DIMS_ORDER[1]
#   array dim 3: signed FFT     (z), length Nz  — FFT_DIMS_ORDER[2]
#   array dim 4: signed FFT     (t), length Nt  — FFT_DIMS_ORDER[3]
#
# This exercises the three-FFT-dim code path in make_wavenumber_dict
# (field :dim2, :dim3, :dim4) and the D=4 reshape in the kernel.

struct QuadGrid <: AbstractGrid{Float64, 4, (2, 1, 3, 4), (2, 3, 4)}
    Ny :: Int
    Nx :: Int
    Nz :: Int
    Nt :: Int
    ws :: Vector{Float64}
end

QuadGrid(Ny, Nx, Nz, Nt) = QuadGrid(Ny, Nx, Nz, Nt, ones(Ny))

Base.size(g::QuadGrid)                            = (g.Ny, g.Nx, g.Nz, g.Nt)
ReSolverFlowsBase.weights(g::QuadGrid)                      = g.ws
ReSolverFlowsBase.wavenumber_scale(::QuadGrid, ::Int)       = 1.0

println("=== Correctness checks ===")
let g = TripleGrid(5, 8, 6)
    check_correctness(g, Val(2), "3D rfft dim (dim 2)")
    check_correctness(g, Val(3), "3D signed FFT dim (dim 3)")
end
println()
let g = QuadGrid(5, 8, 6, 4)
    check_correctness(g, Val(2), "4D rfft dim (dim 2)")
    check_correctness(g, Val(3), "4D signed FFT dim (dim 3)")
    check_correctness(g, Val(4), "4D signed FFT dim (dim 4)")
end
println()

# ------------------------------------------------------------------ #
# Benchmark helper                                                    #
# ------------------------------------------------------------------ #

function run_bench(g, dim_val, dim_label, size_label)
    u   = FTField(g)
    out = FTField(g)
    parent(u) .= randn(ComplexF64, size(parent(u)))

    wns = make_wavenumber_dict(g)

    # warm-up
    ddx_generated!(out, u, dim_val)
    ddx_cartesian!(out, u, dim_val)
    ddx_broadcast!(out, u, dim_val, wns)

    b_gen = @benchmark ddx_generated!($out, $u, $dim_val)       samples=200 evals=1
    b_ci  = @benchmark ddx_cartesian!($out, $u, $dim_val)       samples=200 evals=1
    b_bc  = @benchmark ddx_broadcast!($out, $u, $dim_val, $wns) samples=200 evals=1

    t_gen = median(b_gen).time
    t_ci  = median(b_ci).time
    t_bc  = median(b_bc).time
    a_gen = allocs(median(b_gen))
    a_ci  = allocs(median(b_ci))
    a_bc  = allocs(median(b_bc))

    @printf("  %-20s  %-10s  generated: %7.1f µs (%d alloc)   cartesian: %7.1f µs (×%.2f, %d alloc)   broadcast: %7.1f µs (×%.2f, %d alloc)\n",
            size_label, dim_label,
            t_gen/1e3, a_gen,
            t_ci/1e3, t_ci/t_gen, a_ci,
            t_bc/1e3, t_bc/t_gen, a_bc)
end

# ------------------------------------------------------------------ #
# Grid sizes                                                          #
# ------------------------------------------------------------------ #
#
# Three regimes per dimensionality:
#   small  — fits easily in L2 cache
#   medium — fits in L3 cache, representative of a coarse channel run
#   large  — exceeds L3, representative of a production channel run
#
# 4-D sizes use a shorter 4th dimension to keep total array size
# comparable to the 3-D cases (4-D arrays grow as Ny·Nx·Nz·Nt).

sizes_3d = [
    (Ny=16,  Nx=32,  Nz=24,  label="small  (16×32×24)"),
    (Ny=97,  Nx=128, Nz=96,  label="medium (97×128×96)"),
    (Ny=193, Nx=256, Nz=192, label="large  (193×256×192)"),
]

sizes_4d = [
    (Ny=5,  Nx=8,  Nz=6,  Nt=4,  label="small  (5×8×6×4)"),
    (Ny=16, Nx=32, Nz=24, Nt=16, label="medium (16×32×24×16)"),
    (Ny=32, Nx=64, Nz=48, Nt=24, label="large  (32×64×48×24)"),
]

println("=== ddx! benchmarks (3-D TripleGrid) ===")
println()
for s in sizes_3d
    g = TripleGrid(s.Ny, s.Nx, s.Nz)
    run_bench(g, Val(2), "rfft (x, dim 2)", s.label)
    run_bench(g, Val(3), "sgn  (z, dim 3)", s.label)
    println()
end

println("=== ddx! benchmarks (4-D QuadGrid) ===")
println()
for s in sizes_4d
    g = QuadGrid(s.Ny, s.Nx, s.Nz, s.Nt)
    run_bench(g, Val(2), "rfft (x, dim 2)", s.label)
    run_bench(g, Val(3), "sgn  (z, dim 3)", s.label)
    run_bench(g, Val(4), "sgn  (t, dim 4)", s.label)
    println()
end
