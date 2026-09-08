# Benchmark: generated add_homogeneous_laplacian! vs the production
# CartesianIndices implementation.
#
# The generated function below is a copy of the previous production kernel.
# `ReSolverFlowsBase.add_homogeneous_laplacian!` is the current production implementation.
# The benchmark is intentionally limited to the homogeneous Laplacian
# contribution because the full laplacian! also calls the downstream
# grid-specific _inhomogeneous_laplacian!.
#
# Run from the repo root:
#
#   julia --project=benchmark benchmark/bench_laplacian.jl
#
# For a quick smoke test:
#
#   julia --project=benchmark benchmark/bench_laplacian.jl --quick

import BenchmarkTools
import Printf
import Random
import ReSolverFlowsBase

# ------------------------------------------------------------------ #
# Benchmark grids                                                     #
# ------------------------------------------------------------------ #

struct TripleGrid <: ReSolverFlowsBase.AbstractGrid{Float64, 3, (2, 1, 3, nothing), (2, 3)}
    Ny::Int
    Nx::Int
    Nz::Int
    alpha::Float64
    beta::Float64
    ws::Vector{Float64}
end

function TripleGrid(Ny::Integer, Nx::Integer, Nz::Integer;
                    alpha::Real=1.0, beta::Real=1.0)
    return TripleGrid(Int(Ny), Int(Nx), Int(Nz),
                      Float64(alpha), Float64(beta), ones(Float64, Int(Ny)))
end

Base.size(g::TripleGrid) = (g.Ny, g.Nx, g.Nz)
ReSolverFlowsBase.weights(g::TripleGrid) = g.ws
ReSolverFlowsBase.wavenumber_scale(g::TripleGrid, dim::Int) =
    dim == 2 ? g.alpha :
    dim == 3 ? g.beta :
    one(g.alpha)

struct QuadTimeGrid <: ReSolverFlowsBase.AbstractGrid{Float64, 4, (2, 1, 3, 4), (2, 3, 4)}
    Ny::Int
    Nx::Int
    Nz::Int
    Nt::Int
    alpha::Float64
    beta::Float64
    omega::Float64
    ws::Vector{Float64}
end

function QuadTimeGrid(Ny::Integer, Nx::Integer, Nz::Integer, Nt::Integer;
                      alpha::Real=1.0, beta::Real=1.0, omega::Real=1.0)
    return QuadTimeGrid(Int(Ny), Int(Nx), Int(Nz), Int(Nt),
                        Float64(alpha), Float64(beta), Float64(omega),
                        ones(Float64, Int(Ny)))
end

Base.size(g::QuadTimeGrid) = (g.Ny, g.Nx, g.Nz, g.Nt)
ReSolverFlowsBase.weights(g::QuadTimeGrid) = g.ws
ReSolverFlowsBase.wavenumber_scale(g::QuadTimeGrid, dim::Int) =
    dim == 2 ? g.alpha :
    dim == 3 ? g.beta :
    dim == 4 ? g.omega :
    one(g.alpha)

# ------------------------------------------------------------------ #
# Previous generated implementation                                   #
# ------------------------------------------------------------------ #

@generated function add_homogeneous_laplacian_generated!(
        out::ReSolverFlowsBase.FTField{G},
        u::ReSolverFlowsBase.FTField{G}
    ) where {
        T, D, AXES, FFT_DIMS_ORDER,
        G<:ReSolverFlowsBase.AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}

    H = filter(d -> d != AXES[4], FFT_DIMS_ORDER)
    isempty(H) && return :(return out)

    syms = [Symbol("_i", d) for d in 1:D]

    k2_terms = Expr[]
    for d in H
        n_expr = Symbol("_n", d)
        scale = Symbol("_k_scale", d)
        push!(k2_terms, :(($scale * $n_expr)^2))
    end
    k2_expr = length(k2_terms) == 1 ? k2_terms[1] : Expr(:call, :+, k2_terms...)

    body = quote
        @inbounds parent(out)[$(syms...)] -= $k2_expr * parent(u)[$(syms...)]
    end

    signed_dims = H[2:end]
    blocks = Expr[]
    for mask in 0:(1 << length(signed_dims)) - 1
        ranges = [:(1:Base.size(u, $d)) for d in 1:D]
        wnums = Any[:nothing for _ in 1:D]
        wnums[H[1]] = :($(syms[H[1]]) - 1)

        for (j, d) in enumerate(signed_dims)
            if Bool((mask >> (j - 1)) & 1)
                ranges[d] = :((Base.size(u, $d) >> 1) + 2:Base.size(u, $d))
                wnums[d] = :($(syms[d]) - Base.size(u, $d) - 1)
            else
                ranges[d] = :(1:(Base.size(u, $d) >> 1) + 1)
                wnums[d] = :($(syms[d]) - 1)
            end
        end

        block = body
        for d in 1:D
            sym = syms[d]
            rng = ranges[d]
            if d in H
                n_sym = Symbol("_n", d)
                block = :(for $sym in $rng
                              $n_sym = $(wnums[d])
                              $block
                          end)
            else
                block = :(for $sym in $rng
                              $block
                          end)
            end
        end
        push!(blocks, block)
    end

    return Base.remove_linenums!(quote
        $([:($(Symbol("_k_scale", d)) = ReSolverFlowsBase.wavenumber_scale(ReSolverFlowsBase.grid(u), $d)) for d in H]...)
        $(blocks...)
        return out
    end)
end

# ------------------------------------------------------------------ #
# Correctness and benchmark helpers                                   #
# ------------------------------------------------------------------ #

function fill_random!(a)
    Random.randn!(a)
    return a
end

function check_correctness(g, label)
    u = ReSolverFlowsBase.FTField(g)
    seed = similar(parent(u))
    fill_random!(parent(u))
    fill_random!(seed)

    out_generated = ReSolverFlowsBase.FTField(g)
    out_cartesian = ReSolverFlowsBase.FTField(g)
    parent(out_generated) .= seed
    parent(out_cartesian) .= seed

    add_homogeneous_laplacian_generated!(out_generated, u)
    ReSolverFlowsBase.add_homogeneous_laplacian!(out_cartesian, u)

    err = maximum(abs, parent(out_cartesian) .- parent(out_generated))
    ok = err < 1e-12

    # Measure allocations after the first call has compiled each method.
    alloc_generated = @allocated add_homogeneous_laplacian_generated!(out_generated, u)
    alloc_cartesian = @allocated ReSolverFlowsBase.add_homogeneous_laplacian!(out_cartesian, u)

    Printf.@printf("  %-30s  cartesian: %s (err=%.2e)\n",
                   label, ok ? "ok" : "BAD", err)
    Printf.@printf("  %-30s  alloc: generated=%d B  cartesian=%d B\n",
                   "", alloc_generated, alloc_cartesian)
end

function run_bench(g, label; samples::Int)
    u = ReSolverFlowsBase.FTField(g)
    out = ReSolverFlowsBase.FTField(g)
    fill_random!(parent(u))
    fill_random!(parent(out))

    # Warm-up.
    add_homogeneous_laplacian_generated!(out, u)
    ReSolverFlowsBase.add_homogeneous_laplacian!(out, u)

    b_generated = BenchmarkTools.@benchmark add_homogeneous_laplacian_generated!($out, $u) samples=samples evals=1
    b_cartesian = BenchmarkTools.@benchmark ReSolverFlowsBase.add_homogeneous_laplacian!($out, $u) samples=samples evals=1

    m_generated = BenchmarkTools.median(b_generated)
    m_cartesian = BenchmarkTools.median(b_cartesian)
    t_generated = m_generated.time
    t_cartesian = m_cartesian.time

    Printf.@printf("  %-24s  generated: %8.1f us (%d alloc, %d B)   cartesian: %8.1f us (x%.2f, %d alloc, %d B)\n",
                   label,
                   t_generated / 1e3,
                   BenchmarkTools.allocs(m_generated),
                   BenchmarkTools.memory(m_generated),
                   t_cartesian / 1e3,
                   t_cartesian / t_generated,
                   BenchmarkTools.allocs(m_cartesian),
                   BenchmarkTools.memory(m_cartesian))
end

# ------------------------------------------------------------------ #
# Driver                                                              #
# ------------------------------------------------------------------ #

const QUICK = "--quick" in ARGS
const SAMPLES = QUICK ? 20 : 200

Random.seed!(1234)

println("=== add_homogeneous_laplacian! correctness ===")
check_correctness(TripleGrid(5, 8, 6), "3D TripleGrid")
check_correctness(QuadTimeGrid(5, 8, 6, 4), "4D QuadTimeGrid")
println()

sizes_3d = QUICK ? [
    (Ny=16, Nx=32, Nz=24, label="small  (16x32x24)"),
] : [
    (Ny=16,  Nx=32,  Nz=24,  label="small  (16x32x24)"),
    (Ny=97,  Nx=128, Nz=96,  label="medium (97x128x96)"),
    (Ny=193, Nx=256, Nz=192, label="large  (193x256x192)"),
]

sizes_4d = QUICK ? [
    (Ny=5, Nx=8, Nz=6, Nt=4, label="small  (5x8x6x4)"),
] : [
    (Ny=5,  Nx=8,  Nz=6,  Nt=4,  label="small  (5x8x6x4)"),
    (Ny=16, Nx=32, Nz=24, Nt=16, label="medium (16x32x24x16)"),
    (Ny=32, Nx=64, Nz=48, Nt=24, label="large  (32x64x48x24)"),
]

println("=== add_homogeneous_laplacian! benchmarks (3D TripleGrid) ===")
println("samples = ", SAMPLES)
for s in sizes_3d
    run_bench(TripleGrid(s.Ny, s.Nx, s.Nz), s.label; samples=SAMPLES)
end
println()

println("=== add_homogeneous_laplacian! benchmarks (4D QuadTimeGrid) ===")
println("samples = ", SAMPLES)
for s in sizes_4d
    run_bench(QuadTimeGrid(s.Ny, s.Nx, s.Nz, s.Nt), s.label; samples=SAMPLES)
end
