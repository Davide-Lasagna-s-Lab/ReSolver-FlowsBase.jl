# Benchmark: ReSolverFlowsBase.LoopGalerkin vs ReSolverFlowsBase.GemmGalerkin vs a hand-rolled
# loop kernel that hardcodes the ChannelFlow array layout.  All three
# implementations operate on kH-dependent modes with the canonical layout
# documented in `src/galerkin.jl`.
#
# Grid layout matches ChannelFlow:
#   AXES  = (2, 1, 3, 4)  ->  y->dim1, x->dim2, z->dim3, t->dim4
#   FFT_DIMS_ORDER = (2, 3, 4)     ->  rfft on x (dim2), signed FFT on z and t
#   size  = (Ny, Nx, Nz, Nt) in array-dimension order
#   FTField storage = (Ny, Nx_half, Nz, Nt)
#
# Mode storage (kH-dependent), matching the canonical ChannelFlow layout:
#
#   modes[n] :: Array{ComplexF64, 5}
#   size(modes[n]) == (Ny, Nm, Nx_half, Nz, Nt)
#
# The LOOP series is a benchmark-local hand-rolled loop specialised to the
# array layout above.  It is the reference for what a perfectly-specialised
# implementation can achieve and provides a direct comparison against the
# generic ReSolverFlowsBase.LoopGalerkin (NSLOOP) implementation.
#
# Run:
#
#   julia --project=benchmark benchmark/galerkin_algorithms.jl
#
# Optional knobs:
#
#   GALERKIN_SCALE_FACTORS=1,2,4,8
#   GALERKIN_NM_VALS=2,4,8,16,32
#   GALERKIN_MAX_MODE_MIB=8192
#   GALERKIN_BENCHMARK_SAMPLES=3
#   GALERKIN_BENCHMARK_SECONDS=0.2

import BenchmarkTools
import ReSolverFlowsBase
import PyPlot

# ─────────────────────────────────────────────────────────────── #
# Minimal concrete grid — same layout as ChannelFlow             #
# ─────────────────────────────────────────────────────────────── #
struct BenchGrid <: ReSolverFlowsBase.AbstractGrid{Float64, 4, (2, 1, 3, 4), (2, 3, 4)}
    dims :: NTuple{4, Int}
    ws :: Vector{Float64}
end

Base.size(g::BenchGrid) = g.dims
ReSolverFlowsBase.weights(g::BenchGrid)               = g.ws
ReSolverFlowsBase.wavenumber_scale(::BenchGrid, ::Int) = 1.0
ReSolverFlowsBase.points(g::BenchGrid; _...)          = begin
    Ny, Nx, Nz, Nt = size(g)
    (reshape(collect(1.0:Ny), Ny, 1, 1, 1),
     reshape(collect(1.0:Nx), 1, Nx, 1, 1),
     reshape(collect(1.0:Nz), 1, 1, Nz, 1),
     reshape(collect(1.0:Nt), 1, 1, 1, Nt))
end
Base.convert(::Type{Float64}, g::BenchGrid) = g


# ─────────────────────────────────────────────────────────────── #
# Parameters                                                     #
# ─────────────────────────────────────────────────────────────── #
const NCOMP = 3
const REFERENCE_SIZE = (16, 4, 4, 16)

function parse_int_tuple(value)
    return tuple((parse(Int, strip(x)) for x in split(value, ","))...)
end

# Increase every physical dimension by these factors.  Large kH-dependent
# mode tensors are skipped when they exceed MAX_MODE_MIB.
const SCALE_FACTORS = parse_int_tuple(get(ENV, "GALERKIN_SCALE_FACTORS", "1,2,4"))
const Nm_vals = collect(parse_int_tuple(get(ENV, "GALERKIN_NM_VALS", "2,4,8,16,32")))
const MAX_MODE_MIB = parse(Float64, get(ENV, "GALERKIN_MAX_MODE_MIB", "8192"))
const BENCHMARK_SAMPLES = parse(Int, get(ENV, "GALERKIN_BENCHMARK_SAMPLES", "3"))
const BENCHMARK_SECONDS = parse(Float64, get(ENV, "GALERKIN_BENCHMARK_SECONDS", "0.2"))
const OUTPUT_DIR = joinpath(@__DIR__, "output")
const RESULTS_PATH = joinpath(OUTPUT_DIR, "galerkin_scaling_results.tsv")
const FIGURE_PATH = joinpath(OUTPUT_DIR, "galerkin_scaling.png")

scaled_size(scale) = ntuple(i -> REFERENCE_SIZE[i] * scale, Val(4))
rfft_size(Nx) = (Nx >> 1) + 1
homogeneous_size((_, Nx, Nz, Nt)) = rfft_size(Nx) * Nz * Nt

function make_grid(dims)
    Ny = dims[1]
    return BenchGrid(dims, rand(Float64, Ny))
end

# Mode tensor scales as NCOMP * Ny * Nm * NkH complex values.
mode_storage_mib(dims, Nm) =
    NCOMP * dims[1] * Nm * homogeneous_size(dims) * sizeof(ComplexF64) / 2^20

over_mode_budget(dims, Nm) = mode_storage_mib(dims, Nm) > MAX_MODE_MIB

function print_skip(scale, Nm, reason)
    println(rpad("$(scale)x", 8),
            rpad(Nm, 6),
            rpad("skip", 12),
            rpad(reason, 52))
end

function kh_dependent_modes(dims, Nm)
    Ny, Nx, Nz, Nt = dims
    Nx_half = rfft_size(Nx)
    return ntuple(_ -> randn(ComplexF64, Ny, Nm, Nx_half, Nz, Nt), NCOMP)
end



# ─────────────────────────────────────────────────────────────── #
# Hand-rolled layout-specific loop kernels                       #
# ─────────────────────────────────────────────────────────────── #
# These kernels intentionally assume the benchmark storage layout:
#
#   parent(u[n])  :: Array{ComplexF64,4}  # (Ny, Nx_half, Nz, Nt)
#   parent(a)     :: Array{ComplexF64,4}  # (Nm, Nx_half, Nz, Nt)
#   modes[n]      :: Array{ComplexF64,5}  # (Ny, Nm, Nx_half, Nz, Nt)
#
# They provide the performance baseline against which the generic
# ReSolverFlowsBase.LoopGalerkin (NSLOOP) implementation is compared.

function loop_project!(a, u, modes, ws)
    pa = parent(a)
    fill!(pa, 0)

    Nm, Nx_half, Nz, Nt = size(pa)
    Ny = size(modes[1], 1)

    for n in 1:NCOMP
        @inbounds for it in 1:Nt, iz in 1:Nz, ix in 1:Nx_half
            pu = parent(u[n])
            mn = modes[n]
            for j in 1:Ny
                wuj = ws[j] * pu[j, ix, iz, it]
                for m in 1:Nm
                    pa[m, ix, iz, it] += conj(mn[j, m, ix, iz, it]) * wuj
                end
            end
        end
    end
    return a
end

function loop_expand!(u, a, modes)
    pa = parent(a)
    Nm, Nx_half, Nz, Nt = size(pa)
    Ny = size(modes[1], 1)

    @inbounds for n in 1:NCOMP
        pu = parent(u[n])
        mn = modes[n]
        for it in 1:Nt, iz in 1:Nz, ix in 1:Nx_half, j in 1:Ny
            acc = zero(ComplexF64)
            for m in 1:Nm
                acc += mn[j, m, ix, iz, it] * pa[m, ix, iz, it]
            end
            pu[j, ix, iz, it] = acc
        end
    end
    return u
end


# ─────────────────────────────────────────────────────────────── #
# BenchmarkTools helpers                                          #
# ─────────────────────────────────────────────────────────────── #
function benchmark_kernel(thunk)
    # Each call site passes a concrete zero-arg closure capturing its own
    # arguments, e.g. `() -> ReSolverFlowsBase.project!(a, u, ReSolverFlowsBase.LoopGalerkin())`.
    # Avoiding varargs + splatting here keeps Julia's inference happy on the
    # heavily-parametric kernels.
    thunk()  # warmup keeps compilation out of the reported BenchmarkTools trial.
    trial = BenchmarkTools.@benchmark $thunk() evals=1 samples=BENCHMARK_SAMPLES seconds=BENCHMARK_SECONDS
    estimate = BenchmarkTools.median(trial)
    return estimate.time / 1e9, estimate.memory
end

function format_bytes(bytes)
    bytes < 2^10 && return "$(bytes) B"
    bytes < 2^20 && return "$(round(bytes / 2^10, digits=1)) KiB"
    bytes < 2^30 && return "$(round(bytes / 2^20, digits=2)) MiB"
    return "$(round(bytes / 2^30, digits=2)) GiB"
end

function record!(results; scale, dims, op, Nm, alg, time_s, alloc_bytes)
    row = (scale=scale,
           Ny=dims[1],
           Nx=dims[2],
           Nz=dims[3],
           Nt=dims[4],
           NkH=homogeneous_size(dims),
           op=op,
           Nm=Nm,
           alg=alg,
           time_ms=time_s * 1e3,
           alloc_bytes=alloc_bytes)
    push!(results, row)
    return row
end

function print_table_header()
    println(rpad("scale", 8),
            rpad("Nm", 6),
            rpad("LOOP ms", 12),
            rpad("LOOP alloc", 14),
            rpad("NSLOOP ms", 12),
            rpad("NSLOOP alloc", 14),
            rpad("GEMM ms", 12),
            rpad("GEMM alloc", 14),
            rpad("LOOP/NSLOOP", 14),
            rpad("LOOP/GEMM", 12))
    println("-"^126)
end

function print_row(scale, Nm, t_loop, a_loop, t_nsloop, a_nsloop, t_gemm, a_gemm)
    println(rpad("$(scale)x", 8),
            rpad(Nm, 6),
            rpad(round(t_loop   * 1e3, digits=2), 12),
            rpad(format_bytes(a_loop),             14),
            rpad(round(t_nsloop * 1e3, digits=2), 12),
            rpad(format_bytes(a_nsloop),           14),
            rpad(round(t_gemm   * 1e3, digits=2), 12),
            rpad(format_bytes(a_gemm),             14),
            rpad(round(t_loop / t_nsloop, digits=3), 14),
            rpad(round(t_loop / t_gemm,   digits=3), 12))
end

function print_project_header()
    println()
    println("project!  (a[m,k] += sum_j conj(phi[j,m,k]) * w[j] * u[j,k])")
    print_table_header()
end

function print_expand_header()
    println()
    println("expand!  (u[j,k] = sum_m phi[j,m,k] * a[m,k])")
    print_table_header()
end


# ─────────────────────────────────────────────────────────────── #
# Benchmark kernels                                               #
# ─────────────────────────────────────────────────────────────── #
function benchmark_project!(results, scale, dims)
    g = make_grid(dims)

    for Nm in Nm_vals
        if over_mode_budget(dims, Nm)
            reason = "mode tensor $(round(mode_storage_mib(dims, Nm), digits=1)) MiB > $(MAX_MODE_MIB) MiB"
            print_skip(scale, Nm, reason)
            continue
        end

        u_data = ReSolverFlowsBase.VectorField(g)
        u_data .= randn(ComplexF64)

        modes    = kh_dependent_modes(dims, Nm)
        a_loop   = ReSolverFlowsBase.ProjectedField(g, modes)
        a_nsloop = ReSolverFlowsBase.ProjectedField(g, modes)
        a_gemm   = ReSolverFlowsBase.ProjectedField(g, modes)

        # Correctness: all three implementations must agree.
        loop_project!(a_loop, u_data, modes, ReSolverFlowsBase.weights(g))
        ReSolverFlowsBase.project!(a_nsloop, u_data, ReSolverFlowsBase.LoopGalerkin())
        ReSolverFlowsBase.project!(a_gemm,   u_data, ReSolverFlowsBase.GemmGalerkin())
        @assert maximum(abs, parent(a_loop) .- parent(a_nsloop)) < 1e-8 "project! NSLOOP mismatch (scale=$scale, Nm=$Nm)"
        @assert maximum(abs, parent(a_loop) .- parent(a_gemm))   < 1e-8 "project! GEMM mismatch (scale=$scale, Nm=$Nm)"

        ws = ReSolverFlowsBase.weights(g)
        t_loop,   alloc_loop   = benchmark_kernel(() -> loop_project!(a_loop, u_data, modes, ws))
        t_nsloop, alloc_nsloop = benchmark_kernel(() -> ReSolverFlowsBase.project!(a_nsloop, u_data, ReSolverFlowsBase.LoopGalerkin()))
        t_gemm,   alloc_gemm   = benchmark_kernel(() -> ReSolverFlowsBase.project!(a_gemm,   u_data, ReSolverFlowsBase.GemmGalerkin()))

        record!(results; scale, dims, op="project", Nm, alg="LOOP",   time_s=t_loop,   alloc_bytes=alloc_loop)
        record!(results; scale, dims, op="project", Nm, alg="NSLOOP", time_s=t_nsloop, alloc_bytes=alloc_nsloop)
        record!(results; scale, dims, op="project", Nm, alg="GEMM",   time_s=t_gemm,   alloc_bytes=alloc_gemm)
        print_row(scale, Nm, t_loop, alloc_loop, t_nsloop, alloc_nsloop, t_gemm, alloc_gemm)
        GC.gc()
    end
end

function benchmark_expand!(results, scale, dims)
    g = make_grid(dims)

    for Nm in Nm_vals
        if over_mode_budget(dims, Nm)
            reason = "mode tensor $(round(mode_storage_mib(dims, Nm), digits=1)) MiB > $(MAX_MODE_MIB) MiB"
            print_skip(scale, Nm, reason)
            continue
        end

        u_loop   = ReSolverFlowsBase.VectorField(g)
        u_nsloop = ReSolverFlowsBase.VectorField(g)
        u_gemm   = ReSolverFlowsBase.VectorField(g)

        modes    = kh_dependent_modes(dims, Nm)
        a_loop   = ReSolverFlowsBase.ProjectedField(g, modes)
        a_nsloop = ReSolverFlowsBase.ProjectedField(g, modes)
        a_gemm   = ReSolverFlowsBase.ProjectedField(g, modes)

        coeffs = randn(ComplexF64, size(parent(a_loop)))
        parent(a_loop)   .= coeffs
        parent(a_nsloop) .= coeffs
        parent(a_gemm)   .= coeffs

        # Correctness: all three implementations must agree.
        loop_expand!(u_loop, a_loop, modes)
        ReSolverFlowsBase.expand!(u_nsloop, a_nsloop, ReSolverFlowsBase.LoopGalerkin())
        ReSolverFlowsBase.expand!(u_gemm,   a_gemm,   ReSolverFlowsBase.GemmGalerkin())
        for n in 1:NCOMP
            @assert maximum(abs, parent(u_loop[n]) .- parent(u_nsloop[n])) < 1e-8 "expand! NSLOOP mismatch (scale=$scale, Nm=$Nm, component=$n)"
            @assert maximum(abs, parent(u_loop[n]) .- parent(u_gemm[n]))   < 1e-8 "expand! GEMM mismatch (scale=$scale, Nm=$Nm, component=$n)"
        end

        t_loop,   alloc_loop   = benchmark_kernel(() -> loop_expand!(u_loop, a_loop, modes))
        t_nsloop, alloc_nsloop = benchmark_kernel(() -> ReSolverFlowsBase.expand!(u_nsloop, a_nsloop, ReSolverFlowsBase.LoopGalerkin()))
        t_gemm,   alloc_gemm   = benchmark_kernel(() -> ReSolverFlowsBase.expand!(u_gemm,   a_gemm,   ReSolverFlowsBase.GemmGalerkin()))

        record!(results; scale, dims, op="expand", Nm, alg="LOOP",   time_s=t_loop,   alloc_bytes=alloc_loop)
        record!(results; scale, dims, op="expand", Nm, alg="NSLOOP", time_s=t_nsloop, alloc_bytes=alloc_nsloop)
        record!(results; scale, dims, op="expand", Nm, alg="GEMM",   time_s=t_gemm,   alloc_bytes=alloc_gemm)
        print_row(scale, Nm, t_loop, alloc_loop, t_nsloop, alloc_nsloop, t_gemm, alloc_gemm)
        GC.gc()
    end
end


# ─────────────────────────────────────────────────────────────── #
# Output                                                          #
# ─────────────────────────────────────────────────────────────── #
function write_results(path, results)
    mkpath(dirname(path))
    columns = (:scale, :Ny, :Nx, :Nz, :Nt, :NkH, :op, :Nm, :alg, :time_ms, :alloc_bytes)
    open(path, "w") do io
        println(io, join(string.(columns), '\t'))
        for row in results
            println(io, join((string(getfield(row, col)) for col in columns), '\t'))
        end
    end
end

function plot_metric!(ax, results, op, metric)
    markers = Dict("LOOP" => "^", "NSLOOP" => "v", "GEMM" => "s")
    styles  = Dict("LOOP" => "-.", "NSLOOP" => ":", "GEMM" => "-")
    color_values = length(Nm_vals) == 1 ? (0.72,) : range(0.45, 0.9, length=length(Nm_vals))
    color_maps = Dict("LOOP"   => PyPlot.cm.Greens,
                      "NSLOOP" => PyPlot.cm.Oranges,
                      "GEMM"   => PyPlot.cm.Blues)

    for (i, Nm) in enumerate(Nm_vals)
        for alg in ("LOOP", "NSLOOP", "GEMM")
            rows = sort([row for row in results
                         if row.op == op && row.Nm == Nm && row.alg == alg],
                        by = row -> row.scale)
            isempty(rows) && continue
            x = [row.scale for row in rows]
            y = metric == :time_ms ? [row.time_ms for row in rows] :
                metric == :alloc_bytes ? [row.alloc_bytes for row in rows] :
                error("unsupported metric $metric")
            ax.plot(x, y;
                    color=color_maps[alg](color_values[i]),
                    linestyle=styles[alg],
                    marker=markers[alg],
                    linewidth=1.5,
                    markersize=4,
                    label="$(alg), Nm=$(Nm)")
        end
    end
    ax.set_yscale("log")
    ax.set_xticks(collect(SCALE_FACTORS))
    ax.set_xticklabels(["$(scale)x" for scale in SCALE_FACTORS])
    ax.grid(true; which="both", alpha=0.25)
end

function plot_speedup!(ax, results, op)
    markers = Dict("NSLOOP" => "v", "GEMM" => "s")
    styles  = Dict("NSLOOP" => ":", "GEMM" => "-")
    color_values = length(Nm_vals) == 1 ? (0.72,) : range(0.45, 0.9, length=length(Nm_vals))
    color_maps = Dict("NSLOOP" => PyPlot.cm.Oranges,
                      "GEMM"   => PyPlot.cm.Blues)

    for (i, Nm) in enumerate(Nm_vals)
        loop_rows = Dict(row.scale => row for row in results
                         if row.op == op && row.Nm == Nm && row.alg == "LOOP")

        for alg in ("NSLOOP", "GEMM")
            rows = sort([row for row in results
                         if row.op == op && row.Nm == Nm && row.alg == alg && haskey(loop_rows, row.scale)],
                        by = row -> row.scale)
            isempty(rows) && continue

            x = [row.scale for row in rows]
            y = [loop_rows[row.scale].time_ms / row.time_ms for row in rows]
            ax.plot(x, y;
                    color=color_maps[alg](color_values[i]),
                    linestyle=styles[alg],
                    marker=markers[alg],
                    linewidth=1.5,
                    markersize=4,
                    label="$(alg), Nm=$(Nm)")
        end
    end

    ax.axhline(1.0; color="0.35", linewidth=0.9, linestyle=":")
    ax.set_yscale("log")
    ax.set_xticks(collect(SCALE_FACTORS))
    ax.set_xticklabels(["$(scale)x" for scale in SCALE_FACTORS])
    ax.grid(true; which="both", alpha=0.25)
end

function make_figure(path, results)
    mkpath(dirname(path))
    PyPlot.ioff()
    fig, axes = PyPlot.subplots(3, 2; figsize=(10, 11), sharex=true)
    panels = ("project", "expand")

    for (col, op) in enumerate(panels)
        ax_time    = axes[1, col]
        ax_speedup = axes[2, col]
        ax_alloc   = axes[3, col]

        plot_metric!(ax_time,  results, op, :time_ms)
        plot_speedup!(ax_speedup, results, op)
        plot_metric!(ax_alloc, results, op, :alloc_bytes)

        ax_time.set_title(op)
        col == 1 && ax_time.set_ylabel("time (ms)")
        col == 1 && ax_speedup.set_ylabel("speedup vs hand-rolled LOOP")
        col == 1 && ax_alloc.set_ylabel("allocation (bytes)")
        ax_alloc.set_xlabel("grid scale")
    end

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.suptitle("Galerkin algorithm scaling from reference size $(REFERENCE_SIZE)", y=0.98)
    fig.legend(handles, labels;
               loc="lower center",
               bbox_to_anchor=(0.5, 0.01),
               ncol=5,
               fontsize=7)
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    fig.savefig(path; dpi=200)
    PyPlot.close(fig)
end


# ─────────────────────────────────────────────────────────────── #
# Run                                                             #
# ─────────────────────────────────────────────────────────────── #
# Wrapped in `main()` so the inner benchmark calls run in a typed
# function context — top-level loops trip Julia 1.12's inference of
# heavily-parametric @generated kernels and emit a noisy (harmless)
# internal-error trace.
function main()
    results = NamedTuple[]

    println("reference size: Ny=$(REFERENCE_SIZE[1])  Nx=$(REFERENCE_SIZE[2])  Nz=$(REFERENCE_SIZE[3])  Nt=$(REFERENCE_SIZE[4])")
    println("scale factors: $(join((string(scale) * "x" for scale in SCALE_FACTORS), ", "))")
    println("Nm values: $(join(Nm_vals, ", "))")
    println("mode tensor budget: $(MAX_MODE_MIB) MiB")
    println("BenchmarkTools budget: samples=$(BENCHMARK_SAMPLES), seconds=$(BENCHMARK_SECONDS)")

    for scale in SCALE_FACTORS
        dims   = scaled_size(scale)
        NkH    = homogeneous_size(dims)
        max_Nm = maximum(Nm_vals)

        println()
        println("size $(scale)x: Ny=$(dims[1])  Nx=$(dims[2])  Nz=$(dims[3])  Nt=$(dims[4])  NkH=$NkH")
        println("largest mode tensor: $(round(mode_storage_mib(dims, max_Nm), digits=1)) MiB")

        print_project_header()
        benchmark_project!(results, scale, dims)

        print_expand_header()
        benchmark_expand!(results, scale, dims)
    end

    write_results(RESULTS_PATH, results)
    make_figure(FIGURE_PATH, results)

    println()
    println("wrote results: $RESULTS_PATH")
    println("wrote figure:  $FIGURE_PATH")
end

main()
