# Comprehensive benchmark comparing dev vs CartesianIndices implementations.
#
# Usage:
#   julia --project=benchmark benchmark/compare_branches.jl --save dev_results.json
#   julia --project=benchmark benchmark/compare_branches.jl --save cart_results.json
#   julia --project=benchmark benchmark/compare_branches.jl --compare dev_results.json cart_results.json

using BenchmarkTools
using LinearAlgebra
using Printf
using ReSolverFlowsBase
using JSON

include("../test/test_grids.jl")

# ------------------------------------------------------------------ #
# Grid layouts (inhomogeneous dimension at every possible position)  #
# ------------------------------------------------------------------ #
struct Layout1 <: AbstractGrid{Float64, 4, (2, 1, 3, 4), (2, 3, 4)}
    Ny::Int; Nx::Int; Nz::Int; Nt::Int; ws::Vector{Float64}
end
struct Layout2 <: AbstractGrid{Float64, 4, (1, 2, 3, 4), (1, 3, 4)}
    Nx::Int; Ny::Int; Nz::Int; Nt::Int; ws::Vector{Float64}
end
struct Layout3 <: AbstractGrid{Float64, 4, (1, 3, 2, 4), (1, 2, 4)}
    Nx::Int; Nz::Int; Ny::Int; Nt::Int; ws::Vector{Float64}
end
struct Layout4 <: AbstractGrid{Float64, 4, (1, 4, 2, 3), (1, 2, 3)}
    Nx::Int; Nz::Int; Nt::Int; Ny::Int; ws::Vector{Float64}
end

Base.size(g::Layout1) = (g.Ny, g.Nx, g.Nz, g.Nt)
Base.size(g::Layout2) = (g.Nx, g.Ny, g.Nz, g.Nt)
Base.size(g::Layout3) = (g.Nx, g.Nz, g.Ny, g.Nt)
Base.size(g::Layout4) = (g.Nx, g.Nz, g.Nt, g.Ny)

ReSolverFlowsBase.weights(g::Union{Layout1,Layout2,Layout3,Layout4}) = g.ws
ReSolverFlowsBase.wavenumber_scale(::Union{Layout1,Layout2,Layout3,Layout4}, ::Int) = 1.0

make_grid(::Type{Layout1}, Ny, Nx, Nz, Nt) = Layout1(Ny, Nx, Nz, Nt, ones(Ny))
make_grid(::Type{Layout2}, Ny, Nx, Nz, Nt) = Layout2(Nx, Ny, Nz, Nt, ones(Ny))
make_grid(::Type{Layout3}, Ny, Nx, Nz, Nt) = Layout3(Nx, Nz, Ny, Nt, ones(Ny))
make_grid(::Type{Layout4}, Ny, Nx, Nz, Nt) = Layout4(Nx, Nz, Nt, Ny, ones(Ny))

LAYOUTS       = [Layout1, Layout2, Layout3, Layout4]
LAYOUT_LABELS = ["(y,x,z,t)", "(x,y,z,t)", "(x,z,y,t)", "(x,z,t,y)"]

# ------------------------------------------------------------------ #
# Grid sizes (320 → ~50 M spectral elements)                         #
# ------------------------------------------------------------------ #
sizes = [
    (Ny=4,  Nx=6,   Nz=5,   Nt=4),    #    ~320
    (Ny=6,  Nx=10,  Nz=8,   Nt=6),    #   ~1.7k
    (Ny=8,  Nx=16,  Nz=12,  Nt=8),    #   ~6.9k
    (Ny=12, Nx=24,  Nz=16,  Nt=12),   #  ~30.0k
    (Ny=16, Nx=32,  Nz=24,  Nt=16),   # ~104.4k
    (Ny=20, Nx=48,  Nz=36,  Nt=24),   # ~432.0k
    (Ny=24, Nx=64,  Nz=48,  Nt=32),   #   ~1.2M
]

n_elements(Ny, Nx, Nz, Nt) = Ny * ((Nx >> 1) + 1) * Nz * Nt

SHIFTS  = (0.1, 0.2, 0.15)
NM_VALS = [5, 20, 50]   # mode counts for Galerkin benchmarks
Nv      = 3             # velocity components

# ------------------------------------------------------------------ #
# Function list (one entry per subplot)                               #
# ------------------------------------------------------------------ #
const FUNCTION_NAMES = [
    "ddx!_x  (rfft)",
    "ddx!_z  (sFFT1)",
    "ddx!_t  (sFFT2)",
    "dot_FTField",
    "normdiff_FTField",
    "shift!_FTField",
    "project!_Loop  Nm=5",
    "project!_Loop  Nm=20",
    "project!_Loop  Nm=50",
    "project!_Gemm  Nm=5",
    "project!_Gemm  Nm=20",
    "project!_Gemm  Nm=50",
    "expand!_Loop   Nm=5",
    "expand!_Loop   Nm=20",
    "expand!_Loop   Nm=50",
    "expand!_Gemm   Nm=5",
    "expand!_Gemm   Nm=20",
    "expand!_Gemm   Nm=50",
    "dot_ProjectedField",
    "normdiff_ProjectedField",
    "shift!_ProjectedField",
]

# ------------------------------------------------------------------ #
# Helper: build mode arrays for a given Nm                           #
# ------------------------------------------------------------------ #
function make_modes(Nm, Ny, rfft_sz, Nz, Nt, on_dev)
    if on_dev
        ntuple(_ -> randn(ComplexF64, Ny, Nm, rfft_sz, Nz, Nt), Nv)
    else
        ntuple(_ -> randn(ComplexF64, Nm, Ny, rfft_sz, Nz, Nt), Nv)
    end
end

# ------------------------------------------------------------------ #
# Main benchmark                                                      #
# ------------------------------------------------------------------ #
function run_benchmark()
    println("Benchmarking actual implementations...")
    println()

    results = Dict{String, Any}(
        "layouts"   => LAYOUT_LABELS,
        "functions" => FUNCTION_NAMES,
        "sizes"     => [string(n_elements(s.Ny, s.Nx, s.Nz, s.Nt)) for s in sizes],
        "timings"   => Dict{String, Any}(),
    )

    for (li, LT) in enumerate(LAYOUTS)
        println("Layout $li/$(length(LAYOUTS)): $(LAYOUT_LABELS[li])")

        for s in sizes
            Ny, Nx, Nz, Nt = s.Ny, s.Nx, s.Nz, s.Nt
            ne           = n_elements(Ny, Nx, Nz, Nt)
            g            = make_grid(LT, Ny, Nx, Nz, Nt)
            rfft_sz      = (Nx >> 1) + 1

            @printf("  Ny=%2d Nx=%3d Nz=%3d Nt=%3d (%8d elems)\n", Ny, Nx, Nz, Nt, ne)

            # FTField fixtures
            u   = FTField(g); parent(u) .= randn(ComplexF64, size(parent(u)))
            v   = FTField(g); parent(v) .= randn(ComplexF64, size(parent(v)))
            out = FTField(g)
            q   = VectorField(ntuple(_ -> FTField(g), Nv)...)
            for n in 1:Nv; parent(q[n]) .= randn(ComplexF64, size(parent(u))); end
            out_q = VectorField(ntuple(_ -> zero(FTField(g)), Nv)...)

            # Detect branch: probe with dim1=5, dim2=3.
            # dev reads Nm from axis Ninh+1=2 → pa first dim = 3.
            # current branch reads from axis 1  → pa first dim = 5.
            _probe = ntuple(_ -> zeros(ComplexF64, 5, 3, rfft_sz, Nz, Nt), 1)
            on_dev = size(parent(ProjectedField(g, _probe)), 1) == 3

            # ProjectedField fixtures for each Nm value
            pfields = map(NM_VALS) do Nm
                ms  = make_modes(Nm, Ny, rfft_sz, Nz, Nt, on_dev)
                a   = ProjectedField(g, ms)
                pa2 = ProjectedField(g, ms); parent(pa2) .= randn(ComplexF64, size(parent(pa2)))
                oq  = VectorField(ntuple(_ -> zero(FTField(g)), Nv)...)
                (a=a, pa2=pa2, out_q=oq)
            end
            a5,  pa25, oq5  = pfields[1].a, pfields[1].pa2, pfields[1].out_q
            a20,       oq20 = pfields[2].a,                 pfields[2].out_q
            a50,       oq50 = pfields[3].a,                 pfields[3].out_q

            size_label   = "$(ne)_elems"
            layout_label = LAYOUT_LABELS[li]

            fns = [
                # 1-3: spectral derivatives
                (() -> ddx!(out, u)),
                (() -> ddz!(out, u)),
                (() -> ddt!(out, u)),
                # 4-6: FTField inner products and shift
                (() -> dot(u, v)),
                (() -> normdiff(u, v)),
                (() -> (uc = copy(u); shift!(uc, SHIFTS))),
                # 7-9: project! LoopGalerkin, varying Nm
                (() -> project!(a5,  q, LoopGalerkin())),
                (() -> project!(a20, q, LoopGalerkin())),
                (() -> project!(a50, q, LoopGalerkin())),
                # 10-12: project! GemmGalerkin, varying Nm
                (() -> project!(a5,  q, GemmGalerkin())),
                (() -> project!(a20, q, GemmGalerkin())),
                (() -> project!(a50, q, GemmGalerkin())),
                # 13-15: expand! LoopGalerkin, varying Nm
                (() -> expand!(oq5,  a5,  LoopGalerkin())),
                (() -> expand!(oq20, a20, LoopGalerkin())),
                (() -> expand!(oq50, a50, LoopGalerkin())),
                # 16-18: expand! GemmGalerkin, varying Nm
                (() -> expand!(oq5,  a5,  GemmGalerkin())),
                (() -> expand!(oq20, a20, GemmGalerkin())),
                (() -> expand!(oq50, a50, GemmGalerkin())),
                # 19-21: ProjectedField inner products and shift (Nm=5)
                (() -> dot(a5, pa25)),
                (() -> normdiff(a5, pa25)),
                (() -> (ac = copy(a5); shift!(ac, SHIFTS))),
            ]

            for (fi, f) in enumerate(fns)
                fname = FUNCTION_NAMES[fi]
                key   = "$(fname)_$(layout_label)_$(size_label)"

                for _ in 1:3; f(); end   # warm up

                # Fewer samples for large Nm or large grids
                Nm_fi = fi in 7:18 ? NM_VALS[((fi - 7) % 3) + 1] : 5
                nsamples = if ne < 10_000
                    300
                elseif ne < 100_000
                    150
                elseif ne < 1_000_000
                    Nm_fi >= 50 ? 40 : 75
                elseif ne < 10_000_000
                    Nm_fi >= 50 ? 20 : 40
                else
                    Nm_fi >= 50 ? 10 : 20
                end

                params = BenchmarkTools.Parameters(samples=nsamples, evals=1)
                bmark  = @benchmarkable $f() seconds=600 gcsample=true
                trial  = BenchmarkTools.run(bmark, params)

                results["timings"][key] = Dict(
                    "function" => fname,
                    "layout"   => layout_label,
                    "nelems"   => ne,
                    "time_ns"  => minimum(trial.times),
                    "allocs"   => trial.allocs,
                    "memory"   => trial.memory,
                )
            end
        end
        println()
    end

    return results
end

function main()
    args = ARGS

    if length(args) == 0
        println("Usage:")
        println("  julia compare_branches.jl --save <output.json>")
        println("  julia compare_branches.jl --compare <dev.json> <current.json>")
        return
    end

    if args[1] == "--save"
        length(args) < 2 && error("--save requires output filename")
        results = run_benchmark()
        results["branch"] = get(ENV, "BRANCH", "unknown")
        open(args[2], "w") do f; JSON.print(f, results, 2); end
        println("\nResults saved to $(args[2])")

    elseif args[1] == "--compare"
        length(args) < 3 && error("--compare requires two JSON files")
        dev_data  = JSON.parsefile(args[2])
        cart_data = JSON.parsefile(args[3])

        for fname in dev_data["functions"]
            println("\nFunction: $fname")
            println("─" ^ 70)
            @printf "%20s %15s %15s %10s\n" "Layout" "Dev (μs)" "Cart (μs)" "Ratio"
            println("─" ^ 70)
            for layout in dev_data["layouts"]
                times = Float64[]
                for size_str in dev_data["sizes"]
                    dk = "$(fname)_$(layout)_$(size_str)_elems"
                    ck = "$(fname)_$(layout)_$(size_str)_elems"
                    haskey(dev_data["timings"], dk) && haskey(cart_data["timings"], ck) || continue
                    dt = dev_data["timings"][dk]["time_ns"] / 1000
                    ct = cart_data["timings"][ck]["time_ns"] / 1000
                    dt > 0 && ct > 0 && push!(times, dt / ct)
                end
                isempty(times) && continue
                avg = mean(times)
                lk_d = "$(fname)_$(layout)_$(dev_data["sizes"][end])_elems"
                lk_c = "$(fname)_$(layout)_$(cart_data["sizes"][end])_elems"
                dt_l = dev_data["timings"][lk_d]["time_ns"] / 1000
                ct_l = cart_data["timings"][lk_c]["time_ns"] / 1000
                ratio_str = avg >= 1 ? "$(round(avg; digits=2))× faster" : "$(round(1/avg; digits=2))× slower"
                @printf "%20s %15.3f %15.3f  %s\n" layout dt_l ct_l ratio_str
            end
        end
    end
end

main()
