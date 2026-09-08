using BenchmarkTools
using PyPlot
using Printf
using ReSolverFlowsBase

# ------------------------------------------------------------------ #
# cases: (dimensionality, order of transformed dims, sizes to sweep)  #
# ------------------------------------------------------------------ #
cases = [
    (D=1, order=(1,),     sizes=[16, 32, 64, 128, 256]),
    (D=2, order=(1, 2),   sizes=[16, 32, 64, 128]),
    (D=3, order=(1, 2, 3),sizes=[16, 32, 64]),
]

# ------------------------------------------------------------------ #
# benchmark loop                                                       #
# ------------------------------------------------------------------ #
results = [
    [(ns=Float64[], kb=Float64[]) for _ in 1:4]
    for _ in eachindex(cases)
]

for (ci, c) in enumerate(cases)
    println("\n── D=$(c.D), order=$(c.order) ──────────────────────────────")
    @printf("  %-8s  %-20s  %-20s  %-20s  %-20s\n",
            "n", "copy_to_padded!", "copy_from_padded!", "add_from_padded!", "apply_mask!")

    for n in c.sizes
        sz      = ntuple(_ -> n, c.D)
        pad_sz  = ReSolverFlowsBase.get_padded_size(sz, c.order)
        spec_sz = ReSolverFlowsBase._get_transform_size(sz,     c.order[1])
        cach_sz = ReSolverFlowsBase._get_transform_size(pad_sz, c.order[1])

        u     = rand(ComplexF64, spec_sz)
        cache = rand(ComplexF64, cach_sz)

        b1 = @benchmark ReSolverFlowsBase._copy_to_padded!(  buf, $u,  $c.order) setup=(buf=copy($cache)) evals=1
        b2 = @benchmark ReSolverFlowsBase._copy_from_padded!($u,  buf, $c.order) setup=(buf=copy($cache)) evals=1
        b3 = @benchmark ReSolverFlowsBase._add_from_padded!( $u,  buf, $c.order) setup=(buf=copy($cache)) evals=1
        b4 = @benchmark ReSolverFlowsBase._apply_mask!(buf)                             setup=(buf=copy($cache)) evals=1

        for (fi, b) in enumerate((b1, b2, b3, b4))
            push!(results[ci][fi].ns, median(b).time)
            push!(results[ci][fi].kb, median(b).memory / 1024)
        end

        @printf("  n=%-4d  %7.3f ms / %5.1f B  %7.3f ms / %5.1f B  %7.3f ms / %5.1f B  %7.3f ms / %5.1f B\n",
                n,
                median(b1).time/1e6, median(b1).memory,
                median(b2).time/1e6, median(b2).memory,
                median(b3).time/1e6, median(b3).memory,
                median(b4).time/1e6, median(b4).memory)
    end
end

# ------------------------------------------------------------------ #
# figure: 3 columns (D=1,2,3), 2 rows (time, allocations)            #
# ------------------------------------------------------------------ #
labels = ["copy_to_padded!", "copy_from_padded!", "add_from_padded!", "apply_mask!"]
colors = ["steelblue", "darkorange", "green", "crimson"]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))

for (ci, c) in enumerate(cases)
    ax_t = axes[1, ci]
    ax_a = axes[2, ci]

    for (fi, (lbl, col)) in enumerate(zip(labels, colors))
        ax_t.plot(c.sizes, results[ci][fi].ns ./ 1e6, "o-", color=col,
                  label=lbl, linewidth=1.5, markersize=5)
        ax_a.plot(c.sizes, results[ci][fi].kb, "o-", color=col,
                  label=lbl, linewidth=1.5, markersize=5)
    end

    for ax in (ax_t, ax_a)
        ax.set_xscale("log", base=2)
        # ax.set_yscale("log")
        ax.set_xticks(c.sizes)
        ax.xaxis.set_major_formatter(PyPlot.matplotlib.ticker.ScalarFormatter())
        ax.grid(true, which="both", linestyle="--", alpha=0.4)
        ax.set_xlabel("n  (side length)")
        ax.set_title("D=$(c.D), order=$(c.order)")
    end
    ax_t.set_yscale("log")
    ax_t.set_ylabel("median time  (ms)")
    ax_a.set_ylabel("allocations  (KB)")
    ax_t.legend(fontsize=7)
end

fig.suptitle("_copy_to/from_padded!, _add_from_padded!, _apply_mask! — performance", y=1.01)
fig.tight_layout()

outpath = joinpath(@__DIR__, "fft_copy_padded.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
println("\nFigure saved to $outpath")
