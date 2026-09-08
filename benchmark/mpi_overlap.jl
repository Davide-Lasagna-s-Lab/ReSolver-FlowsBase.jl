# Benchmark: halo communication / computation overlap strategies for the 3D
# nonlinear, forward-linearised, and discrete-adjoint operators on a y-decomposed
# mock channel grid.
#
# Four strategies (see issue #21):
#   1) :noov   — no overlap: post the halo swap, wait immediately, then compute
#                everything (interior + boundary + FFT) once the halo is present.
#   2) :cur    — current branch code: interior derivs + FFT, THEN wait, THEN boundary.
#   3) :test   — :cur plus MPI progress polling (MPI.Iprobe) in the overlap window.
#   4) :async  — :cur code, but launched with the MPI library's async-progress
#                thread enabled and MPI initialised at THREAD_MULTIPLE.
#
# Strategies 1–3 are timed in a normal launch; strategy 4 is the SAME code as
# :cur but run in a second launch with async progress enabled (BENCH_PROGRESS=async
# + MPICH_ASYNC_PROGRESS=1). Timing is wall-clock per RHS, reduced as the MAX over
# ranks (the slowest rank sets the wall time), median of repeated barriered runs.
#
# Run (from the package root, test project has MPI/HaloArrays/FDGrids):
#   N=4
#   # strategies 1–3:
#   julia --project=test -e 'using MPI; run(`$(mpiexec()) -n '$N' '$(Base.julia_cmd())' --project=test benchmarks/mpi_overlap.jl`)'
#   # strategy 4 (async progress):
#   MPICH_ASYNC_PROGRESS=1 BENCH_PROGRESS=async julia --project=test -e 'using MPI; run(`$(mpiexec()) -n '$N' '$(Base.julia_cmd())' --project=test benchmarks/mpi_overlap.jl`)'
#
# A driver at the bottom of this file runs both launches for you when invoked as
#   julia --project=test benchmarks/mpi_overlap.jl --driver N

throw(error("This script is broken - needs updating for newer interface"))

import FDGrids, HaloArrays, MPI, ReSolverFlowsBase, Printf

# ----- driver: spawn both mpiexec launches, then exit ----------------------- #
if !isempty(ARGS) && ARGS[1] == "--driver"
    nranks = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
    jl     = Base.julia_cmd()
    proj   = "--project=" * dirname(Base.active_project())
    self   = @__FILE__
    mpiexec = MPI.mpiexec()
    println("\n========== strategies 1–3 (normal launch, $nranks ranks) ==========")
    run(`$mpiexec -n $nranks $jl $proj --startup-file=no $self`)
    println("\n========== strategy 4 (async progress, $nranks ranks) ==========")
    run(addenv(`$mpiexec -n $nranks $jl $proj --startup-file=no $self`,
               "BENCH_PROGRESS" => "async", "MPICH_ASYNC_PROGRESS" => "1"))
    exit(0)
end

using ReSolverFlowsBase: init_requests!, wait_requests!,
               init_ddx!, init_ddy!, init_ddz!,
               complete_ddx!, complete_ddy!, complete_ddz!,
               init_laplacian!, complete_laplacian!,
               FTField, Field, VectorField, FFTPlans,
               CartesianPrimitive3DNSE, CartesianPrimitive3DLNSE,
               Forward, AdjointDiscrete, FFTW

const ASYNC = get(ENV, "BENCH_PROGRESS", "") == "async"
MPI.Initialized() || MPI.Init(threadlevel = ASYNC ? :multiple : :serialized)

include(joinpath(@__DIR__, "..", "test", "ext", "MPIExt", "grid.jl"))   # MockChannelGrid + MPIExt

const COMM = MPI.COMM_WORLD
const NR   = MPI.Comm_size(COMM)
const RANK = MPI.Comm_rank(COMM)

# Drive the MPI progress engine without touching our halo requests.
_poke!() = (MPI.Iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG, COMM); nothing)

# ------------------------------------------------------------------ #
# Strategy-parameterised RHS (Val{S}: :noov | :cur | :test | :async) #
# ------------------------------------------------------------------ #
# :async uses the :cur ordering (the difference is the launch environment).
@inline _is_noov(::Val{:noov}) = true
@inline _is_noov(::Val)        = false
@inline _is_test(::Val{:test}) = true
@inline _is_test(::Val)        = false

function nse!(eq, u, out, vs::Val)
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    req = init_requests!(u)
    _is_noov(vs) && wait_requests!(req)
    init_laplacian!(out, u); init_ddx!(dudx, u); init_ddy!(dudy, u); init_ddz!(dudz, u)
    _is_test(vs) && _poke!()
    eq.plans(U, u)
    _is_test(vs) && _poke!()
    _is_noov(vs) || wait_requests!(req)
    complete_laplacian!(out, u); complete_ddx!(dudx, u); complete_ddy!(dudy, u); complete_ddz!(dudz, u)
    out .*= 1/eq.Re
    eq.plans(dUdx, dudx); eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)
    for n in 1:3
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n] - U[3]*dUdz[n]
    end
    eq.plans(out, dUdx, add=true)
    return out
end

function fwd!(eq, v, out, vs::Val)
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]; dvdz = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; dVdx = eq.pcache[6]; dVdy = eq.pcache[7]; dVdz = eq.pcache[8]
    req = init_requests!(v)
    _is_noov(vs) && wait_requests!(req)
    init_laplacian!(out, v); init_ddx!(dvdx, v); init_ddy!(dvdy, v); init_ddz!(dvdz, v)
    _is_test(vs) && _poke!()
    eq.plans(V, v)
    _is_test(vs) && _poke!()
    _is_noov(vs) || wait_requests!(req)
    complete_laplacian!(out, v); complete_ddx!(dvdx, v); complete_ddy!(dvdy, v); complete_ddz!(dvdz, v)
    out .*= 1/eq.Re
    eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy); eq.plans(dVdz, dvdz)
    for n in 1:3
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n] - U[3]*dVdz[n]
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n] + V[3]*dUdz[n]
    end
    eq.plans(out, dVdx, add=true)
    return out
end

function adj!(eq, v, out, vs::Val)
    dudx = eq.scache[1]; u1v = eq.scache[2]; u2v = eq.scache[3]; u3v = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; U1V = eq.pcache[6]; U2V = eq.pcache[7]; U3V = eq.pcache[8]
    req = init_requests!(v)
    _is_noov(vs) && wait_requests!(req)
    init_laplacian!(out, v; adjoint=true)
    _is_test(vs) && _poke!()
    eq.plans(V, v)
    _is_test(vs) && _poke!()
    _is_noov(vs) || wait_requests!(req)
    complete_laplacian!(out, v; adjoint=true)
    out .*= 1/eq.Re
    for n in 1:3
        @. U1V[n] = U[1]*V[n]; @. U2V[n] = U[2]*V[n]; @. U3V[n] = U[3]*V[n]
    end
    eq.plans(u1v, U1V); eq.plans(u2v, U2V); eq.plans(u3v, U3V)
    eq.plans(dUdx, dudx)
    for n in 1:3
        r1 = init_requests!(u1v[n]); r2 = init_requests!(u2v[n]); r3 = init_requests!(u3v[n])
        if _is_noov(vs)
            wait_requests!(r1); wait_requests!(r2); wait_requests!(r3)
        end
        init_ddx!(dudx[1], u1v[n]; adjoint=true)
        init_ddy!(dudx[2], u2v[n]; adjoint=true)
        init_ddz!(dudx[3], u3v[n]; adjoint=true)
        _is_test(vs) && _poke!()
        if !_is_noov(vs)
            wait_requests!(r1); wait_requests!(r2); wait_requests!(r3)
        end
        complete_ddx!(dudx[1], u1v[n]; adjoint=true)
        complete_ddy!(dudx[2], u2v[n]; adjoint=true)
        complete_ddz!(dudx[3], u3v[n]; adjoint=true)
        out[n] .-= dudx[1] .+ dudx[2] .+ dudx[3]
    end
    U1V .= 0
    for n in 1:3
        @. U1V[1] -= V[n]*dUdx[n]; @. U1V[2] -= V[n]*dUdy[n]; @. U1V[3] -= V[n]*dUdz[n]
    end
    eq.plans(out, U1V, add=true)
    return out
end

# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
vfield(g, fs) = (up = VectorField([Field(g, f) for f in fs]...);
                 u  = VectorField([FTField(g) for _ in 1:length(fs)]...);
                 return (up, u))

# wall time per call, median over `nrep` barriered blocks, reduced MAX over ranks
function timeit(f; nrep=5, niter=5, nwarm=2)
    for _ in 1:nwarm; f(); end
    ts = Float64[]
    for _ in 1:nrep
        MPI.Barrier(COMM); t0 = MPI.Wtime()
        for _ in 1:niter; f(); end
        MPI.Barrier(COMM)
        push!(ts, (MPI.Wtime() - t0) / niter)
    end
    local_med = sort(ts)[(nrep+1) ÷ 2]
    return MPI.Allreduce(local_med, MPI.MAX, COMM)
end

f1(y,x,z,t) = (1-y^2) * cos(2π*x) * cos(5.8z) * cos(t)
f2(y,x,z,t) = (1-y^2) * sin(2π*x) * cos(5.8z) * sin(t)
f3(y,x,z,t) = (1-y^2)^2 * cos(2π*x) * sin(5.8z) * cos(t)

# async run reuses the :cur ordering (only the launch environment differs)
const STRATS = ASYNC ? (:cur,) : (:noov, :cur, :test)
const CASES  = [(15, 32, 15, 9), (15, 64, 15, 9), (21, 96, 21, 9)]

RANK == 0 && Printf.@printf("\n# MPI overlap benchmark — %d ranks, y-decomposed, async=%s\n",
                            NR, ASYNC)
RANK == 0 && Printf.@printf("%-22s %-8s %12s %12s %12s\n",
                            "case (Nx,Ny,Nz,Nt)", "op", "noov [ms]", "cur [ms]", "test [ms]")
RANK == 0 && println(repeat("-", 74))

for (Nx, Ny, Nz, Nt) in CASES
    gp = MockChannelGrid(Ny, Nx, Nz, Nt; stencil_width=5, α=2π, β=5.8)
    g  = MPIExt.distributed(gp, MPI.Comm_dup(COMM);
                            decomposed_physical_dims=(:y,), nprocesses=(NR,), nhalo=(2,))
    plans = FFTPlans(g; dealias=false, flags=FFTW.ESTIMATE)   # init only; ops keep dealiased plans

    up, u = vfield(g, (f1, f2, f3)); plans(u, up)
    bp, base = vfield(g, (f1, f2, f3)); plans(base, bp)
    out = VectorField([FTField(g) for _ in 1:3]...)

    # build, set up, and time one operator at a time, freeing it before the next
    # (each LNSE/NSE holds 8 dealiased physical caches; keeping all three alive
    # blows up memory at large sizes).
    results = Dict{Tuple{String,Symbol},Float64}()
    let nse = CartesianPrimitive3DNSE(g, 1000.0; flags=FFTW.ESTIMATE)
        for s in STRATS; results[("nse", s)] = timeit(() -> nse!(nse, u, out, Val(s))); end
    end
    GC.gc(); MPI.Barrier(COMM)
    let lf = CartesianPrimitive3DLNSE(g, 1000.0; mode=Forward(), flags=FFTW.ESTIMATE)
        lf(0.0, base, u, out)                            # cache base flow (3-arg)
        for s in STRATS; results[("fwd", s)] = timeit(() -> fwd!(lf, u, out, Val(s))); end
    end
    GC.gc(); MPI.Barrier(COMM)
    let la = CartesianPrimitive3DLNSE(g, 1000.0; mode=AdjointDiscrete(), flags=FFTW.ESTIMATE)
        la(0.0, base, u, out)
        for s in STRATS; results[("adj", s)] = timeit(() -> adj!(la, u, out, Val(s))); end
    end
    GC.gc(); MPI.Barrier(COMM)

    if RANK == 0
        cs = Printf.@sprintf("(%d,%d,%d,%d)", Nx, Ny, Nz, Nt)
        for op in ("nse", "fwd", "adj")
            g3(s) = haskey(results, (op, s)) ? 1e3 * results[(op, s)] : NaN
            Printf.@printf("%-22s %-8s %12.3f %12.3f %12.3f\n",
                           op == "nse" ? cs : "", op, g3(:noov), g3(:cur), g3(:test))
        end
    end
end

if RANK == 0 && ASYNC
    println("\n(async run: the single column is the :cur ordering under MPI async progress = strategy 4)")
end
MPI.Finalize()
