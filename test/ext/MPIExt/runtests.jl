# Orchestrator for the MPIExt test suite.
#
# Each `test_<source_file>.jl` is a self-contained MPI program that
# exercises one part of the ReSolverFlowsBase MPI extension. The runner spawns
# every test script via `mpiexec` with the requested number of ranks and
# checks the subprocess exit status.
#
# Most tests run with both 1 rank (smoke / interface) and 4 ranks
# (multi-rank halo exchange + correctness). Adjust the `nprocs` entries
# below to widen or narrow the matrix.

# Each entry: (test file, descriptive name, number of MPI ranks).
const TEST_FILES_MPI = [
    ("ext/MPIExt/test_abstractgrid.jl",   "DecomposedGrid interface (single rank)",  1),
    ("ext/MPIExt/test_abstractgrid.jl",   "DecomposedGrid interface (4 ranks)",      4),
    ("ext/MPIExt/test_haloarrays.jl",     "Halo storage and exchange (single rank)", 1),
    ("ext/MPIExt/test_haloarrays.jl",     "Halo storage and exchange (4 ranks)",     4),
    ("ext/MPIExt/test_distributed.jl",    "DecomposedGrid wrapper (single rank)",    1),
    ("ext/MPIExt/test_distributed.jl",    "DecomposedGrid wrapper (4 ranks)",        4),
    ("ext/MPIExt/test_ftfield.jl",        "FTField constructors (4 ranks)",          4),
    ("ext/MPIExt/test_field.jl",          "Field constructors (4 ranks)",            4),
    ("ext/MPIExt/test_fft.jl",            "FFTPlans and round-trip (4 ranks)",       4),
    ("ext/MPIExt/test_derivatives.jl",    "Symbol-dispatched derivatives (4 ranks)", 4),
    ("ext/MPIExt/test_galerkin.jl",       "Galerkin projection (4 ranks)",           4),
]

@testset "$(rpad("$name", 68))" for (file, name, nprocs) in TEST_FILES_MPI
    p = run(ignorestatus(cmd(file, nprocs)))
    @test success(p)
end
