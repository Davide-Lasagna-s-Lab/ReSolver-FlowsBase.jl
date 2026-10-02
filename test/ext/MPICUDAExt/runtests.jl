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
const TEST_FILES_MPICUDA = [
    ("ext/MPICUDAExt/test_gridandfields.jl",   "Decomposed CUDA grid                                              ", 4),
    ("ext/MPICUDAExt/test_fft.jl",             "Decomposed cuFFT                                                  ", 4),
    ("ext/MPICUDAExt/test_derivatives.jl",     "Decomposed CUDA derivatives                                       ", 4),
    ("ext/MPICUDAExt/test_galerkin.jl",        "Decomposed CUDA galerkin method                                   ", 4),
    ("ext/MPICUDAExt/test_dot.jl",             "Decomposed CUDA dot product                                       ", 4),
]

@testset "$(rpad("$name", 68))" for (file, name, nprocs) in TEST_FILES_MPICUDA
    p = run(ignorestatus(cmd(file, nprocs)))
    @test success(p)
end
