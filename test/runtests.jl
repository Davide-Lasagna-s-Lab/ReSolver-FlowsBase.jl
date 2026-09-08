using Test

using FFTW
using HCubature
using LinearAlgebra
using CUDA
using MPI
using ReSolverFlowsBase
using FDGrids

# run MPI test file in it's own Julia instance
test_project = dirname(Base.active_project())
cmd(file, nprocs) = addenv(
    `$(mpiexec()) -n $nprocs --oversubscribe $(Base.julia_cmd()) --project=$test_project --startup-file=no $(joinpath(@__DIR__, file))`,
)

# utility function to check if CUDA.jl functions on this system
function cuda_available()
    try
        CUDA.functional()
    catch
        false
    end
end

# Shared test fixtures: `fake.jl` defines `FakeGrid` (a 2-D grid with one
# inhomogeneous and one rfft dimension), `test_grids.jl` defines
# `TripleGrid` (a 3-D grid with one inhomogeneous, one rfft and one signed
# FFT dimension) for tests that need multiple homogeneous directions.
include("fake.jl")
include("test_grids.jl")
include("mock_channel_grid.jl")

# Generic interface / utility tests — exercise every public function in
# ReSolverFlowsBase against the documented contract, not its implementation.
include("test_notimplementederror.jl")
include("test_abstractgrid.jl")
include("test_wavenumbervector.jl")
include("test_broadcasting.jl")
include("test_field.jl")
include("test_ftfield.jl")
include("test_vectorfield.jl")
include("test_projectedfield.jl")
include("test_fields.jl")
include("test_fftplans.jl")
include("test_derivatives.jl")
include("test_axis_utils.jl")
include("test_shifts.jl")
include("test_norms.jl")
include("test_weighting.jl")
include("test_galerkin.jl")
include("test_io.jl")

# Integration tests for the bundled equations module.
include("test_operators.jl")

# Allocation tests — check that no unexpected allocations occur
include("test_allocations.jl")

# test extensions
include("ext/MPIExt/runtests.jl")
if cuda_available()
    include("ext/CUDAExt/runtests.jl")
    MPI.Init()
    if MPI.has_cuda()
        include("ext/MPICUDAExt/runtests.jl")
    else
        rank == 0 && @warn "Skipping MPI+CUDA tests - MPI backend is not CUDA aware"
    end
else
    @warn "Skipping GPU tests - CUDA not functional"
end
