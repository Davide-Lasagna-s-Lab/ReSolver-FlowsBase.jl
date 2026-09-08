import CUDA: i32

const CUDAExt = Base.get_extension(ReSolverFlowsBase, :CUDAExt)

# construct mock grid used in tests
Nx = 15; Ny = 16; Nz = 15; Nt = 15;
g = MockChannelGrid(Ny, Nx, Nz, Nt)

include("test_utils.jl")
include("test_gpugrid.jl")
include("test_gpufields.jl")
include("test_fft.jl")
include("test_derivatives.jl")
include("test_dot.jl")
include("test_galerkin.jl")
