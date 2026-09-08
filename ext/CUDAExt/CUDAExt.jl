module CUDAExt

using CUDA,
      Adapt,
      LinearAlgebra

import CUDA: i32
import Adapt: adapt_structure

import ReSolverFlowsBase,
       FDGrids

# make sure CUDA is functional
__init__() = @assert CUDA.functional(true)

"""
Toggle for extra info for tuning performed the first time
some methods are executed.
"""
const TUNING_INFO = Ref(false)

"""
How many benchmark samples to take for the purpose of tuning
CUDA kernels.
"""
const TUNING_SAMPLES = Ref(5)

"""
Global state parameter storing optimal kernel threads for
given input types.
"""
const LAUNCH_PARAMS = Dict{Tuple{Type, NTuple}, Int32}()

include("utils.jl")
include("gpugrid.jl")
include("gpufields.jl")
include("fft.jl")
include("derivatives.jl")
include("dot.jl")
include("galerkin.jl")

end
