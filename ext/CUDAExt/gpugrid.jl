# ------------------------------------------------------------------ #
# GPUGrid - Generic wrapper for grid stored on device                #
# ------------------------------------------------------------------ #
# 
# `GPUGrid` is a utility type that wraps an `AbstractGrid` which has already
# been moved onto device via `Adapt.adapt_structure`. This special type
# facilitates simpler dispatch for fields that use this grid such that CUDA
# kernels are used instead of the standard methods.
# 
#   ChannelGrid              - host-side grid defined by the user
#   CUDA.cu(g)               - returns a `GPUGrid` wrapping the parent `g`

"""
    GPUGrid{T, D, AXES, FFT_DIMS_ORDER, GP} <:
        AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}

Generic wrapper for an instance of an [`AbstractGrid`](@ref) that lives on
the device.

This wrapper is very lightweight, only defining the necessary parts of the
[`AbstractGrid`](@ref) interface for correct dispatch in hot loops. Otherwise,
this wrapper type allows the simpler definition of specialised constructors
and methods for the various fields, plans, and operators.
"""
struct GPUGrid{T,
               D,
               AXES,
               FFT_DIMS_ORDER,
               GP<:ReSolverFlowsBase.AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}
              } <: ReSolverFlowsBase.AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}
    parent::GP

    GPUGrid(g::GP) where {T, D, AXES, FFT_DIMS_ORDER, GP<:ReSolverFlowsBase.AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} =
        new{T, D, AXES, FFT_DIMS_ORDER, GP}(g)
end

# required to be able to pass custom field types over `GPUGrid`'s directly to kernels
Adapt.adapt_structure(to, g::GPUGrid) = GPUGrid(adapt_structure(to, parent(g)))

CUDA.cu(g::ReSolverFlowsBase.AbstractGrid) = GPUGrid(adapt_structure(CuArray{Float32}, g))
Adapt.adapt_structure(to, g::ReSolverFlowsBase.AbstractGrid) = throw(ReSolverFlowsBase.NotImplementedError(to, g))

"""
    parent(g::GPUGrid) -> ReSolverFlowsBase.AbstractGrid

Return the underlying grid wrapped by `g`.
"""
Base.parent(g::GPUGrid) = g.parent

Base.size(g::GPUGrid) = size(parent(g))
ReSolverFlowsBase.points(g::GPUGrid; dealias::Bool=false) = CUDA.cu.(ReSolverFlowsBase.points(parent(g); dealias=dealias))
ReSolverFlowsBase.weights(g::GPUGrid) = ReSolverFlowsBase.weights(parent(g))
ReSolverFlowsBase.wavenumber_scale(g::GPUGrid, dim::Int) = ReSolverFlowsBase.wavenumber_scale(parent(g), dim)

"""
    derivative_matrix(g::GPUGrid, stor_dim::Int, ::Val{ORDER}, [DiscreteAdjoint()])

Return the FD differentiation matrix of order `ORDER` along storage
dimension `stor_dim`, plain or, with `DiscreteAdjoint()`, its discrete adjoint.

Downstream single-domain grid types implement this on `parent(g)`:
```
ReSolverFlowsBase.derivative_matrix(::ParentType, stor_dim::Int, ::Val{ORDER}, mode::AbstractDerivativeMode)
```
"""
ReSolverFlowsBase.derivative_matrix(g::GPUGrid,
                   stor_dim::Int,
                           ::Val{ORDER},
                       mode::ReSolverFlowsBase.AbstractDerivativeMode) where {ORDER} =
    ReSolverFlowsBase.derivative_matrix(parent(g), stor_dim, Val(ORDER), mode)
