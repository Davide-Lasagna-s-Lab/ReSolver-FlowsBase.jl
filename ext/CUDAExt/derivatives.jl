# GPU kernels for derivatives of FTFields wrapping on a GPUGrid.
# 
# These methods are extensions to the base packages spectral differentiation
# components. The inhomogeneous derivatives are handled via `ReSolverFlowsBase._inhomogeneous_dd!`
# which relies on the user implemented `ReSolverFlowsBase.derivative_matrix`. Any GPU
# specialisations of derivatives over non-transformed directions is assumed to handled
# by the user defining the behaviour `mul!(out, A, u, ::Val{STORAGE_DIM})` for their
# dirivative operator type `A`.

ReSolverFlowsBase._spectral_dd!(out::F,
                        u::F,
                         ::Val{STORAGE_DIM},
                     mode::ReSolverFlowsBase.OperatorMode=ReSolverFlowsBase.Forward()) where {STORAGE_DIM, F<:Union{GPUFTField, GPUProjectedField}} =
    _cuda_spectral_dd!(out, u, Val(STORAGE_DIM), Val(ReSolverFlowsBase.rfft_storage_dim(ReSolverFlowsBase.grid(u))), mode)

function _cuda_spectral_dd!(out, u, ::Val{STORAGE_DIM}, ::Val{RFFT_DIM}, mode::ReSolverFlowsBase.OperatorMode) where {STORAGE_DIM, RFFT_DIM}
    # kernel arguments
    sz     = Int32.(size(u))
    nelem  = Int32(prod(sz))
    _ddx_sign  = mode isa ReSolverFlowsBase.AdjointDiscrete ? -1im*one(real(eltype(u))) : 1im*one(real(eltype(u)))
    _ddx_scale = ReSolverFlowsBase.wavenumber_scale(ReSolverFlowsBase.grid(u), STORAGE_DIM)

    # launch kernel
    kernel_args = (parent(out),
                   parent(u),
                   sz, nelem,
                   _ddx_sign, _ddx_scale,
                   Val(Int32(STORAGE_DIM)),
                   Val(Int32(RFFT_DIM)))
    nthreads = _get_launch_params(_spectral_dd_kernel!, kernel_args...)
    @cuda threads=nthreads blocks=Int32(cld(nelem, nthreads)) _spectral_dd_kernel!(kernel_args...)

    return out
end

function _spectral_dd_kernel!(out,
                                u,
                               sz::NTuple,          nelem::Int32,
                        _ddx_sign::ComplexF32, _ddx_scale::Float32,
                                 ::Val{DIM},
                                 ::Val{RFFT_DIM}) where {DIM, RFFT_DIM}
    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    idx > nelem && return nothing

    I = _linear_to_cart(idx, sz)
    n = _get_freq(I, sz, Val(DIM), Val(RFFT_DIM))

    @inbounds out[I] = _ddx_sign*n*_ddx_scale*u[I]
    return nothing
end

"""
    _get_freq(I::CartesianIndex, sz::NTuple, ::Val{DIM}, ::Val{RFFT_DIM})

Obtain the wave number for dimension `DIM` of the index `I` applied to
a matrix with size `sz`, with `RFFT_DIM` being the first transformed
direction of the array.
"""
@generated function _get_freq(I::CartesianIndex,
                             sz::NTuple,
                               ::Val{DIM},
                               ::Val{RFFT_DIM}) where {DIM, RFFT_DIM}
    return if DIM == RFFT_DIM
        :(@inline; I[$DIM] - 1)
    else
        :(@inline; ifelse(I[$DIM] ≤ (sz[$DIM] >> 1) + 1, I[$DIM] - 1, I[$DIM] - 1 - sz[$DIM]))
    end
end


ReSolverFlowsBase._add_homogeneous_laplacian!(out::F, u::F) where {F<:GPUField} =
    _cuda_add_homogeneous_laplacian!(out, u)

function _cuda_add_homogeneous_laplacian!(out, u)
    # kernel arguments
    sz     = Int32.(size(u))
    nelem  = Int32(prod(sz))
    scales = map(d -> ReSolverFlowsBase.wavenumber_scale(ReSolverFlowsBase.grid(u), d), ReSolverFlowsBase.spatial_fft_storage_dims(ReSolverFlowsBase.grid(u)))

    # launch kernel
    kernel_args = (parent(out),
                   parent(u),
                   sz, nelem,
                   scales,
                   Val(Int32.(ReSolverFlowsBase.spatial_fft_storage_dims(ReSolverFlowsBase.grid(u)))),
                   Val(Int32(ReSolverFlowsBase.rfft_storage_dim(ReSolverFlowsBase.grid(u)))))
    nthreads = _get_launch_params(_add_homogeneous_laplacian_kernel!, kernel_args...)
    @cuda threads=nthreads blocks=Int32(cld(nelem, nthreads)) _add_homogeneous_laplacian_kernel!(kernel_args...)

    return out
end

function _add_homogeneous_laplacian_kernel!(out,
                                              u,
                                             sz::NTuple, nelem::Int32,
                                         scales::NTuple,
                                               ::Val{SPATIAL_FFT_DIMS_ORDER},
                                               ::Val{RFFT_DIM}) where {SPATIAL_FFT_DIMS_ORDER, RFFT_DIM}
    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    idx > nelem && return nothing

    I  = _linear_to_cart(idx, sz)
    k² = _get_laplacian_freq(I, sz, scales, Val(SPATIAL_FFT_DIMS_ORDER), Val(RFFT_DIM))

    @inbounds out[I] -= k²*u[I]
    return nothing
end

"""
    _get_laplacian_freq(I::CartesianIndex, sz::NTuple, scales::NTuple, ::Val{SPATIAL_FFT_DIMS_ORDER}, ::Val{RFFT_DIM})

Compute the (squared) frequency contribution to the Laplacian operator
for all the homogeneous directions at index `I` of an array with size `sz`.
"""
@generated function _get_laplacian_freq(I::CartesianIndex,
                                       sz::NTuple,
                                   scales::NTuple,
                                         ::Val{SPATIAL_FFT_DIMS_ORDER},
                                         ::Val{RFFT_DIM}) where {SPATIAL_FFT_DIMS_ORDER, RFFT_DIM}
    terms = map(enumerate(SPATIAL_FFT_DIMS_ORDER)) do (i, d)
        n = if d == RFFT_DIM
            :(Int32(I[$d] - 1))
        else
            :(ifelse(I[$d] ≤ (sz[$d] >> 1) + 1, Int32(I[$d] - 1), Int32(I[$d] - 1 - sz[$d])))
        end
        :(scales[$i]*$n)
    end

    sum_expr = mapreduce(
        t      -> :($t^2),
        (a, b) -> :($a + $b),
        terms
    )

    return :(@inline; Float32($sum_expr))
end
