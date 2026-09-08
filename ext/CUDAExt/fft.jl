# Specialised constructor for FFTPlans using cuFFT for the backend.

# FFT style to use
struct cuFFTStyle <: ReSolverFlowsBase.FFTPlanStyle end
ReSolverFlowsBase.FFTPlanStyle(::Type{<:GPUGrid})                          = cuFFTStyle()
ReSolverFlowsBase.array_constructor(::cuFFTStyle)                          = CUDA.zeros
ReSolverFlowsBase.transform_backend(::cuFFTStyle)                          = cuFFT
ReSolverFlowsBase.construct_plan(::cuFFTStyle, args...; kwargs...)         = cuFFT.plan_rfft(args...)
ReSolverFlowsBase.construct_inverse_plan(::cuFFTStyle, args...; kwargs...) = cuFFT.plan_brfft(args...)

"""
    _loopblk!(dest::CuArray, ar, src::CuArray, br, ::Val{VADD})

Launch `_loopblk_kernel!` for a single contiguous block.
`ar` and `br` are `NTuple{D, UnitRange{Int}}` - constructed on the host,
never passed to the device.
"""
@inline function ReSolverFlowsBase._loopblk!(dest::CuArray,
                                     ar::NTuple{D},
                                    src::CuArray,
                                     br::NTuple{D},
                                   vadd::Val) where {D}
    # All range arithmetic happens here on the host
    dest_starts = Int32.(ntuple(k ->  first(ar[k]) - 1           , Val(D)))
    dest_sizes  = Int32.(ntuple(k -> length(ar[k])               , Val(D)))
    src_offsets = Int32.(ntuple(k ->  first(br[k]) - first(ar[k]), Val(D)))

    kernel_args = (dest, src, dest_starts, dest_sizes, src_offsets, vadd)
    nthreads = _get_launch_params(_loopblk_kernel!, kernel_args...)
    @cuda threads=nthreads blocks=Int32(cld(prod(dest_sizes), nthreads)) _loopblk_kernel!(kernel_args...)
    return nothing
end

"""
    _loopblk_kernel!(dest, src, dest_starts, dest_sizes, src_offsets, ::Val{ADD})

GPU kernel: each thread handles one CartesianIndex in the destination block.
`dest_starts` is an `NTuple{D, Int}` of 0-based start indices for the destination block.
`dest_sizes` is an `NTuple{D, Int}` of element counts in each dimension.
`src_offsets` is an `NTuple{D, Int}` where `src_idx = dest_idx + src_offset`.
No heap allocation: all index arithmetic is register-based.
"""
@generated function _loopblk_kernel!(dest, src,
                                     dest_starts::NTuple{D, Int32},
                                      dest_sizes::NTuple{D, Int32},
                                     src_offsets::NTuple{D, Int32},
                                                ::Val{ADD}) where {D, ADD}
    # Unroll the D-dimensional linear-index → CartesianIndex decomposition
    # entirely at code-generation time. The emitted kernel has no loops over
    # dimensions — just straight-line index arithmetic.
    index_exprs = quote
        linear = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
        linear > prod(dest_sizes) && return nothing
    end

    # Generate the strided decomposition: recover each dimension's local index
    # from `linear` using precomputed strides.
    decomp = Expr(:block)
    push!(decomp.args, :(rem_idx = linear - 1i32))
    for d in 1:D
        if d < D
            push!(decomp.args, quote
                $(Symbol(:loc_, d)) = rem_idx % dest_sizes[$d] + 1i32
                rem_idx = rem_idx ÷ dest_sizes[$d]
            end)
        else
            push!(decomp.args, :($(Symbol(:loc_, d)) = rem_idx + 1i32))
        end
    end

    # Build the CartesianIndex expressions for dest and src
    dest_idx = :(CartesianIndex($(
        [:(dest_starts[$d] + $(Symbol(:loc_, d))) for d in 1:D]...
    )))
    src_idx  = :(CartesianIndex($(
        [:(dest_starts[$d] + $(Symbol(:loc_, d)) + src_offsets[$d]) for d in 1:D]...
    )))

    # The assignment — branch eliminated by Val{ADD} at code-generation time
    assign = ADD ? :(@inbounds dest[$dest_idx] += src[$src_idx]) :
                   :(@inbounds dest[$dest_idx]  = src[$src_idx])

    return quote
        $index_exprs
        $decomp
        $assign
        return nothing
    end
end
