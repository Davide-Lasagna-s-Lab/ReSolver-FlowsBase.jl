# General utilities useful throughout the module.

"""
    ReSolverFlowsBase.show_tuning_info!(show_info::Bool)

Toggle extra information when performing kernel tuning for Galerkin methods
and dot product of `ProjectedField`.
"""
ReSolverFlowsBase.show_tuning_info!(show_info::Bool) = (TUNING_INFO[] = show_info; return nothing)

"""
    ReSolverFlowsBase.set_tuning_samples!(no_of_samples::Int)

Set how many benchmark samples are taken during autotuning of CUDA kernels.
"""
function ReSolverFlowsBase.set_tuning_samples!(no_of_samples::Int)
    no_of_samples > 0 || throw(ArgumentError("number of samples must be larger than 0"))
    TUNING_SAMPLES[] = no_of_samples

    return nothing
end

"""
    _get_launch_params(kernel_f, kernel_args...) -> Int32

Query global `LAUNCH_PARAMS` for number of threads to use for the CUDA
kernel `kernel_f` with the arguments `kernel_args`.

If a key in `LAUNCH_PARAMS` exists with the specific types of the kernel
arguments, i.e. `map(typeof, kernel_args)`, then the associated value is
returned. If the key does not exist then the optimal threads for `kernel_f`
are determined via [`CUDA.launch_configuration`](@ref), see 
https://cuda.juliagpu.org/stable/lib/cudadrv/#CUDACore.launch_configuration
for more details.
"""
function _get_launch_params(kernel_f::F, kernel_args...) where {F}
    key = (F, map(typeof, kernel_args))
    get!(LAUNCH_PARAMS, key) do
        kernel = @cuda launch=false kernel_f(kernel_args...)
        Int32(CUDA.launch_configuration(kernel.fun).threads)
    end
end

"""
    reset_launch_params!()

Clear the `LAUNCH_PARAMS` variable. Useful when benchmarking different
methods explicitly, or after moving to a different GPU with different
performance characteristics.
"""
ReSolverFlowsBase.reset_launch_params!() = empty!(LAUNCH_PARAMS)

"""
    _linear_to_cart(idx::Int32, sz::NTuple{D, Int32}) -> CartesianIndex{D}

Convert a linear index `idx` to the corresponding `CartesianIndex` for a
multidimensional array with a size of `sz`.
"""
# FIXME: change to a tuple to retain Int32 types
@generated function _linear_to_cart(idx::Int32, sz::NTuple{D, Int32}) where {D}
    return quote
        @inline
        rem = idx - 1i32
        $(ntuple(d -> d < D ? quote
            $(Symbol(:i_, d)) = rem % sz[$d] + 1i32
            rem = rem ÷ sz[$d]
        end : quote
            $(Symbol(:i_, D)) = rem + 1i32
        end, Val(D))...)
        return CartesianIndex($(ntuple(d -> Symbol(:i_, d), Val(D))...))
    end
end
