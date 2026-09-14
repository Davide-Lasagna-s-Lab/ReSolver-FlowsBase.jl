# Inner-product for ProjectedField stored on device.

# ---------------- #
# reduction method #
# ---------------- #
abstract type DotMethod end

"""
    DotTwoStage <: DotMethod

Two-stage dot product method for GPU-resident `ProjectedField` arrays.

Computes the weighted Hermitian inner product by first forming an elementwise
weighted product into a pre-allocated intermediate array, then reducing with
`sum`. The weight accounts for the rfft Hermitian symmetry: indices along the
rfft dimension with `i=1` receive weight 1, all others receive weight 2. The
result is divided by 2 to give the correct inner product.

Pre-allocating both the weight and intermediate arrays at construction time
avoids any allocation cost at the call site.

# Fields
- `cache::NTuple`: tuple of `(weights, intermediate)` — both `CuArray`s
  of the same size and element type as the input fields

# Constructors
    DotTwoStage(a::GPUProjectedField)

- `a`: `ProjectedField` the dot product is computed from

    DotTwoStage(sz::NTuple{D, Int}, order::NTuple, ::Type{T}=Float32)

- `sz`: size of the underlying parent array
- `order`: FFT dimension ordering — `order[1]` is the rfft dimension
- `T`: real element type, defaults to `Float32`

# Example
```julia
method = DotTwoStage(a)
dot(a, b, method)
```
"""
struct DotTwoStage{A} <: DotMethod
    cache::NTuple{2, A}

    function DotTwoStage(sz::NTuple{D, Int}, order::NTuple, ::Type{T}=Float32) where {D, T}
        weight_vec   = CuArray{T}([i == 1 ? one(T) : T(2) for i in 1:sz[order[1]]])
        shape        = ntuple(d -> d == order[1] ? sz[order[1]] : 1, D)
        weights      = CUDA.zeros(T, sz)
        weights     .= reshape(weight_vec, shape)
        intermediate = CUDA.zeros(T, sz)
        new{typeof(weights)}((weights, intermediate))
    end
end
DotTwoStage(a::ReSolverFlowsBase.ProjectedField) = DotTwoStage(size(a), ReSolverFlowsBase.fft_storage_dims(ReSolverFlowsBase.grid(a)), real(eltype(a)))

"""
    DotAtomic <: DotMethod

DotAtomic accumulation dot product method for GPU-resident [`ProjectedField`](@ref)
arrays.

Each thread computes one weighted elementwise product and accumulates its
contribution into a single scalar result via `CUDA.@atomic`.

Best suited for small-to-medium problem sizes where atomic contention on the
single accumulator is not a bottleneck. For large arrays the [`DotShared`](@ref)
or `DotTwoStage` methods are likely faster - use [`initialise_dot!`](@ref) to
autotune.

# Fields
- `result::A`: pre-allocated single-element `CuArray` accumulator
- `sz::NTuple{D, Int32}`: size of `ProjectedField`
- `nelem::Int32`: total number of elements of `ProjectedField` (`nelem=prod(sz)`)

# Constructor
    DotAtomic(a::GPUProjectedField)

- `a`: `ProjectedField` the dot product is computed from

# Example
```julia
method = DotAtomic(a)
dot(a, b, method)
```
"""
# ! remove sz and nelem as cached variables
struct DotAtomic{RFFT_DIM, D, A} <: DotMethod
    result::A
        sz::NTuple{D, Int32}
     nelem::Int32

    function DotAtomic(a::ReSolverFlowsBase.ProjectedField)
        result = CUDA.zeros(real(eltype(a)), 1)
        sz = Int32.(size(a))
        rfft_dim = ReSolverFlowsBase.rfft_storage_dim(ReSolverFlowsBase.grid(a))
        new{rfft_dim, length(sz), typeof(result)}(result, sz, Int32(prod(sz)))
    end
end

"""
    DotShared{THREADS} <: DotMethod

DotShared memory tree-reduction dot product method for GPU-resident [`ProjectedField`](@ref)
arrays. THIS METHOD ONLY WORKS WHEN THE TOTAL NUMBER OF ELEMENTS IS LARGER THAN
THE EXPECTED THREADS BEING USED. THIS ENSURES THAT THE SHARED DATA ARRAY CAN BE
REDUCED PROPERLY. (BUG FIX?)

Each thread block performs a local tree reduction into shared memory, then
contributes a single atomic add to the global result, reducing atomic contention
from `O(N)` to `O(N/THREADS)` compared to the [`DotAtomic`](@ref) method. The shared memory
size is determined by `THREADS` which is encoded as a type parameter and must be
a power of 2.

The optimal thread count is determined at construction time via
[`launch_configuration`](@ref) with shared memory pressure accounted for, and rounded
down to the nearest power of 2 as required by the tree reduction algorithm.

Best suited for medium problem sizes. For very large arrays the [`DotTwoStage`](@ref)
method may still win due to its optimised memory access pattern - use
[`initialise_dot!`](@ref) to autotune.

# Fields
- `result`: pre-allocated single-element `CuArray` accumulator
- `sz::NTuple{D, Int32}`: size of `ProjectedField`
- `nelem::Int32`: total number of elements of `ProjectedField` (`nelem=prod(sz)`)

# Constructor
    DotShared(a::GPUProjectedField)

- `a`: `ProjectedField` the dot product is computed from

# Example
```julia
method = DotShared(a)
dot(a, b, method)
```
"""
struct DotShared{THREADS, RFFT_DIM, D, A} <: DotMethod
    result::A
        sz::NTuple{D, Int32}
     nelem::Int32

    function DotShared(a::ReSolverFlowsBase.ProjectedField)
        pa = parent(a)
        result = CUDA.zeros(real(eltype(a)), 1)
        rfft_dim = ReSolverFlowsBase.rfft_storage_dim(ReSolverFlowsBase.grid(a))
        sz = Int32.(size(pa))
        nelem = Int32(prod(sz))
        kernel  = @cuda launch=false _dot_shared_kernel!(
            result, pa, pa, nelem, sz, Val(rfft_dim), Val(256)  # dummy Val — replaced below
        )
        _nthreads = let config = launch_configuration(kernel.fun;
                                    shmem = t -> t * sizeof(real(eltype(a))))
            # round down to nearest power of 2 — required for tree reduction
            prev_pow2 = 2^floor(Int32, log2(config.threads))
            min(prev_pow2, nelem)
        end
        new{_nthreads, rfft_dim, length(sz), typeof(result)}(result, sz, nelem)
    end
end


# --------------------------------------- #
# autotuning for optimal method dispatch #
# --------------------------------------- #
const DOT_METHODS = Dict{Type, DotMethod}()

"""
    dot_method(a::GPUProjectedField) -> DotMethod

Return the [`DotMethod`](@ref) associated with the concrete type of `a`,
autotuning if this type has not been seen before. Results are cached in
`DOT_METHODS` keyed on `typeof(a)`.

    dot_method(a) -> cached_method or autotune_dot(a)
"""
function dot_method(a::ReSolverFlowsBase.ProjectedField)
    get!(DOT_METHODS, typeof(a)) do
        autotune_dot(a)
    end
end

"""
    autotune_dot(a::GPUProjectedField) -> DotMethod

Benchmark all available [`DotMethod`](@ref) implementations against a dummy
field of the same type as `a` and return the fastest. Each candidate is warmed
up once to trigger compilation before timing. Timing uses [`CUDA.@elapsed`](@ref)
to measure device-side execution time, taking the minimum over 5 trials to reduce
noise.

If [`show_tuning_info`](@ref) has been set to `true` then the winner and all trial
times via `@info`. The number of samples used to benchmark to the kernels can be
controlled via [`set_tuning_samples`](@ref).

    autotune_dot(a) -> best::DotMethod
"""
function autotune_dot(a::ReSolverFlowsBase.ProjectedField)
    b = similar(a)  # dummy field for benchmarking

    # Construct all candidate methods
    candidates = DotMethod[
        DotTwoStage(a),
        DotAtomic(a),
        DotShared(a),
    ]

    # Warmup all candidates — triggers compilation
    for method in candidates
        dot(a, b, method)
    end
    CUDA.synchronize()

    # Time each candidate
    times = map(candidates) do method
        minimum(1:TUNING_SAMPLES[]) do _
            CUDA.@elapsed dot(a, b, method)
        end
    end

    best = candidates[argmin(times)]
    TUNING_INFO[] && (@info "Auto-tuned dot product" best=typeof(best) times_ns=times)
    return best
end


"""
    initialise_dot!(a::GPUProjectedField)

Eagerly autotune and cache the optimal [`DotMethod`](@ref) for the concrete
type of `a`. After this call, all `dot(a, b)` invocations for fields of the
same type will use the cached optimal method with no autotuning overhead.

Should be called once per field type during program initialisation, before
entering any performance-critical loops.

# Example
```julia
a = ProjectedField(...)
b = ProjectedField(...)
initialise_dot!(a)   # benchmarks all methods, caches the winner

# all subsequent calls use the cached optimal method
for i in 1:nsteps
    s = dot(a, b)
end
```

See also: [`reset_dot_cache!`](@ref)
"""
function ReSolverFlowsBase.initialise_dot!(a::ReSolverFlowsBase.ProjectedField)
    dot_method(a)  # triggers autotune and caches result
    return nothing
end

"""
    reset_dot_cache!()
    reset_dot_cache!(a::GPUProjectedField)

Clear the autotune cache for all field types, or for the specific type of `a`.
The next `dot` call after resetting will trigger autotuning again.

Useful when benchmarking different methods explicitly, or after moving to a
different GPU with different performance characteristics.

# Example
```julia
reset_dot_cache!()     # clear all cached methods
reset_dot_cache!(a)    # clear only the method cached for `typeof(a)`
```

See also: [`initialise_dot!`](@ref)
"""
ReSolverFlowsBase.reset_dot_cache!() = empty!(DOT_METHODS)
ReSolverFlowsBase.reset_dot_cache!(::P) where {P<:ReSolverFlowsBase.ProjectedField} = delete!(DOT_METHODS, P)


# ------------------------------- #
# top-level inner-product methods #
# ------------------------------- #
"""
    dot(a::GPUProjectedField, b::GPUProjectedField) -> real(eltype(a))

Compute the weighted Hermitian inner product of two GPU-resident `GPUProjectedField`
arrays using the autotuned optimal method for their concrete type.

The inner product accounts for the rfft Hermitian symmetry by assigning weight 1
to the first index along the rfft dimension and weight 2 to all others. This gives
the correct energy-norm inner product for pseudo-spectral representations of
real-valued fields.

The method used is determined by [`dot_method(a)`](@ref), which autotunes on the first
call and returns the cached result on all subsequent calls. Call [`initialise_dot!(a)`](@ref)
before entering performance-critical loops to ensure the autotuning cost is paid upfront.

# Arguments
- `a`, `b`: `GPUProjectedField`s.

# Returns
- A host-side scalar of the real element type `real(eltype(a))`.

# Example
```julia
initialise_dot!(a)
s = dot(a, b) # uses cached optimal method, returns Float32
```

See also: [`initialise_dot!`](@ref), [`DotTwoStage`](@ref), [`DotShared`](@ref),
[`DotAtomic`](@ref)
"""
LinearAlgebra.dot(a::GPUProjectedField, b::GPUProjectedField) =
    dot(a, b, dot_method(a))

"""
    dot(a::GPUProjectedField, b::GPUProjectedField, method::DotMethod) -> real(eltype(a))

Compute the weighted Hermitian inner product of `a` and `b` using the
explicitly supplied `method`, bypassing the autotune cache.

Useful for benchmarking specific methods or for one-off computations where
constructing and caching a method is not warranted.

# Arguments
- `a`, `b`: `GPUProjectedField`s.
- `method`: a pre-constructed `DotMethod` - one of `DotTwoStage`, `DotShared`,
  or `DotAtomic`.

# Returns
- A host-side scalar of the real element type `real(eltype(a))`.

# Example
```julia
method = DotTwoStage(a)
s = dot(a, b, method)
```
"""
# TODO: don't pass parent and get RFFT_DIM from grid type directly rather than type parameter of method
LinearAlgebra.dot(a::GPUProjectedField, b::GPUProjectedField, method::DotMethod) =
    _dot(parent(a), parent(b), method)

"""
    _dot(a::CuArray{T}, b::CuArray{T}, cache::DotTwoStage) -> T

Two-stage weighted reduction: elementwise weighted product into `intermediate`,
then `sum`. Stays entirely on device until the final scalar transfer.
"""
function _dot(a::CuArray{T}, b::CuArray{T}, cache::DotTwoStage) where {T}
       weights, intermediate = cache.cache
    @. intermediate          = weights*real(dot(a, b))
    return sum(intermediate)
end

"""
    _dot(a::CuArray{T}, b::CuArray{T}, cache::DotAtomic) -> T

Single-pass reduction using per-thread `CUDA.@atomic` accumulation into a
scalar. Thread count `THREADS` is a compile-time constant from the type parameter.
"""
function _dot(a::CuArray{T}, b::CuArray{T}, cache::DotAtomic{RFFT_DIM}) where {T, RFFT_DIM}
    sz      = cache.sz
    nelem   = cache.nelem
    result  = cache.result
    CUDA.fill!(result, zero(T))

    kernel_args = (result, a, b, nelem, sz, Val(RFFT_DIM))
    nthreads = _get_launch_params(_dot_atomic_kernel!, kernel_args...)
    @cuda threads=nthreads blocks=Int32(cld(nelem, nthreads)) _dot_atomic_kernel!(kernel_args...)

    return Array(result)[1]
end

"""
    _dot(a::CuArray{T}, b::CuArray{T}, cache::DotShared{THREADS}) -> T

Two-level reduction: shared memory tree reduction within each block, then one
`CUDA.@atomic` per block into the scalar result. `THREADS` must be a power of 2.
"""
function _dot(a::CuArray{T}, b::CuArray{T}, cache::DotShared{THREADS, RFFT_DIM}) where {T, THREADS, RFFT_DIM}
    sz     = cache.sz
    nelem  = cache.nelem
    result = cache.result
    CUDA.fill!(result, zero(T))

    kernel_args = (result, a, b, nelem, sz, Val(RFFT_DIM), Val(THREADS))
    @cuda threads=THREADS blocks=Int32(cld(nelem, THREADS)) _dot_shared_kernel!(kernel_args...)

    return Array(result)[1]
end


# ----------- #
# dot kernels #
# ----------- #
"""
    _dot_atomic_kernel!(result, a, b, nelem, sz)

GPU kernel: each thread computes one weighted elementwise product and
accumulates into `result` via `CUDA.@atomic`. Weight is 1 for `I[2]==1`,
2 otherwise.
"""
function _dot_atomic_kernel!(result,
                                  a::CuDeviceArray,
                                  b::CuDeviceArray,
                              nelem::Int32,
                                 sz::NTuple,
                                   ::Val{RFFT_DIM}) where {RFFT_DIM}
    idx = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    idx > nelem && return nothing

    I       = _linear_to_cart(idx, sz)
    w       = I[RFFT_DIM] == 1i32 ? one(Float32) : Float32(2)
    contrib = w * real(dot(@inbounds(a[I]), @inbounds(b[I])))

    CUDA.@atomic result[] += contrib
    return nothing
end

"""
    _dot_shared_kernel!(result, a, b, nelem, sz, ::Val{THREADS})

GPU kernel: shared memory tree reduction within each block followed by a single
`CUDA.@atomic` per block. `THREADS` must be a power of 2 and match the launch
thread count. Shared memory is statically allocated as `THREADS` × `sizeof(T)`.
"""
function _dot_shared_kernel!(result,
                                  a::CuDeviceArray,
                                  b::CuDeviceArray,
                              nelem::Int32,
                                 sz::NTuple,
                                   ::Val{RFFT_DIM},
                                   ::Val{THREADS}) where {RFFT_DIM, THREADS}
    shared = CuStaticSharedArray(Float32, THREADS)
    tid    = threadIdx().x
    idx    = (blockIdx().x - 1i32) * blockDim().x + tid

    shared[tid] = if idx <= nelem
        I = _linear_to_cart(idx, sz)
        w = I[RFFT_DIM] == 1i32 ? one(Float32) : Float32(2)
        w * real(dot(@inbounds(a[I]), @inbounds(b[I])))
    else
        zero(Float32)
    end
    sync_threads()

    stride = Int32(THREADS) >> 1i32
    while stride > 0i32
        if tid <= stride
            shared[tid] += shared[tid + stride]
        end
        sync_threads()
        stride >>= 1i32
    end

    tid == 1i32 && CUDA.@atomic result[] += shared[1]
    return nothing
end
