# Galerkin methods used to convert between ProjectedField and FTField
# completely on device.

# TODO: refactor to keep static parts of the computation, make sure the user knows
# TODO: that the assumption is that the same inputs will be given to avoid problems

# -------------- #
# project method #
# -------------- #
abstract type ProjectMethod end

"""
    ProjectLoop <: ProjectMethod

Galerkin projection method that performs the reduction at each frequency on a
single thread.

Each thread computes a single accumulated sum as the result of the inner-product
between a single mode and channel profile at a given frequency and mode number.
The accumulated sum on each thread is then assigned to a single element of the
output projected field.

# Constructor
    ProjectLoop()

# Example
```julia
project!(a, u, ProjectLoop())
```
"""
struct ProjectLoop <: ProjectMethod end

"""
    ProjectShared <: ProjectMethod

Galerkin projection method that performs the reduction at each frequency on a
single thread.

Very similar to [`ProjectLoop`](@ref) with the addition of assigning the
quadrature weights to static shared memory. Might be favourable if number of
global reads to the quadrature weights is a bottleneck in the [`ProjectLoop`](@ref)
method.

# Constructor
    ProjectShared(a, u)

- `a`: GPU resident `ProjectedField` which will be used for the galerkin projection
- `u`: GPU resident `VectorField` of Fourier transformed `FTField`'s which will be
       used for the galerkin projection

# Example
```julia
project!(a, u, ProjectShared(a, u))
```
"""
struct ProjectShared{WS_SZ} <: ProjectMethod
    ProjectShared(a::ReSolverFlowsBase.ProjectedField, ::ReSolverFlowsBase.VectorField) =
        new{map(Int32, size(ReSolverFlowsBase.weights(ReSolverFlowsBase.grid(a))))}()
end

# TODO: implement these and benchmark
"""
    ProjectSharedTiled <: ProjectMethod

Galerkin projection that performs the reduction parallelised over the frequencies
and tiles of the quadrature weights.

The quadrature weights are split up over multiple shared memory tiles which are
computed sequentially. This makes sure that the shared memory for each block is
not exceeded so that memory read latency is minimised.
"""
struct ProjectSharedTiled <: ProjectMethod end

"""
    ProjectSharedTree <: ProjectMethod

Galerkin projection that performs the reduction parallelised over the frequencies
using tree reduction to accumulate the result over the quadrature weights.
"""
struct ProjectSharedTree <: ProjectMethod end


# --------------------------------------- #
# autotuning for optimal method dispatch #
# --------------------------------------- #
const PROJECT_METHODS = Dict{Tuple{Type, Type}, ProjectMethod}()

"""
    project_method(a::GPUProjectedField, u::GPUVectorField) -> ProjectMethod

Return the `ProjectMethod` associated with the concrete type of `a` and `u`,
autotuning if these types have not been seen before. Results are cached in
`PROJECT_METHODS` keyed on `(typeof(a), typeof(u))`.

    project_method(a, u) -> cached_method or autotune_project(a, u)
"""
function project_method(a, u)
    get!(PROJECT_METHODS, (typeof(a), typeof(u))) do
        autotune_project(a, u)
    end
end

"""
    autotune_project(a::GPUProjectedField, u::GPUVectorField) -> ProjectMethod

Benchmark all available `ProjectMethod` implementations against a dummy field of
the same types as `a` and `u` and return the fastest. Each candidate is warmed up
once to trigger compilation before timing. Timing uses `CUDA.@elapsed` to measure
device-side execution time, taking the minimum over 5 trials to reduce noise.

If [`show_tuning_info`](@ref) has been set to `true` then the winner and all trial
times via `@info`. The number of samples used to benchmark to the kernels can be
controlled via [`set_tuning_samples`](@ref).

    autotune_project(a, u) -> best::ProjectMethod
"""
function autotune_project(a, u)
    # Construct all candidate methods
    candidates = ProjectMethod[
        ProjectLoop(),
        ProjectShared(a, u),
    ]

    # Warmup all candidates - triggers compilation
    for method in candidates
        ReSolverFlowsBase.project!(a, u, method)
    end
    CUDA.synchronize()

    # Time each candidate
    times = map(candidates) do method
        minimum(1:TUNING_SAMPLES[]) do _
            CUDA.@elapsed ReSolverFlowsBase.project!(a, u, method)
        end
    end

    best = candidates[argmin(times)]
    TUNING_INFO[] && (@info "autotuned galerkin projection" best=typeof(best) times_ns=times)
    return best
end

"""
    initialise_project!(a::GPUProjectedField, u::GPUVectorField)

Eagerly autotune and cache the optimal `ProjectMethod` for the concrete types of
`a` and `u`. After this call, all `project!(a, u)` invocations for fields of the same
type will use the cached optimal method with no autotuning overhead.

Should be called once per field type during program initialisation, before
entering any performance-critical loops.

# Example
```julia
a = ProjectedField(...)
u = VectorField(...)
initialise_project!(a, u)   # benchmarks all methods, caches the winner

# all subsequent calls use the cached optimal method
for i in 1:nsteps
    project!(similar(a), u)
end
```

See also: [`reset_project_cache!`](@ref)
"""
function ReSolverFlowsBase.initialise_project!(a, u)
    project_method(a, u)
    return nothing
end

"""
    reset_project_cache!()
    reset_project_cache!(a::GPUProjectedField, u::GPUVectorField)

Clear the autotune cache for all field types, or for the specific types of `a`
and `u`. The next `project!` call after resetting will trigger autotuning again.

Useful when benchmarking different methods explicitly, or after moving to a
different GPU with different performance characteristics.

# Example
```julia
reset_project_cache!()     # clear all cached methods
reset_project_cache!(a, u) # clear only the method cached for `(typeof(a), typeof(u))`
```

See also: [`initialise_project!`](@ref)
"""
ReSolverFlowsBase.reset_project_cache!() = empty!(PROJECT_METHODS)
ReSolverFlowsBase.reset_project_cache!(::P, ::V) where {P, V} = delete!(PROJECT_METHODS, (P, V))


# ---------------------------- #
# top-level projection methods #
# ---------------------------- #
"""
    project!(a::GPUProjectedField, u::GPUVectorField{N}) -> a

Compute the Galerkin projection of the GPU-resident `VectorField` onto a set of
orthonormal modes stored in the provided `ProjectedField`, using autotuned optimal
method for their concrete type.

The method used is determined by `project_method(a, u)`, which autotunes on the first
call and returns the cached result on all subsequent calls. Call
`initialise_project!(a, u)` before entering performance-critical loops to ensure the
autotuning cost is paid upfront.

# Arguments
- `a`: `ProjectedField` stored on the GPU.
- `u`: `VectorField` of `FTField`'s also stored on the GPU.

# Returns
- `a` with each element asigned the accumulated constribution of `u` for each mode

# Example
```julia
initialise_project!(a, u)
project!(a, u) # uses cached optimal method, returns `a`
```

```julia
project!(a, u) # autotune optimal method and cache for later use, returns `a`
```

See also: [`initialise_project!`](@ref), [`ProjectLoop`](@ref),
[`ProjectShared`](@ref)
""" # TODO: define GPUVectorFTField for this dispatch, this permits Field types to be passed
ReSolverFlowsBase.project!(a::GPUProjectedField,
                 u::GPUVectorField) = ReSolverFlowsBase.project!(a, u, project_method(a, u))

"""
    project!(a::GPUProjectedField, u::GPUVectorField{N}, method::ProjectMethod) -> a

Compute the Galerkin projection of the GPU-resident `VectorField` onto a set of
orthonormal modes stored in the provided `ProjectedField`, bypassing the autotune
cache.

Useful for benchmarking specific methods or for one-off computations where
constructing and caching a method is not warranted.

# Arguments
- `a`: `ProjectedField` stored on the GPU.
- `u`: `VectorField` of `FTField`'s also stored on the GPU.
- `method`: a pre-constructed `ProjectMethod` - either `ProjectLoop`,
   or `ProjectShared`.

# Returns
- `a` with each element asigned the accumulated constribution of `u` for each mode

# Example
```julia
method = ProjectLoop(a, u)
project!(a, u, method)
```
"""
ReSolverFlowsBase.project!(a, u, method::ProjectMethod) = _project!(a, u, method)

"""
    _project!(a::GPUProjectedField, u::GPUVectorField, ::ProjectLoop) -> a

Compute Galerkin projection of `u` onto modes stored in `a` using
[_project_loop_kernel!](@ref).
"""
function _project!(a, u, ::ProjectLoop)
    sz    = map(Int32, size(a))
    nelem = Int32(prod(sz))

    kernel_args = (a, ReSolverFlowsBase.modes(a), u, ReSolverFlowsBase.weights(ReSolverFlowsBase.grid(a)), sz, nelem)
    nthreads = _get_launch_params(_project_loop_kernel!, kernel_args...)
    @cuda threads=nthreads blocks=Int32(cld(nelem, nthreads)) _project_loop_kernel!(kernel_args...)

    return a
end

"""
    _project!(a::GPUProjectedField, u::GPUVectorField, ::ProjectShared) -> a

Compute Galerkin projection of `u` onto modes stored in `a` using
[_project_shared_kernel!](@ref).
"""
function _project!(a, u, ::ProjectShared{WS_SZ}) where {WS_SZ}
    sz    = map(Int32, size(a))
    nelem = Int32(prod(sz))

    kernel_args = (a, ReSolverFlowsBase.modes(a), u, ReSolverFlowsBase.weights(ReSolverFlowsBase.grid(a)), sz, nelem, Val(WS_SZ))
    nthreads = _get_launch_params(_project_shared_kernel!, kernel_args...)
    shmem_bytes = length(ReSolverFlowsBase.weights(ReSolverFlowsBase.grid(u)))*sizeof(real(eltype(a)))
    @cuda threads=nthreads blocks=Int32(cld(nelem, nthreads)) shmem=shmem_bytes _project_shared_kernel!(kernel_args...)

    return a
end


# ------------------ #
# projection kernels #
# ------------------ #
"""
    _project_loop_kernel!(a, modes, u, ws, sz, nelem)

GPU kernel: compute projection for each element of `a` on each thread using loops,
which are statically unrolled.
"""
function _project_loop_kernel!(a::ReSolverFlowsBase.ProjectedField,
                           modes::NTuple,
                               u::ReSolverFlowsBase.VectorField{N},
                              ws::CuDeviceArray,
                              sz::NTuple,
                           nelem::Int32) where {N}
    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    idx > nelem && return nothing

    # get grid
    g = ReSolverFlowsBase.grid(u)

    # get cartesian index
    I = _linear_to_cart(idx, sz)
    # I_int32 = map(Int32, Tuple(I))

    # each thread computes the full inner reduction for its index
    acc = zero(eltype(a))
    for n in 1:N
        for Inh in CartesianIndices(ReSolverFlowsBase.inhomogeneous_axes(u[n]))
            # Inh_int32 = map(Int32, Tuple(Inh))
            J = CartesianIndex(      ReSolverFlowsBase.combine_indices(g, Inh, CartesianIndex(Base.tail(Tuple(I))))...)
            K = CartesianIndex(I[1], ReSolverFlowsBase.combine_indices(g, Inh, CartesianIndex(Base.tail(Tuple(I))))...)
            acc += ws[Inh]*dot(modes[n][K], u[n][J])
        end
    end

    # assign to output
    @inbounds a[I] = acc
    return nothing
end

"""
    _project_shared_kernel!(a, modes, u, ws, sz, nelem, ::Val{WS_SZ})

GPU kernel: compute projection for each element of `a` on each thread using loops,
which are statically unrolled. The quadrature weights are assigned to static shared
memory to try to optimise global memory reads.
"""
function _project_shared_kernel!(a::ReSolverFlowsBase.ProjectedField,
                             modes::NTuple,
                                 u::ReSolverFlowsBase.VectorField{N},
                                ws::CuDeviceArray,
                                sz::NTuple,
                             nelem::Int32,
                                  ::Val{WS_SZ}) where {N, WS_SZ}
    # load weights into shared memory so all threads in the block share it
    nthds::Int32 = blockDim().x
    tid::Int32 = threadIdx().x
    nw::Int32 = prod(WS_SZ)
    shmem = CuStaticSharedArray(Float32, WS_SZ)
    i = tid
    while i ≤ nw
        @inbounds shmem[i] = ws[i]
        i += nthds
    end
    sync_threads()

    idx = (blockIdx().x - 1i32)*nthds + tid
    idx > nelem && return nothing

    # get grid
    g = ReSolverFlowsBase.grid(u)

    # get cartesian index
    I = _linear_to_cart(idx, sz)

    # each thread computes the full inner reduction for its index
    acc = zero(eltype(a))
    for n in 1:N
        for Inh in CartesianIndices(ReSolverFlowsBase.inhomogeneous_axes(u[n]))
            J = CartesianIndex(      ReSolverFlowsBase.combine_indices(g, Inh, CartesianIndex(Base.tail(Tuple(I))))...)
            K = CartesianIndex(I[1], ReSolverFlowsBase.combine_indices(g, Inh, CartesianIndex(Base.tail(Tuple(I))))...)
            acc += shmem[Inh]*dot(modes[n][K], u[n][J])
        end
    end

    @inbounds a[I] = acc
    return nothing
end

function _project_tiled_kernel!(a::ReSolverFlowsBase.ProjectedField,
                            modes::NTuple,
                                u::ReSolverFlowsBase.VectorField{N},
                               ws::CuDeviceArray,
                               sz::NTuple,
                            nelem::Int32,
                                 ::Val{TILE}) where {N, TILE}
    throw(error("this method has not been implemented yet"))
    nthds = blockDim().x
    tid = threadIdx().x
    nw = length(ws)

    shmem = CuStaticSharedArray(Float32, TILE)

    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    active = idx <= nelem

    g = ReSolverFlowsBase.grid(u)
    I = active ? _linear_to_cart(idx, sz) : _linear_to_cart(1i32, sz)  # dummy for inactive threads

    acc = zero(eltype(a))

    # assumes all modes n share the same reference quadrature shape as `ws`
    ax = ReSolverFlowsBase.homogeneous_axes(u[1])
    cart = CartesianIndices(ax)
    @assert length(cart) == nw  # sanity check outside hot path ideally

    tile_start = 1i32
    while tile_start <= nw
        tile_end = min(tile_start + TILE - 1i32, nw)
        tile_len = tile_end - tile_start + 1i32

        # cooperative load of this tile
        for i in tid:nthds:tile_len
            @inbounds shmem[i] = ws[tile_start + i - 1i32]
        end
        sync_threads()

        if active
            for lidx in tile_start:tile_end
                Inh = cart[lidx]
                w = @inbounds shmem[lidx - tile_start + 1i32]
                J = CartesianIndex(ReSolverFlowsBase.combine_indices(g, Inh, Base.tail(I)))
                K = CartesianIndex(I[1], ReSolverFlowsBase.combine_indices(g, Inh, Base.tail(I))...)
                for n in 1:N
                    @inbounds acc += w * dot(modes[n][K], u[n][J])
                end
            end
        end

        sync_threads()  # ensure no thread is still reading shmem before next tile overwrites it
        tile_start += TILE
    end

    if active
        @inbounds a[I] = acc
    end
    return nothing
end

function _project_tree_kernel!(a::ReSolverFlowsBase.ProjectedField,
                           modes::NTuple,
                               u::ReSolverFlowsBase.VectorField{N},
                              ws::CuDeviceArray,
                              sz::NTuple,
                           nelem::Int32,
                                ::Val{G}) where {N, G}   # G = group size, e.g. 32 (one warp)
    throw(error("this method has not been implemented yet"))
    lane = (threadIdx().x - 1i32) % G + 1i32          # position within group, 1:G
    group_in_block = (threadIdx().x - 1i32) ÷ G       # which output this group handles
    groups_per_block = blockDim().x ÷ G

    idx = (blockIdx().x - 1i32) * groups_per_block + group_in_block + 1i32
    active = idx <= nelem

    g = ReSolverFlowsBase.grid(u)
    I = active ? _linear_to_cart(idx, sz) : _linear_to_cart(1i32, sz)

    ax = ReSolverFlowsBase.homogeneous_axes(u[1])
    cart = CartesianIndices(ax)
    nw = length(cart)

    # each lane accumulates a strided partial sum - no shared memory needed for ws at all,
    # relying on L2/L1 caching since ws is small and reused heavily across blocks
    partial = zero(eltype(a))
    if active
        for lidx in lane:G:nw
            Inh = cart[lidx]
            w = @inbounds ws[lidx]
            J = CartesianIndex(ReSolverFlowsBase.combine_indices(g, Inh, Base.tail(I)))
            K = CartesianIndex(I[1], ReSolverFlowsBase.combine_indices(g, Inh, Base.tail(I))...)
            for n in 1:N
                @inbounds partial += w * dot(modes[n][K], u[n][J])
            end
        end
    end

    # warp-shuffle reduction across the G lanes (requires G <= 32, power of two)
    offset = G >> 1
    while offset > 0
        partial += CUDA.shfl_down_sync(0xffffffff, partial, offset)
        offset >>= 1
    end

    if active && lane == 1i32
        @inbounds a[I] = partial
    end
    return nothing
end


# ------------- #
# expand method #
# ------------- #
abstract type ExpandMethod end

"""
    ExpandModal <: ExpandMethod

Galerkin expansion method for GPU-resident `ProjectedField` and `VectorField`
arrays using GPU native kernels to compute the linear sums in parallel.

Computes the Galerkin expansion of a projected field using the stored set of
orthonormal modes by accumulating the linear sum for each mode number individually
and for every co-location coordinate and frequency in parallel. The option
`over_vector` allows the option to parallelise over the vector field component
as well as the individual points making up the fields of `u`. If it is set to
`false` then the vector field components are looped over in serial by every
thread instead.

# Constructor
    ExpandModal(; over_vector::Bool=false)

- `over_vector`: optional argument to decide whether the computation is
                 parallelised over the vector components of `u`.

# Example
```julia
method = ExpandModal(over_vector=true)
expand!(u, a, method)
```
"""
struct ExpandModal{VECTOR} <: ExpandMethod

    ExpandModal(;over_vector::Bool=false) =
        new{over_vector}()
end

# --------------------------------------- #
# autotuning for optimal method dispatch #
# --------------------------------------- #
const EXPAND_METHODS = Dict{Tuple{Type, Type}, ExpandMethod}()

"""
    expand_method(u::GPUVectorField, a::GPUProjectedField) -> ExpandMethod

Return the `ExpandMethod` associated with the concrete type of `u` and `a`,
autotuning if these types have not been seen before. Results are cached in
`EXPAND_METHODS` keyed on `(typeof(u), typeof(a))`.

    expand_method(u, a) -> cached_method or autotune_expand(u, a)
"""
function expand_method(u, a)
    get!(EXPAND_METHODS, (typeof(u), typeof(a))) do
        autotune_expand(u, a)
    end
end

"""
    autotune_expand(u::GPUVectorField, a::GPUProjectedField) -> ExpandMethod

Benchmark all available `ExpandMethod` implementations against a dummy field of
the same types as `u` and `a` and return the fastest. Each candidate is warmed up
once to trigger compilation before timing. Timing uses `CUDA.@elapsed` to measure
device-side execution time, taking the minimum over 5 trials to reduce noise.

If [`show_tuning_info`](@ref) has been set to `true` then the winner and all trial
times via `@info`. The number of samples used to benchmark to the kernels can be
controlled via [`set_tuning_samples`](@ref).

    autotune_expand(u, a) -> best::ExpandMethod
"""
function autotune_expand(u, a)
    # Construct all candidate methods
    candidates = ExpandMethod[
        ExpandModal(over_vector=false),
        ExpandModal(over_vector=true),
    ]

    # Warmup all candidates - triggers compilation
    for method in candidates
        ReSolverFlowsBase.expand!(u, a, method)
    end
    CUDA.synchronize()

    # Time each candidate
    times = map(candidates) do method
        minimum(1:TUNING_SAMPLES[]) do _
            CUDA.@elapsed ReSolverFlowsBase.expand!(u, a, method)
        end
    end

    best = candidates[argmin(times)]
    TUNING_INFO[] && @info "autotuned galerkin expansion" best=typeof(best) times_ns=times
    return best
end

"""
    initialise_expand!(u::GPUVectorField, a::GPUProjectedField)

Eagerly autotune and cache the optimal `ExpandMethod` for the concrete types of
`u` and `a`. After this call, all `expand!(u, a)` invocations for fields of the same
type will use the cached optimal method with no autotuning overhead.

Should be called once per field type during program initialisation, before
entering any performance-critical loops.

# Example
```julia
u = VectorField(...)
a = ProjectedField(...)
initialise_expand!(u, a)   # benchmarks all methods, caches the winner

# all subsequent calls use the cached optimal method
for i in 1:nsteps
    expand!(similar(u), a)
end
```

See also: [`reset_expand_cache!`](@ref)
"""
function ReSolverFlowsBase.initialise_expand!(u, a)
    expand_method(u, a)
    return nothing
end

"""
    reset_expand_cache!()
    reset_expand_cache!(u::GPUVectorField, a::GPUProjectedField)

Clear the autotune cache for all field types, or for the specific types of `u`
and `a`. The next `expand!` call after resetting will trigger autotuning again.

Useful when benchmarking different methods explicitly, or after moving to a
different GPU with different performance characteristics.

# Example
```julia
reset_expand_cache!()     # clear all cached methods
reset_expand_cache!(u, a) # clear only the method cached for (typeof(u), typeof(a))
```

See also: [`initialise_expand!`](@ref)
"""
ReSolverFlowsBase.reset_expand_cache!() = empty!(EXPAND_METHODS)
ReSolverFlowsBase.reset_expand_cache!(::V, ::P) where {V, P} = delete!(EXPAND_METHODS, (V, P))


# --------------------------- #
# top-level expansion methods #
# --------------------------- #
"""
    expand!(u::GPUVectorField, a::GPUProjectedField) -> u

Compute the Galerkin expansion of the GPU-resident `ProjectedField` storing the
result in the provided `VectorField`, using autotuned optimal method for their
concrete type.

The method used is determined by `expand_method(u, a)`, which autotunes on the first
call and returns the cached result on all subsequent calls. Call
`initialise_expand!(u, a)` before entering performance-critical loops to ensure the
autotuning cost is paid upfront.

# Arguments
- `a`: `ProjectedField` stored on the GPU.
- `u`: `VectorField` of `FTField`'s also stored on the GPU.

# Returns
- `u` with each element assigned the weighted sum of the orthonormal modes stored
      in `modes(a)`

# Example
```julia
initialise_expand!(u, a)
expand!(u, a) # uses cached optimal method, returns `u`
```

```julia
expand!(u, a) # autotune optimal method and cache for later use, returns `u`
```

See also: [`initialise_expand!`](@ref), [`ExpandModal`](@ref)
"""
ReSolverFlowsBase.expand!(u::GPUVectorField, a::GPUProjectedField) =
    ReSolverFlowsBase.expand!(u, a, expand_method(u, a))

"""
    expand!(u::GPUVectorField, a::GPUProjectedField, method::ExpandMethod) -> u

Compute the Galerkin expansion of the GPU-resident `ProjectedField` storing the
result in the provided `VectorField`, bypassing the autotune cache.

Useful for benchmarking specific methods or for one-off computations where
constructing and caching a method is not warranted.

# Arguments
- `a`: `ProjectedField` stored on the GPU.
- `u`: `VectorField` of `FTField`'s also stored on the GPU.
- `method`: a pre-constructed `ExpandMethod`

# Returns
- `u` with each element assigned the weighted sum of the orthonormal modes stored
      in `modes(a)`

# Example
```julia
method = ExpandModal(u, a)
expand!(u, a, method)
```
"""
ReSolverFlowsBase.expand!(u, a, method::ExpandMethod) =
    _expand!(u, a, method)

"""
    _expand!(u::GPUVectorField, a::GPUProjectedField, ::ExpandModal) -> u

Compute Galerkin expansion of `a` into the provided field `u` using
[_expand_modal_1_kernel!](@ref).

Using `over_vector` upon the construction of the [ExpandModal](@ref) cache,
the user can choose to use either of the GPU kernels [_expand_modal_1_kernel](@ref)
or [_expand_modal_2_kernel](@ref). With `over_vector=true` the vector field 
component is included in the parallel execution of the kernel, and
`over_vector=false` means that the kernel loops the vector field components.
"""
function _expand!(u::ReSolverFlowsBase.VectorField{N}, a, ::ExpandModal{true}) where {N}
    sz               = map(Int32, size(u[1]))
    nelem::Int32     = prod(sz)
    nelem_vec::Int32 = N*nelem

    kernel_args = (u, a, ReSolverFlowsBase.modes(a), sz, nelem, nelem_vec)
    nthreads = _get_launch_params(_expand_modal_1_kernel!, kernel_args...)
    @cuda threads=nthreads blocks=Int32(cld(nelem_vec, nthreads)) _expand_modal_1_kernel!(kernel_args...)

    return u
end

function _expand!(u, a, ::ExpandModal{false})
    sz           = map(Int32, size(u[1]))
    nelem::Int32 = prod(sz)

    kernel_args = (u, a, ReSolverFlowsBase.modes(a), sz, nelem)
    nthreads = _get_launch_params(_expand_modal_2_kernel!, kernel_args...)
    @cuda threads=nthreads blocks=Int32(cld(nelem, nthreads)) _expand_modal_2_kernel!(kernel_args...)

    return u
end

# ----------------- #
# expansion kernels #
# ----------------- #
"""
    _expand_modal_1_kernel!(u, a, modes, sz, nelem, nelem_vec)

GPU kernel: compute the expansion of the coefficients `a` into a vectorfield `u`,
parallelising over the vector field components `N` in addition to the individual
field elements of `u`.
"""
function _expand_modal_1_kernel!(u, a, modes, sz, nelem, nelem_vec)
    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    idx > nelem_vec && return nothing

    # TODO: what's the performance like if `n` is indexed from the back? So it increments slowest?
    n  ::Int32 = (idx - 1i32)÷nelem + 1i32
    rem::Int32 = (idx - 1i32)%nelem + 1i32
    I = _linear_to_cart(rem, sz)

    acc = zero(eltype(a))
    for m in axes(a, 1)
        J = CartesianIndex(m, ReSolverFlowsBase.homogeneous_indices(I, ReSolverFlowsBase.grid(u))...)
        K = CartesianIndex(m, I)
        @inbounds acc += a[J]*modes[n][K]
    end
    @inbounds u[n][I] = acc

    return nothing
end

"""
    _expand_modal_2_kernel!(u, a, modes, sz, nelem)

GPU kernel: compute the expansion of the coefficients `a` into a vectorfield `u`,
treating each vector field component of `u` serially on each thread.
"""
function _expand_modal_2_kernel!(u::ReSolverFlowsBase.VectorField{N}, a, modes, sz, nelem) where {N}
    idx = (blockIdx().x - 1i32)*blockDim().x + threadIdx().x
    idx > nelem && return nothing

    I = _linear_to_cart(idx, sz)

    for n in 1:N
        acc = zero(eltype(a))
        for m in axes(a, 1)
            J = CartesianIndex(m, ReSolverFlowsBase.homogeneous_indices(I, ReSolverFlowsBase.grid(u))...)
            K = CartesianIndex(m, I)
            @inbounds acc += a[J]*modes[n][K]
        end
        @inbounds u[n][I] = acc
    end

    return nothing
end
