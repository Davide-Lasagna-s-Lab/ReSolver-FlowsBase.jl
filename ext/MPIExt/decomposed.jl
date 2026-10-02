# ------------------------------------------------------------------ #
# DecomposedGrid — generic decomposed wrapper                        #
# ------------------------------------------------------------------ #
#
# `DecomposedGrid` lets any single-domain (Undecomposed) grid serve as a
# decomposed grid for distributed computations without the downstream package
# having to define a parallel grid type. The wrapper carries a reference to
# the parent grid and the Cartesian MPI communicator, and delegates every
# query that does not depend on the per-rank partition straight back to the
# parent.
#
#   ChannelGrid               - single-domain grid the user already has
#   distributed(g, comm; ...) - returns a DecomposedGrid wrapping it
#
# Per-rank queries (`size`, `points`, `weights`, `local_size`) slice the
# parent's data along the decomposition axes based on the rank's Cartesian
# coordinates. `wavenumber_scale` and `derivative_matrix` are pure pass-throughs.
# `growto` and `convert` rebuild the wrapper while preserving the communicator
# and decomposition metadata.
#
# Users choose decomposed directions with physical symbols (`:x`, `:y`, `:z`,
# `:t`); public accessors also accept physical symbols via `Symbol` overloads.
# Internal helpers and derivative kernels always work with `Int` storage dims.
#
# The only method the parent grid must add beyond the standard ReSolverFlowsBase interface
# is `ReSolverFlowsBase.derivative_matrix(parent, stor_dim::Int, ::Val{ORDER}, ::Val{ADJ})`.

# ------------------------------------------------------------------ #
# Private helpers                                                    #
# ------------------------------------------------------------------ #

function _check_plain_intracommunicator(comm::MPI.Comm)
    flag = Ref{Cint}()
    MPI.API.MPI_Comm_test_inter(comm, flag)
    flag[] == 0 ||
        throw(ArgumentError("comm must be an MPI intracommunicator"))

    status = Ref{Cint}()
    MPI.API.MPI_Topo_test(comm, status)
    status[] == MPI.API.MPI_UNDEFINED[] ||
        throw(ArgumentError("comm must not already have an MPI topology; pass a plain communicator because distributed creates its own Cartesian communicator"))

    return comm
end

"""
    DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP, W, P} <:
        ReSolverFlowsBase.AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}

Generic decomposed-grid wrapper around a single-domain parent grid.

Whether a grid is decomposed is *not* recorded in the ReSolverFlowsBase `AbstractGrid`
type — `DecomposedGrid` carries the partition metadata itself. The partitioned
storage dimensions are the `DDIMS` type parameter, exposed via
[`decomposition_storage_dims`](@ref).

Type parameters:

  - `T, D, AXES, FFT_DIMS_ORDER`: identical to the parent grid's.
  - `DDIMS`: tuple of storage-dimension indices partitioned across ranks.
  - `NHALO`: length-`D` tuple of halo widths; 0 on non-decomposed dims.
  - `S`: local interior storage size as a length-`D` tuple.
  - `GP`: concrete parent grid type; recovered via `parent(g)`.
  - `W`: type of the cached per-rank quadrature weights array.
  - `P`: type of the cached per-rank inhomogeneous coordinate tuple.

Construct via [`distributed`](@ref). Every field, FFT, derivative, and
equation method in this package dispatches on `DecomposedGrid`.

**Stencil constraint**: halo exchanges are always face-only (`economic=true` in
HaloArrays). Finite-difference stencils must not read corner or edge halo
cells — only the ghost layers along the decomposed (wall-normal) direction.
"""
struct DecomposedGrid{T,
                      D,
                      AXES,
                      FFT_DIMS_ORDER,
                      DDIMS,
                      NHALO,
                      S,
                      GP<:ReSolverFlowsBase.AbstractGrid{T, D, AXES, FFT_DIMS_ORDER},
                      W,
                      P,
                      COMM} <: ReSolverFlowsBase.AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}
    # The single-domain (serial) grid this wrapper exposes as decomposed.
    parent     :: GP
    # Full-Cartesian MPI communicator built by `distributed(...)` (one Cart
    # direction per storage dim, with rank 1 on non-decomposed dims). This
    # is the shape `HaloArrays` requires, and the comm returned by `comm(g)`.
    comm       :: COMM
    # Per-rank quadrature weights, sliced from the parent at construction.
    # `weights(g)` is called inside hot loops (projection, dot products), so
    # lazy `getindex` slicing here would allocate per call.
    weights    :: W
    # Per-rank coordinate vectors along each inhomogeneous storage dim, in
    # `ReSolverFlowsBase.inhomogeneous_storage_dims(g)` order. The k-th entry is a `Vector`
    # representing storage dim `inhomogeneous_storage_dims(g)[k]`. Cached for the
    # same reason as `weights`: `points(g)` is called every time a Field is
    # initialised by broadcasting, and re-slicing the parent's stored
    # coordinate vector per call would allocate.
    inh_points :: P
end

DecomposedGrid(gp::GP,
               DDIMS, NHALO, LOCAL_SIZE,
               weights::W, inh_points::P,
               comm::COMM) where {
        T, D, AXES, FFT_DIMS_ORDER, GP<:ReSolverFlowsBase.AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}, W, P, COMM} =
    DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, LOCAL_SIZE, GP, W, P, COMM}(gp, comm, weights, inh_points)

"""
    distributed(g::ReSolverFlowsBase.AbstractGrid, comm::MPI.Comm;
                decomposed_physical_dims::NTuple{K, Symbol},
                nprocesses::NTuple{K, Int},
                nhalo::NTuple{K, Int}) -> DecomposedGrid

Wrap the single-domain (undecomposed) grid `g` as a decomposed grid for
distributed computations.

`comm` must be a plain MPI intracommunicator with no attached topology, such as
`MPI.COMM_WORLD` or a duplicate/split of it. `distributed` constructs the
full Cartesian communicator required by HaloArrays itself; passing an already
Cartesian communicator would stack one topology construction on top of another
and obscure the rank-to-grid mapping.

Three parallel `K`-tuples describe the decomposition, with one entry per
decomposed direction:

  - `decomposed_physical_dims[k]` — the physical-coordinate symbol
    (`:x`, `:y`, `:z`, or `:t`) of the `k`-th decomposed direction; must
    be a spatial inhomogeneous direction (FFT-transformed cannot be decomposed).
  - `nprocesses[k]` — number of ranks along that direction.
  - `nhalo[k]` — halo width along that direction (the FD stencil
    half-width).

Halo exchange is always face-only (`economic=true` in HaloArrays). Stencils
must not read corner or edge halo cells — only the ghost layers along the
decomposed direction.

Decomposed directions are always non-periodic in this package, so no
`isperiodic` argument is exposed — `MPI.Cart_create` is called with
`periodic=(false, ...)`.

Validation:

  - `prod(nprocesses) == MPI.Comm_size(comm)`.
  - Every `decomposed_physical_dims[k]` is a spatial inhomogeneous
    direction (see [`ReSolverFlowsBase.spatial_inhomogeneous_physical_dims`](@ref)).
  - For each `k`, `size(g, storage_dim(g, decomposed_physical_dims[k]))`
    is divisible by `nprocesses[k]` (uniform decomposition).

`distributed` calls `MPI.Cart_create(comm, ...)` once internally to
build the full-Cartesian comm that `HaloArrays` requires. The new comm
is what `comm(g)` returns.

# Example

```julia
g  = ChannelGrid(...)                          # Undecomposed
dg = distributed(g, MPI.COMM_WORLD;
                 decomposed_physical_dims=(:y,),
                 nprocesses=(MPI.Comm_size(MPI.COMM_WORLD),),
                 nhalo=(1,))
```
"""
function ReSolverFlowsBase.distributed(               g::ReSolverFlowsBase.AbstractGrid{T, D},
                                         comm::COMM;
                     decomposed_physical_dims::NTuple{K, Symbol},
                                   nprocesses::NTuple{K, Int},
                                        nhalo::NTuple{K, Int}) where {T, D, K, COMM<:MPI.Comm}
    # --------------------------------------------------------------------- #
    # Validation
    # --------------------------------------------------------------------- #

    # Validate the decomposed-direction labels and translate to storage
    # dims once for the `Decomposed{DIMS}` type-param payload. Time is
    # always homogeneous in this package, so the inhomogeneous_physical_dims
    # set is automatically free of `:t` and the user-facing "spatial"
    # qualifier would be redundant.
    allowed = ReSolverFlowsBase.inhomogeneous_physical_dims(g)
    all(d in allowed for d in decomposed_physical_dims) ||
        throw(ArgumentError("decomposed_physical_dims=$decomposed_physical_dims must be a subset of inhomogeneous_physical_dims=$allowed"))
    length(unique(decomposed_physical_dims)) == K ||
        throw(ArgumentError("decomposed_physical_dims contains duplicate entries"))
    all(>(0), nhalo) ||
        throw(ArgumentError("nhalo entries must be positive along decomposed dimensions"))

    decomposed_storage_dims = ntuple(k -> ReSolverFlowsBase.storage_dim(g, decomposed_physical_dims[k]), Val(K))

    # `prod(nprocesses)` must consume every rank in `comm`.
    prod(nprocesses) == MPI.Comm_size(comm) ||
        throw(ArgumentError("prod(nprocesses)=$(prod(nprocesses)) does not match MPI.Comm_size(comm)=$(MPI.Comm_size(comm))"))

    # Require divisibility along every decomposed direction. Without it
    # the per-rank slab sizes below would differ between ranks; we keep
    # the simpler "every rank owns the same shape" contract instead, and
    # surface a clear error at construction rather than failing later
    # inside FFTW or the slicing helpers.
    for k in 1:K
        d = decomposed_storage_dims[k]
        size(g, d) % nprocesses[k] == 0 ||
            throw(ArgumentError("parent size $(size(g, d)) along $(repr(decomposed_physical_dims[k])) is not divisible by $(nprocesses[k]) ranks"))
    end


    # --------------------------------------------------------------------- #
    # Storage-order decomposition metadata
    # --------------------------------------------------------------------- #

    # Build D-length tuples for HaloArrays and the Cart communicator.
    # For each storage dim d, find its position in decomposed_storage_dims;
    # dims not in the decomposition get 1 rank and 0 halo.
    nprocesses_full = ntuple(Val(D)) do d
        idx = findfirst(==(d), decomposed_storage_dims)
        isnothing(idx) ? 1 : nprocesses[idx]
    end
    nhalo_full = ntuple(Val(D)) do d
        idx = findfirst(==(d), decomposed_storage_dims)
        isnothing(idx) ? 0 : nhalo[idx]
    end


    # --------------------------------------------------------------------- #
    # Cartesian communicator
    # --------------------------------------------------------------------- #

    # `reorder=false` keeps every rank's identity stable; `periodic=false`
    # in every dim because decomposed directions are wall-normal (non-periodic).
    _check_plain_intracommunicator(comm)
    full_cart_comm = MPI.Cart_create(comm, nprocesses_full;
                                     periodic=ntuple(_ -> false, Val(D)),
                                     reorder=false)


    # --------------------------------------------------------------------- #
    # Local interior size
    # --------------------------------------------------------------------- #

    # Per-rank interior size in storage order: decomposed dims shrink,
    # others keep the parent size.
    local_size = ntuple(d -> size(g, d) ÷ nprocesses_full[d], Val(D))


    # --------------------------------------------------------------------- #
    # Per-rank field precomputation
    # --------------------------------------------------------------------- #

    # Precompute the two per-rank fields stored on `DecomposedGrid`:
    # quadrature weights and inhomogeneous coordinate vectors. Both are read
    # in hot paths (`weights(g)` by inner-product code, `points(g)` by field
    # initialisers), so lazy slicing would allocate per call.
    cart             = _cart_topology(full_cart_comm)
    local_weights    = _local_weights(g, cart)
    local_inh_points = _local_inhomogeneous_points(g, cart)

    # --------------------------------------------------------------------- #
    # Wrapper construction
    # --------------------------------------------------------------------- #
    return DecomposedGrid(g,
                          decomposed_storage_dims, nhalo_full, local_size,
                          local_weights, local_inh_points,
                          full_cart_comm)
end

# ------------------------------------------------------------------ #
# Accessors                                                          #
# ------------------------------------------------------------------ #

"""
    parent(g::DecomposedGrid) -> ReSolverFlowsBase.AbstractGrid

Return the underlying single-domain grid wrapped by `g`. This is exactly the
object passed to [`distributed`](@ref).
"""
Base.parent(g::DecomposedGrid) = g.parent

"""
    decomposition_storage_dims(g::DecomposedGrid) -> Tuple{Int, ...}

Return the storage dimensions along which `g` is partitioned, i.e. the `DDIMS`
type parameter. These accessors live in MPIExt rather than ReSolverFlowsBase: the
core `AbstractGrid` type is agnostic to decomposition, and only the MPI wrapper
knows a grid is partitioned.
"""
decomposition_storage_dims(::DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS}) where
    {T, D, AXES, FFT_DIMS_ORDER, DDIMS} = DDIMS

"""
    decomposition_physical_dims(g::DecomposedGrid) -> Tuple{Vararg{Symbol}}

Physical-coordinate symbols for the partitioned storage dimensions of `g` —
the `Symbol` counterpart of [`decomposition_storage_dims`](@ref).
"""
decomposition_physical_dims(g::DecomposedGrid) =
    map(d -> ReSolverFlowsBase.physical_dim(g, d), decomposition_storage_dims(g))

"""
    ndecomposed_dims(g::DecomposedGrid) -> Int

Return the number of storage dimensions along which `g` is partitioned.
"""
ndecomposed_dims(g::DecomposedGrid) = length(decomposition_storage_dims(g))

# ------------------------------------------------------------------ #
# ReSolverFlowsBase grid interface — per-rank queries                          #
# ------------------------------------------------------------------ #

"""
    size(g::DecomposedGrid) -> NTuple{D, Int}

Return the **local interior** storage size on this rank — i.e. the size
of the data each rank actually owns, not the global grid size.

Along each storage dim `d`:

  - if `d ∈ DDIMS` (a decomposed dim), the result is
    `size(parent(g), d) ÷ nprocesses[k]`, where `k` is `d`'s position in
    `DDIMS`;
  - otherwise it equals `size(parent(g), d)` (every rank owns the full
    extent along non-decomposed dims).

Halo cells are **not** included; use `nhalo(g)` to recover them.
The global undecomposed size is `global_size(g) == size(parent(g))`.
"""
# The local interior size is baked into the type parameter `S` at
# construction time (see the `local_size` block in `distributed`), so this
# accessor is a compile-time constant.
Base.size(::DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S}) where
    {T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S} = S

"""
    points(g::DecomposedGrid; dealias=false) -> Tuple

Return per-rank coordinate arrays for `g`.

The parent's `points(parent; dealias)` provides global broadcastable
coordinate arrays in storage-dim order. For FFT-transformed dims the
parent's entry is kept as-is, since FFT dims are global by construction
and the parent computes them correctly for either `dealias` setting.
For inhomogeneous dims the per-rank vector is read from `g.inh_points`
(cached at construction by `distributed(...)`) and reshaped to broadcast
shape — no slicing or copying happens here.
"""
function ReSolverFlowsBase.points(g::DecomposedGrid{T, D}; dealias::Bool=false) where {T, D}
    points = collect(ReSolverFlowsBase.points(g.parent; dealias=dealias))
    for (k, d) in pairs(ReSolverFlowsBase.inhomogeneous_storage_dims(g))
        points[d] = _broadcast_axis(g.inh_points[k], d, Val(D))
    end
    return Tuple(points)
end

"""
    weights(g::DecomposedGrid) -> AbstractArray

Return the per-rank quadrature weights for `g`.

The returned array is the local slice of `weights(parent(g))`: axes whose
inhomogeneous storage dim is in `DDIMS` are truncated to this rank's local
range, the others are kept in full.

The slice is computed **once** at construction time and stored on the grid
(`g.weights`). Inner-product loops (`project!`, `dot`, weighted norms) can
therefore call `weights(g)` without allocating.
"""
ReSolverFlowsBase.weights(g::DecomposedGrid) = g.weights

# ------------------------------------------------------------------ #
# ReSolverFlowsBase grid interface — pure parent delegations                   #
# ------------------------------------------------------------------ #

"""
    wavenumber_scale(g::DecomposedGrid, phys_dim::Symbol) -> Real
    wavenumber_scale(g::DecomposedGrid, stor_dim::Integer) -> Real

Return the wavenumber scale for an FFT-transformed direction.

`phys_dim` is the user-facing `Symbol` form (`:x`, `:y`, `:z`,
`:t`). `stor_dim::Integer` is the internal contract ReSolverFlowsBase reaches
for inside its `dd!(out, u, ::Val{STORAGE_DIM})` primitive.

Decomposition does not change which physical period a transformed
direction covers, so both forms delegate to `parent(g)` after
translating to the storage axis as needed.
"""
ReSolverFlowsBase.wavenumber_scale(g::DecomposedGrid, phys_dim::Symbol) =
    ReSolverFlowsBase.wavenumber_scale(g.parent, ReSolverFlowsBase.storage_dim(g, phys_dim))

ReSolverFlowsBase.wavenumber_scale(g::DecomposedGrid, stor_dim::Int) =
    ReSolverFlowsBase.wavenumber_scale(g.parent, stor_dim)

"""
    growto(g::DecomposedGrid, target_size) -> DecomposedGrid

Return a `DecomposedGrid` whose parent has been resized to `target_size`.

The parent is regrown via `ReSolverFlowsBase.growto(parent(g), target_size)` and the
result is re-wrapped with the **same** communicator, decomposition dims, and
halo widths as `g`. Only FFT-transformed dims change size, so the
divisibility precondition along the decomposed (inhomogeneous) dims still
holds.
"""
function ReSolverFlowsBase.growto(g::DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S},
                        target_size::NTuple{N, Int}) where
        {T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, N}
    new_parent = ReSolverFlowsBase.growto(g.parent, target_size)

    # `growto` only resizes FFT-transformed dims. Decomposed dims keep
    # their per-rank size unchanged; the new local size therefore takes
    # the current rank's size on decomposed axes and the new parent's
    # global size everywhere else.
    new_local_size = ntuple(Val(D)) do d
        d in DDIMS ? size(g, d) : size(new_parent, d)
    end

    # Weights and inhomogeneous points only depend on inhomogeneous
    # storage dims, which `growto` does not touch — reuse the caches as-is.
    return DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER,
                           DDIMS, NHALO, new_local_size, typeof(new_parent),
                           typeof(g.weights), typeof(g.inh_points), typeof(g.comm)}(
        new_parent, g.comm, g.weights, g.inh_points)
end

"""
    convert(::Type{T}, g::DecomposedGrid) -> DecomposedGrid

Convert `g` to scalar element type `T`.

The parent grid is converted via `Base.convert(T, parent(g))`; the
cached local weights and inhomogeneous coordinate vectors are converted
to `T` as well. The Cartesian communicator, decomposition metadata, halo
widths, and local storage size are preserved. Returns `g` unchanged if
it already has eltype `T`.
"""
Base.convert(::Type{T}, g::DecomposedGrid{T}) where {T<:Real} = g

function Base.convert(::Type{T}, g::DecomposedGrid{T0, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S}) where
        {T<:Real, T0<:Real, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S}
    new_parent = Base.convert(T, g.parent)
    new_weights = T.(g.weights)
    new_inh_points = map(v -> T.(v), g.inh_points)
    return DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, typeof(new_parent),
                          typeof(new_weights), typeof(new_inh_points), typeof(g.comm)}(
        new_parent, g.comm, new_weights, new_inh_points)
end

# ------------------------------------------------------------------ #
# MPIExt decomposed-grid interface                                   #
# ------------------------------------------------------------------ #

"""
    comm(g::DecomposedGrid) -> MPI.Comm

Return the Cartesian MPI communicator stored on `g`.

This is the communicator created internally by [`distributed`](@ref) via
`MPI.Cart_create` — not the plain communicator the caller passed in. It
carries the full-Cartesian topology that HaloArrays requires and is the
object `HaloArray` fields constructed from `g` communicate through.
"""
comm(g::DecomposedGrid) = g.comm

# Keep halo widths in the type. They determine the HaloArray storage layout and
# the interior/boundary split used by derivative kernels, so making them a
# compile-time property mirrors HaloArrays and keeps `nhalo(g)` allocation
# free. Runtime decomposition metadata, if needed later, should be added as
# separate fields without moving `NHALO` out of the type.
"""
    nhalo(g::DecomposedGrid) -> NTuple{D, Int}
    nhalo(g::DecomposedGrid, phys_dim::Symbol) -> Int
    nhalo(g::DecomposedGrid, stor_dim::Int) -> Int

Return the halo width(s) for `g`.

The no-arg form returns the full length-`D` tuple in storage-dim order.
Entries outside the decomposed dims are always 0; entries for decomposed
dims are the values supplied to [`distributed`](@ref). Baked into type
parameter `NHALO`, so the no-arg form is a compile-time constant.
"""
nhalo(::DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO}) where
    {T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO} = NHALO

nhalo(g::DecomposedGrid, phys_dim::Symbol) =
    nhalo(g)[ReSolverFlowsBase.storage_dim(g, phys_dim)]

nhalo(g::DecomposedGrid, stor_dim::Int) = nhalo(g)[stor_dim]

"""
    global_size(g::DecomposedGrid) -> NTuple{D, Int}
    global_size(g::DecomposedGrid, phys_dim::Symbol) -> Int
    global_size(g::DecomposedGrid, stor_dim::Int) -> Int

Return the global (undecomposed) storage size of `g`, i.e. `size(parent(g))`.

Contrast with `size(g)` / `local_size(g)`, which return the per-rank
interior size on this rank.
"""
global_size(g::DecomposedGrid) = size(g.parent)

global_size(g::DecomposedGrid, phys_dim::Symbol) =
    global_size(g)[ReSolverFlowsBase.storage_dim(g, phys_dim)]
    
global_size(g::DecomposedGrid, stor_dim::Int) = global_size(g)[stor_dim]

"""
    derivative_matrix(g::DecomposedGrid, stor_dim::Int,
                      ::Val{ORDER}, [mode])

Return the FD differentiation matrix of order `ORDER` along storage
dimension `stor_dim`, plain or, with `DiscreteAdjoint()`, its discrete adjoint.

Downstream single-domain grid types implement this on `parent(g)`:
```
ReSolverFlowsBase.derivative_matrix(::ParentType, stor_dim::Int, ::Val{ORDER}, mode::AbstractDerivativeMode)
```
"""
ReSolverFlowsBase.derivative_matrix(g::DecomposedGrid,
                   stor_dim::Int,
                           ::Val{ORDER},
                       mode::AbstractDerivativeMode) where {ORDER} =
    ReSolverFlowsBase.derivative_matrix(g.parent, stor_dim, Val(ORDER), mode)

# Neighbour predicates. Periodic directions always have neighbours;
# non-periodic ones do not at the first/last rank.
function _has_neighbor(g::DecomposedGrid, stor_dim::Int, step::Int)
    cart = _cart_topology(comm(g))
    cart.nprocesses[stor_dim] == 1 && return false
    return cart.periods[stor_dim] ||
           (step < 0 ? cart.coords[stor_dim] > 0 :
                       cart.coords[stor_dim] < cart.nprocesses[stor_dim] - 1)
end

_has_lower_neighbor(g::DecomposedGrid, stor_dim::Int) = _has_neighbor(g, stor_dim, -1)
_has_upper_neighbor(g::DecomposedGrid, stor_dim::Int) = _has_neighbor(g, stor_dim,  1)

# ------------------------------------------------------------------ #
# Public per-rank topology / range accessors                         #
# ------------------------------------------------------------------ #

"""
    comm_size(g::DecomposedGrid) -> Int

Total number of MPI ranks in `comm(g)`.
"""
comm_size(g::DecomposedGrid) = MPI.Comm_size(comm(g))

"""
    comm_rank(g::DecomposedGrid) -> Int

Zero-based rank of the calling process in `comm(g)`.
"""
comm_rank(g::DecomposedGrid) = MPI.Comm_rank(comm(g))

"""
    local_size(g::DecomposedGrid) -> NTuple{D,Int}
    local_size(g::DecomposedGrid, phys_dim::Symbol) -> Int
    local_size(g::DecomposedGrid, stor_dim::Int) -> Int

Return the local interior storage size owned by this rank.

Identical to `size(g)` (and its scalar overloads). Provided as an
explicit named counterpart to `global_size` for call sites where the
local/global distinction matters. Halo cells are not included.
"""
local_size(g::DecomposedGrid) = size(g)

local_size(g::DecomposedGrid, phys_dim::Symbol) =
    size(g, ReSolverFlowsBase.storage_dim(g, phys_dim))
local_size(g::DecomposedGrid, stor_dim::Int) = size(g, stor_dim)

"""
    local_physical_size(g::DecomposedGrid; dealias=false) -> NTuple{D,Int}

Return the per-rank physical-space interior size. With `dealias=true`, the FFT
storage dimensions are padded by the same 3/2-rule used by `ReSolverFlowsBase.Field`.
Halo cells are not included.
"""
local_physical_size(g::DecomposedGrid; dealias::Bool=false) =
    dealias ? ReSolverFlowsBase.get_padded_size(local_size(g), ReSolverFlowsBase.fft_storage_dims(g)) :
              local_size(g)

"""
    local_transform_size(g::DecomposedGrid) -> NTuple{D,Int}

Return the per-rank spectral interior size corresponding to
`local_size(g)`. The rfft storage dimension is halved in FFTW's
real-to-complex layout; all other dimensions keep their local extent.
Halo cells are not included.
"""
local_transform_size(g::DecomposedGrid) =
    ReSolverFlowsBase._get_transform_size(local_size(g), ReSolverFlowsBase.rfft_storage_dim(g))

"""
    global_first_index(g::DecomposedGrid, phys_dim::Symbol) -> Int
    global_first_index(g::DecomposedGrid, stor_dim::Int) -> Int

1-based global index corresponding to local index 1 along the given direction.
Assumes a uniform Cartesian decomposition; passed to FDGrids `mul!` so
local array indices map to the correct rows of the global FD matrix.
"""
function global_first_index(g::DecomposedGrid, phys_dim::Symbol)
    global_first_index(g, ReSolverFlowsBase.storage_dim(g, phys_dim))
end
function global_first_index(g::DecomposedGrid, stor_dim::Int)
    cart = _cart_topology(comm(g))
    cart.nprocesses[stor_dim] == 1 && return 1
    n_local = local_size(g, stor_dim)
    first   = cart.coords[stor_dim] * n_local + 1
    first + n_local - 1 <= global_size(g, stor_dim) ||
        throw(ArgumentError("local size exceeds global size along storage dim $stor_dim"))
    return first
end

"""
    local_interior_range(g::DecomposedGrid, phys_dim::Symbol) -> UnitRange{Int}
    local_interior_range(g::DecomposedGrid, stor_dim::Int) -> UnitRange{Int}

Local row indices along the given direction that can be differentiated before
halo communication completes. Excludes `nhalo(g)[stor_dim]` rows on each side
that borders a neighbouring rank; rows next to a physical domain boundary are
always included.
"""
function local_interior_range(g::DecomposedGrid, phys_dim::Symbol)
    local_interior_range(g, ReSolverFlowsBase.storage_dim(g, phys_dim))
end
function local_interior_range(g::DecomposedGrid, stor_dim::Int)
    h = nhalo(g, stor_dim)
    n = local_size(g, stor_dim)
    first = _has_lower_neighbor(g, stor_dim) ? h + 1 : 1
    last  = _has_upper_neighbor(g, stor_dim) ? n - h : n
    first <= last ||
        throw(ArgumentError("halo width $h leaves no interior points along storage dim $stor_dim (local size $n)"))
    return first:last
end

"""
    local_boundary_ranges(g::DecomposedGrid, phys_dim::Symbol) -> NTuple{2,UnitRange{Int}}
    local_boundary_ranges(g::DecomposedGrid, stor_dim::Int) -> NTuple{2,UnitRange{Int}}

Local row ranges along the given direction that require fresh halo data before
they can be differentiated. Returns `(lower, upper)`; an entry is the empty
range `1:0` when no communication is required on that side.
"""
function local_boundary_ranges(g::DecomposedGrid, phys_dim::Symbol)
    local_boundary_ranges(g, ReSolverFlowsBase.storage_dim(g, phys_dim))
end
function local_boundary_ranges(g::DecomposedGrid, stor_dim::Int)
    h = nhalo(g, stor_dim)
    n = local_size(g, stor_dim)
    lower = _has_lower_neighbor(g, stor_dim) ? (1:h) : (1:0)
    upper = _has_upper_neighbor(g, stor_dim) ? ((n - h + 1):n) : (1:0)
    return lower, upper
end

# Val{N} delegators — allow callers holding a compile-time Val{STORAGE_DIM}
# (e.g. from physical_to_storage_dim) to pass it directly without unpacking.
local_interior_range(g::DecomposedGrid, ::Val{N}) where {N} = local_interior_range(g, N)
local_boundary_ranges(g::DecomposedGrid, ::Val{N}) where {N} = local_boundary_ranges(g, N)

# ------------------------------------------------------------------ #
# Slicing helpers                                                    #
# ------------------------------------------------------------------ #

@inline _broadcast_axis(v, d::Int, ::Val{D}) where {D} =
    reshape(v, ntuple(j -> j == d ? length(v) : 1, Val(D)))

function _local_weights(g::ReSolverFlowsBase.AbstractGrid, cart)
    parent_weights = ReSolverFlowsBase.weights(g)
    inh_storage_dims = ReSolverFlowsBase.inhomogeneous_storage_dims(g)
    slices = ntuple(length(inh_storage_dims)) do axis
        _local_axis_indices(size(parent_weights, axis), inh_storage_dims[axis], cart)
    end
    return parent_weights[slices...]
end

function _local_inhomogeneous_points(g::ReSolverFlowsBase.AbstractGrid, cart)
    parent_points = ReSolverFlowsBase.points(g)
    return map(ReSolverFlowsBase.inhomogeneous_storage_dims(g)) do stor_dim
        global_vec = Vector(vec(parent_points[stor_dim]))
        axis_inds  = _local_axis_indices(length(global_vec), stor_dim, cart)
        axis_inds isa Colon ? global_vec : global_vec[axis_inds]
    end
end

@inline function _local_axis_indices(n_global::Int, stor_dim::Int, cart)
    cart.nprocesses[stor_dim] == 1 && return Colon()
    return _local_range(n_global, stor_dim, cart.coords, cart.nprocesses)
end

# Shared range arithmetic for uniformly decomposed axes. Constructor-time
# slicing and finite-difference indexing use the same helper so the local-range
# formula has one home.
"""
    _local_range(n_global::Int, k::Int, coords::Tuple, nprocesses::Tuple) -> UnitRange{Int}

Return this rank's owned index range within a global axis of length
`n_global` that is decomposed along Cartesian direction `k`.

`coords[k]` is this rank's zero-based coordinate and `nprocesses[k]` is
the rank count along direction `k`. A uniform decomposition is assumed
(`n_global % nprocesses[k] == 0`), checked by the caller.

# Example

A 12-row global axis split 3 ways, this rank at Cartesian coord 1:

```julia
_local_range(12, 1, (1,), (3,))  # → 5:8
```
"""
@inline function _local_range(n_global::Int, k::Int,
                              coords::Tuple, nprocesses::Tuple)
    n_local = n_global ÷ nprocesses[k]
    first   = coords[k] * n_local + 1
    return first:(first + n_local - 1)
end

"""
    _cart_topology(comm::MPI.Comm) -> NamedTuple

Return a NamedTuple `(nprocesses, periods, coords)` describing the
Cartesian topology of `comm`:

  - `nprocesses`: number of ranks along each Cartesian dimension.
  - `periods`: periodicity flags (all `false` for wall-normal decompositions).
  - `coords`: this rank's zero-based Cartesian coordinates.
"""
@inline function _cart_topology(comm::MPI.Comm)
    nprocesses, periods, coords = MPI.Cart_get(comm)
    return (nprocesses = Tuple(Int.(nprocesses)),
            periods    = Tuple(Bool.(periods)),
            coords     = Tuple(Int.(coords)))
end
