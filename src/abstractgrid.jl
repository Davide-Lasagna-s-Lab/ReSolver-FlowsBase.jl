# Abstract interface for all computational grids in ReSolverFlowsBase.
#
# `AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}` is the single point of coupling between
# field types (`FTField`, `Field`, `VectorField`, `ProjectedField`) and the
# underlying geometry.  Every field stores a reference to its grid and
# dispatches size queries, FFT index ranges, quadrature weights, and coordinate
# arrays through this interface.
#
# Downstream packages (e.g. Resolver-ChannelFlow.jl) implement a concrete grid subtype
# by defining a small set of required methods (`size`, `points`,
# `wavenumber_scale`, `weights`).  All other grid-aware behaviour - spectral
# loop generation, transform sizing, base-flow injection, derivative operators -
# is derived automatically from the compile-time type parameters without
# any runtime overhead.
#
# The four type parameters encode:
#   T              - real scalar type (Float64 by default)
#   D              - number of array dimensions
#   AXES           - 4-tuple mapping physical coordinates (x,y,z,t) to storage dims
#   FFT_DIMS_ORDER - ordered tuple of array dimensions that are FFT-transformed;
#                    FFT_DIMS_ORDER[1] is always the rfft dimension
#
# Whether a grid is domain-decomposed for distributed execution is *not* recorded
# in the grid type. Downstream MPI packages (e.g. MPIExt) wrap a serial
# grid in their own `DecomposedGrid` type that holds the partition metadata; the
# core abstract grid stays agnostic to decomposition.

"""
    AbstractGrid{T, D, AXES, FFT_DIMS_ORDER} where {T<:Real}

Abstract type that represents a generic computational grid of a
`D`-dimensional domain.

Type parameters:
- `T`: scalar real type used by physical-space fields on this grid.
- `D`: number of array dimensions.
- `AXES`: four-entry Cartesian axis layout `(x_dim, y_dim, z_dim, t_dim)`.
  ReSolverFlowsBase assumes this tuple always has four entries, one for each logical
  Cartesian coordinate.  Each entry is the array dimension occupied by that
  coordinate, or `nothing` when the coordinate is absent.  The non-`nothing`
  entries should be a permutation of `1:D`.  For example, a three-dimensional
  grid stored as `(y, x, z)` should use `AXES = (2, 1, 3, nothing)`.
- `FFT_DIMS_ORDER`: tuple of statistically homogeneous array dimensions. These are
  transformed by FFTs; `FFT_DIMS_ORDER[1]` is the rfft dimension.

# Required downstream methods

Concrete grid packages must implement:

- `Base.size(grid)` returning an `NTuple{D, Int}`.
- `points(grid; dealias=false)` returning one coordinate array per dimension,
   in array-dimension order.
- `wavenumber_scale(grid, dim)` for each transformed dimension in
  `fft_storage_dims(grid)`.
- `weights(grid)` returning quadrature weights for the inhomogeneous
  dimensions.
- `_inhomogeneous_dd!(out, u, Val(dim), [DiscreteAdjoint()])` for first derivatives
  along each non-Fourier storage dimension.
- `inhomogeneous_laplacian!(out, u, [DiscreteAdjoint()])` for the sum of spatial
  second derivatives along non-Fourier dimensions.
- `Base.convert(::Type{S}, grid)` if fields should support changing scalar
   precision via `similar(field, S)`.

# Optional downstream methods

Implement `growto(grid, target_size)` if the package supports changing the
homogeneous resolution.  Implementing grid growth is also required for
`FFT(field, target_size)` and `IFFT(ftfield, target_size)`.

Implement `derivative_matrix(grid, storage_dim, Val(order), Direct())` and
`derivative_matrix(grid, storage_dim, Val(order), DiscreteAdjoint())` when a
decomposed-grid wrapper needs to retrieve local first- and second-derivative
stencils from its serial parent. The two methods may return different concrete
types; dispatch on the tag keeps the accessor type stable without any runtime
branch.

# Provided defaults (override only if needed)

- `Base.size(grid, dim)`: indexes `size(grid)`.
- `fft_norm(grid)`: returns `map(d -> size(grid, d), fft_storage_dims(grid))`, the mode
  counts used to normalise forward transforms.
- `transform_size(grid)`: returns the size of the corresponding `FTField`.
"""
abstract type AbstractGrid{T<:Real, D, AXES, FFT_DIMS_ORDER} end

"""
    fft_storage_dims(grid::AbstractGrid) -> Tuple{Int, …}

Return the tuple of FFT-transformed array dimensions in transform order,
i.e. the grid type parameter `FFT_DIMS_ORDER`.

`fft_storage_dims(g)[1]` is the rfft dimension (only non-negative wavenumbers are
stored); the remaining entries are full-spectrum complex FFT dimensions.
"""
fft_storage_dims(::AbstractGrid{<:Any, <:Any, <:Any, FFT_DIMS_ORDER}) where {FFT_DIMS_ORDER} = FFT_DIMS_ORDER

"""
    spatial_fft_storage_dims(grid::AbstractGrid) -> Tuple{Int, …}

Return the FFT-transformed array dimensions that correspond to spatial
coordinates, i.e. `fft_dims(grid)` with the temporal dimension excluded.

For a steady grid this is the same tuple as [`fft_storage_dims`](@ref).  For a
space-time grid whose logical time coordinate is also transformed, the time
dimension is omitted.
"""
@generated function spatial_fft_storage_dims(::AbstractGrid{<:Any, <:Any, AXES, FFT_DIMS_ORDER}) where {AXES, FFT_DIMS_ORDER}
    :($(Tuple(d for d in FFT_DIMS_ORDER if d != AXES[4])))
end

"""
    inhomogeneous_storage_dims(grid::AbstractGrid) -> Tuple

Return the array dimensions that are NOT transformed by FFTs, i.e. the
complement of `fft_storage_dims(grid)` within `1:D`.  These are the directions over
which quadrature weights are needed for inner products.
"""
@generated function inhomogeneous_storage_dims(::AbstractGrid{<:Any, D, <:Any, FFT_DIMS_ORDER}) where {D, FFT_DIMS_ORDER}
    :($(Tuple(d for d in 1:D if d ∉ FFT_DIMS_ORDER)))
end

"""
    spatial_inhomogeneous_storage_dims(grid::AbstractGrid) -> Tuple

Return the inhomogeneous array dimensions that correspond to spatial (not
temporal) coordinates - i.e. dimensions in `inhomogeneous_storage_dims(grid)` that
are not the time axis `AXES[4]`.  These are the dimensions for which a
finite-difference or collocation derivative must be applied.
"""
@generated function spatial_inhomogeneous_storage_dims(::AbstractGrid{<:Any, D, AXES, FFT_DIMS_ORDER}) where {D, AXES, FFT_DIMS_ORDER}
    :($(Tuple(d for d in 1:D if d ∉ FFT_DIMS_ORDER && d != AXES[4])))
end

"""
    rfft_storage_dim(grid::AbstractGrid) -> Int

Return the array dimension that is transformed by the real-to-complex FFT,
i.e. the first entry of `fft_storage_dims(grid)`.

`rfft_storage_dim(g)` is the rfft dimension (only non-negative wavenumbers are
stored); the remaining entries are full-spectrum complex FFT dimensions.
"""
rfft_storage_dim(::AbstractGrid{<:Any,<:Any,<:Any,FFT_DIMS_ORDER}) where {FFT_DIMS_ORDER} = FFT_DIMS_ORDER[1]

# ------------------------------------------------------------------------- #
# Physical-direction variants                                                #
# ------------------------------------------------------------------------- #

"""
    fft_physical_dims(grid::AbstractGrid) -> Tuple{Vararg{Symbol}}

Physical-coordinate symbols for the FFT-transformed storage dimensions
of `grid`, in transform order - the `Symbol` counterpart of
[`fft_storage_dims`](@ref).

`fft_physical_dims(g)[1]` is the physical coordinate that is rfft'd; the
remaining entries are the signed-FFT coordinates.
"""
fft_physical_dims(g::AbstractGrid) =
    map(d -> physical_dim(g, d), fft_storage_dims(g))

"""
    spatial_fft_physical_dims(grid::AbstractGrid) -> Tuple{Vararg{Symbol}}

Physical-coordinate symbols for the spatial FFT-transformed storage
dimensions of `grid` - the `Symbol` counterpart of
[`spatial_fft_storage_dims`](@ref). A transformed logical time
coordinate is excluded.
"""
spatial_fft_physical_dims(g::AbstractGrid) =
    map(d -> physical_dim(g, d), spatial_fft_storage_dims(g))

"""
    inhomogeneous_physical_dims(grid::AbstractGrid) -> Tuple{Vararg{Symbol}}

Physical-coordinate symbols for the inhomogeneous (non-FFT) storage
dimensions of `grid` - the `Symbol` counterpart of
[`inhomogeneous_storage_dims`](@ref).
"""
inhomogeneous_physical_dims(g::AbstractGrid) =
    map(d -> physical_dim(g, d), inhomogeneous_storage_dims(g))

"""
    spatial_inhomogeneous_physical_dims(grid::AbstractGrid) -> Tuple{Vararg{Symbol}}

Physical-coordinate symbols for the spatial inhomogeneous storage
dimensions of `grid` - the `Symbol` counterpart of
[`spatial_inhomogeneous_storage_dims`](@ref). The time coordinate is
excluded.
"""
spatial_inhomogeneous_physical_dims(g::AbstractGrid) =
    map(d -> physical_dim(g, d), spatial_inhomogeneous_storage_dims(g))

"""
    rfft_physical_dim(grid::AbstractGrid) -> Symbol

Physical-coordinate symbol for the real-to-complex FFT storage dimension
- the `Symbol` counterpart of [`rfft_storage_dim`](@ref). Equal to
`fft_physical_dims(grid)[1]`.
"""
rfft_physical_dim(g::AbstractGrid) = physical_dim(g, rfft_storage_dim(g))

"""
    one_or_two(I::CartesianIndex, g::AbstractGrid)      -> Int
    one_or_two(I::CartesianIndex, ::Val{FFT_DIMS_ORDER}) -> Int

Return `1` if the rfft storage index in `I` corresponds to the zero
wavenumber, and `2` otherwise.  Used to apply the correct Hermitian
multiplicity weight in inner products and norms, which are iterated
with `CartesianIndices`.

The grid form extracts `FFT_DIMS_ORDER` from `g` and delegates to the
`Val` form; the `Val` form is the primary implementation and can be used
without a grid instance.
"""
one_or_two(I::CartesianIndex, g::AbstractGrid) = one_or_two(I, Val(fft_storage_dims(g)))
one_or_two(I::CartesianIndex, ::Val{FFT_DIMS_ORDER}) where {FFT_DIMS_ORDER} =
    I[FFT_DIMS_ORDER[1]] == 1 ? 1 : 2


"""
    inhomogeneous_indices(I::CartesianIndex, g::AbstractGrid)     -> Tuple
    inhomogeneous_indices(I::CartesianIndex, ::Val{FFT_DIMS_ORDER}) -> Tuple

Extract the non-FFT indices from the Cartesian index `I` and return them
as a tuple, i.e. the entries of `I` along the inhomogeneous dimensions.
"""
inhomogeneous_indices(I::CartesianIndex, g::AbstractGrid) = inhomogeneous_indices(I, Val(fft_storage_dims(g)))

@generated function inhomogeneous_indices(I::CartesianIndex{D}, 
                                           ::Val{FFT_DIMS_ORDER}) where {D, FFT_DIMS_ORDER}
    idxs = [d for d in 1:D if d ∉ FFT_DIMS_ORDER]
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, (:(I[$idx]) for idx in idxs)...))
end

"""
    homogeneous_indices(I::CartesianIndex, g::AbstractGrid) -> Tuple

Extract the FFT indices from the Cartesian index `I` and return them as a
tuple, i.e. the entries of `I` along the homogeneous dimensions.
"""
homogeneous_indices(I::CartesianIndex, g::AbstractGrid) = homogeneous_indices(I, Val(fft_storage_dims(g)))

@generated function homogeneous_indices(I::CartesianIndex{D},
                                         ::Val{FFT_DIMS_ORDER}) where {D, FFT_DIMS_ORDER}
    idxs = [d for d in 1:D if d ∈ FFT_DIMS_ORDER]
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, (:(I[$idx]) for idx in idxs)...))
end

"""
    combine_indices(grid::AbstractGrid, Inh::CartesianIndex{Dnh}, Ih::CartesianIndex{Dh}) -> Tuple
    combine_indices(::Val{FFT_DIMS_ORDER}, Inh::CartesianIndex{Dnh}, Ih::CartesianIndex{Dh}) -> Tuple

Combine inhomogeneous indices `Inh` and homogeneous indices `Ih` into a
single index tuple of length `D = Dnh + Dh`, interleaving them according to
`FFT_DIMS_ORDER`.  Dimensions in `FFT_DIMS_ORDER` draw from `Ih`; all
others draw from `Inh`.
"""
combine_indices(   ::AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}, 
                Inh::CartesianIndex, 
                 Ih::CartesianIndex) where {T, D, AXES, FFT_DIMS_ORDER} =
    combine_indices(Val(FFT_DIMS_ORDER), Inh, Ih)

@generated function combine_indices(   ::Val{FFT_DIMS_ORDER},
                                    Inh::CartesianIndex{Dnh},
                                     Ih::CartesianIndex{Dh}) where {Dnh, Dh, FFT_DIMS_ORDER}
    D = Dnh + Dh
    inds = []
    k = 1
    i = 1
    for d in 1:D
        if d ∈ FFT_DIMS_ORDER
            push!(inds, :(Ih[$k]))
            k += 1
        else
            push!(inds, :(Inh[$i]))
            i += 1
        end
    end
    return :(@inline; ($(inds...),))
end

"""
    combine_indices(grid, ::Colon, Ih) -> Tuple

Combine homogeneous indices `Ih` with `Colon` for the non-homogeneous dimensions, returning a tuple 
of length `D` suitable for indexing into a `FTField`.  This is mostly used to create views of `FTField`s
that span all indices at a particular homogeneous index.
"""
@generated function combine_indices(::AbstractGrid{T,D,AXES,FFT_DIMS_ORDER}, ::Colon, Ih) where {T,D,AXES,FFT_DIMS_ORDER}
    inds = []
    k = 1
    for d in 1:D
        if d ∈ FFT_DIMS_ORDER
            push!(inds, :(Ih[$k]))
            k += 1
        else
            push!(inds, :(Colon()))
        end
    end
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, inds...))
end

"""
    storage_dim(grid, physical_dim) -> Int | nothing

Return the storage-array dimension used for physical direction `physical_dim`.

`physical_dim` may be one of `:x`, `:y`, `:z`, or `:t`, or `Val` of one of those
symbols. A missing physical direction returns `nothing`, which lets derivative
wrappers for absent coordinates become no-ops.

# Examples

For `AXES = (2, 1, 3, nothing)`, `storage_dim(grid, :x) == 2`,
`storage_dim(grid, :y) == 1`, and `storage_dim(grid, :t) === nothing`.
"""
# `where {T<:Real, D}` matches the constraint on the abstract type's
# first parameter explicitly; without it Julia (1.12+) sees the literal
# `Val{:x}` methods as ambiguous with the catch-all `Val{DIM}` fallback
# below when the input grid passes through an indirect type hierarchy.
storage_dim(::AbstractGrid{T, D, AXES} where {T<:Real, D}, ::Val{:x}) where {AXES} = AXES[1]
storage_dim(::AbstractGrid{T, D, AXES} where {T<:Real, D}, ::Val{:y}) where {AXES} = AXES[2]
storage_dim(::AbstractGrid{T, D, AXES} where {T<:Real, D}, ::Val{:z}) where {AXES} = AXES[3]
storage_dim(::AbstractGrid{T, D, AXES} where {T<:Real, D}, ::Val{:t}) where {AXES} = AXES[4]
storage_dim(grid::AbstractGrid, physical_dim::Symbol) = storage_dim(grid, Val(physical_dim))
function storage_dim(::AbstractGrid, ::Val{DIM}) where {DIM}
    throw(ArgumentError("physical direction must be one of :x, :y, :z, or :t; got $(repr(DIM))"))
end

"""
    physical_dim(grid, storage_dim) -> Symbol | nothing

Return the physical direction represented by storage-array dimension `storage_dim`.

`storage_dim` may be an integer storage dimension or `Val` of one. `nothing` is
propagated so callers can normalize optional directions.

# Examples

For `AXES = (2, 1, 3, nothing)`, `physical_dim(grid, 1) == :y`,
`physical_dim(grid, 2) == :x`, and `physical_dim(grid, nothing) === nothing`.
"""
physical_dim(::AbstractGrid, ::Nothing) = nothing
physical_dim(::AbstractGrid, ::Val{nothing}) = nothing
physical_dim(grid::AbstractGrid, storage_dim::Integer) = physical_dim(grid, Val(storage_dim))

@generated function physical_dim(::AbstractGrid{<:Any, D, AXES}, ::Val{DIM}) where {D, AXES, DIM}
    isnothing(DIM) && return :(nothing)
    if DIM isa Symbol
        if DIM in (:x, :y, :z, :t)
            return QuoteNode(DIM)
        end
        msg = "physical direction must be one of :x, :y, :z, or :t; got $(repr(DIM))"
        return :(throw(ArgumentError($msg)))
    end
    if !(DIM isa Integer)
        msg = "dimension must be a physical direction, storage integer, or nothing; got $(repr(DIM))"
        return :(throw(ArgumentError($msg)))
    end
    if !(1 <= DIM <= D)
        msg = "storage dimension $DIM is outside 1:$D"
        return :(throw(ArgumentError($msg)))
    end
    directions = (:x, :y, :z, :t)
    i = findfirst(==(DIM), AXES)
    isnothing(i) && return :(nothing)
    return QuoteNode(directions[i])
end

"""
    physical_to_storage_dim(grid, physical_dim) -> Val{N} | Val{nothing}

Return the storage-array dimension for `physical_dim`, wrapped in a `Val`
so it can feed type-stably into other `Val`-dispatched functions (e.g.
`FDGrids.mul!`).

`physical_dim` may be a coordinate symbol (`:x`, `:y`, `:z`, `:t`) or a
`Val` of one. The result is `Val{N}` where `N` is the storage dimension
hosting that physical direction, or `Val{nothing}` when the grid omits
that coordinate.

This is a `@generated` function: the `AXES` lookup is folded to a
literal `Val(N)` at compile time, so the call has no runtime cost at
literal call sites such as `physical_to_storage_dim(g, Val(:x))`. Use it
in preference to `Val(storage_dim(g, dim))`, which constructs the `Val`
from a runtime `Int` and breaks the type-stable chain.

# Examples

For `AXES = (2, 1, 3, nothing)`, `physical_to_storage_dim(g, Val(:x))`
returns `Val(2)`, `physical_to_storage_dim(g, Val(:y))` returns
`Val(1)`, and `physical_to_storage_dim(g, Val(:t))` returns
`Val(nothing)`.
"""
@generated function physical_to_storage_dim(::AbstractGrid{<:Any, <:Any, AXES},
                                            ::Val{PHYSICAL_DIM}) where {AXES, PHYSICAL_DIM}
    idx = if PHYSICAL_DIM === :x; 1
      elseif PHYSICAL_DIM === :y; 2
      elseif PHYSICAL_DIM === :z; 3
      elseif PHYSICAL_DIM === :t; 4
    else
        msg = "physical direction must be :x, :y, :z, or :t; got $(repr(PHYSICAL_DIM))"
        return :(throw(ArgumentError($msg)))
    end
    storage_dim_value = AXES[idx]
    return :(Val($(QuoteNode(storage_dim_value))))
end

physical_to_storage_dim(grid::AbstractGrid, physical_dim::Symbol) =
    physical_to_storage_dim(grid, Val(physical_dim))

"""
    to_storage_order(values, grid) -> Tuple

Permute a tuple from logical coordinate order into storage/array-dimension
order.

The axis layout is read from the grid type parameter `AXES`, where `AXES[i]`
is the array dimension occupied by logical coordinate `i`.  `values` must have
the four Cartesian coordinate slots `(X, Y, Z, T)`.  Coordinates whose `AXES`
entry is `nothing` are omitted from the result.  For example, if
`values = (X, Y, Z, T)` and `AXES = (2, 1, 3, 4)`, then
`to_storage_order(values, grid)` returns `(Y, X, Z, T)`.
"""
@generated function to_storage_order(values::Tuple{Vararg{Any, 4}},
                                     ::AbstractGrid{<:Any, D, AXES}) where {D, AXES}
    present = [(coordinate, axis) for (coordinate, axis) in pairs(AXES) if !isnothing(axis)]
    order = first.(sort(present; by=last))
    return Expr(:tuple, (:(values[$coordinate]) for coordinate in order)...)
end

# ------------------ #
# required interface #
# ------------------ #
"""
    Base.convert(::Type{S}, grid::AbstractGrid{T}) -> AbstractGrid{S}

Convert `grid` to scalar real type `S`.
"""
Base.convert(::Type{S}, grid::AbstractGrid{T}) where {S, T} = throw(NotImplementedError(grid, S))

"""
    Base.size(grid::AbstractGrid) -> Dims{D}
    Base.size(grid::AbstractGrid, dim::Int) -> Int

Return the physical-space array size associated with `grid`, in array-dimension
order.  This is required by `Field`, `FTField`, `FFTPlans`, and generated
operators. Optionally provide a `dim` argument to extract the size of the grid
in a particular direction.
"""
Base.size(grid::AbstractGrid) = throw(NotImplementedError(grid))
Base.size(grid::AbstractGrid, dim::Int) = size(grid)[dim]

"""
    points(grid::AbstractGrid; dealias=false) -> Tuple

Return one coordinate array per array dimension, suitable for broadcasting a
function into a `Field`.

When `dealias=true`, transformed dimensions should use the physical padded
sizes expected by `FFTPlans(grid; dealias=true)`.
"""
points(grid::AbstractGrid; dealias=false) = throw(NotImplementedError(grid))

"""
    wavenumber_scale(grid::AbstractGrid, dim::Int) -> Real

Return the physical wavenumber scaling factor for homogeneous dimension `dim`.
For a spatial direction with period `L` the factor is `2π/L`; for a unit-period
temporal direction, return `1`.

Downstream packages must extend this for each dimension in `fft_storage_dims(grid)`.
"""
wavenumber_scale(grid::AbstractGrid, dim::Int) = throw(NotImplementedError(grid, dim))

"""
    weights(grid::AbstractGrid) -> AbstractArray

Return the quadrature weights for the inhomogeneous directions of `grid`.
The returned array has one axis per dimension in `inhomogeneous_storage_dims(grid)`,
in ascending array-dimension order: a `Vector` when there is one inhomogeneous
direction, a `Matrix` when there are two, and so on.

Used by `dot(u::FTField, v::FTField)` to weight each inhomogeneous index
combination.
"""
weights(grid::AbstractGrid) = throw(NotImplementedError(grid))


# ------------------ #
# optional interface #
# ------------------ #
"""
    growto(grid::AbstractGrid, target_size)

Return an equivalent grid with a new homogeneous resolution.

`target_size` contains one physical-space size per homogeneous direction, in
`fft_storage_dims(grid)` order. Implementations should preserve inhomogeneous
directions and return a grid whose transformed dimensions match those sizes.

This is optional.  Packages should only implement it if they support
resolution-changing transforms such as `FFT(u, target_size)` and
`IFFT(û, target_size)`.
"""
growto(grid::AbstractGrid, target_size) = throw(NotImplementedError(grid, target_size))


# ------------------ #
# derived quantities #
# ------------------ #
"""
    transform_size(g::AbstractGrid)

Size of the spectral (FTField) array: `FFT_DIMS_ORDER[1]` (rfft) becomes `(N÷2)+1`,
all other dimensions are unchanged.
"""
transform_size(g::AbstractGrid{<:Any, D, <:Any, FFT_DIMS_ORDER}) where {D, FFT_DIMS_ORDER} =
    ntuple(d -> d == FFT_DIMS_ORDER[1] ? (size(g, d) >> 1) + 1 : size(g, d), D)

"""
    fft_norm(g::AbstractGrid)

Number of grid points in each FFT dimension `FFT_DIMS_ORDER`. Used to normalise
forward transforms. Default: reads from `size(g)`.
"""
fft_norm(g::AbstractGrid) = map(d -> size(g, d), fft_storage_dims(g))

# ------------------------------------------------------------------ #
# Hermitian-symmetry enforcement on raw coefficient arrays           #
# ------------------------------------------------------------------ #

"""
    _average_complex(z1, z2) -> z

Return the unique complex number `z` such that `z` and `conj(z)` are the
nearest conjugate pair to `z1` and `z2`:

```
re(z) = (re(z1) + re(z2)) / 2   # real parts should agree → average them
im(z) = (im(z1) - im(z2)) / 2   # imaginary parts should be negatives → split the difference
```

After the call, `data[I] = z` and `data[Ineg] = conj(z)` enforce
`data[I] = conj(data[Ineg])` to floating-point precision.
"""
@inline function _average_complex(z1::T, z2::T) where {T}
    _re = 0.5 * (real(z1) + real(z2))
    _im = 0.5 * (imag(z1) - imag(z2))
    return _re + im * _im
end

"""
    apply_symmetry!(data::AbstractArray, ::Val{FFT_DIMS_ORDER}) -> data
    apply_symmetry!(data::AbstractArray, dims::Tuple)           -> data

Enforce Hermitian symmetry in-place on a raw spectral coefficient array.

# Background

A real-valued physical field `f(x)` has a Fourier transform satisfying
`F̂(-k) = conj(F̂(k))` for every wavenumber vector `k`.  When the transform
is performed with `rfft` along the first homogeneous axis (`FFT_DIMS_ORDER[1]`),
only non-negative wavenumbers are stored along that axis, so there are no
conjugate-pair redundancies there.  The remaining axes in
`FFT_DIMS_ORDER[2:end]` use a full signed FFT and store both positive and
negative wavenumbers.

The constraint `F̂(-k) = conj(F̂(k))` still applies to the signed-FFT axes,
but **only at the zero-frequency slice of the rfft axis** (storage index 1,
i.e. rfft wavenumber = 0).  At any nonzero rfft wavenumber `kx > 0` the
negative-`kx` entry is not stored, so no symmetry constraint can be applied.

# What this function does

1. **Restrict to the DC plane**: loops only over positions where the rfft axis
   index is 1 (storage index for rfft wavenumber = 0), while iterating all
   other axes freely.

2. **Find conjugate partners**: for each such index `I`, constructs `Ineg` by
   flipping each signed-FFT axis using FFTW's wrap-around convention:
   - index 1 (wavenumber 0) maps to itself (self-conjugate),
   - index `i > 1` (wavenumber `n = i - 1 > 0`) maps to `size(data, d) - i + 2`
     (the index storing wavenumber `-n`).
   Axes not in `FFT_DIMS_ORDER` (e.g. wall-normal, mode index) are kept
   identical between `I` and `Ineg` - they are not transformed.

3. **Average each pair once**: the `LI[I] <= LI[Ineg]` guard ensures each
   `{I, Ineg}` pair is visited exactly once.  The `_average_complex` helper
   computes the unique value `a` satisfying `a = (z1 + conj(z2)) / 2` in the
   real part and `a = (z1 - conj(z2)) / 2` in the imaginary part, then writes
   `data[I] = a` and `data[Ineg] = conj(a)`, enforcing `F̂(-k) = conj(F̂(k))`
   exactly.

# Arguments

- `FFT_DIMS_ORDER` (via `Val`): tuple of transformed array dimensions in
  transform order - `FFT_DIMS_ORDER[1]` is the rfft dim, the rest are
  signed-FFT dims.

No-op when `length(FFT_DIMS_ORDER) ≤ 1` (only an rfft dim, no partners).
"""
function apply_symmetry!(data::AbstractArray{<:Any, D}, ::Val{FFT_DIMS_ORDER}) where {D, FFT_DIMS_ORDER}
    # No signed-FFT dimensions: nothing to symmetrize.
    length(FFT_DIMS_ORDER) <= 1 && return data

    # Guard against offset arrays whose index-1 slot may not be the DC bin.
    Base.require_one_based_indexing(data)

    # Split FFT_DIMS_ORDER into the rfft dim and the signed-FFT dims.
    rfft_storage_dim           = FFT_DIMS_ORDER[1]
    secondary_fft_dims = Base.tail(FFT_DIMS_ORDER)

    # Build the loop ranges: the rfft axis is pinned to index 1 (DC plane);
    # all other axes (signed-FFT and untransformed) run their full range.
    ranges = ntuple(Val(D)) do d
        d == rfft_storage_dim ? (1:1) : axes(data, d)
    end

    # LinearIndices lets us guard against visiting each {I, Ineg} pair twice
    # without storing a visited-set: we process the pair only when LI[I] <= LI[Ineg].
    LI = LinearIndices(data)

    @inbounds for I in CartesianIndices(ranges)

        # Find the index of the conjugate partner: flip only the secondary_fft_dims Fourier dimensions.
        Ineg = CartesianIndex(ntuple(Val(D)) do d
            if d in secondary_fft_dims
                # zero-frequency plane: partner is the same index, so no conjugation.
                if I[d] == 1
                    return I[d]
                end
                # negative wavenumber partner index, using FFTW's wrap-around order.
                return size(data, d) - I[d] + 2
            else
                return I[d]
            end
        end)

        # only handle each pair once
        if LI[I] <= LI[Ineg]
            av = _average_complex(data[I], data[Ineg])
            data[I]    = av
            data[Ineg] = conj(av)
        end
    end

    return data
end

# Convenience overload so callers can pass a plain tuple without wrapping in Val.
apply_symmetry!(data::AbstractArray, dims::Tuple) = apply_symmetry!(data, Val(dims))
