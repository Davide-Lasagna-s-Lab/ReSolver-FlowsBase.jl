# Ordered tuple of scalar fields forming a velocity (or other multi-component) field.
#
# A `VectorField` bundles N scalar fields — either `FTField` (spectral) or
# `Field` (physical) — into a single indexable container.  All components must
# share the same grid and the same scalar field type.  The struct acts as a
# 1-D AbstractVector{S} over its components so that generic algorithms (norms,
# broadcasts, dot products) can iterate over components uniformly.
#
# The design separates concern of "multiple components" from "field data":
# VectorField only manages component ordering; all arithmetic is delegated to
# the underlying scalar fields through the broadcasting interface in
# broadcasting.jl.  This makes it easy to extend ReSolverFlowsBase to 2D (N=2) or
# arbitrary-component flows without changing any core logic.

"""
    VectorField{N, S} where {S<:Union{FTField, Field}}

An ordered tuple of `N` scalar fields, all of the same type `S` (either
`FTField` or `Field`) and all defined on the same grid.

Implements `AbstractVector{S}`, so `q[n]` retrieves the `n`-th component,
iteration works out of the box, and broadcasting propagates component-wise
through the `FieldType` broadcasting hooks in broadcasting.jl.

# Fields
- `elements`: the `NTuple{N, S}` of scalar field components
"""
struct VectorField{N, S} <: AbstractVector{S}
    elements::NTuple{N, S}

    function VectorField(elements::Vararg{S, N}) where {S<:Union{FTField, Field}, N}
        new{N, S}(elements)
    end
end

"""
    add_base_flow!(u::VectorField, base::NTuple{N}) -> u

Add the laminar base flow `base` to the zero-wavenumber slice of the spectral
vector field `u` in-place, recovering the full (base + perturbation) velocity.

`base` has one entry per velocity component.  Each entry is either a vector of
length `Ny` (values at the inhomogeneous grid points) or `nothing` (component
is skipped, i.e. no base flow in that direction).

# Index construction

For a field stored as a `D`-dimensional spectral array, the zero-wavenumber
slice is selected by an index pattern derived from the grid's `fft_storage_dims`:

- FFT dimensions (homogeneous, transformed): index `1`, the DC bin.
  FFTW always places the zero-frequency coefficient first, for both the real
  FFT (rfft) and the full complex FFT.
- Inhomogeneous dimensions: `Colon()`, selecting all grid points.

For a 4D channel grid with `CHANNEL_FFT_ORDER = (2, 3, 4)` this emits
`parent(u[n])[:, 1, 1, 1]` — the full wall-normal profile at zero streamwise,
spanwise, and temporal wavenumber.

# Example

```julia
# Channel flow: Couette base U(y)=y, no base in v or w.
add_base_flow!(u, (U, nothing, nothing))

# Equivalent to the original hand-written channel specialisation:
#   u[1][:, 1, 1, 1] .+= U
```
"""
@generated function add_base_flow!(u::VectorField{N, <:FTField{G}},
                                   base::Tuple{Vararg{Any, N}}) where {N, T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}
    idx = Any[(d ∈ FFT_DIMS_ORDER ? 1 : :(Colon())) for d in 1:D]

    return quote
        Base.Cartesian.@nexprs $N n -> begin
            if base[n] !== nothing
                Base.@views $(Expr(:ref, :(parent(u[n])), idx...)) .+= base[n]
            end
        end
        return u
    end
end

"""
    grid(u::VectorField) -> AbstractGrid

Return the grid shared by the scalar field components of `u`.
"""
grid(u::VectorField) = grid(u[1])

"""
    growto(u::VectorField, target_size)

Return an equivalent vector field with a new homogeneous resolution.

Each component is grown independently via [`growto(::FTField, target_size)`](@ref),
so the returned field covers the same spectral content as `u` but on a grid
with the physical-space sizes specified by `target_size` (one entry per
homogeneous dimension in `FFT_DIMS_ORDER` order).

This is optional.  Packages should only implement it if they support
resolution-changing transforms such as `FFT(u, target_size)` and
`IFFT(û, target_size)`.
"""
growto(u::VectorField{N, <:FTField}, target_size) where {N} = VectorField([growto(u[n], target_size) for n in 1:N]...)


# ------------------- #
# constructor methods #
# ------------------- #
"""
    VectorField(g::AbstractGrid, ::Type{S}=FTField; N::Int=3, kwargs...) -> VectorField

Allocate a zero-initialised `N`-component vector field of type `S` on grid `g`.
`kwargs` are forwarded to each `S(g; kwargs...)` constructor call, allowing
dealiasing flags or other grid-specific options to be set.  Defaults to
`FTField` (spectral) components and `N=3` components.
"""
VectorField(g::AbstractGrid, ::Type{S}=FTField; N::Int=3, kwargs...) where {S} = VectorField([S(g; kwargs...) for _ in 1:N]...)

"""
    VectorField(g::AbstractGrid, funcs...; dealias::Bool=false) -> VectorField

Construct a `VectorField` of physical-space `Field` components by evaluating
each function in `funcs` at the collocation points of `g`.  The number of
components equals `length(funcs)`.  `dealias=true` uses the padded grid size.
"""
VectorField(g::AbstractGrid, funcs...; dealias::Bool=false) = VectorField([Field(g, f; dealias=dealias) for f in funcs]...)


# ------------- #
# array methods #
# ------------- #
"""
    IndexStyle(::Type{<:VectorField})

`VectorField` uses linear (1-D) indexing: `q[n]` retrieves the `n`-th
component scalar field.
"""
Base.IndexStyle(::Type{<:VectorField}) = Base.IndexLinear()

"""
    parent(q::VectorField) -> NTuple

Return the underlying `NTuple{N, S}` of scalar field components.
"""
Base.parent(q::VectorField) = q.elements

"""
    q[i::Int] -> S

Return the `i`-th scalar field component of `q`.
"""
Base.getindex(q::VectorField, i::Int) = parent(q)[i]

"""
    q[i::Int] = v

Copy the contents of scalar field `v` into the `i`-th component of `q`
in-place (equivalent to `q[i] .= v`).
"""
Base.setindex!(q::VectorField{N, F}, v::F, i::Int) where {N, F} = (parent(q)[i] .= v; return v)

"""
    size(::VectorField{N}) -> (N,)

Return the number of scalar field components as a 1-tuple.
"""
Base.size(::VectorField{N}) where {N} = (N,)

"""
    eltype(::VectorField{N, F}) -> Type{F}

Return the scalar field type `F` (either `FTField` or `Field`).
"""
Base.eltype(::VectorField{N, F}) where {N, F} = F

"""
    similar(q::VectorField[, ::Type{T}]) -> VectorField

Allocate a new `VectorField` with the same structure as `q`.  Optionally
provide an element type `T` to change the scalar precision of each component.
"""
Base.similar(q::VectorField{N}, ::Type{T}=eltype(q[1])) where {N, T} = VectorField([Base.similar(q.elements[n], T) for n in 1:N]...)

"""
    copy(q::VectorField) -> VectorField

Return a deep copy of `q`: a new `VectorField` whose components are independent
copies of those in `q`, all sharing the same grid object.
"""
Base.copy(q::VectorField{N}) where {N} = VectorField([copy(q.elements[n]) for n in 1:N]...)

"""
    zero(q::VectorField) -> VectorField

Return a zero-valued `VectorField` with the same structure as `q`.
"""
Base.zero(q::VectorField{N}) where {N} = VectorField([zero(q.elements[n]) for n in 1:N]...)
