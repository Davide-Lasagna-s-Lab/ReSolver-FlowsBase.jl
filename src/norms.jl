# Inner products, norms, and distance functions for all field types.
#
# Every field type (FTField, VectorField of FTField, ProjectedField) gets a
# consistent set of three operations:
#
#   dot(u, v)        — L2 inner product, weighted by rfft Hermitian multiplicity
#                      and, for FTField, the grid quadrature weights.
#   norm(u)          — induced norm sqrt(dot(u, u)).
#   normdiff(u, v)   — norm of the difference ‖u − v‖, optionally after shifting
#                      v along the homogeneous directions by a continuous phase shift.
#
# Each operation accumulates directly into a plain scalar `s`; no Ref boxing or
# closures are needed because the loops are written as explicit for-loops.
#
# All three inner products are consistent: dot(VectorField) = sum_n dot(u_n, v_n),
# norm(VectorField) = sqrt(dot(u,u)) = sqrt(sum_n norm(u_n)^2), and
# normdiff(VectorField) = sqrt(sum_n normdiff(u_n, v_n)^2) matches the induced norm.
#
# `minnormdiff` searches over a grid of candidate continuous shifts and returns
# the minimum ‖u − shift(v, s)‖ together with the minimising shift tuple `s`.
# It is implemented for both FTField and VectorField inputs.

# --------- #
# FTField   #
# --------- #
"""
    dot(u::FTField, v::FTField) -> Real

Inner product of two spectral fields on the same grid.

In terms of the physical field `u(j, z, t)`, this computes the quadrature-weighted
domain average over the full 4D domain:

```math
\\langle u, v \\rangle =
  \\frac{1}{V} \\int w(\\mathbf{j})\\, u(\\mathbf{j}, \\mathbf{z})\\,
  v(\\mathbf{j}, \\mathbf{z})\\, d\\mathbf{j}\\, d\\mathbf{z}
```

where `j` are the inhomogeneous coordinates, `z` are the periodic coordinates,
`w(j)` are the quadrature weights returned by [`weights`](@ref), and `V` is the
product of the periods in the homogeneous directions.

In spectral space this is:

```math
\\langle u, v \\rangle =
  \\sum_{\\mathbf{k}} c_{k_1}\\, w(\\mathbf{j})\\,
  \\operatorname{Re}\\bigl(\\bar{u}_{\\mathbf{k},\\mathbf{j}}\\, v_{\\mathbf{k},\\mathbf{j}}\\bigr)
```

where `c_{k_1}` is `1` for the zero rfft wavenumber and `2` otherwise (rfft
Hermitian symmetry: each stored non-DC mode represents both the `+k` and `-k`
contributions).  The forward FFT normalization by `1/V` combined with Parseval
gives the domain-averaged inner product.
"""
function LinearAlgebra.dot(u::FTField{G}, v::FTField{G}) where {FFT_DIMS_ORDER, G<:AbstractGrid{<:Any,<:Any,<:Any,FFT_DIMS_ORDER}}
    return _dot(parent(u), parent(v), weights(grid(u)), Val(FFT_DIMS_ORDER))
end

# Helper function for the dot product. This is used by downstream packages (e.g. ReSolverFlowsBaseMpiExt)
function _dot(u::AbstractArray, v::AbstractArray, ws::AbstractArray, ::Val{FFT_DIMS_ORDER}) where {FFT_DIMS_ORDER}
    s = zero(real(eltype(u)))
    @inbounds for I in CartesianIndices(u)
        s += one_or_two(I, Val(FFT_DIMS_ORDER)) * ws[inhomogeneous_indices(I, Val(FFT_DIMS_ORDER))...] * real(conj(u[I]) * v[I])
    end
    return s
end

"""
    norm(u::FTField) -> Real

Norm induced from [`dot(u::FTField, v::FTField)`](@ref).
"""
LinearAlgebra.norm(u::FTField) = sqrt(dot(u, u))

@inline function _check_shift_length(M::Int, FFT_DIMS_ORDER)
    expected = length(FFT_DIMS_ORDER)
    M == expected ||
        throw(DimensionMismatch("shifts must have one entry per homogeneous dimension; got $(M), expected $(expected)"))
    return nothing
end

"""
    normdiff(u::FTField, v::FTField) -> Real
    normdiff(u::FTField, v::FTField, shifts::NTuple) -> Real
    normdiff(u::FTField, v::FTField, shifts::NTuple, tmp::FTField) -> Real
    normdiff(u::VectorField, v::VectorField) -> Real
    normdiff(u::VectorField, v::VectorField, shifts::NTuple) -> Real
    normdiff(u::VectorField, v::VectorField, shifts::NTuple, tmp::FTField) -> Real

Return `‖u − shift(v, shifts)‖`, the norm of the difference after optionally
shifting `v` along the homogeneous directions.

`shifts` is a tuple with one entry per homogeneous dimension in
`fft_storage_dims(grid(u)) = FFT_DIMS_ORDER` order, defaulting to all zeros (no shift).

`tmp` is an optional pre-allocated `FTField` workspace.  For `FTField` inputs,
the explicit workspace method copies `v` into `tmp`, shifts it in place when
needed, and then delegates to `normdiff(u, tmp)`.  For `VectorField` inputs,
the same scalar workspace is reused one component at a time.  If `shifts` is
passed without `tmp`, a temporary is allocated only when a non-zero shift is
requested.
"""
function normdiff(u::FTField{G}, v::FTField{G}) where {G<:AbstractGrid}
    s = zero(real(eltype(u)))
    g = grid(u)
    ws = weights(g)
    @inbounds for I in CartesianIndices(u)
        s += one_or_two(I, g) * ws[inhomogeneous_indices(I, g)...] * abs2(u[I] - v[I])
    end
    return sqrt(s)
end

function normdiff(u::FTField{G}, v::FTField{G},
                  shifts::NTuple{M, Real}) where {M, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    _check_shift_length(M, FFT_DIMS_ORDER)
    any(!iszero, shifts) || return normdiff(u, v)
    return normdiff(u, v, shifts, zero(v))
end

function normdiff(u::FTField{G}, v::FTField{G},
                  shifts::NTuple{M, Real},
                  tmp::FTField{G}) where {M, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    _check_shift_length(M, FFT_DIMS_ORDER)
    tmp .= v
    any(!iszero, shifts) && shift!(tmp, shifts)
    return normdiff(u, tmp)
end

normdiff(u::FTField, v::FTField, shifts::NTuple{M, Real}, ::Nothing) where {M} =
    normdiff(u, v, shifts)


# ----------- #
# VectorField #
# ----------- #
"""
    dot(q::VectorField, p::VectorField) -> Real

Inner product of two vector fields: sum of [`dot`](@ref) over components.
"""
function LinearAlgebra.dot(q::VectorField{N, <:FTField{G}},
                           p::VectorField{N, <:FTField{G}}) where {N, G<:AbstractGrid}
    s = zero(real(eltype(q[1])))
    for n in 1:N
        s += dot(q[n], p[n])
    end
    return s
end

LinearAlgebra.dot(q::VectorField{N}, p::VectorField{N}) where {N} = sum(dot(q[n], p[n]) for n in 1:N)

"""
    norm(q::VectorField) -> Real

Norm induced from [`dot(q::VectorField, p::VectorField)`](@ref).
"""
LinearAlgebra.norm(q::VectorField) = sqrt(dot(q, q))

"""
    normdiff(u::VectorField, v::VectorField) -> Real
    normdiff(u::VectorField, v::VectorField, shifts::NTuple) -> Real
    normdiff(u::VectorField, v::VectorField, shifts::NTuple, tmp::FTField) -> Real

Return `‖u − shift(v, shifts)‖` for vector fields: the Euclidean combination
of per-component [`normdiff`](@ref) values.

For shifted differences, `tmp` may be a single pre-allocated [`FTField`](@ref)
workspace.  It is reused component by component, so a full `VectorField`
workspace is not required.
"""
function normdiff(u::VectorField{N, <:FTField{G}},
                  v::VectorField{N, <:FTField{G}}) where {N, G<:AbstractGrid}
    s = zero(real(eltype(u[1])))
    for n in 1:N
        d = normdiff(u[n], v[n])
        s += d * d
    end
    return sqrt(s)
end

function normdiff(u::VectorField{N, <:FTField{G}},
                  v::VectorField{N, <:FTField{G}},
                  shifts::NTuple{M, Real}) where {N, M, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    _check_shift_length(M, FFT_DIMS_ORDER)
    any(!iszero, shifts) || return normdiff(u, v)
    return normdiff(u, v, shifts, zero(v[1]))
end

function normdiff(u::VectorField{N, <:FTField{G}},
                  v::VectorField{N, <:FTField{G}},
                  shifts::NTuple{M, Real},
                  tmp::FTField{G}) where {N, M, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    _check_shift_length(M, FFT_DIMS_ORDER)
    s = zero(real(eltype(u[1])))
    for n in 1:N
        tmp .= v[n]
        any(!iszero, shifts) && shift!(tmp, shifts)
        d = normdiff(u[n], tmp)
        s += d * d
    end
    return sqrt(s)
end

normdiff(u::VectorField, v::VectorField, shifts::NTuple{M, Real}, ::Nothing) where {M} =
    normdiff(u, v, shifts)

"""
    minnormdiff(u, v, N[, tmp]) -> (Real, NTuple)
    minnormdiff(u, v[, tmp]) -> (Real, NTuple)

Return `(min_diff, shifts)`: the minimum of `‖u − shift(v, shifts)‖` over a
regular grid of candidate shifts covering one full period in each transform
dimension.  `N` has one entry per transform dimension in `fft_storage_dims(grid(u))`
order; when omitted, each transform dimension uses 32 samples.

If `fft_storage_dims(grid(u)) == (2, 3)`, for example, then `N = (Nx, Nz)` samples the
first transform dimension with `Nx` shifts and the second transform dimension
with `Nz` shifts.  The returned `shifts` tuple has the same order and length as
`N`.

`tmp` is an optional workspace of the same type as `v`. It holds the shifted
copy of `v` for each candidate shift, so providing it avoids allocating that
workspace inside repeated calls.
"""
function minnormdiff(u::F,
                     v::F,
                     N::NTuple{M, Int},
                     tmp::F=zero(v)) where {M, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER},
                                             F<:Union{FTField{G}, VectorField{<:Any, <:FTField{G}}}} where {T, D, AXES, FFT_DIMS_ORDER}
    M == length(FFT_DIMS_ORDER) ||
        throw(DimensionMismatch("N must have one entry per transform dimension; got $(M), expected $(length(FFT_DIMS_ORDER))"))

    g        = grid(u)
    min_diff = typemax(real(eltype(u isa VectorField ? u[1] : u)))
    s_min    = ntuple(k -> zero(T), Val(M))
    s         = zero(min_diff)

    # The shift increment in each transform dimension is one physical period
    # divided by the number of samples requested in that dimension.
    steps = ntuple(k -> 2π / (wavenumber_scale(g, FFT_DIMS_ORDER[k]) * N[k]), Val(M))

    # Enumerate the candidate grid directly as zero-based shift counts in the
    # same order as `fft_storage_dims(g)`.
    ranges = ntuple(k -> 0:(N[k] - 1), Val(M))
    for shift_counts in Iterators.product(ranges...)
        shifts = ntuple(k -> steps[k] * shift_counts[k], Val(M))

        tmp .= v
        shift!(tmp, shifts)
        s = normdiff(u, tmp)

        if s < min_diff
            min_diff = s
            s_min    = shifts
        end
    end

    return min_diff, s_min
end

function minnormdiff(u::F,
                     v::F,
                     tmp::F=zero(v)) where {G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER},
                                            F<:Union{FTField{G}, VectorField{<:Any, <:FTField{G}}}} where {T, D, AXES, FFT_DIMS_ORDER}
    # Use the default resolution of 32 shifts for each transform dimension
    M = length(FFT_DIMS_ORDER)
    return minnormdiff(u, v, ntuple(Returns(32), Val(M)), tmp)
end


# -------------- #
# ProjectedField #
# -------------- #
"""
    dot(a::ProjectedField, b::ProjectedField)

Inner product of two projected fields, exploiting the rfft Hermitian symmetry.
Wavenumbers with rfft index `> 1` (wavenumber `nx > 0`) are stored once but represent
both `+nx` and `-nx`, so they contribute with weight 2; the `nx = 0` plane has
weight 1.
"""
function LinearAlgebra.dot(a::ProjectedField{G}, b::ProjectedField{G}) where {G<:AbstractGrid{T, D}} where {T, D}
    s = zero(real(eltype(a)))
    # ProjectedField parent has D+1 dimensions: 1 mode dim + D grid dims.
    # rfft dimension is always dim 2 (first grid dim after the mode index).
    # Split into two branch-free loops so the compiler can vectorise the
    # inner loop over dim 1 (mode index, stride-1 in storage).
    ix1 = ntuple(d -> d == 2 ? (1:1)          : axes(parent(a), d), Val(D+1))
    ix2 = ntuple(d -> d == 2 ? (2:size(a, 2)) : axes(parent(a), d), Val(D+1))
    @inbounds for I in CartesianIndices(ix1)
        s += real(conj(a[I]) * b[I])
    end
    @inbounds for I in CartesianIndices(ix2)
        s += 2 * real(conj(a[I]) * b[I])
    end
    return s
end

"""
    norm(a::ProjectedField)

Norm induced from [`dot(a::ProjectedField, b::ProjectedField)`](@ref).
"""
LinearAlgebra.norm(a::ProjectedField) = sqrt(dot(a, a))

"""
    normdiff(a::ProjectedField, b::ProjectedField) -> Real

Return `‖a − b‖`.

Follows the same rfft Hermitian weighting as [`dot`](@ref): rfft wavenumbers
with `nx > 0` contribute with weight 2 (they represent both `±nx`).
"""
function normdiff(a::ProjectedField{G}, b::ProjectedField{G}) where {G<:AbstractGrid{T, D}} where {T, D}
    s = zero(real(eltype(a)))
    # D+1: one mode dim + D grid dims; rfft on dim 2 (first grid dim).
    ix1 = ntuple(d -> d == 2 ? (1:1)          : axes(parent(a), d), Val(D+1))
    ix2 = ntuple(d -> d == 2 ? (2:size(a, 2)) : axes(parent(a), d), Val(D+1))
    @inbounds for I in CartesianIndices(ix1)
        s += abs2(a[I] - b[I])
    end
    @inbounds for I in CartesianIndices(ix2)
        s += 2 * abs2(a[I] - b[I])
    end
    return sqrt(s)
end

"""
    normdiff(a::ProjectedField, b::ProjectedField, shifts, tmp::ProjectedField) -> Real

Return `‖a − shift(b, shifts)‖`.

`shifts` is a tuple with one entry per homogeneous dimension in
`fft_storage_dims(grid(a)) = FFT_DIMS_ORDER` order.

`tmp` is a pre-allocated `ProjectedField` workspace of the same type as `a`
and `b`; `b` is copied into `tmp` and (if needed) shifted in place, so `b`
itself is not modified.  Providing `tmp` avoids any allocations: use this form
in performance-critical loops.
"""
function normdiff(a::ProjectedField{G}, b::ProjectedField{G},
                  shifts::NTuple{M, Real},
                  tmp::ProjectedField{G}) where {M, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    _check_shift_length(M, FFT_DIMS_ORDER)
    tmp .= b
    any(!iszero, shifts) && shift!(tmp, shifts)
    return normdiff(a, tmp)
end
