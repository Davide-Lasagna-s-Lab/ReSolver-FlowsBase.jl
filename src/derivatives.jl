# Spectral differentiation operators for FTField, ProjectedField, and VectorField.
#
# Entry points are `ddx!`, `ddy!`, `ddz!`, `ddt!` — one per physical coordinate.
# Each resolves its direction to a compile-time `Val{STORAGE_DIM}` via
# `physical_to_storage_dim` and delegates to `dd!(out, u, ::Val{STORAGE_DIM})`.
#
# `dd!(out, u, ::Val{STORAGE_DIM})` is the core primitive, implemented with
# CartesianIndices loops.  For the rfft dimension all stored wavenumbers are
# non-negative so a single loop over the full array suffices.  For signed-FFT
# dimensions the index range is split into a positive block [1:(N÷2)+1] and a
# negative block [(N÷2)+2:N] so that the signed wavenumber can be computed
# branch-free from the loop index inside each block.
#
# When a physical direction is absent from the grid its storage dimension is
# `nothing` and the kernel is a no-op.
#
# Inhomogeneous (non-FFT) directions are NOT handled here — `dd!` throws
# `NotImplementedError` for those, and downstream packages must extend it
# with a grid-specific method (typically a matrix–vector multiply).
#
# The full Laplacian combines `_inhomogeneous_laplacian!` (provided by
# downstream) with `_add_homogeneous_laplacian!` (provided here), which
# subtracts the spatial ‖k‖² · u contribution from each spectral coefficient.

derivative_matrix(g::AbstractGrid, ::Integer, ::Val, ::OperatorMode) = throw(NotImplementedError(g))

"""
    ddx!(out, u, mode=Forward()) -> out
Differentiate `u` along physical direction `x`, storing the result in `out`.
The wrapper resolves the direction to a `Val{STORAGE_DIM}` at the
call site and delegates to the low-level [`dd!`](@ref)`(out, u, ::Val)` primitive.

`mode` selects the operator variant: `Forward()` (default) applies the forward
derivative, `AdjointDiscrete()` its discrete adjoint. The tag participates in
dispatch, so each variant compiles to a concrete operator with no runtime
branch.

For an absent direction (e.g. `:z` on a 2D grid) the call is a compile-time no-op.
"""
ddx!(out, u, mode::OperatorMode=Forward()) = dd!(out, u, physical_to_storage_dim(grid(u), Val(:x)), mode)

"""Differentiate `u` along physical direction `y`; see [`ddx!`](@ref)."""
ddy!(out, u, mode::OperatorMode=Forward()) = dd!(out, u, physical_to_storage_dim(grid(u), Val(:y)), mode)

"""Differentiate `u` along physical direction `z`; see [`ddx!`](@ref)."""
ddz!(out, u, mode::OperatorMode=Forward()) = dd!(out, u, physical_to_storage_dim(grid(u), Val(:z)), mode)

"""Differentiate `u` along physical direction `t`; see [`ddx!`](@ref)."""
ddt!(out, u, mode::OperatorMode=Forward()) = dd!(out, u, physical_to_storage_dim(grid(u), Val(:t)), mode)

dd!(out::VectorField{N}, u::VectorField{N}, sd::Val, mode::OperatorMode=Forward()) where {N} =
    (for n in 1:N; dd!(out[n], u[n], sd, mode); end; return out)

"""
    dd!(out, u, ::Val{STORAGE_DIM}, mode=Forward())

In-place derivative of `u` along the storage dimension encoded by
`Val(STORAGE_DIM)`.

For `STORAGE_DIM in FFT_DIMS_ORDER` the derivative is multiplication by
`±im * n * wavenumber_scale(grid, STORAGE_DIM)` where `n` is the signed
wavenumber. `Forward()` (default) gives `+im·n·scale·u`;
`AdjointDiscrete()` gives `-im·n·scale·u`.

For an inhomogeneous storage dimension the method throws
`NotImplementedError`. Downstream packages should extend
[`derivative_matrix`](@ref) for each inhomogeneous direction.

`AdjointContinuous` is not part of this API: the continuous adjoint is
expressed in the equation methods through forward derivatives.

For `Val(nothing)` the function is a no-op.
"""
function dd!(out::F, u::F, ::Val{STORAGE_DIM},
             mode::OperatorMode=Forward()) where {
        STORAGE_DIM, T, D, AXES, FFT_DIMS_ORDER,
        G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER},
        F<:Union{FTField{G}, ProjectedField{G}}}

    isnothing(STORAGE_DIM) && return out
    STORAGE_DIM ∉ FFT_DIMS_ORDER ? _inhomogeneous_dd!(out, u, Val(STORAGE_DIM), mode) :
                                        _spectral_dd!(out, u, Val(STORAGE_DIM), mode)

    return out
end

"""
    _spectral_dd!(out, u, ::Val{STORAGE_DIM}, mode=Forward())

In-place derivative of `u` along the storage dimension where
`STORAGE_DIM ∈ FFT_DIMS_ORDER` using spectral methods.
"""
function _spectral_dd!(out::F,
                         u::F,
                          ::Val{STORAGE_DIM},
                      mode::OperatorMode=Forward()) where {
        STORAGE_DIM, T, D, AXES, FFT_DIMS_ORDER,
        G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER},
        F<:Union{FTField{G}, ProjectedField{G}}}

    scale = wavenumber_scale(grid(u), STORAGE_DIM)
    coeff = mode isa AdjointDiscrete ? -im * T(scale) : im * T(scale)
    Nd    = size(u, STORAGE_DIM)
    pu    = parent(u)
    pout  = parent(out)

    if STORAGE_DIM == FFT_DIMS_ORDER[1]
        @inbounds for I in CartesianIndices(pu)
            pout[I] = (coeff * (I[STORAGE_DIM] - 1)) * pu[I]
        end
    else
        pos_ranges = ntuple(d -> d == STORAGE_DIM ? (1:(Nd >> 1) + 1)  : Base.OneTo(size(pu, d)), Val(D))
        neg_ranges = ntuple(d -> d == STORAGE_DIM ? ((Nd >> 1) + 2:Nd) : Base.OneTo(size(pu, d)), Val(D))
        @inbounds for I in CartesianIndices(pos_ranges)
            pout[I] = (coeff * (I[STORAGE_DIM] - 1)) * pu[I]
        end
        @inbounds for I in CartesianIndices(neg_ranges)
            pout[I] = (coeff * (I[STORAGE_DIM] - Nd - 1)) * pu[I]
        end
    end
    return out
end

"""
    _inhomogeneous_dd!(out, u, ::Val{STORAGE_DIM}; adjoint=false)

Compute the derivative of the field `u` along the inhomogeneous
direction using a stored differentation operator, storing the result
in `out`.

Requires `ReSolverFlowsBase.derivative_matrix` to be defined for input types.
"""
function _inhomogeneous_dd!(out::FTField{G},
                              u::FTField{G},
                               ::Val{STORAGE_DIM},
                           mode::OperatorMode=Forward()) where {G<:AbstractGrid, STORAGE_DIM}

    A = derivative_matrix(grid(u), STORAGE_DIM, Val(1), mode)
    LinearAlgebra.mul!(parent(out), A, parent(u), Val(STORAGE_DIM))

    return out
end


"""
    _add_homogeneous_laplacian!(out::FTField, u::FTField)
    _add_homogeneous_laplacian!(out::VectorField, u::VectorField)

Add the homogeneous Laplacian contribution of `u` to `out`:

    out[mode] -= (∑_{d∈spatial_fft_storage_dims(g)} (wavenumber_scale(g, d) · n_d)²) · u[mode]

Call after computing the non-homogeneous (e.g. wall-normal) second derivative.
If the grid includes a transformed logical time coordinate, that direction is
not part of the spatial Laplacian.
"""
function _add_homogeneous_laplacian!(out::FTField{G}, u::FTField{G}) where {T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}
    g = grid(u)

    H = spatial_fft_storage_dims(g)
    scales = map(dim -> dim in H ? T(wavenumber_scale(g, dim)) : zero(T), FFT_DIMS_ORDER)
    pu = parent(u)
    pout = parent(out)

    @inbounds for Ih in CartesianIndices(homogeneous_axes(u))
        k² = zero(T)
        k = to_wavenumber_vector(g, Ih)
        for j in eachindex(FFT_DIMS_ORDER)
            k² += (scales[j] * k[j])^2
        end

        for Iinh in CartesianIndices(inhomogeneous_axes(u))
            I = combine_indices(g, Iinh, Ih)
            pout[I...] -= k² * pu[I...]
        end
    end
    return out
end
_add_homogeneous_laplacian!(out::VectorField{N}, u::VectorField{N}) where {N} =
    (for n in 1:N; _add_homogeneous_laplacian!(out[n], u[n]); end; return out)

"""
    inhomogeneous_laplacian!(out::FTField, u::FTField, mode=Forward()) -> out

Apply the inhomogeneous (non-FFT) part of the Laplacian of `u` to `out`.

This contribution contains the second derivatives along directions that are
not in [`fft_storage_dims`](@ref), such as wall-normal collocation directions. The
full spatial Laplacian is the sum of this contribution and the homogeneous
spectral contribution from [`_add_homogeneous_laplacian!`](@ref).

Requires `ReSolverFlowsBase.derivative_matrix` to be defined for input types.
"""
function _inhomogeneous_laplacian!(out::FTField{G},
                                     u::FTField{G},
                                  mode::OperatorMode=Forward()) where {G<:AbstractGrid}
    inh_spatial_dims = inhomogeneous_storage_dims(grid(u))
    isempty(inh_spatial_dims) && (out .*= 0; return out)

    A = derivative_matrix(grid(u), inh_spatial_dims[1], Val(2), mode)
    LinearAlgebra.mul!(parent(out), A, parent(u), Val(inh_spatial_dims[1]))

    for dim in Base.tail(inh_spatial_dims)
        A = derivative_matrix(grid(u), dim, Val(2), mode)
        LinearAlgebra.mul!(parent(out), A, parent(u), Val(dim), Val(true))
    end

    return out
end

"""
    laplacian!(out::FTField{G}, u::FTField{G}, mode=Forward())
    laplacian!(out::VectorField{N}, u::VectorField{N}, mode=Forward())

Compute the full Laplacian of `u` in-place, storing the result in `out`:

    out[mode] = (inhomogeneous second derivatives
                 - ∑_{d∈spatial_fft_storage_dims(g)} (wavenumber_scale(g, d) · n_d)²) · u[mode]

Only spatial transformed directions enter the homogeneous sum; a transformed
logical time coordinate is excluded through [`spatial_fft_storage_dims`](@ref).

`mode` selects the operator variant for the finite-difference part;
the homogeneous −‖k‖² contribution is self-adjoint and unaffected.
"""
function laplacian!(out::FTField{G}, u::FTField{G}, mode::OperatorMode=Forward()) where {G}
    _inhomogeneous_laplacian!(out, u, mode)
    _add_homogeneous_laplacian!(out, u)
    return out
end
laplacian!(out::VectorField{N}, u::VectorField{N}, mode::OperatorMode=Forward()) where {N} =
    (for n in 1:N; laplacian!(out[n], u[n], mode); end; return out)
