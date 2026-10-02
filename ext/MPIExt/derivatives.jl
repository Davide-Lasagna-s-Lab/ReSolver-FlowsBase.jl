# Halo exchange and finite-difference derivatives for decomposed grids.
#
# Concrete decomposed grids must implement
#   derivative_matrix(g, stor_dim::Int, ::Val{ORDER}, mode::AbstractDerivativeMode)
# for every inhomogeneous spatial direction and every derivative order they support.


# ------------------------------------------------------------------ #
# init_requests! / wait_requests!                                    #
# ------------------------------------------------------------------ #
init_requests!(u::DecomposedScalarField) = HaloArrays.haloswap!(parent(u), false)
init_requests!(u::DecomposedVectorField{N}) where {N} =
    ntuple(n -> HaloArrays.haloswap!(parent(u[n]), false), Val(N))

wait_requests!(requests::MPI.AbstractMultiRequest) = (MPI.Waitall(requests); nothing)
wait_requests!(tokens::Tuple) = (foreach(wait_requests!, tokens); nothing)


# ------------------------------------------------------------------ #
# laplacian! — blocking override for decomposed grids                #
# ------------------------------------------------------------------ #
"""
    ReSolverFlowsBase.laplacian!(out::DecomposedFTField,
                         u::DecomposedFTField,
                      [DiscreteAdjoint()]) -> DecomposedFTField

Compute the full Laplacian of the distributed FTField `u` in-place,
storing the result in `out`.

The computation is split between the interior part of the decomposed
field which does not depend on the halo swap between processes, and
the boundary computations that can only be completed after the halo
swaps have completed.
"""
function ReSolverFlowsBase.laplacian!(out::DecomposedFTField,
                              u::DecomposedFTField,
                           mode::AbstractDerivativeMode=Direct())
    # start comm
    requests = init_requests!(u)

    # do interior work
    interior_laplacian!(out, u, mode)

    # do boundary work once halo swap has concluded
    wait_requests!(requests)
    boundary_laplacian!(out, u, mode)

    return out
end

"""
    interior_laplacian!(out::DecomposedFTField,
                          u::DecomposedFTField,
                       [DiscreteAdjoint()]) -> DecomposedFTField

Compute the Laplacian operator on the parts of the distributed field
`u` that do not depend on data from the halo regions of the underlying
arrays.
"""
function interior_laplacian!(out::DecomposedFTField,
                               u::DecomposedFTField,
                            mode::AbstractDerivativeMode=Direct())
    # santise input
    out .= zero(eltype(u))

    g = ReSolverFlowsBase.grid(u)
    fd_stor_dims = ReSolverFlowsBase.spatial_inhomogeneous_storage_dims(g)

    first_sd = first(fd_stor_dims)
    _dd_over!(out, u, g, Val(first_sd), Val(2),
              (local_interior_range(g, first_sd),), mode;
              accumulate=Val(false))

    # ! this is more efficient than zero-ing the whole output first, but doesn't
    # ! play well when using CUDA
    # Zero boundary bands of the first FD direction before accumulating tail
    # dimensions, which span all first-dimension indices including halo bands.
    # let a = parent(out)
    #     for rng in local_boundary_ranges(g, first_sd)
    #         isempty(rng) && continue
    #         fill!(selectdim(a, first_sd, rng), zero(eltype(a)))
    #     end
    # end

    for sd in Base.tail(fd_stor_dims)
        _dd_over!(out, u, g, Val(sd), Val(2),
                  (local_interior_range(g, sd),), mode;
                  accumulate=Val(true))
    end

    ReSolverFlowsBase._add_homogeneous_laplacian!(out, u)
    return out
end

"""
    boundary_laplacian!(out::DecomposedFTField,
                          u::DecomposedFTField,
                       [DiscreteAdjoint()]) -> DecomposedFTField

Compute the remaining parts of the Laplacian of the distributed field
`u` that depend on the data in the halo regions of the underlying arrays.
This function should only be called after `wait_requests!(requests)` has
completed.
"""
function boundary_laplacian!(out::DecomposedFTField,
                               u::DecomposedFTField,
                            mode::AbstractDerivativeMode=Direct())
    g = ReSolverFlowsBase.grid(u)
    for sd in ReSolverFlowsBase.spatial_inhomogeneous_storage_dims(g)
        _dd_over!(out, u, g, Val(sd), Val(2),
                  local_boundary_ranges(g, sd), mode;
                  accumulate=Val(true))
    end
    return out
end


# ------------------------------------------------------------------ #
# distributed derivatives                                            #
# ------------------------------------------------------------------ #
"""
    dd!(out::DecomposedFTField,
          u::DecomposedFTField,
           ::Val{STORAGE_DIM},
       [DiscreteAdjoint()]) -> DecomposedFTField

In-place derivative of the distributed field `u` along the storage
dimension encoded by `Val(STORAGE_DIM)`.

For `STORAGE_DIM ∉ DDIMS` the code falls back to the standard
derivative method [`dd!`](@ref) which uses the default non-distributed
routines for the derivative computation.
"""
function ReSolverFlowsBase.dd!(out::F,
                       u::F,
                        ::Val{STORAGE_DIM},
                    mode::AbstractDerivativeMode=Direct()) where {
    STORAGE_DIM, FFT_DIMS_ORDER, DDIMS, T, D, AXES,
    G<:DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS},
    F<:Union{ReSolverFlowsBase.FTField{G}, ReSolverFlowsBase.ProjectedField{G}}} # ! how do I get rid of the ProjectedField part?

    # ! can redefine ReSolverFlowsBase._inhomogeneous_dd! in the extension to use _distributed_dd! or basic version depending of `STORAGE_DIM`?
    isnothing(STORAGE_DIM) && return out
    if STORAGE_DIM ∈ DDIMS
        _distributed_dd!(out, u, Val(STORAGE_DIM), mode)
    else
        if STORAGE_DIM ∈ FFT_DIMS_ORDER
            ReSolverFlowsBase._spectral_dd!(out, u, Val(STORAGE_DIM), mode)
        else
            ReSolverFlowsBase._inhomogeneous_dd!(out, u, Val(STORAGE_DIM), mode)
        end
    end

    return out
end

"""
    _distributed_dd!(out::DecomposedFTField,
                       u::DecomposedFTField,
                        ::Val{STORAGE_DIM},
                    [DiscreteAdjoint()]) -> DecomposedFTField

Compute the derivative of a distributed field `u` along one of
decomposed storage dimensions, i.e. `STORAGE_DIM ∈ DDIMS`.
"""
function _distributed_dd!(out::Union{DecomposedFTField, DecomposedField},
                            u::Union{DecomposedFTField, DecomposedField},
                             ::Val{STORAGE_DIM},
                         mode::AbstractDerivativeMode=Direct()) where {STORAGE_DIM}
    # start comm
    requests = init_requests!(u)

    # do interior work
    interior_dd!(out, u, Val(STORAGE_DIM), mode)

    # do boundary work once halo swap has concluded
    wait_requests!(requests)
    boundary_dd!(out, u, Val(STORAGE_DIM), mode)

    return out
end

"""
    interior_dd!(out::DecomposedFTField,
                   u::DecomposedFTField,
                    ::Val{STORAGE_DIM},
                [DiscreteAdjoint()]) -> DecomposedFTField

Compute the derivative of the parts of the distributed field `u` along
the storage dimension `STORAGE_DIM` that do not depend on data from
the halo regions of the underlying arrays.
"""
interior_dd!(out::DecomposedFTField,
               u::DecomposedFTField,
                ::Val{STORAGE_DIM},
            mode::AbstractDerivativeMode=Direct()) where {STORAGE_DIM} =
    _dd_over!(out, u, ReSolverFlowsBase.grid(u), Val(STORAGE_DIM), Val(1),
              (local_interior_range(ReSolverFlowsBase.grid(u), STORAGE_DIM),), mode)

"""
    boundary_dd!(out::DecomposedFTField,
                   u::DecomposedFTField,
                    ::Val{STORAGE_DIM},
                [DiscreteAdjoint()]) -> DecomposedFTField

Compute the remaining parts of the derivative of the distributed field
`u` that depend on the data in the halo regions of the underlying arrays.
This function should only be called after `wait_requests!(requests)` has
completed.
"""
boundary_dd!(out::DecomposedFTField,
               u::DecomposedFTField,
                ::Val{STORAGE_DIM},
            mode::AbstractDerivativeMode=Direct()) where {STORAGE_DIM} =
    _dd_over!(out, u, ReSolverFlowsBase.grid(u), Val(STORAGE_DIM), Val(1),
              local_boundary_ranges(ReSolverFlowsBase.grid(u), STORAGE_DIM), mode)


# ------------------------------------------------------------------ #
# _dd_over! — FD kernel                                              #
# ------------------------------------------------------------------ #
"""
    _dd_over!(out, u, g::DecomposedGrid, ::Val{STORAGE_DIM}, ::Val{ORDER},
              ranges, [DiscreteAdjoint()]; accumulate::Val=Val(false))

Compute the derivative of a distributed field `u` in-place along the
storage dimension `STORAGE_DIM ∈ DDIMS` and assign the
result to `out`.

Only a slice of the [`DiffMatrix`](@ref) is used for the computation,
corresponding to `range`, determined by both the particular part of the
overal decomposed field this is being computed on, as well as the particular
parts of the field locally that want to be computed.

Optionally, by setting `accumulate=Val(true)` the result can be accumulated
into `out`, instead of overwritting it with the result.

The mode tag participates in dispatch, so the selected forward or adjoint
operator has one concrete type in the matrix kernel.
"""
@inline function _dd_over!(out, u, g::DecomposedGrid, ::Val{STORAGE_DIM}, ::Val{ORDER},
                           ranges,
                           mode::AbstractDerivativeMode=Direct();
                           accumulate::Val{B}=Val(false)) where {STORAGE_DIM, ORDER, B}
    A = ReSolverFlowsBase.derivative_matrix(g, STORAGE_DIM, Val(ORDER), mode)
    g_first = global_first_index(g, STORAGE_DIM)
    for rng in ranges
        isempty(rng) && continue
        LinearAlgebra.mul!(parent(out), A, parent(u),
                           Val(STORAGE_DIM), g_first, rng, accumulate)
    end
    return out
end
