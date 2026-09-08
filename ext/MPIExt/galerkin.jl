"""
    ReSolverFlowsBase.project!(a, u)

MPI-aware override of [`ReSolverFlowsBase.project!`](@ref) for decomposed grids.

The input argument `a` is zero-ed before computations begin.

Each rank holds only a local slab of the inhomogeneous dimension, so
`_project_component!` produces a partial inner product over that slab.
`MPI.Allreduce!` then sums the partial modal coefficients across all ranks,
recovering the correct global projection.
"""
function ReSolverFlowsBase.project!(a::DecomposedProjectedField,
                          u::DecomposedFTVectorField)
    # ! get rid of the commented stuff if I can
    # fill!(parent(a), zero(eltype(a)))
    # for n in 1:N
    #     ReSolverFlowsBase._project_component!(parent(a),
    #                                 parent(u[n]),
    #                                 ReSolverFlowsBase.modes(a)[n],
    #                                 ReSolverFlowsBase.weights(ReSolverFlowsBase.grid(u)),
    #                                 Val(ReSolverFlowsBase.fft_storage_dims(ReSolverFlowsBase.grid(u))))
    # end

    ReSolverFlowsBase.project!(a, u, ReSolverFlowsBase.LoopGalerkin())

    # Sum the per-rank partial projections into the global modal coefficients
    MPI.Allreduce!(parent(a), MPI.SUM, comm(ReSolverFlowsBase.grid(u)))
    return a
end

"""
    ReSolverFlowsBase.project(u::DecomposedFTVectorField, modes) -> ProjectedField

Allocate a `ReSolverFlowsBase.ProjectedField` over `modes` and project a decomposed `u`
onto it. See [`project!`](@ref) for the in-place form.
"""
ReSolverFlowsBase.project(u::DecomposedFTVectorField, modes) =
    ReSolverFlowsBase.project!(ReSolverFlowsBase.ProjectedField(ReSolverFlowsBase.grid(u), modes), u)

# `expand!` does not need an MPI override: it reads from `a` (which is already
# globally consistent after `project!`) and writes to `u` locally.
