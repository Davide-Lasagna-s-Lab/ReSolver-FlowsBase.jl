# MPI-reduced inner product for decomposed grids.
#
# ReSolverFlowsBase's `LinearAlgebra.dot(u::FTField, v::FTField)` accumulates a
# weighted sum over every inhomogeneous index this rank owns. On a
# decomposed grid each rank owns only a slab, so the local accumulation
# is a partial sum; `MPI.Allreduce` combines partial sums into the
# global inner product on every rank. The `VectorField` overload in
# ReSolverFlowsBase reduces to a sum over per-component scalar `dot` calls; each
# call is already globally summed by the override below, so the vector
# version is correct without further changes.
#
# `norm(u) = sqrt(dot(u, u))` and any caller that derives an aggregate
# quantity from `dot` likewise inherits the global semantics — no
# separate `norm` override is needed.

"""
    LinearAlgebra.dot(u::FTField{<:DecomposedGrid}, v::FTField{<:DecomposedGrid}) -> Real

MPI-reduced weighted inner product of two distributed spectral fields.

Each rank computes the same partial sum that ReSolverFlowsBase computes for a
single-domain field, then `MPI.Allreduce` sums the partial values into
the global inner product. Every rank returns the same value.

`u` and `v` must share the same `DecomposedGrid` (so the partition,
weights, and Hermitian-symmetry multipliers agree across ranks).
"""
function LinearAlgebra.dot(u::F, v::F) where {F<:DecomposedFTField}
    g = ReSolverFlowsBase.grid(u)
    local_dot = ReSolverFlowsBase._dot(parent(u),
                             parent(v),
                             ReSolverFlowsBase.weights(g),
                             Val(ReSolverFlowsBase.fft_storage_dims(g)))
    return MPI.Allreduce(real(local_dot), MPI.SUM, comm(g))
end
