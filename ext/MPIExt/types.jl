# Convenience aliases for field wrappers on decomposed grids.

# Scalar fields.
const DecomposedField            = ReSolverFlowsBase.Field{<:DecomposedGrid}
const DecomposedFTField          = ReSolverFlowsBase.FTField{<:DecomposedGrid}
const DecomposedProjectedField   = ReSolverFlowsBase.ProjectedField{<:DecomposedGrid}
const DecomposedScalarField      = Union{DecomposedField, DecomposedFTField}

# Vector fields.
const DecomposedVectorField{N}   = ReSolverFlowsBase.VectorField{N, <:DecomposedScalarField}
const DecomposedFTVectorField{N} = ReSolverFlowsBase.VectorField{N, <:DecomposedFTField}

# Operation groups.
const DecomposedSpectralField    = Union{DecomposedFTField, DecomposedFTVectorField}
