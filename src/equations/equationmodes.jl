# Equation-mode tags: Nonlinear, Linearised, AdjointDiscrete, AdjointContinuous.
# They select, through dispatch, which operator is applied, so the compiler
# generates a specialised method for each and no branch is taken at run time.
# The derivative API has its own tags, Direct and DiscreteAdjoint (derivatives.jl).

# ---------- #
# mode types #
# ---------- #
"""
    AbstractEquationMode

Abstract supertype of the equation-mode tags. Concrete subtypes select which
operator is applied: nonlinear, linearised or adjoint.
"""
abstract type AbstractEquationMode end

"""
    Nonlinear <: AbstractEquationMode

Tag selecting the nonlinear Navier–Stokes operator `N(u)` of a formulation.
"""
struct Nonlinear         <: AbstractEquationMode end

"""
    Linearised <: AbstractEquationMode

Tag selecting the linearised Navier–Stokes operator `L v` of a formulation.
"""
struct Linearised        <: AbstractEquationMode end

"""
    AdjointDiscrete <: AbstractEquationMode

Tag selecting the discrete adjoint of the forward linearised operator.
The discrete adjoint is derived by transposing the discrete operator exactly —
it satisfies ⟨L·v, w⟩ = ⟨v, L*·w⟩ with respect to the discrete inner product
used in [`dot`](@ref).
"""
struct AdjointDiscrete   <: AbstractEquationMode end

"""
    AdjointContinuous <: AbstractEquationMode

Tag selecting the continuous adjoint of the linearised operator.  The
continuous adjoint is derived by integration-by-parts before discretisation,
which gives a different operator from the discrete adjoint for finite
resolution.
"""
struct AdjointContinuous <: AbstractEquationMode end

"""
    AnyLinear

Union of the modes of the linear operators of a formulation:
[`Linearised`](@ref), [`AdjointDiscrete`](@ref) and [`AdjointContinuous`](@ref).
"""
const AnyLinear = Union{Linearised, AdjointDiscrete, AdjointContinuous}
