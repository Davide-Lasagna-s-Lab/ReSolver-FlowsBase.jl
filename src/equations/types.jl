# Mode tags. They select, through dispatch, which operator is applied, so the
# compiler generates a specialised method for each and no branch is taken at
# run time.
#
#     Nonlinear, Linearised, AdjointDiscrete, AdjointContinuous
#         modes of the equation operators (see NavierStokes)
#     Forward, AdjointDiscrete
#         modes of the derivative API (ddx!, laplacian!, ...) and of body forces

# ---------- #
# mode types #
# ---------- #
"""
    AbstractEquationMode

Abstract supertype of the mode tags. Concrete subtypes select which operator is
applied: nonlinear, linearised or adjoint. `Forward` is here only until the
derivative API gets its own mode type.
"""
abstract type AbstractEquationMode end

"""
    Forward <: AbstractEquationMode

Tag selecting the forward derivative operators (`ddx!`, `laplacian!`, ...) and
the forward action of body forces.
"""
struct Forward           <: AbstractEquationMode end

"""
    Nonlinear <: AbstractEquationMode

Tag selecting the nonlinear Navier–Stokes operator `N(u)` of a formulation.
"""
struct Nonlinear         <: AbstractEquationMode end

"""
    Linearised <: AbstractEquationMode

Tag selecting the linearised Navier–Stokes operator `L v` of a formulation.
The derivative API keeps [`Forward`](@ref) for the same role.
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

"""
    OperatorMode

Union of the mode tags accepted by the low-level derivative API (`dd!`,
`ddx!`, `laplacian!`, `derivative_matrix`, ...): [`Forward`](@ref) selects the
forward operators and [`AdjointDiscrete`](@ref) their caller-supplied discrete
adjoints. [`AdjointContinuous`](@ref) is deliberately excluded — the
continuous adjoint has no discrete operator realisation and is written out in
the equation methods using forward derivatives.
"""
const OperatorMode = Union{Forward, AdjointDiscrete}
