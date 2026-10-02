# Equation-mode tags, the no-op body force, and the compound force combinator.
#
# Mode tags control which flavour of the linearised Navier-Stokes operator is
# applied.  Each concrete NSE/LNSE operator is parameterised on a mode type so
# that the compiler generates specialised, zero-overhead method bodies for each
# adjoint variant rather than branching at runtime.
#
# `NoForce` is the default body-force callable: it receives `(out, u, mode)` and
# returns `out` unchanged, imposing no additional forcing.
#
# `CompoundForcing` chains multiple body-force callables in sequence.  The call
# is unrolled at compile time via `@nexprs` so that dynamic dispatch is avoided
# even when the individual force types differ.

# ---------- #
# mode types #
# ---------- #
"""
    Mode

Abstract supertype for equation-mode tags.  Concrete subtypes select which
variant of the linearised Navier-Stokes operator is applied.
"""
abstract type Mode end

"""
    Forward <: Mode

Tag selecting the forward linearised operator L·v, i.e. the linearisation of
the NSE around the current base flow in the forward-time direction.
"""
struct Forward           <: Mode end

"""
    Nonlinear <: Mode

Tag selecting the nonlinear Navier–Stokes operator `N(u)` of a formulation.
"""
struct Nonlinear         <: Mode end

"""
    Linearised <: Mode

Tag selecting the linearised Navier–Stokes operator `L v` of a formulation.
The derivative API keeps [`Forward`](@ref) for the same role.
"""
struct Linearised        <: Mode end

"""
    AdjointDiscrete <: Mode

Tag selecting the discrete adjoint of the forward linearised operator.
The discrete adjoint is derived by transposing the discrete operator exactly —
it satisfies ⟨L·v, w⟩ = ⟨v, L*·w⟩ with respect to the discrete inner product
used in [`dot`](@ref).
"""
struct AdjointDiscrete   <: Mode end

"""
    AdjointContinuous <: Mode

Tag selecting the continuous adjoint of the linearised operator.  The
continuous adjoint is derived by integration-by-parts before discretisation,
which gives a different operator from the discrete adjoint for finite
resolution.
"""
struct AdjointContinuous <: Mode end

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
