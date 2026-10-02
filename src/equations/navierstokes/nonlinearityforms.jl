# Forms of the nonlinear (advection) term of the Navier–Stokes equations.
# They differ by gradients, absorbed in the pressure, so projected onto
# divergence-free modes they give the same operators up to aliasing.

"""
    AbstractNonlinearityForm

Supertype of the forms of the nonlinear term: [`Convective`](@ref) and
[`Rotational`](@ref).
"""
abstract type AbstractNonlinearityForm end

"""
    Convective()

Convective form of the nonlinearity, `-(u·∇)u`.
"""
struct Convective <: AbstractNonlinearityForm end

"""
    Rotational()

Rotational form of the nonlinearity, `u × ω` with `ω = ∇ × u`. It differs
from the convective form by the gradient `∇(|u|²/2)`, absorbed in the pressure.
"""
struct Rotational <: AbstractNonlinearityForm end
