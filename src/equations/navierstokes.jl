# Navier–Stokes operator, generic over mode, geometry and form of the advection.
#
#     N(u)  = ν∇²u  + B(u, u)                  + f(u)      Nonlinear
#     L v   = ν∇²v  + B(U, v) + B(v, U)        + f(v)      Linearised
#     L* w  = ν∇²⁺w + B(U, ·)* w + B(·, U)* w  + f*(w)     AdjointDiscrete / AdjointContinuous
#
# The call method below is written once. A geometry (geometries.jl) provides
# `viscous!`; each geometry–form pair provides `advection!`,
# `fill_linearised_state!` and the workspace sizes (navierstokes_<form>.jl). Kernels take the fields explicitly, so other systems
# (e.g. Boussinesq, MHD) can reuse them on the velocity components.
#
# NOTE — possible future generalisation: a "quadratic system" framework.
# Navier–Stokes, Boussinesq (state u, θ) and MHD (state u, b) all have the form
#
#     ∂t q = A q + B(q, q) + f
#
# with A linear and B bilinear. The four modes then follow generically:
# nonlinear A q + B(q, q), linearised A v + B(Q, v) + B(v, Q), and the adjoints
# by transposing A and the two partial maps of B. A system would supply only A,
# B and its transposes; the skeleton below, the workspace and the projection
# would be shared. Worth implementing once a second system is needed.


# ============================================================================ #
# Advection forms                                                              #
# ============================================================================ #

"""
    Convective()

Convective form of the advection, `-(u·∇)u`.
"""
struct Convective end

"""
    Rotational()

Rotational form of the advection, `u × ω` with `ω = ∇ × u`. It differs from the
convective form by the gradient `∇(|u|²/2)`, absorbed in the pressure.
"""
struct Rotational end


# ============================================================================ #
# Mode maps                                                                    #
# ============================================================================ #

# derivative mode used by the Laplacian
_laplacian_mode(::Union{Nonlinear, Linearised, AdjointContinuous}) = Forward()
_laplacian_mode(::AdjointDiscrete)                                 = AdjointDiscrete()

# mode passed to the body force
_force_mode(::Union{Nonlinear, Linearised}) = Forward()
_force_mode(mode::Union{AdjointDiscrete, AdjointContinuous}) = mode


# ============================================================================ #
# Operator                                                                     #
# ============================================================================ #

"""
    NavierStokes(mode, geometry, form, work, Re; force=NoForce())

Navier–Stokes operator in `mode` ([`Nonlinear`](@ref), [`Linearised`](@ref),
[`AdjointDiscrete`](@ref) or [`AdjointContinuous`](@ref)), for a `geometry`
such as [`Cartesian`](@ref) and an advection `form` ([`Convective`](@ref) or
[`Rotational`](@ref)). `work` is
the shared [`Workspace`](@ref). Called as `op(t, u, out)` on spectral
`VectorField`s; the linearised modes act about the point set by
[`linearise_about!`](@ref).
"""
mutable struct NavierStokes{MODE, GEOM, FORM, T, W, BF}
             Re::T    # Reynolds number
    const  mode::MODE # Nonlinear, Linearised, AdjointDiscrete or AdjointContinuous
    const  geom::GEOM # geometry, e.g. Cartesian{3}
    const  form::FORM # advection form, Convective or Rotational
    const  work::W    # shared Workspace: plans, scratch, linearised state
    const force::BF   # body force, called as force(out, u, mode)
end

function NavierStokes(mode::Mode, geom, form, work::Workspace, Re; force=NoForce())
    return NavierStokes(_realtype(work)(Re), mode, geom, form, work, force)
end

ncomp(op::NavierStokes) = ncomp(op.geom)


# ---------------------------------------------------------------------------- #
# the operator: viscous term, advection, body force                            #
# ---------------------------------------------------------------------------- #
function (op::NavierStokes)(::Real, u::VectorField, out::VectorField)

    # ---- viscous term ----
    viscous!(out, u, op.mode, op.geom, op.Re)

    # ---- advection, form- and mode-specific ----
    advection!(out, u, op.mode, op.geom, op.form, op.work)

    # ---- body force ----
    op.force(out, u, _force_mode(op.mode))

    return out
end


# ---------------------------------------------------------------------------- #
# linearisation point                                                          #
# ---------------------------------------------------------------------------- #
function linearise_about!(op::NavierStokes, u::VectorField)
    fill_linearised_state!(op.work.linearised_state, u, op.geom, op.form, op.work)
    return op
end
