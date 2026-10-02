# Navier–Stokes operator, generic over mode, formulation and nonlinearity form.
#
#     N(u)  = ν∇²u  + B(u, u)                  + f(u)      Nonlinear
#     L v   = ν∇²v  + B(U, v) + B(v, U)        + f(v)      Linearised
#     L* w  = ν∇²⁺w + B(U, ·)* w + B(·, U)* w  + f*(w)     AdjointDiscrete / AdjointContinuous
#
# The call method below is written once. A formulation (formulations.jl)
# provides `viscous!`; each formulation–nonlinearity form pair provides
# `advection!`, `fill_linearised_state!` and the workspace sizes
# (navierstokes/advection/<formulation>/<form>.jl). Kernels take the fields explicitly, so other
# systems (e.g. Boussinesq, MHD) can reuse them on the velocity components.
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
# Mode maps                                                                    #
# ============================================================================ #

# derivative mode used by the Laplacian
_laplacian_mode(::Union{Nonlinear, Linearised, AdjointContinuous}) = Direct()
_laplacian_mode(::AdjointDiscrete)                                 = DiscreteAdjoint()


# ============================================================================ #
# Operator                                                                     #
# ============================================================================ #

"""
    NavierStokes(mode, formulation, nlform, work, Re; force=NoForce())

Navier–Stokes operator in `mode` ([`Nonlinear`](@ref), [`Linearised`](@ref),
[`AdjointDiscrete`](@ref) or [`AdjointContinuous`](@ref)), for a `formulation`
such as [`Cartesian`](@ref) and a nonlinearity form `nlform`
([`Convective`](@ref) or [`Rotational`](@ref)). `work` is the shared
[`Workspace`](@ref). Called as `op(t, u, out)` on spectral `VectorField`s; the
linearised modes act about the point set by [`linearise_about!`](@ref).
"""
struct NavierStokes{MODE, FORMULATION, NLFORM<:AbstractNonlinearityForm, T, W, BF}
             Re::T           # Reynolds number
           mode::MODE        # Nonlinear, Linearised, AdjointDiscrete or AdjointContinuous
    formulation::FORMULATION # e.g. Cartesian(3), Cylindrical(grid)
         nlform::NLFORM      # nonlinearity form, Convective() or Rotational()
           work::W           # shared Workspace: plans, scratch, linearised state
          force::BF          # body force, called as force(out, u, mode)
end

function NavierStokes(mode::AbstractEquationMode,
                      formulation,
                      nlform::AbstractNonlinearityForm,
                      work::Workspace,
                      Re;
                      force=NoForce())
    return NavierStokes(_realtype(work)(Re), mode, formulation, nlform, work, force)
end

ncomp(op::NavierStokes) = ncomp(op.formulation)


# ---------------------------------------------------------------------------- #
# the operator: viscous term, advection, body force                            #
# ---------------------------------------------------------------------------- #
function (op::NavierStokes)(::Real, u::VectorField, out::VectorField)

    # ---- viscous term ----
    viscous!(out, u, op.mode, op.formulation, op.Re, op.work)

    # ---- advection, specific to the nonlinearity form and mode ----
    advection!(out, u, op.mode, op.formulation, op.nlform, op.work)

    # ---- body force ----
    op.force(out, u, op.mode)

    return out
end


# ---------------------------------------------------------------------------- #
# linearisation point                                                          #
# ---------------------------------------------------------------------------- #
linearise_about!(op::NavierStokes, u::VectorField) =
    (fill_linearised_state!(op.work.linearised_state,
                            u,
                            op.formulation,
                            op.nlform,
                            op.work); op)
