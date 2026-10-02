# Rotational form of the nonlinearity, Cartesian formulation:
#
#     B(a, b) = a × (∇ × b)
#
#     Nonlinear           u × ω
#     Linearised          U × (∇ × v) + v × Ω
#     AdjointDiscrete     curl⁺(w × U) + Ω × w
#     AdjointContinuous   curl (w × U) + Ω × w
#
# with ω = ∇ × u and Ω = ∇ × U. The two adjoints differ only in the curl: built
# from the adjoint derivatives for the discrete adjoint, from the forward ones
# for the continuous adjoint. In 2D the vorticity is the scalar ω = ∂x u₂ - ∂y u₁,
# stored in the first component of a VectorField, and a × (ζ e_z) = (a₂ζ, -a₁ζ).
#
# Linearised state: U and Ω in physical space.
# Physical-space fields are written in capitals; `plans(Phys, spec)` goes to
# physical space, `plans(spec, Phys)` back, `add=true` accumulates.

# workspace sizes: linearised state, spectral and physical scratch
_workspace_sizes(::Cartesian{3}, ::Rotational) = (2, 3, 3)
_workspace_sizes(::Cartesian{2}, ::Rotational) = (2, 3, 3)

# derivative mode of the curl in the adjoints
_curl_mode(::AdjointDiscrete)   = DiscreteAdjoint()
_curl_mode(::AdjointContinuous) = Direct()


# ============================================================================ #
# 3D                                                                           #
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# curl: ω = ∇ × u, or with DiscreteAdjoint its exact transpose                 #
# ---------------------------------------------------------------------------- #
#     (curl⁺w)₁ = ∂z⁺w₂ - ∂y⁺w₃
#     (curl⁺w)₂ = ∂x⁺w₃ - ∂z⁺w₁
#     (curl⁺w)₃ = ∂y⁺w₁ - ∂x⁺w₂
function _curl!(   ω::VectorField{3},
                   u::VectorField{3},
                 tmp::FTField,
                    ::Cartesian{3},
                mode::AbstractDerivativeMode)
    ddy!(ω[1], u[3], mode); ddz!(tmp, u[2], mode); ω[1] .-= tmp
    ddz!(ω[2], u[1], mode); ddx!(tmp, u[3], mode); ω[2] .-= tmp
    ddx!(ω[3], u[2], mode); ddy!(tmp, u[1], mode); ω[3] .-= tmp

    # the transpose of a difference of derivatives swaps its sign
    mode isa DiscreteAdjoint && (ω .*= -1)

    return ω
end


# ---------------------------------------------------------------------------- #
# linearised state: U, Ω                                                       #
# ---------------------------------------------------------------------------- #
function fill_linearised_state!(      state::Vector{<:VectorField{3}},
                                          u::VectorField{3},
                                formulation::Cartesian{3},
                                           ::Rotational,
                                       work::Workspace)
    U, Ω   = state
    ω, tmp = work.scache

    # ---- vorticity in spectral space ----
    _curl!(ω, u, tmp[1], formulation, Direct())

    # ---- to physical space ----
    work.plans(U, u)
    work.plans(Ω, ω)

    return state
end


# ---------------------------------------------------------------------------- #
# nonlinear: u × ω                                                             #
# ---------------------------------------------------------------------------- #
function advection!(        out::VectorField{3},
                              u::VectorField{3},
                               ::Nonlinear,
                    formulation::Cartesian{3},
                               ::Rotational,
                           work::Workspace)
    ω, tmp  = work.scache
    U, W, R = work.pcache

    # ---- u and ω in physical space ----
    _curl!(ω, u, tmp[1], formulation, Direct())

    work.plans(U, u)
    work.plans(W, ω)

    # ---- u × ω ----
    @. R[1] = U[2]*W[3] - U[3]*W[2]
    @. R[2] = U[3]*W[1] - U[1]*W[3]
    @. R[3] = U[1]*W[2] - U[2]*W[1]

    work.plans(out, R, add=true)

    return out
end


# ---------------------------------------------------------------------------- #
# linearised: U × ζ + v × Ω, with ζ = ∇ × v                                    #
# ---------------------------------------------------------------------------- #
function advection!(        out::VectorField{3},
                              v::VectorField{3},
                               ::Linearised,
                    formulation::Cartesian{3},
                               ::Rotational,
                           work::Workspace)
    # assumes linearise_about! has been called before this function
    U, Ω    = work.linearised_state
    ζ, tmp  = work.scache
    V, Z, R = work.pcache

    # ---- v and ζ in physical space ----
    _curl!(ζ, v, tmp[1], formulation, Direct())

    work.plans(V, v)
    work.plans(Z, ζ)

    # ---- U × ζ + v × Ω ----
    @. R[1] = U[2]*Z[3] - U[3]*Z[2] + V[2]*Ω[3] - V[3]*Ω[2]
    @. R[2] = U[3]*Z[1] - U[1]*Z[3] + V[3]*Ω[1] - V[1]*Ω[3]
    @. R[3] = U[1]*Z[2] - U[2]*Z[1] + V[1]*Ω[2] - V[2]*Ω[1]

    work.plans(out, R, add=true)

    return out
end


# ---------------------------------------------------------------------------- #
# adjoints: curl⁺(w × U) + Ω × w, forward curl for the continuous adjoint    #
# ---------------------------------------------------------------------------- #
function advection!(        out::VectorField{3},
                              w::VectorField{3},
                           mode::Union{AdjointDiscrete, AdjointContinuous},
                    formulation::Cartesian{3},
                               ::Rotational,
                           work::Workspace)
    # assumes linearise_about! has been called before this function
    U, Ω        = work.linearised_state
    wxU, c, tmp = work.scache
    W, R        = work.pcache

    # ---- w in physical space ----
    work.plans(W, w)

    # ---- curl⁺(w × U) ----
    @. R[1] = W[2]*U[3] - W[3]*U[2]
    @. R[2] = W[3]*U[1] - W[1]*U[3]
    @. R[3] = W[1]*U[2] - W[2]*U[1]

    work.plans(wxU, R)
    _curl!(c, wxU, tmp[1], formulation, _curl_mode(mode))
    out .+= c

    # ---- Ω × w ----
    @. R[1] = Ω[2]*W[3] - Ω[3]*W[2]
    @. R[2] = Ω[3]*W[1] - Ω[1]*W[3]
    @. R[3] = Ω[1]*W[2] - Ω[2]*W[1]

    work.plans(out, R, add=true)

    return out
end


# ============================================================================ #
# 2D                                                                           #
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# curl: scalar ω = ∂x u₂ - ∂y u₁                                               #
# ---------------------------------------------------------------------------- #
function _curl!(ω::FTField, u::VectorField{2}, tmp::FTField, ::Cartesian{2})
    ddx!(ω, u[2])
    ddy!(tmp, u[1])
    ω .-= tmp
    return ω
end

# ---------------------------------------------------------------------------- #
# transpose of the scalar curl, from a scalar c to a vector:                   #
#     discrete (-∂y⁺c, ∂x⁺c),   continuous (∂y c, -∂x c)                       #
# ---------------------------------------------------------------------------- #
function _curl_transpose!( out::VectorField{2},
                             c::FTField,
                              ::Cartesian{2},
                          mode::Union{AdjointDiscrete, AdjointContinuous})
    m = _curl_mode(mode)

    ddy!(out[1], c, m)
    ddx!(out[2], c, m)
    out[2] .*= -1

    # the transpose of a difference of derivatives swaps its sign
    mode isa AdjointDiscrete && (out .*= -1)

    return out
end


# ---------------------------------------------------------------------------- #
# linearised state: U, Ω (scalar, first component)                             #
# ---------------------------------------------------------------------------- #
function fill_linearised_state!(      state::Vector{<:VectorField{2}},
                                          u::VectorField{2},
                                formulation::Cartesian{2},
                                           ::Rotational,
                                       work::Workspace)
    U, Ω   = state
    ω, tmp = work.scache

    # ---- vorticity in spectral space ----
    _curl!(ω[1], u, tmp[1], formulation)

    # ---- to physical space ----
    work.plans(U, u)
    work.plans(Ω, ω)

    return state
end


# ---------------------------------------------------------------------------- #
# nonlinear: u × ω = (u₂ω, -u₁ω)                                               #
# ---------------------------------------------------------------------------- #
function advection!(        out::VectorField{2},
                              u::VectorField{2},
                               ::Nonlinear,
                    formulation::Cartesian{2},
                               ::Rotational,
                           work::Workspace)
    ω, tmp  = work.scache
    U, W, R = work.pcache

    # ---- u and ω in physical space ----
    _curl!(ω[1], u, tmp[1], formulation)

    work.plans(U, u)
    work.plans(W, ω)

    # ---- u × ω ----
    @. R[1] =  U[2]*W[1]
    @. R[2] = -U[1]*W[1]

    work.plans(out, R, add=true)

    return out
end


# ---------------------------------------------------------------------------- #
# linearised: U × ζ + v × Ω = (U₂ζ + v₂Ω, -U₁ζ - v₁Ω)                           #
# ---------------------------------------------------------------------------- #
function advection!(        out::VectorField{2},
                              v::VectorField{2},
                               ::Linearised,
                    formulation::Cartesian{2},
                               ::Rotational,
                           work::Workspace)
    # assumes linearise_about! has been called before this function
    U, Ω    = work.linearised_state
    ζ, tmp  = work.scache
    V, Z, R = work.pcache

    # ---- v and ζ in physical space ----
    _curl!(ζ[1], v, tmp[1], formulation)

    work.plans(V, v)
    work.plans(Z, ζ)

    # ---- U × ζ + v × Ω ----
    @. R[1] =  U[2]*Z[1] + V[2]*Ω[1]
    @. R[2] = -U[1]*Z[1] - V[1]*Ω[1]

    work.plans(out, R, add=true)

    return out
end


# ---------------------------------------------------------------------------- #
# adjoints: curl⁺(w₁U₂ - w₂U₁) + (-w₂Ω, w₁Ω)                                   #
# ---------------------------------------------------------------------------- #
function advection!(        out::VectorField{2},
                              w::VectorField{2},
                           mode::Union{AdjointDiscrete, AdjointContinuous},
                    formulation::Cartesian{2},
                               ::Rotational,
                           work::Workspace)
    # assumes linearise_about! has been called before this function
    U, Ω      = work.linearised_state
    s, c = work.scache
    W, R = work.pcache

    # ---- w in physical space ----
    work.plans(W, w)

    # ---- curl⁺ of the scalar w₁U₂ - w₂U₁ ----
    @. R[1] = W[1]*U[2] - W[2]*U[1]
    R[2] .= 0

    work.plans(s, R)
    _curl_transpose!(c, s[1], formulation, mode)
    out .+= c

    # ---- (-w₂Ω, w₁Ω) ----
    @. R[1] = -W[2]*Ω[1]
    @. R[2] =  W[1]*Ω[1]

    work.plans(out, R, add=true)

    return out
end
