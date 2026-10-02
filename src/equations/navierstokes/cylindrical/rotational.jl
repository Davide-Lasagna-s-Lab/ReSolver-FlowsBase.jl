# Rotational form of the nonlinearity, cylindrical formulation (r, θ, z):
#
#     B(a, b) = a × (∇ × b)
#
#     Nonlinear           u × ω
#     Linearised          U × (∇ × v) + v × Ω
#     AdjointDiscrete     curl⁺(w × U) + Ω × w
#     AdjointContinuous   curl (w × U) + Ω × w
#
# The basis (e_r, e_θ, e_z) is orthonormal, so cross products are as in
# Cartesian coordinates; only the curl changes:
#
#     ω_r = (1/r)∂θ u_z - ∂z u_θ
#     ω_θ = ∂z u_r - ∂r u_z
#     ω_z = ∂r u_θ + u_θ/r - (1/r)∂θ u_r
#
# Its exact transpose, with adjoint derivatives taken with respect to the grid
# weights (which include r dr), is
#
#     (curl⁺c)_r = ∂z⁺c_θ - (1/r)∂θ⁺c_z
#     (curl⁺c)_θ = ∂r⁺c_z + c_z/r - ∂z⁺c_r
#     (curl⁺c)_z = (1/r)∂θ⁺c_r - ∂r⁺c_θ
#
# The continuous adjoint uses the forward curl, the curl being self-adjoint.
#
# Linearised state: U and Ω in physical space.
# SKETCH: the radial operators near the axis are not treated (see Cylindrical).

# workspace sizes: linearised state, spectral and physical scratch
_workspace_sizes(::Cylindrical, ::Rotational) = (2, 3, 3)


# ---------------------------------------------------------------------------- #
# curl: ω = ∇ × u                                                              #
# ---------------------------------------------------------------------------- #
function _curl!(  ω::VectorField{3},
                  u::VectorField{3},
                tmp::FTField,
                cyl::Cylindrical,
                   ::Forward)

    # ---- ω_r = (1/r)∂θ u_z - ∂z u_θ ----
    ddy!(ω[1], u[3])
    parent(ω[1]) .*= cyl.r⁻¹
    ddz!(tmp, u[2])
    ω[1] .-= tmp

    # ---- ω_θ = ∂z u_r - ∂r u_z ----
    ddz!(ω[2], u[1])
    ddx!(tmp, u[3])
    ω[2] .-= tmp

    # ---- ω_z = ∂r u_θ + u_θ/r - (1/r)∂θ u_r ----
    ddx!(ω[3], u[2])
    ddy!(tmp, u[1])
    parent(ω[3]) .+= cyl.r⁻¹ .* (parent(u[2]) .- parent(tmp))

    return ω
end


# ---------------------------------------------------------------------------- #
# discrete transpose of the curl: curl⁺c                                       #
# ---------------------------------------------------------------------------- #
function _curl!(  o::VectorField{3},
                  c::VectorField{3},
                tmp::FTField,
                cyl::Cylindrical,
                   ::AdjointDiscrete)
    m = AdjointDiscrete()

    # ---- (curl⁺c)_r = ∂z⁺c_θ - (1/r)∂θ⁺c_z ----
    ddz!(o[1], c[2], m)
    ddy!(tmp,  c[3], m)
    parent(o[1]) .-= cyl.r⁻¹ .* parent(tmp)

    # ---- (curl⁺c)_θ = ∂r⁺c_z + c_z/r - ∂z⁺c_r ----
    ddx!(o[2], c[3], m)
    ddz!(tmp,  c[1], m)
    parent(o[2]) .+= cyl.r⁻¹ .* parent(c[3]) .- parent(tmp)

    # ---- (curl⁺c)_z = (1/r)∂θ⁺c_r - ∂r⁺c_θ ----
    ddy!(o[3], c[1], m)
    parent(o[3]) .*= cyl.r⁻¹
    ddx!(tmp, c[2], m)
    o[3] .-= tmp

    return o
end


# ---------------------------------------------------------------------------- #
# linearised state: U, Ω                                                       #
# ---------------------------------------------------------------------------- #
function fill_linearised_state!(state::Vector{<:VectorField{3}},
                                    u::VectorField{3},
                                  cyl::Cylindrical,
                                     ::Rotational,
                                 work::Workspace)
    U, Ω   = state
    ω, tmp = work.scache

    # ---- vorticity in spectral space ----
    _curl!(ω, u, tmp[1], cyl, Forward())

    # ---- to physical space ----
    work.plans(U, u)
    work.plans(Ω, ω)

    return state
end


# ---------------------------------------------------------------------------- #
# nonlinear: u × ω                                                             #
# ---------------------------------------------------------------------------- #
function advection!( out::VectorField{3},
                       u::VectorField{3},
                        ::Nonlinear,
                     cyl::Cylindrical,
                        ::Rotational,
                    work::Workspace)
    ω, tmp  = work.scache
    U, W, R = work.pcache

    # ---- u and ω in physical space ----
    _curl!(ω, u, tmp[1], cyl, Forward())

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
function advection!( out::VectorField{3},
                       v::VectorField{3},
                        ::Linearised,
                     cyl::Cylindrical,
                        ::Rotational,
                    work::Workspace)
    # assumes linearise_about! has been called before this function
    U, Ω    = work.linearised_state
    ζ, tmp  = work.scache
    V, Z, R = work.pcache

    # ---- v and ζ in physical space ----
    _curl!(ζ, v, tmp[1], cyl, Forward())

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
# adjoints: curl⁺(w × U) + Ω × w, forward curl for the continuous adjoint      #
# ---------------------------------------------------------------------------- #
function advection!( out::VectorField{3},
                       w::VectorField{3},
                    mode::Union{AdjointDiscrete, AdjointContinuous},
                     cyl::Cylindrical,
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
    _curl!(c, wxU, tmp[1], cyl, _curl_mode(mode))
    out .+= c

    # ---- Ω × w ----
    @. R[1] = Ω[2]*W[3] - Ω[3]*W[2]
    @. R[2] = Ω[3]*W[1] - Ω[1]*W[3]
    @. R[3] = Ω[1]*W[2] - Ω[2]*W[1]

    work.plans(out, R, add=true)

    return out
end
