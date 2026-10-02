# Convective form of the nonlinearity, cylindrical formulation (r, θ, z):
#
#     N(u)_n = -Σⱼ uⱼ Gⱼ(u)_n + C(u, u)_n,   Gᵣ = ∂r, G_θ = (1/r)∂θ, G_z = ∂z
#
# with the curvature terms C(u, u) = (u_θ²/r, -u_r u_θ/r, 0). Then
#
#     Nonlinear           -Σⱼ uⱼ Gⱼ(u)            + (u_θ²/r, -u_r u_θ/r, 0)
#     Linearised          -Σⱼ Uⱼ Gⱼ(v) - Σⱼ vⱼ Gⱼ(U) + M v
#     AdjointDiscrete     -Σⱼ Gⱼ⁺(Uⱼ w) - (GU)ᵀ w   + Mᵀ w
#     AdjointContinuous   +Σⱼ Uⱼ Gⱼ(w)  - (GU)ᵀ w   + Mᵀ w
#
# where M is the pointwise linearisation of the curvature terms,
#
#     M v  = ( 2U_θ v_θ/r,  -(U_r v_θ + U_θ v_r)/r,  0 )
#     Mᵀ w = ( -U_θ w_θ/r,  (2U_θ w_r - U_r w_θ)/r,  0 ),
#
# and (GU)ᵀ w has components -Σₙ wₙ Gᵢ(U)ₙ. Pointwise maps have the same
# continuous and discrete adjoint; the factor 1/r of G_θ is applied in physical
# space and commutes with the products.
#
# Linearised state: U, Gᵣ(U), G_θ(U), G_z(U) in physical space.
# SKETCH: the radial operators near the axis are not treated (see Cylindrical).

# workspace sizes: linearised state, spectral and physical scratch
_workspace_sizes(::Cylindrical, ::Convective) = (4, 4, 4)


# ---------------------------------------------------------------------------- #
# linearised state: U, ∂r U, (1/r)∂θ U, ∂z U                                   #
# ---------------------------------------------------------------------------- #
function fill_linearised_state!(state::Vector{<:VectorField{3}},
                                    u::VectorField{3},
                                  cyl::Cylindrical,
                                     ::Convective,
                                 work::Workspace)
    U, GrU, GθU, GzU = state
    durr, duθ, duz   = work.scache

    # ---- derivatives in spectral space ----
    ddx!(durr, u)
    ddy!(duθ,  u)
    ddz!(duz,  u)

    # ---- to physical space, 1/r on the azimuthal derivative ----
    work.plans(U,   u)
    work.plans(GrU, durr)
    work.plans(GθU, duθ)
    work.plans(GzU, duz)

    for n in 1:3
        parent(GθU[n]) .*= cyl.r⁻¹
    end

    return state
end


# ---------------------------------------------------------------------------- #
# nonlinear: -Σⱼ uⱼ Gⱼ(u) + (u_θ²/r, -u_r u_θ/r, 0)                            #
# ---------------------------------------------------------------------------- #
function advection!( out::VectorField{3},
                       u::VectorField{3},
                        ::Nonlinear,
                     cyl::Cylindrical,
                        ::Convective,
                    work::Workspace)
    durr, duθ, duz   = work.scache
    U, GrU, GθU, GzU = work.pcache

    # ---- u and its derivatives in physical space ----
    ddx!(durr, u)
    ddy!(duθ,  u)
    ddz!(duz,  u)

    work.plans(U,   u)
    work.plans(GrU, durr)
    work.plans(GθU, duθ)
    work.plans(GzU, duz)

    for n in 1:3
        parent(GθU[n]) .*= cyl.r⁻¹
    end

    # ---- -Σⱼ uⱼ Gⱼ(u), overwriting GrU ----
    for n in 1:3
        @. GrU[n] = -U[1]*GrU[n] - U[2]*GθU[n] - U[3]*GzU[n]
    end

    # ---- curvature terms ----
    parent(GrU[1]) .+= cyl.r⁻¹ .* parent(U[2]) .* parent(U[2])
    parent(GrU[2]) .-= cyl.r⁻¹ .* parent(U[1]) .* parent(U[2])

    work.plans(out, GrU, add=true)

    return out
end


# ---------------------------------------------------------------------------- #
# linearised: -Σⱼ Uⱼ Gⱼ(v) - Σⱼ vⱼ Gⱼ(U) + M v                                  #
# ---------------------------------------------------------------------------- #
function advection!( out::VectorField{3},
                       v::VectorField{3},
                        ::Linearised,
                     cyl::Cylindrical,
                        ::Convective,
                    work::Workspace)
    # assumes linearise_about! has been called before this function
    U, GrU, GθU, GzU = work.linearised_state
    dvr, dvθ, dvz    = work.scache
    V, GrV, GθV, GzV = work.pcache

    # ---- v and its derivatives in physical space ----
    ddx!(dvr, v)
    ddy!(dvθ, v)
    ddz!(dvz, v)

    work.plans(V,   v)
    work.plans(GrV, dvr)
    work.plans(GθV, dvθ)
    work.plans(GzV, dvz)

    for n in 1:3
        parent(GθV[n]) .*= cyl.r⁻¹
    end

    # ---- -Σⱼ Uⱼ Gⱼ(v) - Σⱼ vⱼ Gⱼ(U), overwriting GrV ----
    for n in 1:3
        @. GrV[n]  = -U[1]*GrV[n] - U[2]*GθV[n] - U[3]*GzV[n]
        @. GrV[n] -=  V[1]*GrU[n] + V[2]*GθU[n] + V[3]*GzU[n]
    end

    # ---- curvature: M v ----
    parent(GrV[1]) .+= cyl.r⁻¹ .* 2 .* parent(U[2]) .* parent(V[2])
    parent(GrV[2]) .-= cyl.r⁻¹ .* (parent(U[1]) .* parent(V[2]) .+ parent(U[2]) .* parent(V[1]))

    work.plans(out, GrV, add=true)

    return out
end


# ---------------------------------------------------------------------------- #
# discrete adjoint: -Σⱼ Gⱼ⁺(Uⱼ w) - (GU)ᵀ w + Mᵀ w                             #
# ---------------------------------------------------------------------------- #
function advection!( out::VectorField{3},
                       w::VectorField{3},
                        ::AdjointDiscrete,
                     cyl::Cylindrical,
                        ::Convective,
                    work::Workspace)
    # assumes linearise_about! has been called before this function
    U, GrU, GθU, GzU   = work.linearised_state
    u1w, u2w, u3w, tmp = work.scache
    W, U1W, U2W, U3W   = work.pcache

    # ---- products Uⱼ w in physical space (1/r with U_θ), back to spectral ----
    work.plans(W, w)

    for n in 1:3
        @. U1W[n] = U[1]*W[n]
        @. U2W[n] = U[2]*W[n]
        @. U3W[n] = U[3]*W[n]
        parent(U2W[n]) .*= cyl.r⁻¹
    end

    work.plans(u1w, U1W)
    work.plans(u2w, U2W)
    work.plans(u3w, U3W)

    # ---- -Σⱼ ∂ⱼ⁺(·), with adjoint derivatives ----
    for n in 1:3
        ddx!(tmp[1], u1w[n], DiscreteAdjoint())
        ddy!(tmp[2], u2w[n], DiscreteAdjoint())
        ddz!(tmp[3], u3w[n], DiscreteAdjoint())
        out[n] .-= tmp[1] .+ tmp[2] .+ tmp[3]
    end

    # ---- -(GU)ᵀ w + Mᵀ w, reusing U1W ----
    _transpose_terms!(U1W, W, U, GrU, GθU, GzU, cyl)
    work.plans(out, U1W, add=true)

    return out
end


# ---------------------------------------------------------------------------- #
# continuous adjoint: +Σⱼ Uⱼ Gⱼ(w) - (GU)ᵀ w + Mᵀ w                            #
# ---------------------------------------------------------------------------- #
function advection!( out::VectorField{3},
                       w::VectorField{3},
                        ::AdjointContinuous,
                     cyl::Cylindrical,
                        ::Convective,
                    work::Workspace)
    # assumes linearise_about! has been called before this function
    U, GrU, GθU, GzU = work.linearised_state
    dwr, dwθ, dwz    = work.scache
    W, GrW, GθW, GzW = work.pcache

    # ---- w and its derivatives in physical space ----
    ddx!(dwr, w)
    ddy!(dwθ, w)
    ddz!(dwz, w)

    work.plans(W,   w)
    work.plans(GrW, dwr)
    work.plans(GθW, dwθ)
    work.plans(GzW, dwz)

    for n in 1:3
        parent(GθW[n]) .*= cyl.r⁻¹
    end

    # ---- +Σⱼ Uⱼ Gⱼ(w), overwriting GrW ----
    for n in 1:3
        @. GrW[n] = U[1]*GrW[n] + U[2]*GθW[n] + U[3]*GzW[n]
    end
    work.plans(out, GrW, add=true)

    # ---- -(GU)ᵀ w + Mᵀ w, reusing GzW ----
    _transpose_terms!(GzW, W, U, GrU, GθU, GzU, cyl)
    work.plans(out, GzW, add=true)

    return out
end


# ---------------------------------------------------------------------------- #
# pointwise terms shared by both adjoints: R = -(GU)ᵀ w + Mᵀ w                 #
# ---------------------------------------------------------------------------- #
function _transpose_terms!(  R::VectorField{3},
                             W::VectorField{3},
                             U::VectorField{3},
                           GrU::VectorField{3},
                           GθU::VectorField{3},
                           GzU::VectorField{3},
                           cyl::Cylindrical)

    # ---- -(GU)ᵀ w, component i = -Σₙ wₙ Gᵢ(U)ₙ ----
    R .= 0
    for n in 1:3
        @. R[1] -= W[n]*GrU[n]
        @. R[2] -= W[n]*GθU[n]
        @. R[3] -= W[n]*GzU[n]
    end

    # ---- Mᵀ w ----
    parent(R[1]) .-= cyl.r⁻¹ .* parent(U[2]) .* parent(W[2])
    parent(R[2]) .+= cyl.r⁻¹ .* (2 .* parent(U[2]) .* parent(W[1]) .- parent(U[1]) .* parent(W[2]))

    return R
end
