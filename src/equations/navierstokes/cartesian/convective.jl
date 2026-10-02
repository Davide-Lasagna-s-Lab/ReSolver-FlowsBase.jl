# Convective form of the nonlinearity, Cartesian formulation:
#
#     B(a, b) = -(a·∇) b
#
#     Nonlinear           -(u·∇)u
#     Linearised          -(U·∇)v - (v·∇)U
#     AdjointDiscrete     -∇⁺·(U ⊗ w) - (∇U)ᵀ w
#     AdjointContinuous   +(U·∇)w     - (∇U)ᵀ w
#
# Linearised state: U and its gradient in physical space.
# Physical-space fields are written in capitals; `plans(Phys, spec)` goes to
# physical space, `plans(spec, Phys)` back, `add=true` accumulates.

# workspace sizes: linearised state, spectral and physical scratch
_workspace_sizes(::Cartesian{3}, ::Convective) = (4, 4, 4)
_workspace_sizes(::Cartesian{2}, ::Convective) = (3, 3, 3)


# ============================================================================ #
# 3D                                                                           #
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# linearised state: U, ∂U/∂x, ∂U/∂y, ∂U/∂z                                     #
# ---------------------------------------------------------------------------- #
function fill_linearised_state!(state::Vector{<:VectorField{3}},
                                    u::VectorField{3},
                                     ::Cartesian{3},
                                     ::Convective,
                                 work::Workspace)
    U, dUdx, dUdy, dUdz = state
    dudx, dudy, dudz    = work.scache

    # ---- gradient in spectral space ----
    ddx!(dudx, u)
    ddy!(dudy, u)
    ddz!(dudz, u)

    # ---- to physical space ----
    work.plans(U,    u)
    work.plans(dUdx, dudx)
    work.plans(dUdy, dudy)
    work.plans(dUdz, dudz)

    return state
end


# ---------------------------------------------------------------------------- #
# nonlinear: -(u·∇)u                                                           #
# ---------------------------------------------------------------------------- #
function advection!( out::VectorField{3},
                       u::VectorField{3},
                        ::Nonlinear,
                        ::Cartesian{3},
                        ::Convective,
                    work::Workspace)
    dudx, dudy, dudz    = work.scache
    U, dUdx, dUdy, dUdz = work.pcache

    # ---- u and its gradient in physical space ----
    ddx!(dudx, u)
    ddy!(dudy, u)
    ddz!(dudz, u)

    work.plans(U,    u)
    work.plans(dUdx, dudx)
    work.plans(dUdy, dudy)
    work.plans(dUdz, dudz)

    # ---- -(u·∇)u, overwriting dUdx ----
    for n in 1:3
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n] - U[3]*dUdz[n]
    end
    work.plans(out, dUdx, add=true)

    return out
end


# ---------------------------------------------------------------------------- #
# linearised: -(U·∇)v - (v·∇)U                                                 #
# ---------------------------------------------------------------------------- #
function advection!( out::VectorField{3},
                       v::VectorField{3},
                        ::Linearised,
                        ::Cartesian{3},
                        ::Convective,
                    work::Workspace)
    # assumes linearise_about! has been called before this function
    U, dUdx, dUdy, dUdz = work.linearised_state
    dvdx, dvdy, dvdz    = work.scache
    V, dVdx, dVdy, dVdz = work.pcache

    # ---- v and its gradient in physical space ----
    ddx!(dvdx, v)
    ddy!(dvdy, v)
    ddz!(dvdz, v)

    work.plans(V,    v)
    work.plans(dVdx, dvdx)
    work.plans(dVdy, dvdy)
    work.plans(dVdz, dvdz)

    # ---- -(U·∇)v - (v·∇)U, overwriting dVdx ----
    for n in 1:3
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n] - U[3]*dVdz[n]
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n] + V[3]*dUdz[n]
    end
    work.plans(out, dVdx, add=true)

    return out
end


# ---------------------------------------------------------------------------- #
# discrete adjoint: -∇⁺·(U ⊗ w) - (∇U)ᵀ w                                      #
# ---------------------------------------------------------------------------- #
function advection!( out::VectorField{3},
                       w::VectorField{3},
                        ::AdjointDiscrete,
                        ::Cartesian{3},
                        ::Convective,
                    work::Workspace)
    # assumes linearise_about! has been called before this function
    U, dUdx, dUdy, dUdz = work.linearised_state
    u1w, u2w, u3w, tmp  = work.scache
    W, U1W, U2W, U3W    = work.pcache

    # ---- products U_j w in physical space, back to spectral ----
    work.plans(W, w)

    for n in 1:3
        @. U1W[n] = U[1]*W[n]
        @. U2W[n] = U[2]*W[n]
        @. U3W[n] = U[3]*W[n]
    end

    work.plans(u1w, U1W)
    work.plans(u2w, U2W)
    work.plans(u3w, U3W)

    # ---- -∇⁺·(U ⊗ w), with adjoint derivatives ----
    for n in 1:3
        ddx!(tmp[1], u1w[n], AdjointDiscrete())
        ddy!(tmp[2], u2w[n], AdjointDiscrete())
        ddz!(tmp[3], u3w[n], AdjointDiscrete())
        out[n] .-= tmp[1] .+ tmp[2] .+ tmp[3]
    end

    # ---- -(∇U)ᵀ w, component i = -Σₙ wₙ ∂ᵢUₙ, reusing U1W ----
    U1W .= 0
    for n in 1:3
        @. U1W[1] -= W[n]*dUdx[n]
        @. U1W[2] -= W[n]*dUdy[n]
        @. U1W[3] -= W[n]*dUdz[n]
    end
    work.plans(out, U1W, add=true)

    return out
end


# ---------------------------------------------------------------------------- #
# continuous adjoint: +(U·∇)w - (∇U)ᵀ w                                        #
# ---------------------------------------------------------------------------- #
function advection!( out::VectorField{3},
                       w::VectorField{3},
                        ::AdjointContinuous,
                        ::Cartesian{3},
                        ::Convective,
                    work::Workspace)
    # assumes linearise_about! has been called before this function
    U, dUdx, dUdy, dUdz = work.linearised_state
    dwdx, dwdy, dwdz    = work.scache
    W, dWdx, dWdy, dWdz = work.pcache

    # ---- w and its gradient in physical space ----
    ddx!(dwdx, w)
    ddy!(dwdy, w)
    ddz!(dwdz, w)

    work.plans(W,    w)
    work.plans(dWdx, dwdx)
    work.plans(dWdy, dwdy)
    work.plans(dWdz, dwdz)

    # ---- +(U·∇)w, overwriting dWdx ----
    for n in 1:3
        @. dWdx[n] = U[1]*dWdx[n] + U[2]*dWdy[n] + U[3]*dWdz[n]
    end

    # ---- -(∇U)ᵀ w, reusing dWdz ----
    dWdz .= 0
    for i in 1:3
        @. dWdz[1] -= W[i]*dUdx[i]
        @. dWdz[2] -= W[i]*dUdy[i]
        @. dWdz[3] -= W[i]*dUdz[i]
    end

    work.plans(out, dWdx, add=true)
    work.plans(out, dWdz, add=true)

    return out
end


# ============================================================================ #
# 2D                                                                           #
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# linearised state: U, ∂U/∂x, ∂U/∂y                                            #
# ---------------------------------------------------------------------------- #
function fill_linearised_state!(state::Vector{<:VectorField{2}},
                                    u::VectorField{2},
                                     ::Cartesian{2},
                                     ::Convective,
                                 work::Workspace)
    U, dUdx, dUdy = state
    dudx, dudy    = work.scache

    # ---- gradient in spectral space ----
    ddx!(dudx, u)
    ddy!(dudy, u)

    # ---- to physical space ----
    work.plans(U,    u)
    work.plans(dUdx, dudx)
    work.plans(dUdy, dudy)

    return state
end


# ---------------------------------------------------------------------------- #
# nonlinear: -(u·∇)u                                                           #
# ---------------------------------------------------------------------------- #
function advection!( out::VectorField{2},
                       u::VectorField{2},
                        ::Nonlinear,
                        ::Cartesian{2},
                        ::Convective,
                    work::Workspace)
    dudx, dudy    = work.scache
    U, dUdx, dUdy = work.pcache

    # ---- u and its gradient in physical space ----
    ddx!(dudx, u)
    ddy!(dudy, u)

    work.plans(U,    u)
    work.plans(dUdx, dudx)
    work.plans(dUdy, dudy)

    # ---- -(u·∇)u, overwriting dUdx ----
    for n in 1:2
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n]
    end
    work.plans(out, dUdx, add=true)

    return out
end


# ---------------------------------------------------------------------------- #
# linearised: -(U·∇)v - (v·∇)U                                                 #
# ---------------------------------------------------------------------------- #
function advection!( out::VectorField{2},
                       v::VectorField{2},
                        ::Linearised,
                        ::Cartesian{2},
                        ::Convective,
                    work::Workspace)
    # assumes linearise_about! has been called before this function
    U, dUdx, dUdy = work.linearised_state
    dvdx, dvdy    = work.scache
    V, dVdx, dVdy = work.pcache

    # ---- v and its gradient in physical space ----
    ddx!(dvdx, v)
    ddy!(dvdy, v)

    work.plans(V,    v)
    work.plans(dVdx, dvdx)
    work.plans(dVdy, dvdy)

    # ---- -(U·∇)v - (v·∇)U, overwriting dVdx ----
    for n in 1:2
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n]
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n]
    end
    work.plans(out, dVdx, add=true)

    return out
end


# ---------------------------------------------------------------------------- #
# discrete adjoint: -∇⁺·(U ⊗ w) - (∇U)ᵀ w                                      #
# ---------------------------------------------------------------------------- #
function advection!( out::VectorField{2},
                       w::VectorField{2},
                        ::AdjointDiscrete,
                        ::Cartesian{2},
                        ::Convective,
                    work::Workspace)
    # assumes linearise_about! has been called before this function
    U, dUdx, dUdy = work.linearised_state
    u1w, u2w, tmp = work.scache
    W, U1W, U2W   = work.pcache

    # ---- products U_j w in physical space, back to spectral ----
    work.plans(W, w)

    for n in 1:2
        @. U1W[n] = U[1]*W[n]
        @. U2W[n] = U[2]*W[n]
    end

    work.plans(u1w, U1W)
    work.plans(u2w, U2W)

    # ---- -∇⁺·(U ⊗ w), with adjoint derivatives ----
    for n in 1:2
        ddx!(tmp[1], u1w[n], AdjointDiscrete())
        ddy!(tmp[2], u2w[n], AdjointDiscrete())
        out[n] .-= tmp[1] .+ tmp[2]
    end

    # ---- -(∇U)ᵀ w, component i = -Σₙ wₙ ∂ᵢUₙ, reusing U1W ----
    U1W .= 0
    for n in 1:2
        @. U1W[1] -= W[n]*dUdx[n]
        @. U1W[2] -= W[n]*dUdy[n]
    end
    work.plans(out, U1W, add=true)

    return out
end


# ---------------------------------------------------------------------------- #
# continuous adjoint: +(U·∇)w - (∇U)ᵀ w                                        #
# ---------------------------------------------------------------------------- #
function advection!( out::VectorField{2},
                       w::VectorField{2},
                        ::AdjointContinuous,
                        ::Cartesian{2},
                        ::Convective,
                    work::Workspace)
    # assumes linearise_about! has been called before this function
    U, dUdx, dUdy = work.linearised_state
    dwdx, dwdy    = work.scache
    W, dWdx, dWdy = work.pcache

    # ---- w and its gradient in physical space ----
    ddx!(dwdx, w)
    ddy!(dwdy, w)

    work.plans(W,    w)
    work.plans(dWdx, dwdx)
    work.plans(dWdy, dwdy)

    # ---- +(U·∇)w, overwriting dWdx ----
    for n in 1:2
        @. dWdx[n] = U[1]*dWdx[n] + U[2]*dWdy[n]
    end

    # ---- -(∇U)ᵀ w, reusing dWdy ----
    dWdy .= 0
    for i in 1:2
        @. dWdy[1] -= W[i]*dUdx[i]
        @. dWdy[2] -= W[i]*dUdy[i]
    end

    work.plans(out, dWdx, add=true)
    work.plans(out, dWdy, add=true)

    return out
end
