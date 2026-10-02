# Two-component (u, v) planar Cartesian primitive-variable formulation.
# Same operators as the 3D formulation, without w and z-derivatives:
#
#     N(u)  = ν∇²u - (u·∇)u + f(u)
#     L v   = ν∇²v - (U·∇)v - (v·∇)U + f(v)
#     L* w  = ν∇²⁺w - ∇⁺·(U ⊗ w) - (∇U)ᵀ w + f*(w)     discrete adjoint
#     L* w  = ν∇²w  + (U·∇)w     - (∇U)ᵀ w + f*(w)     continuous adjoint
#
# State: U, ∂U/∂x, ∂U/∂y in physical space.

"""
    CartesianPrimitive2D(mode, work, Re; force=NoForce())

Two-component planar Cartesian primitive-variable Navier–Stokes operator. `mode` is
[`Nonlinear`](@ref), [`Linearised`](@ref), [`AdjointDiscrete`](@ref) or
[`AdjointContinuous`](@ref); `work` is the shared [`Workspace`](@ref). Called as
`op(t, u, out)` on spectral `VectorField`s.
"""
mutable struct CartesianPrimitive2D{MODE, T, W, BF}
             Re::T    # Reynolds number
    const  mode::MODE # Nonlinear, Linearised, AdjointDiscrete or AdjointContinuous
    const  work::W    # shared Workspace: plans, scratch, linearisation state
    const force::BF   # body force, called as force(out, u, mode)
end

CartesianPrimitive2D(mode::Mode, work::Workspace, Re; force=NoForce()) =
    CartesianPrimitive2D(_realtype(work)(Re), mode, work, force)

ncomp(       ::Type{<:CartesianPrimitive2D})                    = 2
cache_length(::Type{<:CartesianPrimitive2D}, ::Type{<:FTField}) = 3
cache_length(::Type{<:CartesianPrimitive2D}, ::Type{<:Field})   = 3
state_length(::Type{<:CartesianPrimitive2D})                    = 3

ncomp(op::CartesianPrimitive2D) = ncomp(typeof(op))


# ---------------------------------------------------------------------------- #
# linearisation state: U, ∂U/∂x, ∂U/∂y                                         #
# ---------------------------------------------------------------------------- #
function linearise_about!(op::CartesianPrimitive2D, u::VectorField{2})
    U, dUdx, dUdy = op.work.state
    dudx, dudy    = op.work.scache

    # ---- gradient in spectral space ----
    ddx!(dudx, u)
    ddy!(dudy, u)

    # ---- to physical space ----
    op.work.plans(U,    u)
    op.work.plans(dUdx, dudx)
    op.work.plans(dUdy, dudy)

    return op
end


# ---------------------------------------------------------------------------- #
# nonlinear: N(u) = ν∇²u - (u·∇)u + f(u)                                       #
# ---------------------------------------------------------------------------- #
function (op::CartesianPrimitive2D{Nonlinear})(::Real, u::VectorField{2}, out::VectorField{2})
    dudx, dudy    = op.work.scache
    U, dUdx, dUdy = op.work.pcache

    # ---- viscous term ----
    laplacian!(out, u)
    out .*= 1/op.Re

    # ---- u and its gradient in physical space ----
    ddx!(dudx, u)
    ddy!(dudy, u)

    op.work.plans(U,    u)
    op.work.plans(dUdx, dudx)
    op.work.plans(dUdy, dudy)

    # ---- -(u·∇)u, overwriting dUdx ----
    for n in 1:2
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n]
    end
    op.work.plans(out, dUdx, add=true)

    # ---- body force ----
    op.force(out, u, Forward())

    return out
end


# ---------------------------------------------------------------------------- #
# linearised: L v = ν∇²v - (U·∇)v - (v·∇)U + f(v)                              #
# ---------------------------------------------------------------------------- #
function (op::CartesianPrimitive2D{Linearised})(::Real, v::VectorField{2}, out::VectorField{2})
    U, dUdx, dUdy = op.work.state
    dvdx, dvdy    = op.work.scache
    V, dVdx, dVdy = op.work.pcache

    # ---- viscous term ----
    laplacian!(out, v)
    out .*= 1/op.Re

    # ---- v and its gradient in physical space ----
    ddx!(dvdx, v)
    ddy!(dvdy, v)

    op.work.plans(V,    v)
    op.work.plans(dVdx, dvdx)
    op.work.plans(dVdy, dvdy)

    # ---- -(U·∇)v - (v·∇)U, overwriting dVdx ----
    for n in 1:2
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n]
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n]
    end
    op.work.plans(out, dVdx, add=true)

    # ---- body force ----
    op.force(out, v, Forward())

    return out
end


# ---------------------------------------------------------------------------- #
# discrete adjoint: L* w = ν∇²⁺w - ∇⁺·(U ⊗ w) - (∇U)ᵀ w + f*(w)                #
# ---------------------------------------------------------------------------- #
function (op::CartesianPrimitive2D{AdjointDiscrete})(::Real, w::VectorField{2}, out::VectorField{2})
    U, dUdx, dUdy  = op.work.state
    u1w, u2w, tmp  = op.work.scache
    W, U1W, U2W    = op.work.pcache

    # ---- viscous term ----
    laplacian!(out, w, AdjointDiscrete())
    out .*= 1/op.Re

    # ---- products U_j w in physical space, back to spectral ----
    op.work.plans(W, w)

    for n in 1:2
        @. U1W[n] = U[1]*W[n]
        @. U2W[n] = U[2]*W[n]
    end

    op.work.plans(u1w, U1W)
    op.work.plans(u2w, U2W)

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
    op.work.plans(out, U1W, add=true)

    # ---- body force ----
    op.force(out, w, AdjointDiscrete())

    return out
end


# ---------------------------------------------------------------------------- #
# continuous adjoint: L* w = ν∇²w + (U·∇)w - (∇U)ᵀ w + f*(w)                   #
# ---------------------------------------------------------------------------- #
function (op::CartesianPrimitive2D{AdjointContinuous})(::Real, w::VectorField{2}, out::VectorField{2})
    U, dUdx, dUdy = op.work.state
    dwdx, dwdy    = op.work.scache
    W, dWdx, dWdy = op.work.pcache

    # ---- viscous term ----
    laplacian!(out, w)
    out .*= 1/op.Re

    # ---- w and its gradient in physical space ----
    ddx!(dwdx, w)
    ddy!(dwdy, w)

    op.work.plans(W,    w)
    op.work.plans(dWdx, dwdx)
    op.work.plans(dWdy, dwdy)

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

    op.work.plans(out, dWdx, add=true)
    op.work.plans(out, dWdy, add=true)

    # ---- body force ----
    op.force(out, w, AdjointContinuous())

    return out
end
