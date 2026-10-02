# Three-component (u, v, w) Cartesian primitive-variable formulation.
#
#     N(u)  = ν∇²u - (u·∇)u + f(u)
#     L v   = ν∇²v - (U·∇)v - (v·∇)U + f(v)
#     L* w  = ν∇²⁺w - ∇⁺·(U ⊗ w) - (∇U)ᵀ w + f*(w)     discrete adjoint
#     L* w  = ν∇²w  + (U·∇)w     - (∇U)ᵀ w + f*(w)     continuous adjoint
#
# State: U, ∂U/∂x, ∂U/∂y, ∂U/∂z in physical space.
# Physical-space fields are written in capitals; `plans(Phys, spec)` goes to
# physical space, `plans(spec, Phys)` back, `add=true` accumulates.

"""
    CartesianPrimitive3D(mode, work, Re; force=NoForce())

Three-component Cartesian primitive-variable Navier–Stokes operator. `mode` is
[`Nonlinear`](@ref), [`Linearised`](@ref), [`AdjointDiscrete`](@ref) or
[`AdjointContinuous`](@ref); `work` is the shared [`Workspace`](@ref). Called as
`op(t, u, out)` on spectral `VectorField`s.
"""
mutable struct CartesianPrimitive3D{MODE, T, W, BF}
             Re::T    # Reynolds number
    const  mode::MODE # Nonlinear, Linearised, AdjointDiscrete or AdjointContinuous
    const  work::W    # shared Workspace: plans, scratch, linearisation state
    const force::BF   # body force, called as force(out, u, mode)
end

CartesianPrimitive3D(mode::Mode, work::Workspace, Re; force=NoForce()) =
    CartesianPrimitive3D(_realtype(work)(Re), mode, work, force)

ncomp(       ::Type{<:CartesianPrimitive3D})                    = 3
cache_length(::Type{<:CartesianPrimitive3D}, ::Type{<:FTField}) = 4
cache_length(::Type{<:CartesianPrimitive3D}, ::Type{<:Field})   = 4
state_length(::Type{<:CartesianPrimitive3D})                    = 4

ncomp(op::CartesianPrimitive3D) = ncomp(typeof(op))


# ---------------------------------------------------------------------------- #
# linearisation state: U, ∂U/∂x, ∂U/∂y, ∂U/∂z                                  #
# ---------------------------------------------------------------------------- #
function linearise_about!(op::CartesianPrimitive3D, u::VectorField{3})
    U, dUdx, dUdy, dUdz = op.work.state
    dudx, dudy, dudz    = op.work.scache

    # ---- gradient in spectral space ----
    ddx!(dudx, u)
    ddy!(dudy, u)
    ddz!(dudz, u)

    # ---- to physical space ----
    op.work.plans(U,    u)
    op.work.plans(dUdx, dudx)
    op.work.plans(dUdy, dudy)
    op.work.plans(dUdz, dudz)

    return op
end


# ---------------------------------------------------------------------------- #
# nonlinear: N(u) = ν∇²u - (u·∇)u + f(u)                                       #
# ---------------------------------------------------------------------------- #
function (op::CartesianPrimitive3D{Nonlinear})(::Real, u::VectorField{3}, out::VectorField{3})
    dudx, dudy, dudz    = op.work.scache
    U, dUdx, dUdy, dUdz = op.work.pcache

    # ---- viscous term ----
    laplacian!(out, u)
    out .*= 1/op.Re

    # ---- u and its gradient in physical space ----
    ddx!(dudx, u)
    ddy!(dudy, u)
    ddz!(dudz, u)

    op.work.plans(U,    u)
    op.work.plans(dUdx, dudx)
    op.work.plans(dUdy, dudy)
    op.work.plans(dUdz, dudz)

    # ---- -(u·∇)u, overwriting dUdx ----
    for n in 1:3
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n] - U[3]*dUdz[n]
    end
    op.work.plans(out, dUdx, add=true)

    # ---- body force ----
    op.force(out, u, Forward())

    return out
end


# ---------------------------------------------------------------------------- #
# linearised: L v = ν∇²v - (U·∇)v - (v·∇)U + f(v)                              #
# ---------------------------------------------------------------------------- #
function (op::CartesianPrimitive3D{Linearised})(::Real, v::VectorField{3}, out::VectorField{3})
    U, dUdx, dUdy, dUdz = op.work.state
    dvdx, dvdy, dvdz    = op.work.scache
    V, dVdx, dVdy, dVdz = op.work.pcache

    # ---- viscous term ----
    laplacian!(out, v)
    out .*= 1/op.Re

    # ---- v and its gradient in physical space ----
    ddx!(dvdx, v)
    ddy!(dvdy, v)
    ddz!(dvdz, v)

    op.work.plans(V,    v)
    op.work.plans(dVdx, dvdx)
    op.work.plans(dVdy, dvdy)
    op.work.plans(dVdz, dvdz)

    # ---- -(U·∇)v - (v·∇)U, overwriting dVdx ----
    for n in 1:3
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n] - U[3]*dVdz[n]
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n] + V[3]*dUdz[n]
    end
    op.work.plans(out, dVdx, add=true)

    # ---- body force ----
    op.force(out, v, Forward())

    return out
end


# ---------------------------------------------------------------------------- #
# discrete adjoint: L* w = ν∇²⁺w - ∇⁺·(U ⊗ w) - (∇U)ᵀ w + f*(w)                #
# ---------------------------------------------------------------------------- #
function (op::CartesianPrimitive3D{AdjointDiscrete})(::Real, w::VectorField{3}, out::VectorField{3})
    U, dUdx, dUdy, dUdz = op.work.state
    u1w, u2w, u3w, tmp  = op.work.scache
    W, U1W, U2W, U3W    = op.work.pcache

    # ---- viscous term ----
    laplacian!(out, w, AdjointDiscrete())
    out .*= 1/op.Re

    # ---- products U_j w in physical space, back to spectral ----
    op.work.plans(W, w)

    for n in 1:3
        @. U1W[n] = U[1]*W[n]
        @. U2W[n] = U[2]*W[n]
        @. U3W[n] = U[3]*W[n]
    end

    op.work.plans(u1w, U1W)
    op.work.plans(u2w, U2W)
    op.work.plans(u3w, U3W)

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
    op.work.plans(out, U1W, add=true)

    # ---- body force ----
    op.force(out, w, AdjointDiscrete())

    return out
end


# ---------------------------------------------------------------------------- #
# continuous adjoint: L* w = ν∇²w + (U·∇)w - (∇U)ᵀ w + f*(w)                   #
# ---------------------------------------------------------------------------- #
function (op::CartesianPrimitive3D{AdjointContinuous})(::Real, w::VectorField{3}, out::VectorField{3})
    U, dUdx, dUdy, dUdz = op.work.state
    dwdx, dwdy, dwdz    = op.work.scache
    W, dWdx, dWdy, dWdz = op.work.pcache

    # ---- viscous term ----
    laplacian!(out, w)
    out .*= 1/op.Re

    # ---- w and its gradient in physical space ----
    ddx!(dwdx, w)
    ddy!(dwdy, w)
    ddz!(dwdz, w)

    op.work.plans(W,    w)
    op.work.plans(dWdx, dwdx)
    op.work.plans(dWdy, dwdy)
    op.work.plans(dWdz, dwdz)

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

    op.work.plans(out, dWdx, add=true)
    op.work.plans(out, dWdz, add=true)

    # ---- body force ----
    op.force(out, w, AdjointContinuous())

    return out
end
