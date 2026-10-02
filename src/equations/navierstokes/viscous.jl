# Viscous term of the Navier–Stokes equations, per formulation:
#
#     ν∇²u            Nonlinear, Linearised, AdjointContinuous
#     ν∇²⁺u           AdjointDiscrete
#
# with ν = 1/Re.


# ============================================================================ #
# Cartesian                                                                    #
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# viscous term: ν∇²u (∇²⁺ for the discrete adjoint)                            #
# ---------------------------------------------------------------------------- #
function viscous!(out::VectorField{N},
                    u::VectorField{N},
                 mode::AbstractEquationMode,
                     ::Cartesian{N},
                   Re::Real,
                 work::Workspace) where {N}
    laplacian!(out, u, _laplacian_mode(mode))
    out .*= 1/Re
    return out
end


# ============================================================================ #
# Cylindrical — SKETCH                                                         #
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# viscous term                                                                 #
# ---------------------------------------------------------------------------- #
#     (∇²u)_r = Δu_r - u_r/r² - (2/r²) ∂θu_θ
#     (∇²u)_θ = Δu_θ - u_θ/r² + (2/r²) ∂θu_r
#     (∇²u)_z = Δu_z
#
# with the scalar Laplacian Δf = ∂rr f + (1/r)∂r f + (1/r²)∂θθ f + ∂zz f.
# The coupling block is self-adjoint, and so are the 1/r² terms and ∂θθ, ∂zz;
# the discrete adjoint only changes ∂rr into its adjoint and (1/r)∂r into
# ∂r⁺((1/r) ·).
function viscous!(        out::VectorField{3},
                            u::VectorField{3},
                         mode::AbstractEquationMode,
                  formulation::Cylindrical,
                           Re::Real,
                         work::Workspace)
    t1, t2 = work.scache[1][1], work.scache[1][2]

    # ---- scalar Laplacian of each component ----
    for n in 1:3
        _scalar_laplacian!(out[n], u[n], t1, t2, formulation, mode)
    end

    # ---- curvature terms ----
    ddy!(t1, u[2])
    ddy!(t2, u[1])

    parent(out[1]) .-= formulation.r⁻² .* (parent(u[1]) .+ 2 .* parent(t1))
    parent(out[2]) .-= formulation.r⁻² .* (parent(u[2]) .- 2 .* parent(t2))

    out .*= 1/Re

    return out
end

# Δf = ∂rr f + (1/r)∂r f + (1/r²)∂θθ f + ∂zz f
function _scalar_laplacian!(        out::FTField,
                                      f::FTField,
                                     t1::FTField,
                                     t2::FTField,
                            formulation::Cylindrical,
                                   mode::AbstractEquationMode)
    m = _laplacian_mode(mode)

    # ---- ∂rr f ----
    _inhomogeneous_laplacian!(out, f, m)

    # ---- (1/r)∂r f, or ∂r⁺((1/r) f) for the discrete adjoint ----
    if m isa DiscreteAdjoint
        parent(t2) .= formulation.r⁻¹ .* parent(f)
        ddx!(t1, t2, m)
        parent(out) .+= parent(t1)
    else
        ddx!(t1, f)
        parent(out) .+= formulation.r⁻¹ .* parent(t1)
    end

    # ---- (1/r²)∂θθ f ----
    ddy!(t1, f)
    ddy!(t2, t1)
    parent(out) .+= formulation.r⁻² .* parent(t2)

    # ---- ∂zz f ----
    ddz!(t1, f)
    ddz!(t2, t1)
    parent(out) .+= parent(t2)

    return out
end
