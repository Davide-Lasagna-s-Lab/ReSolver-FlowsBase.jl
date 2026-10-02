# Formulations: coordinate systems the operators act on. A formulation provides
# the number of components and the viscous term; advection kernels are written
# per formulation and nonlinearity form.


# ============================================================================ #
# Cartesian                                                                    #
# ============================================================================ #

"""
    Cartesian(N)

Cartesian formulation with `N` velocity components and spatial directions,
`N = 2` (x, y) or `N = 3` (x, y, z).
"""
struct Cartesian{N} end

Cartesian(N::Integer) = Cartesian{Int(N)}()

ncomp(::Cartesian{N}) where {N} = N


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
#
# Coordinates (r, θ, z), velocity (u_r, u_θ, u_z). The grid stores them in the
# coordinate slots (x, y, z): r ↔ x (inhomogeneous), θ ↔ y, z ↔ z, so ddx!,
# ddy!, ddz! act as ∂r, ∂θ, ∂z. The grid weights include the radial measure r dr,
# and its adjoint radial derivatives are taken with respect to those weights.
#
# Open: on a pipe grid the radial operators near the axis depend on the parity
# of the azimuthal wavenumber and of the component; not handled here.

"""
    Cylindrical(grid)

Cylindrical formulation (r, θ, z) with velocity (u_r, u_θ, u_z), for a grid that
stores r, θ, z in its x, y, z slots. SKETCH: viscous term only.
"""
struct Cylindrical{R}
    r⁻¹::R # 1/r, shaped to broadcast over the storage arrays of the grid
    r⁻²::R # 1/r²

    function Cylindrical(grid::AbstractGrid)
        r = points(grid)[storage_dim(grid, :x)]
        return new{typeof(r)}(1 ./ r, 1 ./ r.^2)
    end
end

ncomp(::Cylindrical) = 3


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
    if m isa AdjointDiscrete
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
