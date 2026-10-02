# Formulations: coordinate systems the operators act on. A formulation provides
# the number of components; the Navier–Stokes viscous term and advection kernels
# are written per formulation in navierstokes/.


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
stores r, θ, z in its x, y, z slots. SKETCH.
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
