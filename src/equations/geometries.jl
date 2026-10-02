# Geometries: coordinate systems the operators act on. A geometry provides the
# number of components and the viscous term; advection kernels are written per
# geometry and form.


# ============================================================================ #
# Cartesian                                                                    #
# ============================================================================ #

"""
    Cartesian(N)

Cartesian geometry with `N` velocity components and spatial directions,
`N = 2` (x, y) or `N = 3` (x, y, z).
"""
struct Cartesian{N} end

Cartesian(N::Integer) = Cartesian{Int(N)}()

ncomp(::Cartesian{N}) where {N} = N


# ---------------------------------------------------------------------------- #
# viscous term: ν∇²u (∇²⁺ for the discrete adjoint)                            #
# ---------------------------------------------------------------------------- #
function viscous!(out, u, mode, ::Cartesian, Re)
    laplacian!(out, u, _laplacian_mode(mode))
    out .*= 1/Re
    return out
end
