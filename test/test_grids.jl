# Helper grid for the generic-interface test suites.
#
# `FakeGrid` (in fake.jl) is a 2-D grid with one inhomogeneous and one rfft
# dimension.  Several tests want a richer layout with multiple homogeneous
# dimensions of mixed kind (rfft + signed FFT) so we can check the
# convention-sensitive code paths in shifts, norms, weighting and
# derivatives.  `TripleGrid` provides exactly that.


# 3-D grid layout:
#
#   array dim 1: inhomogeneous (y), length Ny — wall-normal-like.
#   array dim 2: rfft           (x), length Nx, period 2π/α  (FFT_DIMS_ORDER[1]).
#   array dim 3: signed FFT     (z), length Nz, period 2π/β  (FFT_DIMS_ORDER[2]).
#
# `AXES = (2, 1, 3, nothing)` means coordinate 1 (x) → array dim 2,
# coordinate 2 (y) → array dim 1, coordinate 3 (z) → array dim 3, and
# coordinate 4 (t) is absent.  Cartesian grid test helpers keep all four
# coordinate slots so wrappers such as `ddx_4!` can dispatch to a no-op for
# grids with no time direction.
#
# `points(g)` still returns only `D == 3` arrays, in storage order:
# `(y, x, z)`.
struct TripleGrid <: AbstractGrid{Float64, 3, (2, 1, 3, nothing), (2, 3)}
    Ny :: Int
    Nx :: Int
    Nz :: Int
    α  :: Float64        # 2π/Lx
    β  :: Float64        # 2π/Lz
    ws :: Vector{Float64}
end

# Construct with unit weights by default — sufficient for tests that don't
# depend on a particular quadrature rule.
function TripleGrid(Ny, Nx, Nz; α=1.0, β=1.0)
    TripleGrid(Ny, Nx, Nz, α, β, ones(Ny))
end

Base.size(g::TripleGrid)                                  = (g.Ny, g.Nx, g.Nz)
ReSolverFlowsBase.weights(g::TripleGrid)                            = g.ws
ReSolverFlowsBase.wavenumber_scale(g::TripleGrid, dim::Int)         = dim == 2 ? g.α :
                                                            dim == 3 ? g.β :
                                                            one(g.α)
Base.convert(::Type{Float64}, g::TripleGrid)              = g

# `growto` increases the homogeneous resolution while keeping the inhomogeneous
# size (Ny) and the wavenumber scales (α, β) fixed.
function ReSolverFlowsBase.growto(g::TripleGrid, target_size::NTuple{2, Int})
    Nx_new, Nz_new = target_size
    return TripleGrid(g.Ny, Nx_new, Nz_new, g.α, g.β, g.ws)
end

# Broadcastable coordinate arrays in storage order.  The inhomogeneous
# direction is exposed as a custom non-uniform `g.y` vector; the two
# homogeneous directions cover their full periods with equally-spaced points.
ReSolverFlowsBase.points(g::TripleGrid; dealias=false) = begin
    Ny, Nx, Nz = size(g)
    y = reshape(range(-1, 1, length=Ny) |> collect, Ny, 1, 1)
    x = reshape((0:Nx-1) * (2π/g.α/Nx),             1, Nx, 1)
    z = reshape((0:Nz-1) * (2π/g.β/Nz),             1,  1, Nz)
    return (y, x, z)
end


# 2-D analytic fixture:
#
#   array dim 1: inhomogeneous y, represented by arbitrary collocation points.
#   array dim 2: rfft x, periodic with length Lx.
#
# The derivative matrices are Lagrange-collocation differentiation matrices.
# They differentiate polynomials of degree < Ny exactly at the grid points,
# which lets tests compare `ddx_2!` and `laplacian!` against analytic
# derivatives without depending on ChannelFlow's Chebyshev grid.
struct LagrangeDerivativeMatrix
    mat::Matrix{Float64}
end

LinearAlgebra.adjoint(D::LagrangeDerivativeMatrix) = LagrangeDerivativeMatrix(D.mat')

Base.:(*)(D1::LagrangeDerivativeMatrix, D2::LagrangeDerivativeMatrix) =
    LagrangeDerivativeMatrix(D1.mat*D2.mat)

function LinearAlgebra.mul!(out, D::LagrangeDerivativeMatrix, u, ::Val, ::Val{ADD}=Val(false)) where {ADD}
    for j in axes(u, 2)
        @views mul!(out[:, j], D, u[:, j], Val(ADD))
    end
    return out
end

@inline LinearAlgebra.mul!(out, D::LagrangeDerivativeMatrix, u, ::Val{true})  = mul!(out, D.mat, u, 1, 1)
@inline LinearAlgebra.mul!(out, D::LagrangeDerivativeMatrix, u, ::Val{false}) = mul!(out, D.mat, u)

function _lagrange_derivative_matrix(x::AbstractVector{<:Real})
    N = length(x)
    λ = ones(Float64, N)
    for j in 1:N, m in 1:N
        j != m && (λ[j] /= x[j] - x[m])
    end

    D = zeros(Float64, N, N)
    for i in 1:N, j in 1:N
        i != j && (D[i, j] = λ[j] / (λ[i] * (x[i] - x[j])))
    end
    for i in 1:N
        D[i, i] = -sum(D[i, j] for j in 1:N if j != i)
    end
    return LagrangeDerivativeMatrix(D)
end

struct PolynomialGrid <: AbstractGrid{Float64, 2, (2, 1, nothing, nothing), (2,)}
    y   :: Vector{Float64}
    Nx  :: Int
    Lx  :: Float64
    D1  :: LagrangeDerivativeMatrix
    D2  :: LagrangeDerivativeMatrix
    ws  :: Vector{Float64}
end

function PolynomialGrid(y::AbstractVector{<:Real}, Nx::Integer, Lx::Real=2π)
    yv = Float64.(collect(y))
    D1 = _lagrange_derivative_matrix(yv)
    D2 = D1 * D1
    return PolynomialGrid(yv, Int(Nx), Float64(Lx), D1, D2, ones(length(yv)))
end

Base.size(g::PolynomialGrid)                         = (length(g.y), g.Nx)
ReSolverFlowsBase.weights(g::PolynomialGrid)                   = g.ws
ReSolverFlowsBase.wavenumber_scale(g::PolynomialGrid, dim::Int) = dim == 2 ? 2π / g.Lx : one(g.Lx)
Base.convert(::Type{Float64}, g::PolynomialGrid)     = g

ReSolverFlowsBase.points(g::PolynomialGrid; dealias=false) = begin
    Nx = dealias ? ReSolverFlowsBase.get_padded_size(size(g), ReSolverFlowsBase.fft_storage_dims(g))[2] : g.Nx
    y = reshape(g.y, :, 1)
    x = reshape((0:Nx-1) * (g.Lx / Nx), 1, :)
    return (y, x)
end

function ReSolverFlowsBase.derivative_matrix(g::PolynomialGrid,
                                    ::Int,
                                    ::Val{ORDER},
                                    ::OperatorMode) where {ORDER}
    ORDER == 1 && return g.D1
    ORDER == 2 && return g.D2
    throw(ArgumentError("only orders 1 and 2 are available, got order=$ORDER"))
end


# Minimal grids used by several generic tests.  They intentionally implement
# only `size`; tests that use them exercise code paths that need no concrete
# coordinate arrays, weights, or derivative extensions.
struct SpectralTestGrid{S, D, AXES, FFT_DIMS_ORDER} <: AbstractGrid{Float64, D, AXES, FFT_DIMS_ORDER} end
Base.size(::SpectralTestGrid{S}) where {S} = S

struct GalerkinGrid{S} <: AbstractGrid{Float64, 2, (1, 2, nothing, nothing), (2,)}
    ws::Vector{Float64}
end

Base.size(::GalerkinGrid{S}) where {S} = S
ReSolverFlowsBase.weights(g::GalerkinGrid) = g.ws
ReSolverFlowsBase.wavenumber_scale(::GalerkinGrid, ::Int) = 1.0


# "Flipped" 3-D grid: FFT dimensions come FIRST in storage, inhomogeneous LAST.
#
#   array dim 1: rfft           (x), length Nx  — FFT_DIMS_ORDER[1]
#   array dim 2: signed FFT     (z), length Nz  — FFT_DIMS_ORDER[2]
#   array dim 3: inhomogeneous  (y), length Ny
#
# AXES = (1, 3, 2, nothing): coord x → dim 1, coord y → dim 3, coord z → dim 2.
# FFT_DIMS_ORDER = (1, 2).
#
# The `transform_size` of this grid is ((Nx>>1)+1, Nz, Ny), so FTField storage
# is (rfft, z, y) — opposite of TripleGrid.  Used to test that `homogeneous_axes`
# and `inhomogeneous_axes` work correctly regardless of which dimensions come first.
struct FlippedGrid <: AbstractGrid{Float64, 3, (1, 3, 2, nothing), (1, 2)}
    Nx :: Int
    Nz :: Int
    Ny :: Int
    ws :: Vector{Float64}
end

function FlippedGrid(Nx, Nz, Ny)
    FlippedGrid(Nx, Nz, Ny, ones(Ny))
end

# size(g) returns (Nx, Nz, Ny) — storage-order sizes.
Base.size(g::FlippedGrid) = (g.Nx, g.Nz, g.Ny)
ReSolverFlowsBase.weights(g::FlippedGrid)              = g.ws
ReSolverFlowsBase.wavenumber_scale(::FlippedGrid, ::Int) = 1.0
