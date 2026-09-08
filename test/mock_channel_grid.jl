# Mock channel grid used as the parent (Undecomposed) grid for the
# MPIExt integration tests.
#
# The grid mirrors the production `ReSolverChannelFlow.ChannelGrid` shape:
#
#   - 4D storage in storage-axis order `(y, x, z, t)`
#   - Single inhomogeneous direction (`:y` -> storage dim 1)
#   - Three FFT directions (`:x`, `:z`, `:t` -> storage dims 2, 3, 4)
#   - Wall-normal finite-difference operators `D₁`, `D₂` supplied by FDGrids
#
# It is intentionally minimal: only the ReSolverFlowsBase + MPIExt parent-grid
# hooks needed by `distributed(...)` are implemented. The decomposed test
# subject is always
# `distributed(g, comm; decomposed_physical_dims=(:y,), nprocesses=..., nhalo=...)`.

import FDGrids,
       Adapt

# Physical-coordinate layout for the mock channel grid. The y direction lives
# on storage dim 1, matching the production ChannelGrid convention; x, z, t
# follow on storage dims 2, 3, 4.
const MOCK_AXES = (2, 1, 3, 4)
const MOCK_FFT_ORDER = (2, 3, 4)
const MOCK_INHOMOGENEOUS_DIMS = (1,)

"""
    MockChannelGrid{S,T,D1,D2}

Single-domain mock channel grid. `S = (Nx, Ny, Nz, Nt)` follows the
production physical-coordinate order; `Base.size(grid)` permutes it into
`(Ny, Nx, Nz, Nt)` storage order via `to_storage_order`.

The grid subtypes `ReSolverFlowsBase.AbstractGrid` *directly* (no abstract
intermediate) so that ReSolverFlowsBase's specificity rules for `storage_dim` etc.
resolve unambiguously.

Fields:
- `y`: wall-normal collocation points (length `Ny`)
- `ws`: wall-normal quadrature weights (length `Ny`)
- `D₁`, `D₂`: wall-normal first- and second-derivative `FDGrids.DiffMatrix`
- `α`, `β`: streamwise and spanwise wavenumber scales
"""
struct MockChannelGrid{S, T, D1<:AbstractMatrix{T}, D2<:AbstractMatrix{T}, V<:AbstractVector{T}} <:
       ReSolverFlowsBase.AbstractGrid{T, 4, MOCK_AXES, MOCK_FFT_ORDER}
    y  :: V
    ws :: V
    D₁ :: D1
    D₂ :: D2
    α  :: T
    β  :: T

    function MockChannelGrid{S, T}(y, ws, D₁, D₂, α, β) where {S, T}
        Nx, Ny, Nz, Nt = S
        isodd(Nx) && isodd(Nz) && isodd(Nt) ||
            throw(ArgumentError("Nx, Nz, Nt must all be odd"))
        length(y) == length(ws) == Ny ||
            throw(ArgumentError("y/ws length must equal Ny"))
        size(D₁) == size(D₂) == (Ny, Ny) ||
            throw(ArgumentError("derivative matrices must be Ny x Ny"))
        return new{S, T, typeof(D₁), typeof(D₂), typeof(y)}(
            y, ws, D₁, D₂, T(α), T(β))
    end
end

# `S = (Nx, Ny, Nz, Nt)` is in physical-coordinate order; `to_storage_order`
# permutes it into storage-axis order `(Ny, Nx, Nz, Nt)` per `MOCK_AXES`.
Base.size(g::MockChannelGrid{S}) where {S} = ReSolverFlowsBase.to_storage_order(S, g)

"""
    MockChannelGrid(Ny, Nx, Nz, Nt; T=Float64, stencil_width=3)

Build a `MockChannelGrid` with a uniform wall-normal mesh on `[-1, 1]`,
quadrature weights from `FDGrids.grid`, and `FDGrids.DiffMatrix`
finite-difference operators.
"""
function MockChannelGrid(Ny::Int, Nx::Int, Nz::Int, Nt::Int;
                         T::Type=Float64, stencil_width::Int=3,
                         α::Real=2π, β::Real=2π)
    y, ws = FDGrids.grid(Ny, T(-1), T(1), FDGrids.MappedGrid(T(1)))
    D₁ = FDGrids.DiffMatrix(y, stencil_width, 1)
    D₂ = FDGrids.DiffMatrix(y, stencil_width, 2)
    return MockChannelGrid{(Nx, Ny, Nz, Nt), T}(y, ws, D₁, D₂, α, β)
end

# ------------------------------------------------------------------ #
# ReSolverFlowsBase grid interface                                             #
# ------------------------------------------------------------------ #

# Broadcast-shape coordinate arrays in storage-dim order: y varies along
# storage dim 1, x along 2, z along 3, t along 4.
function ReSolverFlowsBase.points(g::MockChannelGrid{S, T}; dealias::Bool=false) where {S, T}
    Nx, Ny, Nz, Nt = S
    if dealias
        padded_storage_size = ReSolverFlowsBase.get_padded_size(size(g),
                                                       ReSolverFlowsBase.fft_storage_dims(g))
        Nx_eff = padded_storage_size[MOCK_AXES[1]]
        Nz_eff = padded_storage_size[MOCK_AXES[3]]
        Nt_eff = padded_storage_size[MOCK_AXES[4]]
    else
        Nx_eff, Nz_eff, Nt_eff = Nx, Nz, Nt
    end
    Lx = T(2π / g.α)
    Lz = T(2π / g.β)
    x = collect(range(T(0), Lx; length=Nx_eff + 1))[1:end-1]
    z = collect(range(T(0), Lz; length=Nz_eff + 1))[1:end-1]
    t = collect(range(T(0), T(2π); length=Nt_eff + 1))[1:end-1]
    return reshape(g.y, Ny, 1, 1, 1),
           reshape(x, 1, Nx_eff, 1, 1),
           reshape(z, 1, 1, Nz_eff, 1),
           reshape(t, 1, 1, 1, Nt_eff)
end

# Quadrature weights for the single inhomogeneous (wall-normal) dimension.
ReSolverFlowsBase.weights(g::MockChannelGrid) = g.ws

# Wavenumber scales for the three FFT-transformed directions. ReSolverFlowsBase
# expects storage-dim integers here.
ReSolverFlowsBase.wavenumber_scale(g::MockChannelGrid, storage_dim::Int) =
    storage_dim == MOCK_AXES[1] ? g.α :               # x -> dim 2
    storage_dim == MOCK_AXES[3] ? g.β :               # z -> dim 3
    storage_dim == MOCK_AXES[4] ? one(eltype(g.y)) :  # t -> dim 4 (period 2π)
    throw(ArgumentError("MockChannelGrid: dim $storage_dim is not an FFT dim"))

# Scalar-type conversion for `similar(field, T)`. Identity when T matches so
# DecomposedGrid's `convert` short-circuit can return the wrapper unchanged.
Base.convert(::Type{T}, g::MockChannelGrid{S, T}) where {S, T<:Real} = g
Base.convert(::Type{T}, g::MockChannelGrid{S}) where {S, T<:Real} =
    MockChannelGrid{S, T}(Vector{T}(g.y), Vector{T}(g.ws),
                          T.(g.D₁), T.(g.D₂),
                          T(g.α), T(g.β))

# Allow resizing the homogeneous directions (FFT dims). The wall-normal
# data is preserved; only `S` changes.
function ReSolverFlowsBase.growto(g::MockChannelGrid{S, T},
                        target_size::NTuple{3, Int}) where {S, T}
    _, Ny, _, _ = S
    Nx, Nz, Nt = target_size
    return MockChannelGrid{(Nx, Ny, Nz, Nt), T}(
        g.y, g.ws, g.D₁, g.D₂, g.α, g.β)
end

# ------------------------------------------------------------------ #
# MPIExt parent-grid contract                                        #
# ------------------------------------------------------------------ #
#
# `distributed(parent, comm; decomposed_physical_dims, nprocesses, nhalo)` forwards
# `derivative_matrix(parent, storage_dim::Integer, Val{ORDER}, Val{ADJ})`
# to the parent grid. Only the wall-normal direction (storage dim 1) is
# a finite-difference direction here; the FFT directions never call this
# hook.

function ReSolverFlowsBase.derivative_matrix(g::MockChannelGrid,
                                    ::Integer,
                                    ::Val{ORDER},
                                    ::Forward) where {ORDER}
    A = ORDER == 1 ? g.D₁ :
        ORDER == 2 ? g.D₂ :
        throw(ArgumentError("MockChannelGrid: unsupported derivative order $ORDER"))
    return A
end

function ReSolverFlowsBase.derivative_matrix(g::MockChannelGrid,
                                    ::Integer,
                                    ::Val{ORDER},
                                    ::AdjointDiscrete) where {ORDER}
    A = ORDER == 1 ? g.D₁ :
        ORDER == 2 ? g.D₂ :
        throw(ArgumentError("MockChannelGrid: unsupported derivative order $ORDER"))
    return LinearAlgebra.adjoint(A)
end

# ------------------------------------------------------------------ #
# CUDAExt adapt method                                               #
# ------------------------------------------------------------------ #
# ! this doesn't work since the type parameters aren't properly propogated into the function definition
# Adapt.@adapt_structure(MockChannelGrid{S, T} where {S, T})
# eval(@macroexpand(Adapt.@adapt_structure(MockChannelGrid{S, T} where {S, T})))

function Adapt.adapt_structure(to, g::MockChannelGrid{S}) where {S}
    y  = Adapt.adapt_structure(to, g.y)
    ws = Adapt.adapt_structure(to, g.ws)
    D₁ = Adapt.adapt_structure(to, g.D₁)
    D₂ = Adapt.adapt_structure(to, g.D₂)
    return MockChannelGrid{S, Float32}(y, ws, D₁, D₂, Float32(g.α), Float32(g.β))
end

# ------------------------------------------------------------------ #
# Test helpers                                                       #
# ------------------------------------------------------------------ #

# import HaloArrays
# import LinearAlgebra
# import MPI
# import ReSolverFlowsBase

# """
#     distributed_mock(Ny, Nx, Nz, Nt, comm; nhalo=(1,), T=Float64,
#                      decomposed_physical_dims=(:y,),
#                      nprocesses=(MPI.Comm_size(comm),),
#                      stencil_width=3) -> DecomposedGrid

# Build a `MockChannelGrid` and wrap it with
# `ReSolverFlowsBase.distributed(...)` along the wall-normal `:y` direction.
# """
# function distributed_mock(Ny::Int, Nx::Int, Nz::Int, Nt::Int, comm;
#                           nhalo = (1,),
#                           T::Type = Float64,
#                           decomposed_physical_dims = (:y,),
#                           nprocesses = (MPI.Comm_size(comm),),
#                           stencil_width::Int = 3)
#     parent = MockChannelGrid(Ny, Nx, Nz, Nt;
#                              T=T, stencil_width=stencil_width)
#     nhalo_tuple = nhalo isa Integer ?
#         ntuple(_ -> Int(nhalo), length(decomposed_physical_dims)) :
#         nhalo
#     return ReSolverFlowsBase.distributed(parent, comm;
#                                      decomposed_physical_dims = decomposed_physical_dims,
#                                      nprocesses = nprocesses,
#                                      nhalo = nhalo_tuple)
# end
