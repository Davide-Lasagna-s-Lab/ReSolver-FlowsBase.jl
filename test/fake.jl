# ------------- #
# concrete grid #
# ------------- #
struct FakeGrid <: AbstractGrid{Float64, 2, (1, 2, nothing, nothing), (2,)}
    x::Vector{Float64}
    N::Int
    L::Float64
end

  Base.convert(::Type{T}, g::FakeGrid) where {T<:Float64} = FakeGrid(g.x, g.N, g.L)
     Base.size(           g::FakeGrid)                    = (length(g.x), g.N)
ReSolverFlowsBase.points(           g::FakeGrid; dealias=false)     = dealias ? _grid_dealiased_points(g) : _grid_points(g)

ReSolverFlowsBase.wavenumber_scale(g::FakeGrid, ::Int) = 2π/g.L
ReSolverFlowsBase.weights(g::FakeGrid) = ones(length(g.x))

_grid_points(g)           = (reshape(g.x, :, 1), reshape(collect(range(0, g.L*(1 - 1/g.N), length=g.N)), 1, :))
_grid_dealiased_points(g) = (reshape(g.x, :, 1), reshape(collect(range(0, g.L*(1 - 1/(ceil(Int, 1.5*g.N))), length=ceil(Int, 1.5*g.N))), 1, :))


ReSolverFlowsBase.add_base_flow!(u::VectorField{N, <:FTField{FakeGrid}}, base) where {N} = u


# ----------------- #
# field dot methods #
# ----------------- #
function LinearAlgebra.dot(u::FTField{FakeGrid}, v::FTField{FakeGrid})
    Nx, Ny = size(ReSolverFlowsBase.grid(u))
    sum = 0.0
    for nx in 1:Nx
        sum += real(dot(u[nx, 1], v[nx, 1]))
        for ny in 2:((Ny >> 1) + 1)
            sum += 2*real(dot(u[nx, ny], v[nx, ny]))
        end
    end
    return sum
end


# ------------------ #
# derivative methods #
# ------------------ #
struct FakeDerivativeMatrix end

function LinearAlgebra.mul!(out, ::FakeDerivativeMatrix, u, ::Val, ::Val{ADD}=Val(false)) where {ADD}
    !ADD && (out .= u; return out)
     ADD && (out .+= u; return out)
end

ReSolverFlowsBase.derivative_matrix(::FakeGrid, ::Int, ::Val, ::OperatorMode) = FakeDerivativeMatrix()
