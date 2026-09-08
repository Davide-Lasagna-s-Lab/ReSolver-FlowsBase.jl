# Physical-space field constructors for decomposed grids.

"""
    ReSolverFlowsBase.Field(grid::DecomposedGrid; dealias=false) -> Field

Construct a zero-initialised physical-space field on a decomposed grid.

Storage is always `HaloArrays.HaloArray`-backed with face-only halo exchange
(`economic=true`). Decomposed dimensions must have positive halo widths;
non-decomposed and FFT-transformed dimensions keep zero halo width.

The interior size is `local_size(grid)` without dealiasing. With
`dealias=true`, the 3/2-rule padding is applied to the FFT-transformed
dimensions of that local size.
"""
ReSolverFlowsBase.Field(g::DecomposedGrid{T}; dealias::Bool=false) where {T} = 
    ReSolverFlowsBase.Field(g, HaloArrays.HaloArray{T}(comm(g), 
                                             local_physical_size(g; dealias), 
                                             nhalo(g); economic=true))

"""
    ReSolverFlowsBase.Field(grid::DecomposedGrid, func; dealias=false) -> Field

Construct a physical-space field by evaluating `func` at the grid points.

`func` is called as `func(coords...)` where `coords = ReSolverFlowsBase.points(grid;
dealias=dealias)`.  `points` must return per-rank local coordinate arrays
(as required by the decomposed-grid locality contract), so `func`
is evaluated only at the collocation points owned by this rank. The result is
written into the HaloArray interior, with halos left uninitialised until the
next exchange.
"""
function ReSolverFlowsBase.Field(g::DecomposedGrid{T}, func::Function; dealias::Bool=false) where {T}
    u = ReSolverFlowsBase.Field(g; dealias=dealias)
    parent(u) .= func.(ReSolverFlowsBase.points(g; dealias=dealias)...)
    return u
end
