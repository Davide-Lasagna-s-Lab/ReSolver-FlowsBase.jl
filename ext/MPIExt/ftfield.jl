# Spectral field constructors for decomposed grids.

"""
    ReSolverFlowsBase.FTField(grid::DecomposedGrid) -> FTField

Construct a zero-initialised spectral field on a decomposed grid.

Storage is always `HaloArrays.HaloArray`-backed with face-only halo exchange
(`economic=true`). Decomposed dimensions must have positive halo widths;
non-decomposed and FFT-transformed dimensions keep zero halo width. The
interior size is derived from `local_size(grid)`: the first FFT storage
dimension is stored as the rfft half-spectrum, while all other dimensions keep
their local extent.

**Note**: halos are indexed as part of the Fourier coefficient array along
the decomposed (physical-space) dimension.  This is the intended layout for
finite-difference stencils applied in spectral space — each Fourier mode
carries a row of wall-normal coefficients, and the halo stores the ghost rows
needed by the FD stencil at process boundaries.
"""
ReSolverFlowsBase.FTField(g::DecomposedGrid{T}) where {T} =
    ReSolverFlowsBase.FTField(g, HaloArrays.HaloArray{Complex{T}}(comm(g), 
                                                        local_transform_size(g), 
                                                        nhalo(g); economic=true))
