# ReSolverFlowsBase.jl

**ReSolverFlowsBase** is a Julia package that provides the shared spectral-field infrastructure
for Navier-Stokes solvers targeting wall-bounded flows.  It defines the grid
interface, all field types, FFT transforms, spectral derivative operators,
Galerkin projection utilities, and concrete Cartesian primitive-variable NSE
operators.  Downstream packages (e.g. *ChannelFlow.jl*) contribute a concrete
`AbstractGrid` subtype and obtain a fully-featured, allocation-free spectral
solver without reimplementing any of the above.

## Installation

ReSolverFlowsBase registered via [Registry.jl](https://github.com/Davide-Lasagna-s-Lab/Registry.jl.git). Alternatively it can be installed directly from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/Davide-Lasagna-s-Lab/ReSolverFlowsBase.jl")
```

## Quick start

```julia
using ReSolverFlowsBase

# A concrete grid is provided by a downstream package, e.g. ChannelFlow.jl.
# Assuming `grid` is already constructed and `U` holds the laminar base flow:
obj = construct_equations(grid, Re, (U, nothing, nothing); flags = FFTW.MEASURE)

# Evaluate the nonlinear NSE residual projected onto a Galerkin basis
a   = ProjectedField(grid, randn(ComplexF64, M, Nk), modes)
rhs = obj(zero(a), a)          # rhs = P(Δu/Re − (u·∇)u)

# Evaluate the linearised operator around a given base state `b`
rhs_lin = obj(similar(a), a, b)
```

## Documentation

Full documentation — including a concepts & conventions guide and the
complete API reference — lives in [`docs/`](docs/).

## Extending ReSolverFlowsBase

Implement a new grid by subtyping
`AbstractGrid{T, D, AXES, FFT_DIMS_ORDER, DECOMPOSITION}` and defining four
required methods. Use `Undecomposed` for a grid stored on one domain, or
`Decomposed{DIMS}` when storage is partitioned along the dimensions in `DIMS`.

| Method | Purpose |
|--------|---------|
| `Base.size(grid)` | Physical-space array size |
| `points(grid; dealias)` | Collocation coordinate arrays |
| `wavenumber_scale(grid, dim)` | `2π/L` for spatial, `1` for temporal |
| `weights(grid)` | Quadrature weights for inhomogeneous dimensions |

See the [Concepts & Conventions](docs/src/guide.md) page for the full interface
contract and the assumptions ReSolverFlowsBase makes about grid geometry.
