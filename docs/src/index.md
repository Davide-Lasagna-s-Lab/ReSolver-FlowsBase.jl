# ReSolverFlowsBase.jl

**ReSolverFlowsBase** is a Julia package providing the shared spectral-field infrastructure
for Navier-Stokes solvers targeting wall-bounded flows.  It defines:

- a generic **grid interface** (`AbstractGrid`) that downstream packages implement
  for their specific geometry;
- four **field types** (`Field`, `FTField`, `VectorField`, `ProjectedField`) that
  wrap raw arrays and carry their grid;
- **FFT plan management** and lossless physical↔spectral transforms;
- **spectral derivative operators** and a Laplacian;
- **Galerkin projection** onto arbitrary bases;
- **continuous phase shifts**, inner products, norms;
- **concrete Cartesian primitive-variable NSE and LNSE operators** (3-D and 2-D),
  with forward, continuous-adjoint, and discrete-adjoint modes; and
- a `construct_equations` factory that wires all of the above together into a
  ready-to-use `ProjectedNSE` callable.

Downstream packages (e.g. *ChannelFlow.jl*) only need to supply a concrete grid
subtype implementing four required methods.

## Installation

ReSolverFlowsBase is not yet registered. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/Davide-Lasagna-s-Lab/ReSolverFlowsBase.jl")
```

## Quick start

```julia
using ReSolverFlowsBase

# grid is provided by a downstream package (e.g. ChannelFlow.jl)
obj = construct_equations(grid, Re, (U, nothing, nothing); flags = FFTW.MEASURE)

# Galerkin basis and a random coefficient vector
a   = ProjectedField(grid, randn(ComplexF64, M, Nk), modes)

# Evaluate nonlinear NSE: rhs = P(Δu/Re − (u·∇)u)
rhs = obj(zero(a), a)

# Evaluate linearised NSE around base state b acting on perturbation a
rhs_lin = obj(similar(a), a, b)
```

## Contents

```@contents
Pages = ["guide.md", "api.md"]
Depth = 2
```
