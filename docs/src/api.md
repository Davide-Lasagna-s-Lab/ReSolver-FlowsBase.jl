# API Reference

## Grid interface

```@docs
AbstractGrid
GridDecomposition
Undecomposed
Decomposed
decomposition_dims
ndecomposed_dims
fft_dims
spatial_fft_dims
inhomogeneous_dims
spatial_inhomogeneous_dims
to_storage_order
transform_size
fft_norm
growto
points
wavenumber_scale
weights
distributed (requires MPI)
CUDA.cu (requires CUDA)
```

## Wavenumber indexing

```@docs
WaveNumberVector
to_homogeneous_indices
to_wavenumber_vector
```

## Field types

```@docs
Field
FTField
VectorField
ProjectedField
grid
modes
```

## Transforms

```@docs
FFTPlans
FFT
IFFT
```

## Derivatives

```@docs
ddx!
ddx_1!
ddx_2!
ddx_3!
ddx_4!
laplacian!
```

## Shifts

```@docs
shift!
shift
```

## Inner products and norms

```@docs
ReSolverFlowsBase.dot
ReSolverFlowsBase.norm
normdiff
minnormdiff
```

## Galerkin projection

```@docs
LoopGalerkin
GemmGalerkin
project!
project
expand!
expand
```

## Weighting

```@docs
FarazmandWeight
```

## Base-flow utilities

```@docs
add_base_flow!
```

## NSE formulations

```@docs
CartesianPrimitive3D
CartesianPrimitive2D
PolarPrimitive
ncomp
cache_length
nonlinear_operator
linearised_operator
construct_equations
```

## Operators

```@docs
CartesianPrimitive3DNSE
CartesianPrimitive3DLNSE
CartesianPrimitive2DNSE
CartesianPrimitive2DLNSE
ProjectedNSE
```

## Mode tags

```@docs
Mode
Forward
AdjointContinuous
AdjointDiscrete
NoForce
CompoundForcing
```

## IO

```@docs
save_grid
load_grid
save_field
load_field
```
