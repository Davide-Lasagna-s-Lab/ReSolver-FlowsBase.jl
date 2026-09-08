# Concepts & Conventions

This page documents the core assumptions, conventions, and constraints that
ReSolverFlowsBase builds on.  Understanding these is essential both for using the package
correctly and for implementing a new downstream grid.

## The grid interface

Everything in ReSolverFlowsBase is parameterised on a **grid** — a concrete subtype of
`AbstractGrid{T, D, AXES, FFT_DIMS_ORDER, DECOMPOSITION}`.  The five compile-time type parameters
encode all structural information about the domain:

| Parameter | Meaning |
|-----------|---------|
| `T <: Real` | Scalar real type (typically `Float64`) |
| `D` | Number of array dimensions |
| `AXES` | Four-tuple `(x_dim, y_dim, z_dim, t_dim)` mapping logical Cartesian coordinates to array dimensions; use `nothing` for absent coordinates |
| `FFT_DIMS_ORDER` | Ordered tuple of the array dimensions that are FFT-transformed; `FFT_DIMS_ORDER[1]` is **always** the rfft dimension |

Because these parameters are part of the type, the compiler can fully specialise
every loop, index mapping, and generated function at compile time — there is no
runtime dispatch and no runtime allocation in the hot paths.

### Required methods

A concrete grid should implement exactly five methods:

```julia
Base.size(grid)                                                             # → NTuple{D, Int}
points(grid; dealias=false)                                                 # → one array per dimension
wavenumber_scale(grid, dim::Int)                                            # → Real (2π/L for spatial, 1 for temporal)
weights(grid)                                                               # → AbstractArray (quadrature weights)
derivative_matrix(grid, storage_dim::Int, ::Val{ORDER}, mode::OperatorMode) # -> AbstractArray (differentiation operator for inhomogeneous directions)
```

`weights` must have one axis per entry in `inhomogeneous_dims(grid)` in ascending array-dimension order, and `derivative_matrix` should be defined for all `storage_dim ∉ FFT_DIMS_ORDER`.

### Homogeneous vs. inhomogeneous dimensions

ReSolverFlowsBase draws a sharp distinction between two kinds of dimensions:

- **Homogeneous** (a.k.a. FFT-transformed) dimensions — entries of
  `FFT_DIMS_ORDER`.  These are statistically periodic and are represented in
  spectral space as Fourier coefficients.  Derivatives are exact multiplications
  by `im · n · wavenumber_scale`.
- **Inhomogeneous** dimensions — the complement of `FFT_DIMS_ORDER` in `1:D`.
  These are non-periodic (e.g. the wall-normal direction in channel flow).
  ReSolverFlowsBase provides no derivative for them; downstream packages must extend `ddx!`
  with their own matrix-multiply or spectral-element method.

---

## Field types

ReSolverFlowsBase provides four field types.  All are subtypes of `AbstractArray` and store
a reference to their grid alongside their data.

### `Field`

A physical-space scalar field.  The underlying array has shape `size(grid)` and
real element type `T`.  Used as a scratch buffer for de-aliased nonlinear
products.

```julia
u = Field(grid)                      # zero field
u = Field(grid, (x, y) -> sin(x)*y)  # initialise from a function
```

### `FTField`

A spectral-space scalar field — the rfft of a `Field`.  The underlying array has
complex element type and shape `transform_size(grid)`, which is `size(grid)` with
dimension `FFT_DIMS_ORDER[1]` halved to `(N÷2)+1`.

The remaining transformed dimensions keep their full size in **FFTW wrap-around
order**: indices `1:n÷2+1` hold non-negative wavenumbers and indices `n÷2+2:n`
hold negative wavenumbers in reverse order.

Two invariants are **enforced on construction** (when given a plain `Array`):

1. **Hermitian symmetry** — in the zero-rfft-wavenumber plane every coefficient
   at signed wavenumber `(0, k₂, k₃, …)` must satisfy
   `û(0, -k₂, -k₃, …) = conj(û(0, k₂, k₃, …))`.
2. **Real mean** — the fully-zero-wavenumber mode `û(0, 0, …)` must be real.

### `VectorField`

A fixed-length tuple of `FTField`s (or `Field`s) representing a vector-valued
quantity such as a velocity field.  `VectorField{N}` holds exactly `N` scalar
components.  Broadcasting, `copy`, `zero`, `similar`, etc. are all
component-wise.

### `ProjectedField`

A Galerkin-reduced representation of a vector field.  Instead of storing full
spectral coefficients for every wavenumber and every wall-normal point, it stores
`Nm` scalar coefficients per wavenumber:

```
parent(a)[m, i_H1, i_H2, …]   m ∈ 1:Nm, spectral storage indices
```

The mode index occupies axis 1 and the spectral axes follow in `FFT_DIMS_ORDER`
order.  Non-FFT (inhomogeneous) dimensions are **absent** — the basis functions
absorb the wall-normal dependence.  The same Hermitian-symmetry and real-mean
invariants as `FTField` apply.

---

## Wavenumber conventions

### Storage order and signed wavenumbers

The rfft dimension (`FFT_DIMS_ORDER[1]`) stores only non-negative wavenumbers
`0, 1, …, N÷2`.  All other transformed dimensions use FFTW's wrap-around order.
The helper type `WaveNumberVector{N}` converts between storage indices and signed
wavenumber integers, handling the conjugate-symmetry lookup for the rfft axis.

### Physical wavenumber scale

The actual physical wavenumber associated with integer mode `n` in an
`FFT_DIMS_ORDER[j]` dimension is

```
k_physical = n × wavenumber_scale(grid, FFT_DIMS_ORDER[j])
```

For a spatial dimension with period `L` the convention is `wavenumber_scale = 2π/L`.
For a temporal direction with unit period the convention is `wavenumber_scale = 1`.

Downstream grids must implement `wavenumber_scale` for every dimension in
`fft_dims(grid)`.

---

## Derivative convention

`dd!(out, u, Val{STORAGE_DIM})` computes the in-place spectral derivative along array
dimension `STORAGE_DIM ∈ FFT_DIMS_ORDER`:

```
out[k] = +im · n · wavenumber_scale(grid, DIM) · u[k]     (adjoint=false)
out[k] = -im · n · wavenumber_scale(grid, DIM) · u[k]     (adjoint=true)
```

The `adjoint=true` form is the L² adjoint of the spectral derivative operator. For the case that `STORAGE_DIM ∉ FFT_DIMS_ORDER` the direction trying to differentiated is inhomogeneous (non-periodic). For this to work the user is required to have implemented the `derivative_matrix` method, which will fetch the differentiation operator being used and then use it for the differentiation via `LinearAlgebra.mul!`.

Four named wrappers pick the array dimension from `AXES`:

| Wrapper | Logical coordinate |
|---------|--------------------|
| `ddx!` | x (`AXES[1]`) |
| `ddy!` | y (`AXES[2]`) |
| `ddz!` | z (`AXES[3]`) |
| `ddt!` | t (`AXES[4]`) |

When the coordinate is absent (`AXES[j] === nothing`) the wrapper is a compile-time
no-op.

---

## Inner product and norm conventions

The L² inner product is defined as

```math
\langle u, v \rangle
  = \frac{1}{2} \sum_{\mathbf{k}} c_{k_1}\, w(\mathbf{j})\,
    \Re\bigl(\bar{u}_{\mathbf{k},\mathbf{j}}\, v_{\mathbf{k},\mathbf{j}}\bigr),
```

where:

- The sum runs over all stored spectral wavenumbers **k** and inhomogeneous
  indices **j**.
- `c_{k₁}` is **1** for the zero rfft wavenumber and **2** for all others,
  accounting for Hermitian symmetry.
- `w(j)` are the quadrature weights returned by `weights(grid)`.
- The `1/2` factor combined with the `c_{k₁}` multiplier ensures equivalence with
  the continuous L² norm: the rfft discards the negative-wavenumber half of the
  spectrum, which carries equal energy.

`dot(u, v)` implements this for `FTField`, `VectorField{<:FTField}`, and
`ProjectedField`.

---

## Phase shifts

A continuous shift by displacement `s` in a homogeneous direction is an **exact**
spectral operation — no interpolation is needed.  The phase factor applied to mode
`n` is

```
exp(im · n · s · wavenumber_scale(g, dim))
```

`shift!(u, shifts)` applies the product of phase factors over all homogeneous
dimensions in `FFT_DIMS_ORDER` order.  Shifts are given in **physical units**
(same units as the period `L`).

---

## FFT Transforms and Plans

The main mechanism to transform between `FTField`'s and `Field`'s is via `FFT`, `IFFT`, and `FFTPlans`. Both `FFT` and `IFFT` are allocating transforms and are most useful for interactive developement and exploration. If the `growto` method is defined for a subtype of `AbstractGrid`, then fields that are built on this subtype also have access to resolution altering `FFT` and `IFFT` methods. This allow the user to pad or truncate the spectral components (homogeneous directions) of the field, useful for creating better resolved plots.

```julia
u = Field(g) # g isa AbstractGrid
û = FFT(u)
IFFT(û) # ≈ u

û_padded  = FFT(u, target_size) # target_size isa NTuple{N, Int}
u_refined = IFFT(u, target_size) # u_refined is subsampled in the homogeneous directions
```

In hot loops, where operators are evaluated many times over, using `FFTPlans` is strongly preferred as it does not allocate any new data. `FFTPlans` are primarily constructed via a concrete implementation of `AbstractGrid`, and can be optionally be constructed to pad the transforms for the purpose of dealiasing. The primarily backend for all the transforms is FFTW, and therefore FFTW flags (`FFTW.ESTIMATE`, `FFT.TIMELIMIT`, etc.) can be passed to modify the plans being constructed.

```julia
g = ChannelGrid(...) # <: AbstractGrid
F = FFTPlans(u; flags=FFTW.EXHAUSTIVE, timelimit=FFTW.NO_TIMELIMIT)

û = FTField(g)
u = Field(g)
F(û, u) # forward transform u -> û
F(u, û) # backward transform û -> u
```

---

## De-aliasing

Physical-space nonlinear products are computed on a padded grid using the **3/2
rule**: the physical-space arrays in `pcache` are allocated with `dealias=true`,
which extends each homogeneous dimension to `3N/2`.  ReSolverFlowsBase zero-pads the
spectral coefficients before transforming to the padded grid and symmetrically
truncates after transforming back, eliminating aliasing errors from quadratic
nonlinearities.

---

## NSE formulations

ReSolverFlowsBase includes two complete Cartesian primitive-variable formulations:

| Tag struct | Components | Operators |
|-----------|-----------|----------|
| `CartesianPrimitive3D()` | u, v, w (3 components) | `CartesianPrimitive3DNSE`, `CartesianPrimitive3DLNSE{MODE}` |
| `CartesianPrimitive2D()` | u, v (2 components) | `CartesianPrimitive2DNSE`, `CartesianPrimitive2DLNSE{MODE}` |

Each linearised operator `CartesianPrimitive3DLNSE{MODE}` is parameterised on a
`MODE <: Mode` tag:

| `MODE` | Operator computed |
|--------|------------------|
| `Forward` | Standard linearised NSE around base flow |
| `AdjointContinuous` | Continuous adjoint of the linearised operator |
| `AdjointDiscrete` | Discrete (numerical) adjoint — exact transpose of `Forward` |

### Adding a new formulation

Implement a singleton struct (no fields) and the four interface methods:

```julia
struct MyFormulation end
ncomp(              ::MyFormulation)                    = ...   # number of velocity components
cache_length(       ::MyFormulation, ::Type{<:FTField}) = ...   # spectral scratch arrays needed
cache_length(       ::MyFormulation, ::Type{<:Field})   = ...   # physical scratch arrays needed
nonlinear_operator( ::MyFormulation)                    = ...   # return concrete operator type
linearised_operator(::MyFormulation, ::M) where {M}     = ...   # return parametric operator type
```

Then pass `MyFormulation()` to `construct_equations`.

---

## `construct_equations` and `ProjectedNSE`

`construct_equations` is the recommended entry point for building a solver.  It:

1. Allocates two shared scratch pools — `scache` (spectral) and `pcache`
   (physical), sized by `cache_length`.
2. Builds FFTW plans for the grid.
3. Instantiates the nonlinear and linearised operators, wiring them to the same
   scratch pools.
4. Wraps everything in a `ProjectedNSE` struct.

`ProjectedNSE` is callable:

```julia
obj(out, a)       # nonlinear NSE: out = P(Δu/Re − (u·∇)u + force)
obj(out, a, b)    # linearised NSE: out = P(L_a · b)
```

where `a` and `b` are `ProjectedField`s.

---

## Memory and allocation policy

Every hot-path operation in ReSolverFlowsBase is **allocation-free** at steady state.
Cache arrays are pre-allocated once inside the operator structs and reused on
every call.

When writing downstream code, prefer the in-place (`!`) variants of all
operations.  The allocating wrappers (`FFT`, `IFFT`, `shift`, `project`,
`expand`) are convenience forms intended for interactive exploration and tests.

---

# GPU Acceleration

ReSolverFlowsBase supports GPU accelerations, via CUDA, for any subtype of `AbstractGrid`. This is achieved via the `CUDA.cu` method which is responsible for moving all the contents of a grid and all fields built on top it to the GPU. For these routines to work the user must implement the following:

```julia
Adapt.adapt_structure(to, grid)
Base.convert(T, grid)
```

The `CUDA.cu` method is opinionated: it forcefully converts all numeric types to `Float32`. If this is not desired it is also possible use `Adapt.adapt_structure` directly which retains the same numeric types while moving all the data to the GPU. Once grids can be moved onto the GPU then simply calling operator constructors (`CartesianPrimitive3DNSE(::GPUGrid, ...)`, `construct_equations(::GPUGrid, ...)`) will automatically construct all caches and plans required with the appropriate memory placements and backends.

`CUDA.cu` returns a `GPUGrid` type which is used primarily to modify the dispatch path such that the specialised CUDA kernels are used where appropriate. This includes:
 - constructing `FFTPlans` using cuFFT for the backend (instead of FFTW)
 - derivatives, galerkin project and expand, dot products.

The first time some of these methods are run for particular input types, the code automatically benchmarks all the available kernels and chooses the fastest as the preferred method for all future calls with the same input types. This autotuning process can be controlled via the following methods:

```julia
# ProjectedField dot product
initialise_dot!(a::ProjectedField)
reset_dot_cache!() # -> clears all saved tuned methods
reset_dot_cache!(a::ProjectedField) # -> clears the tuned method for only the type `typeof(a)`

# Galerkin projection
initialise_project!(a::ProjectedField)
reset_project_cache!() # -> clears all saved tuned methods
reset_project_cache!(a::ProjectedField) # -> clears the tuned method for only the types `typeof(a)` and `typeof(u)`

# Galerkin expansion
initialise_expand!(a::ProjectedField, u::VectorField)
reset_expand_cache!() # -> clears all saved tuned methods
reset_expand_cache!(a::ProjectedField, u::VectorField) # -> clears the tuned method for only the types `typeof(a)` and `typeof(u)`
```

Alternatively, a particular kernel method can be chosen manually by passing it directly to the method call:

```julia
CUDAExt = Base.get_extension(ReSolverFlowsBase, :CUDAExt)
method = CUDAExt.DotAtomic(a)
dot(a, b, method)

# similarly for galerkin project and expand
```

To toggle the information output from the autotuning process use `ReSolverFlowsBase.show_tuning_info!(true)`. The number of samples taken during autotuning can also be controlled via the `ReSolverFlowsBase.set_tuning_samples!(::Int)` method.

Finally, all kernels are configured before they run with the optimal number of threads given their input types. This stored for proceeding calls to the kernel such that it doesn't have to regenerated anew each time the kernel is run. To reset this global cache use `ReSolverFlowsBase.reset_launch_cache!()`.

---

## MPI and Distributed Computations

MPI parallelisation is supported, and can be accessed by loading [MPI.jl](https://github.com/JuliaParallel/MPI.jl) and [FDGrids.jl](https://github.com/Davide-Lasagna-s-Lab/FDGrids.jl) into the current Julia session alongside ReSolverFlowsBase.

Using `distributed` a `DecomposedGrid` object can be constructed, which allows for the construction of `FTField`'s, `Field`'s, and whole operators that are split up over multiple MPI processes. Only decomposing over the inhomogeneous spatial directions is currently supported. 

Below is an example of a decomposition of a field over the `y` direction:

```julia
comm = MPI.COMM_WORLD
np   = MPI.Comm_size(comm)

g_dist = distributed(g_parent, comm;
                        decomposed_physical_dims=(:y,), nprocesses=(np,), nhalo=(2,))

Re = 100 # Reynolds number
op = CartesianPrimitive3DNSE(g_dist, Re)
```

The variable `nhalo` is used to tell the fields how much to pad the local arrays to accomodate data from neighbouring processes required to compute derivatives. This functionality is enabled via [HaloArrays.jl](https://github.com/Davide-Lasagna-s-Lab/HaloArrays.jl). Note that it is required to use [FDGrids.jl](https://github.com/Davide-Lasagna-s-Lab/FDGrids.jl) for the differentiation over the decomposed directions as the package implements the (somewhat fiddly) methods to compute derivatives using finite-differences with arbitrary stencil widths and thus order of accuracy. See the package for extra details.

It is possible to use both MPI and CUDA together to achieve massively parallel computations, simply by applying the the `CUDA.cu` method directly to the `DecomposedGrid` in the same way as you would any subtype of `AbstractGrid` as discussed above.

```julia
g_dist = distributed(g_parent, comm;
                        decomposed_physical_dims=(:y), nprocesses=(np,), nhalo=(2,))
g_device = CUDA.cu(g_dist)
```

This results in a set of decomposed grid objects, one for each MPI process, where the data and any fields/plans constructed from it (using `FTField(g_device)`, `FFTPlans(g_device)`, etc.) is stored on the primary device for that process (set with `CUDA.device!`).

---
