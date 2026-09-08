module ReSolverFlowsBase

# TODO: add benchmark scripts, profile the cartesian primitive NSE and LNSE implementations
# TODO: @enum might be a useful way to associate spatial dimensions with value types and make code more readable

using LinearAlgebra, FFTW, JLD2

export FFTW

export AbstractGrid
export storage_dim, physical_dim, physical_to_storage_dim, to_storage_order
export rfft_storage_dim, rfft_physical_dim
export points, growto, weights
export fft_storage_dims, fft_physical_dims
export spatial_fft_storage_dims, spatial_fft_physical_dims
export inhomogeneous_storage_dims, inhomogeneous_physical_dims
export spatial_inhomogeneous_storage_dims, spatial_inhomogeneous_physical_dims
export transform_size, fft_norm, wavenumber_scale
export WaveNumberVector, to_homogeneous_indices, to_wavenumber_vector
export FTField, Field, VectorField, grid
export add_base_flow!
export FFTPlans, FFT, IFFT
export ProjectedField, modes, project!, project, expand!, expand
export LoopGalerkin, GemmGalerkin
export dd!, ddx!, ddy!, ddz!, ddt!
export laplacian!
export shift!, shift, normdiff, minnormdiff
export save_grid, load_grid, save_field, load_field
export FarazmandWeight
export CartesianPrimitive3DNSE, CartesianPrimitive3DLNSE
export CartesianPrimitive2DNSE, CartesianPrimitive2DLNSE
export CartesianPrimitive2D3CNSE, CartesianPrimitive2D3CLNSE
export CartesianPrimitive3DBoussinesqNSE, CartesianPrimitive3DBoussinesqLNSE
export Forward, AdjointContinuous, AdjointDiscrete, Mode, OperatorMode
export NoForce, CompoundForcing
export construct_equations, CartesianPrimitive3D, CartesianPrimitive2D, CartesianPrimitive2D3C, PolarPrimitive
export CartesianPrimitive3DBoussinesq
export ncomp, cache_length, nonlinear_operator, linearised_operator
export ProjectedNSE

include("notimplementederror.jl")
include("abstractgrid.jl")
include("wavenumbervector.jl")
include("ftfield.jl")
include("field.jl")
include("vectorfield.jl")
include("projectedfield.jl")
include("fft.jl")
include("galerkin.jl")
include("shifts.jl")
include("norms.jl")
include("weighting.jl")
include("broadcasting.jl")
include("equations/types.jl")
include("derivatives.jl")
include("io.jl")
include("equations/forcing.jl")
include("equations/cartesianprimitive_3d.jl")
include("equations/cartesianprimitive_2d.jl")
include("equations/cartesianprimitive_2d3c.jl")
include("equations/cartesianprimitive_3d_boussinesq.jl")
include("equations/projectednse.jl")
include("equations/shared.jl")


# dummy function definition for MPI extension
export distributed

function distributed end


# dummy function definition for CUDA extension
export initialise_dot!, reset_dot_cache!
export initialise_project!, reset_project_cache!
export initialise_expand!, reset_expand_cache!

function show_tuning_info! end
function set_tuning_samples! end

function reset_launch_params! end

function initialise_dot! end
function reset_dot_cache! end

function initialise_project! end
function reset_project_cache! end

function initialise_expand! end
function reset_expand_cache! end

end
