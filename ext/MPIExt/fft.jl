# FFT hooks for HaloArray-backed fields.
#
# FFTs never act along halo-bearing dimensions. Plans must nevertheless use the
# full parent shape because halo rows are independent batches along the
# homogeneous dimensions, and FFTW's planned strides must match execution.
# We therefore pass the entire halo-inclusive dense parent array to FFTW, not a
# view of the local interior. An interior view would have different strides and
# would not match the plan constructed for the actual storage buffer.

ReSolverFlowsBase._fft_size(g::DecomposedGrid) = size(g) .+ 2 .* nhalo(g)

# ReSolverFlowsBase's field-level FFT methods call `_fft_data(parent(field))`. For a
# HaloArray, the FFTW-compatible storage is the halo-inclusive dense parent.
ReSolverFlowsBase._fft_data(a::HaloArrays.HaloArray) = parent(a)
