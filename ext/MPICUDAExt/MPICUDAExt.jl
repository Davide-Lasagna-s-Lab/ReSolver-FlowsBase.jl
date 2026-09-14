module MPICUDAExt

import CUDA,
       Adapt,
       MPI,
       FDGrids,
       HaloArrays,
       LinearAlgebra,
       ReSolverFlowsBase

# get extension namespaces
const MPIExt  = Base.get_extension(ReSolverFlowsBase, :MPIExt)
const CUDAExt = Base.get_extension(ReSolverFlowsBase, :CUDAExt)


const DecomposedGPUGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S} = MPIExt.DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP} where {GP<:CUDAExt.GPUGrid}
ReSolverFlowsBase.FFTPlanStyle(::Type{<:DecomposedGPUGrid}) = CUDAExt.cuFFTStyle()


struct DecomposedDeviceGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP, W, P} <: ReSolverFlowsBase.AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}
        parent::GP
       weights::W
    inh_points::P

    DecomposedDeviceGrid(gp::GP,
                         DDIMS, NHALO, S,
                         weights::W,
                      inh_points::P) where {
                    T, D, AXES, FFT_DIMS_ORDER,
                    GP<:CUDAExt.GPUGrid{T, D, AXES, FFT_DIMS_ORDER}, W, P} =
        new{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP, W, P}(gp, weights, inh_points)
end


function Adapt.adapt_structure(to, g::DecomposedGPUGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S}) where {T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S}
    g_d          = Adapt.adapt_structure(to, parent(g))
    ws_d         = Adapt.adapt_structure(to, ReSolverFlowsBase.weights(g))
    inh_points_d = Adapt.adapt_structure(to, g.inh_points)
    return DecomposedDeviceGrid(g_d, DDIMS, NHALO, S, ws_d, inh_points_d)
end

function CUDA.cu(g::MPIExt.DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP, W, P, COMM}) where {T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP, W, P, COMM}
    g_d          = CUDA.cu(parent(g))
    ws_d         = CUDA.cu(ReSolverFlowsBase.weights(g))
    inh_points_d = CUDA.cu(g.inh_points)
    return MPIExt.DecomposedGrid(g_d, DDIMS, NHALO, S, ws_d, inh_points_d, g.comm)
end


ReSolverFlowsBase.FTField(g::DecomposedGPUGrid{T}) where {T} =
    ReSolverFlowsBase.FTField(g, CUDA.cu(HaloArrays.HaloArray{Complex{T}}(MPIExt.comm(g),
                                                                MPIExt.local_transform_size(g),
                                                                MPIExt.nhalo(g); economic=true)))

ReSolverFlowsBase.Field(g::DecomposedGPUGrid{T}; dealias::Bool=false) where {T} =
    ReSolverFlowsBase.Field(g, CUDA.cu(HaloArrays.HaloArray{T}(MPIExt.comm(g),
                                                     MPIExt.local_physical_size(g; dealias=dealias),
                                                     MPIExt.nhalo(g); economic=true)))

function ReSolverFlowsBase.ProjectedField(grid::DecomposedGPUGrid{T}, modes) where {T}
    Nm = size(modes[1], 1)
    return ReSolverFlowsBase.ProjectedField(grid,
                                  CUDA.zeros(Complex{T}, Nm,
                                        ReSolverFlowsBase.transform_size(grid)[collect(ReSolverFlowsBase.fft_storage_dims(grid))]...),
                                  modes)
end


ReSolverFlowsBase._spectral_dd!(out::F,
                                u::F,
                                 ::Val{STORAGE_DIM},
                             mode::ReSolverFlowsBase.OperatorMode=ReSolverFlowsBase.Forward()) where {
                            STORAGE_DIM,
                            G<:DecomposedGPUGrid,
                            F<:Union{ReSolverFlowsBase.FTField{G}, ReSolverFlowsBase.ProjectedField{G}}} =
    CUDAExt._cuda_spectral_dd!(out, u, Val(STORAGE_DIM), Val(ReSolverFlowsBase.rfft_storage_dim(ReSolverFlowsBase.grid(u))), mode)

ReSolverFlowsBase._add_homogeneous_laplacian!(out::F, u::F) where {F<:ReSolverFlowsBase.FTField{<:DecomposedGPUGrid}} =
    CUDAExt._cuda_add_homogeneous_laplacian!(out, u)


function ReSolverFlowsBase.project!(a::ReSolverFlowsBase.ProjectedField{G},
                                    u::ReSolverFlowsBase.VectorField{N, <:ReSolverFlowsBase.FTField{G}},
                               method::CUDAExt.ProjectMethod=CUDAExt.project_method(a, u)) where {N, G<:DecomposedGPUGrid}
    CUDAExt._project!(a, u, method)

    # Sum the per-rank partial projections into the global modal coefficients
    CUDA.synchronize() # make sure computation is finished on all processes
    MPI.Allreduce!(parent(a), MPI.SUM, MPIExt.comm(ReSolverFlowsBase.grid(u)))
    return a
end

ReSolverFlowsBase.expand!(u::ReSolverFlowsBase.VectorField{N, <:ReSolverFlowsBase.FTField{G}},
                          a::ReSolverFlowsBase.ProjectedField{G},
                     method::CUDAExt.ExpandMethod=CUDAExt.expand_method(u, a)) where {N, G<:DecomposedGPUGrid} =
    CUDAExt._expand!(u, a, method)


LinearAlgebra.dot(a::ReSolverFlowsBase.ProjectedField{G},
                  b::ReSolverFlowsBase.ProjectedField{G},
             method::CUDAExt.DotMethod=CUDAExt.dot_method(a)) where {G<:DecomposedGPUGrid} =
    CUDAExt._dot(parent(a), parent(b), method)

end
