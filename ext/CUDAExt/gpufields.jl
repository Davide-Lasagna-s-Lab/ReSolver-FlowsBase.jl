# Specialised constructors and methods for fields defined on a GPUGrid.

# useful type aliases
const GPUFTField        = ReSolverFlowsBase.FTField{<:GPUGrid}
const GPUField          = ReSolverFlowsBase.Field{<:GPUGrid}
const GPUProjectedField = ReSolverFlowsBase.ProjectedField{<:GPUGrid}
const GPUVectorField{N} = ReSolverFlowsBase.VectorField{N, <:Union{GPUFTField, GPUField}}

# ------------------------- #
# special CUDA constructors #
# ------------------------- #
"""
    FTField(g::GPUGrid)

Construct a `FTField` such that the parent data is stored
on the device.
"""
ReSolverFlowsBase.FTField(g::GPUGrid{T}) where {T} =
    ReSolverFlowsBase.FTField(g, CUDA.zeros(Complex{T}, ReSolverFlowsBase.transform_size(g)))

"""
    Field(g::GPUGrid)

Construct a `Field` such that the parent data is stored
on the device.
"""
ReSolverFlowsBase.Field(g::GPUGrid{T}; dealias=true) where {T} = dealias ?  ReSolverFlowsBase.Field(g, CUDA.zeros(T, ReSolverFlowsBase.get_padded_size(size(g), ReSolverFlowsBase.fft_storage_dims(g)))) :
                                                                  ReSolverFlowsBase.Field(g, CUDA.zeros(T, size(g)))

"""
    ProjectedField(g::GPUGrid)

Construct a `ProjectedField` such that the parent data and
modes are stored on the device.
"""
function ReSolverFlowsBase.ProjectedField(grid::GPUGrid{T}, modes) where {T}
    Nm = size(modes[1], 1)
    return ReSolverFlowsBase.ProjectedField(grid,
                                  CUDA.zeros(Complex{T}, Nm,
                                        ReSolverFlowsBase.transform_size(grid)[collect(ReSolverFlowsBase.fft_storage_dims(grid))]...),
                                  modes)
end


# ------------------------------ #
# adapt methods for CUDA kernels #
# ------------------------------ #
adapt_structure(to, u::ReSolverFlowsBase.FTField) =
    ReSolverFlowsBase.FTField(adapt_structure(to, ReSolverFlowsBase.grid(u)), adapt_structure(to, parent(u)))
adapt_structure(to, u::ReSolverFlowsBase.Field) =
    ReSolverFlowsBase.Field(adapt_structure(to, ReSolverFlowsBase.grid(u)), adapt_structure(to, parent(u)))
adapt_structure(to, u::ReSolverFlowsBase.VectorField{N}) where {N} =
    ReSolverFlowsBase.VectorField(ntuple(n -> adapt_structure(to, u[n]), Val(N))...)
function adapt_structure(to, a::ReSolverFlowsBase.ProjectedField)
    g = adapt_structure(to, ReSolverFlowsBase.grid(a))
    data = adapt_structure(to, parent(a))
    mds = adapt_structure(to, ReSolverFlowsBase.modes(a))
    return ReSolverFlowsBase.ProjectedField(g, data, mds)
end


# --------------------------------- #
# opinionated CUDA field converters #
# --------------------------------- #
"""
    CUDA.cu(u::FTField)
    CUDA.cu(u::Field)
    CUDA.cu(u::VectorField)
    CUDA.cu(a::ProjectedField)

Opinionated converter for field types that moves all their
respective data to the device.

Automatically converts all numerics type to `Float32`.
"""
CUDA.cu(u::ReSolverFlowsBase.FTField)                  =
    ReSolverFlowsBase.FTField(CUDA.cu(ReSolverFlowsBase.grid(u)), CUDA.cu(parent(u)))
CUDA.cu(u::ReSolverFlowsBase.Field)                    =
    ReSolverFlowsBase.Field(CUDA.cu(ReSolverFlowsBase.grid(u)), CUDA.cu(parent(u)))
CUDA.cu(u::ReSolverFlowsBase.VectorField{N}) where {N} =
    ReSolverFlowsBase.VectorField(ntuple(n -> CUDA.cu(u[n]), Val(N))...)
CUDA.cu(a::ReSolverFlowsBase.ProjectedField)           =
    ReSolverFlowsBase.ProjectedField(CUDA.cu(ReSolverFlowsBase.grid(a)), CUDA.cu(parent(a)), CUDA.cu(ReSolverFlowsBase.modes(a)))
