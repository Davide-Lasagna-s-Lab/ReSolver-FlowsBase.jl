using Test

using LinearAlgebra

import MPI
import FDGrids
import CUDA

using ReSolverFlowsBase

CUDAExt = Base.get_extension(ReSolverFlowsBase, :CUDAExt)

include("../../mock_channel_grid.jl")

# setup MPI environment
MPI.Initialized() || MPI.Init()
comm   = MPI.COMM_WORLD
rank   = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# grid construction parameters
const Ny = 16; const Nx = 5; const Nz = 5; const Nt = 5
const Ny_local = Ny ÷ nranks
const NHALO = 1
const M     = 2
const NCOMP = 2

# construct grid
gp = MockChannelGrid(Ny, Nx, Nz, Nt)
g = CUDA.cu(distributed(gp, comm; decomposed_physical_dims=(:y,),
                                  nprocesses              =(nranks,),
                                  nhalo                   =(NHALO,)))

# construct modes
if rank == 0
    Ψ = ntuple(n -> zeros(ComplexF64, M, Ny, (Nx >> 1) + 1, Nz, Nt), NCOMP)
    for nt in 1:Nt, nz in 1:Nz, nx in 2:(Nx >> 1) + 1
        local tmp = (Diagonal(1 ./ sqrt.(repeat(gp.ws, NCOMP)))*qr(Diagonal(sqrt.(repeat(gp.ws, NCOMP)))*randn(ComplexF64, NCOMP*Ny, M)).Q[:, 1:M])'
        for n in 1:NCOMP
            @views Ψ[n][:, :, nx, nz, nt] .= tmp[:, (1 + (n - 1)*Ny):n*Ny]
        end
    end
    for nz in 2:(Nz >> 1) + 1, nt in 2:Nt
        local tmp = (Diagonal(1 ./ sqrt.(repeat(gp.ws, NCOMP)))*qr(Diagonal(sqrt.(repeat(gp.ws, NCOMP)))*randn(ComplexF64, NCOMP*Ny, M)).Q[:, 1:M])'
        for n in 1:NCOMP
            Ψ[n][:, :, 1,     nz,       nt]   .= tmp[:, (1 + (n - 1)*Ny):n*Ny]
            Ψ[n][:, :, 1, end-nz+2, end-nt+2] .= conj.(Ψ[n][:, :, 1, nz, nt])
        end
    end
    for nz in 2:Nz
        local tmp = (Diagonal(1 ./ sqrt.(repeat(gp.ws, NCOMP)))*qr(Diagonal(sqrt.(repeat(gp.ws, NCOMP)))*randn(ComplexF64, NCOMP*Ny, M)).Q[:, 1:M])'
        for n in 1:NCOMP
            Ψ[n][:, :, 1,     nz,   1] .= tmp[:, (1 + (n - 1)*Ny):n*Ny]
            Ψ[n][:, :, 1, end-nz+2, 1] .= conj.(Ψ[n][:, :, 1, nz, 1])
        end
    end
    for nt in 2:Nt
        local tmp = (Diagonal(1 ./ sqrt.(repeat(gp.ws, NCOMP)))*qr(Diagonal(sqrt.(repeat(gp.ws, NCOMP)))*randn(ComplexF64, NCOMP*Ny, M)).Q[:, 1:M])'
        for n in 1:NCOMP
            Ψ[n][:, :, 1, 1,     nt]   .= tmp[:, (1 + (n - 1)*Ny):n*Ny]
            Ψ[n][:, :, 1, 1, end-nt+2] .= conj.(Ψ[n][:, :, 1, 1, nt])
        end
    end
    local tmp = (Diagonal(1 ./ sqrt.(repeat(gp.ws, NCOMP)))*qr(Diagonal(sqrt.(repeat(gp.ws, NCOMP)))*randn(Float64, NCOMP*Ny, M)).Q[:, 1:M])'
    for n in 1:NCOMP
        Ψ[n][:, :, 1, 1, 1] .= tmp[:, (1 + (n - 1)*Ny):n*Ny]
    end
end

# distribute modes over each process
if rank != 0
    Ψ_slab = ntuple(n -> zeros(ComplexF64, M, Ny_local, (Nx >> 1) + 1, Nz, Nt), NCOMP)
    for n in 1:NCOMP
        tag = rank + (n - 1)*NCOMP
        MPI.Recv!(Ψ_slab[n], comm; source=0, tag=tag)
    end
else
    Ψ_slab = ntuple(NCOMP) do n
        Ψ[n][:, 1:Ny_local, :, :, :]
    end
    for n in 1:NCOMP
        for dest in 1:(nranks - 1)
            tag = dest + (n - 1)*NCOMP
            MPI.Send(Ψ[n][:, (1 + dest*Ny_local):(dest + 1)*Ny_local, :, :, :], comm; dest=dest, tag=tag)
        end
    end
end

# construct fields to re-use
ud = VectorField(g; N=NCOMP)
if rank != 0
    ad = ProjectedField(g, CUDA.cu(Ψ_slab))
    MPI.Recv!(parent(ad), comm; source=0)
else
    ad  = ProjectedField(g, CUDA.cu(Ψ_slab))
    ad .= CUDA.randn(ComplexF32, M, (Nx >> 1) + 1, Nz, Nt)
    for dest in 1:(nranks - 1)
        MPI.Send(parent(ad), comm; dest=dest)
    end
end

# construct all galerkin methods
project_methods = [
                CUDAExt.ProjectLoop(),
                CUDAExt.ProjectShared(ad, ud),
                #    CUDAExt.ProjectSharedTiled(ad, ud), # ! not implemented yet
                #    CUDAExt.ProjectSharedTree(ad, ud),  # ! not implemented yet
                    ]
expand_methods  = [
                CUDAExt.ExpandModal(;over_vector=true),
                CUDAExt.ExpandModal(;over_vector=false),
                    ]

@testset "Decomposed CUDA galerkin methods                                    " for pmethod in project_methods, emethod in expand_methods
    out_d = project!(similar(ad), expand!(ud, ad, emethod), pmethod)
    @test norm(Array(parent(ad - out_d))) < 1e-5
end
