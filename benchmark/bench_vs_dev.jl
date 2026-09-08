# Benchmark: CartesianIndices (this branch) vs @generated (dev branch).
#
# Every function ported from @generated / for_each_* to CartesianIndices is
# measured here for a 3-D flow on a 4-D grid (wall-normal y + periodic x, z, t).
# Four storage layouts cover every position of the inhomogeneous dimension:
#
#   Layout 1: (y, x, z, t)  — inh first,  FFT_DIMS_ORDER = (2, 3, 4)
#   Layout 2: (x, y, z, t)  — inh second, FFT_DIMS_ORDER = (1, 3, 4)
#   Layout 3: (x, z, y, t)  — inh third,  FFT_DIMS_ORDER = (1, 2, 4)
#   Layout 4: (x, z, t, y)  — inh last,   FFT_DIMS_ORDER = (1, 2, 3)
#
# Functions benchmarked (13 subplots):
#   ddx!              rfft dim (x)
#   ddx!              signed-FFT dim 1 (z)
#   ddx!              signed-FFT dim 2 (t)
#   dot               FTField × FTField
#   normdiff          FTField
#   shift!            FTField
#   project!          LoopGalerkin
#   expand!           LoopGalerkin
#   dot               ProjectedField × ProjectedField
#   normdiff          ProjectedField
#   shift!            ProjectedField
#   apply_symmetry!   FTField         (cart=DC-plane only; gen=all kH, as in dev)
#   apply_symmetry!   ProjectedField  (same comparison)
#
# y-axis: ratio t_cart / t_gen  (< 1 → CartesianIndices faster)
# x-axis: total FTField elements (log scale)
#
# Run from the repo root:
#   julia --project=benchmark benchmark/bench_vs_dev.jl

using BenchmarkTools
using LinearAlgebra
using Printf
using PyPlot
using ReSolverFlowsBase

include("../test/fake.jl")
include("../test/test_grids.jl")

# ------------------------------------------------------------------ #
# @generated reference implementations (verbatim from dev branch)    #
# ------------------------------------------------------------------ #
# These are the loop generators that dev used instead of CartesianIndices.

@generated function _for_each_homogeneous_index(
        f, g::AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}) where {T, D, AXES, FFT_DIMS_ORDER}
    N    = length(FFT_DIMS_ORDER)
    syms = [Symbol("_i", k) for k in 1:N]
    body = :(for $(syms[1]) in 1:(Base.size(g, $(FFT_DIMS_ORDER[1])) >> 1) + 1
                 f($(syms[1]) == 1 ? 1 : 2, ($(syms...),))
             end)
    for k in 2:N
        dim  = FFT_DIMS_ORDER[k]
        sym  = syms[k]
        pos  = :(for $sym in 1:(Base.size(g, $dim) >> 1) + 1; $body; end)
        neg  = :(for $sym in (Base.size(g, $dim) >> 1) + 2:Base.size(g, $dim); $body; end)
        body = Expr(:block, pos, neg)
    end
    return body
end

@generated function _for_each_index(
        f, g::AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}) where {T, D, AXES, FFT_DIMS_ORDER}
    syms       = [Symbol("_i", d) for d in 1:D]
    inh        = Tuple(d for d in 1:D if d ∉ FFT_DIMS_ORDER)
    inh_syms   = [syms[d] for d in inh]
    rfft_dim   = FFT_DIMS_ORDER[1]
    one_or_two = :($(syms[rfft_dim]) == 1 ? 1 : 2)
    body       = :(f($one_or_two, ($(inh_syms...),), ($(syms...),)))
    for d in 1:D
        sym = syms[d]
        body = if d == rfft_dim
            :(for $sym in 1:(Base.size(g, $d) >> 1) + 1; $body; end)
        else
            :(for $sym in 1:Base.size(g, $d); $body; end)
        end
    end
    return body
end

# --- ddx! @generated ---

@generated function _ddx_gen!(out::FTField{G}, u::FTField{G}, ::Val{DIM};
                               adjoint::Bool=false) where {
        DIM, T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}
    (isnothing(DIM) || isnothing(AXES[DIM])) && return :(return out)
    DIM ∉ FFT_DIMS_ORDER && return :(throw(ReSolverFlowsBase.NotImplementedError(grid(u), Val($DIM))))
    syms  = [Symbol("_i", d) for d in 1:D]
    n_sym = Symbol("_n", DIM)
    assign = :(@inbounds parent(out)[$(syms...)] =
                   _ddx_sign * $n_sym * _ddx_scale * parent(u)[$(syms...)])
    function mkloop(body, ranges, wavenumbers)
        for d in 1:D
            body = if d == DIM
                :(for $(syms[d]) in $(ranges[d]); $n_sym = $(wavenumbers[d]); $body; end)
            else
                :(for $(syms[d]) in $(ranges[d]); $body; end)
            end
        end
        return body
    end
    ranges = [:(1:Base.size(u, $d)) for d in 1:D]
    wnums  = Any[:nothing for _ in 1:D]
    if DIM == FFT_DIMS_ORDER[1]
        wnums[DIM] = :($(syms[DIM]) - 1)
        body = mkloop(assign, ranges, wnums)
    else
        pos = copy(ranges); neg = copy(ranges)
        pos[DIM] = :(1:(Base.size(u, $DIM) >> 1) + 1)
        neg[DIM] = :((Base.size(u, $DIM) >> 1) + 2:Base.size(u, $DIM))
        wnums[DIM] = :($(syms[DIM]) - 1)
        pb = mkloop(assign, pos, wnums)
        wnums[DIM] = :($(syms[DIM]) - Base.size(u, $DIM) - 1)
        nb = mkloop(assign, neg, wnums)
        body = :($pb; $nb)
    end
    return quote
        _ddx_scale = ReSolverFlowsBase.wavenumber_scale(ReSolverFlowsBase.grid(u), $DIM)
        _ddx_sign  = adjoint ? -1im : 1im
        $body
        return out
    end
end

# --- Galerkin @generated ---

@generated function _proj_gen_inner!(pa, u::VectorField{N}, a::ProjectedField, w,
                                     g::AbstractGrid{<:Any, D, AXES, FFT_DIMS_ORDER}
                                     ) where {N, D, AXES, FFT_DIMS_ORDER}
    inh_dims = [d for d in 1:D if d ∉ FFT_DIMS_ORDER]
    syms     = [Symbol("_i", d) for d in 1:D]
    pa_ref   = Expr(:ref, :pa, :m, (syms[d] for d in FFT_DIMS_ORDER)...)
    pu_ref   = Expr(:ref, :pu, (syms[d] for d in 1:D)...)
    w_ref    = Expr(:ref, :w,  (syms[d] for d in inh_dims)...)
    mn_ref   = Expr(:ref, :modes_n, :m,
                    (syms[d] for d in inh_dims)..., (syms[d] for d in FFT_DIMS_ORDER)...)
    body = Expr(:for, Expr(:(=), :m, :(1:Nm)),
                :($pa_ref += conj($mn_ref) * wuj))
    body = Expr(:block, :(wuj = $w_ref * $pu_ref), body)
    for d in inh_dims
        body = Expr(:for, Expr(:(=), syms[d], :(1:Base.size(g, $d))), body)
    end
    for k in 1:length(FFT_DIMS_ORDER)
        d   = FFT_DIMS_ORDER[k]
        rng = k == 1 ? :(1:(Base.size(g, $d) >> 1) + 1) : :(1:Base.size(g, $d))
        body = Expr(:for, Expr(:(=), syms[d], rng), body)
    end
    body = Expr(:block, :(pu = parent(u[n])), :(modes_n = modes(a)[n]), body)
    body = Expr(:for, Expr(:(=), :n, :(1:$N)), body)
    return quote; Nm = Base.size(pa, 1); @inbounds $body; end
end

@generated function _expd_gen_inner!(u::VectorField{N}, pa, a::ProjectedField,
                                      g::AbstractGrid{<:Any, D, AXES, FFT_DIMS_ORDER}
                                      ) where {N, D, AXES, FFT_DIMS_ORDER}
    inh_dims = [d for d in 1:D if d ∉ FFT_DIMS_ORDER]
    syms     = [Symbol("_i", d) for d in 1:D]
    pa_ref   = Expr(:ref, :pa, :m, (syms[d] for d in FFT_DIMS_ORDER)...)
    pu_ref   = Expr(:ref, :pu, (syms[d] for d in 1:D)...)
    mn_ref   = Expr(:ref, :modes_n, :m,
                    (syms[d] for d in inh_dims)..., (syms[d] for d in FFT_DIMS_ORDER)...)
    body = Expr(:for, Expr(:(=), :m, :(1:Nm)), :(acc += $mn_ref * $pa_ref))
    body = Expr(:block, :(acc = zero(eltype(pa))), body, :($pu_ref = acc))
    for d in inh_dims
        body = Expr(:for, Expr(:(=), syms[d], :(1:Base.size(g, $d))), body)
    end
    for k in 1:length(FFT_DIMS_ORDER)
        d   = FFT_DIMS_ORDER[k]
        rng = k == 1 ? :(1:(Base.size(g, $d) >> 1) + 1) : :(1:Base.size(g, $d))
        body = Expr(:for, Expr(:(=), syms[d], rng), body)
    end
    body = Expr(:block, :(pu = parent(u[n])), :(modes_n = modes(a)[n]), body)
    body = Expr(:for, Expr(:(=), :n, :(1:$N)), body)
    return quote; Nm = Base.size(pa, 1); @inbounds $body; end
end

# --- @generated apply_symmetry! from dev (works on raw arrays) ---
# This is the exact @generated implementation from the dev branch.
@generated function _apply_symmetry_gen!(data::AbstractArray{T, D}, ::Val{H}) where {T, D, H}
    length(H) <= 1 && return :(return data)

    secondary = H[2:end]
    blocks    = Expr[]

    for mask_int in 1:(1 << length(secondary)) - 1
        active_dims = Tuple(secondary[k] for k in eachindex(secondary) if Bool((mask_int >> (k-1)) & 1))
        half_dim    = active_dims[end]

        range_exprs = ntuple(D) do d
            if d == H[1]
                :(1:1)
            elseif d == half_dim
                :(2:(Base.size(data, $d) >> 1) + 1)
            elseif d in active_dims
                :(2:Base.size(data, $d))
            elseif d in secondary
                :(1:1)
            else
                :(1:Base.size(data, $d))
            end
        end

        neg_exprs = ntuple(d -> d in active_dims ? :(Base.size(data, $d) - I[$d] + 2) : :(I[$d]), D)

        push!(blocks, quote
            for I in CartesianIndices($(Expr(:tuple, range_exprs...)))
                neg       = $(Expr(:call, :CartesianIndex, neg_exprs...))
                _av       = ReSolverFlowsBase._average_complex(data[I], data[neg])
                data[I]   = _av
                data[neg] = conj(_av)
            end
        end)
    end

    return quote
        $(blocks...)
        return data
    end
end

# --- apply_symmetry! dev reference: iterates ALL kH (no DC-plane restriction) ---
# This was the dev-branch behaviour; it is also wrong (touches non-DC entries)
# but that is exactly why we restricted it to the DC plane in this branch.

function _apply_sym_ft_gen!(u::FTField{<:AbstractGrid{<:Any,<:Any,<:Any,FFT_DIMS_ORDER}}) where {FFT_DIMS_ORDER}
    LI = LinearIndices(parent(u))
    g  = ReSolverFlowsBase.grid(u)
    for Ih in CartesianIndices(ReSolverFlowsBase.homogeneous_axes(u))   # ALL kH, not just DC
        Ih_neg = CartesianIndex(ntuple(Val(length(FFT_DIMS_ORDER))) do d
            if FFT_DIMS_ORDER[d] == ReSolverFlowsBase.rfft_dim(g)
                return Ih[d]
            elseif Ih[d] == 1
                return Ih[d]
            else
                return size(u, FFT_DIMS_ORDER[d]) - Ih[d] + 2
            end
        end)
        for Inh in CartesianIndices(ReSolverFlowsBase.inhomogeneous_axes(u))
            I    = CartesianIndex(ReSolverFlowsBase.combine_indices(g, Inh, Ih))
            Ineg = CartesianIndex(ReSolverFlowsBase.combine_indices(g, Inh, Ih_neg))
            if LI[I] <= LI[Ineg]
                av = ReSolverFlowsBase._average_complex(parent(u)[I], parent(u)[Ineg])
                parent(u)[I]    = av
                parent(u)[Ineg] = conj(av)
            end
        end
    end
    return u
end

function _apply_sym_pf_gen!(a::ProjectedField{<:AbstractGrid{<:Any,<:Any,<:Any,FFT_DIMS_ORDER}}) where {FFT_DIMS_ORDER}
    Nhom = length(FFT_DIMS_ORDER)
    pa   = parent(a)
    LI   = LinearIndices(pa)
    for Ih in CartesianIndices(ReSolverFlowsBase.homogeneous_axes(a))   # ALL kH, not just DC
        Ih_neg = CartesianIndex(ntuple(Val(Nhom)) do k
            if k == 1
                return Ih[k]
            elseif Ih[k] == 1
                return Ih[k]
            else
                return size(pa, k + 1) - Ih[k] + 2
            end
        end)
        for Im in axes(pa, 1)
            I    = CartesianIndex(Im, Ih.I...)
            Ineg = CartesianIndex(Im, Ih_neg.I...)
            if LI[I] <= LI[Ineg]
                av = ReSolverFlowsBase._average_complex(pa[I], pa[Ineg])
                pa[I]    = av
                pa[Ineg] = conj(av)
            end
        end
    end
    return a
end

# --- Reference wrappers for dot / normdiff / shift! ---

function _dot_ft_gen(u::FTField{G}, v::FTField{G}) where {G<:AbstractGrid}
    g = ReSolverFlowsBase.grid(u); pu = parent(u); pv = parent(v); ws = ReSolverFlowsBase.weights(g)
    s = Ref(zero(real(eltype(u))))
    _for_each_index(g) do ot, inh, idx
        @inbounds s[] += ot * ws[inh...] * real(conj(pu[idx...]) * pv[idx...])
    end
    return s[] / 2
end

function _normdiff_ft_gen(u::FTField{G}, v::FTField{G}) where {G<:AbstractGrid}
    g = ReSolverFlowsBase.grid(u); pu = parent(u); pv = parent(v); ws = ReSolverFlowsBase.weights(g)
    s = Ref(zero(real(eltype(u))))
    _for_each_index(g) do ot, inh, idx
        @inbounds s[] += ot * ws[inh...] * abs2(pu[idx...] - pv[idx...])
    end
    return sqrt(s[] / 2)
end

function _shift_ft_gen!(u::FTField{G}, shifts) where {G<:AbstractGrid}
    any(!iszero, shifts) || return u
    g = ReSolverFlowsBase.grid(u); pu = parent(u)
    inh_dims  = ReSolverFlowsBase.inhomogeneous_dims(g)
    inh_sizes = map(d -> size(g, d), inh_dims)
    _for_each_homogeneous_index(g) do _, hom_idx
        k     = ReSolverFlowsBase.to_wavenumber_vector(g, hom_idx)
        phase = ReSolverFlowsBase._shift_phase(g, shifts, k)
        for I in CartesianIndices(inh_sizes)
            idx = ReSolverFlowsBase.combine_indices(g, Tuple(I), hom_idx)
            @inbounds pu[idx...] *= phase
        end
    end
    return u
end

function _project_gen!(a::ProjectedField{G},
                        u::VectorField{N, <:FTField{G}}) where {N, T, G<:AbstractGrid{T}}
    a .= zero(Complex{T})
    _proj_gen_inner!(parent(a), u, a, weights(ReSolverFlowsBase.grid(u)), ReSolverFlowsBase.grid(u))
    return a
end

function _expand_gen!(u::VectorField{N, <:FTField{G}},
                       a::ProjectedField{G}) where {N, T, G<:AbstractGrid{T}}
    _expd_gen_inner!(u, parent(a), a, ReSolverFlowsBase.grid(u))
    return u
end

function _dot_pf_gen(a::ProjectedField{G}, b::ProjectedField{G}) where {G<:AbstractGrid}
    s = Ref(zero(real(eltype(a))))
    _for_each_homogeneous_index(ReSolverFlowsBase.grid(a)) do ot, hom_idx
        for m in axes(a, 1)
            @inbounds s[] += ot * real(LinearAlgebra.dot(a[m, hom_idx...], b[m, hom_idx...]))
        end
    end
    return s[] / 2
end

function _normdiff_pf_gen(a::ProjectedField{G}, b::ProjectedField{G}) where {G<:AbstractGrid}
    s = Ref(zero(real(eltype(a))))
    _for_each_homogeneous_index(ReSolverFlowsBase.grid(a)) do ot, hom_idx
        for m in axes(a, 1)
            @inbounds s[] += ot * abs2(a[m, hom_idx...] - b[m, hom_idx...])
        end
    end
    return sqrt(s[] / 2)
end

function _shift_pf_gen!(a::ProjectedField{G}, shifts) where {G<:AbstractGrid}
    any(!iszero, shifts) || return a
    g = ReSolverFlowsBase.grid(a)
    _for_each_homogeneous_index(g) do _, hom_idx
        k     = ReSolverFlowsBase.to_wavenumber_vector(g, hom_idx)
        phase = ReSolverFlowsBase._shift_phase(g, shifts, k)
        for m in axes(a, 1)
            @inbounds a[m, hom_idx...] *= phase
        end
    end
    return a
end

# ------------------------------------------------------------------ #
# Grid definitions — 4 layouts for a 4D array                       #
# ------------------------------------------------------------------ #
# Physical coords: x (rfft), y (inh/wall-normal), z (sFFT), t (sFFT)
# The four layouts place y at array dimensions 1, 2, 3, 4 respectively.

struct Layout1 <: AbstractGrid{Float64, 4, (2, 1, 3, 4), (2, 3, 4)}
    Ny::Int; Nx::Int; Nz::Int; Nt::Int; ws::Vector{Float64}
end
struct Layout2 <: AbstractGrid{Float64, 4, (1, 2, 3, 4), (1, 3, 4)}
    Nx::Int; Ny::Int; Nz::Int; Nt::Int; ws::Vector{Float64}
end
struct Layout3 <: AbstractGrid{Float64, 4, (1, 3, 2, 4), (1, 2, 4)}
    Nx::Int; Nz::Int; Ny::Int; Nt::Int; ws::Vector{Float64}
end
struct Layout4 <: AbstractGrid{Float64, 4, (1, 4, 2, 3), (1, 2, 3)}
    Nx::Int; Nz::Int; Nt::Int; Ny::Int; ws::Vector{Float64}
end

Base.size(g::Layout1) = (g.Ny, g.Nx, g.Nz, g.Nt)
Base.size(g::Layout2) = (g.Nx, g.Ny, g.Nz, g.Nt)
Base.size(g::Layout3) = (g.Nx, g.Nz, g.Ny, g.Nt)
Base.size(g::Layout4) = (g.Nx, g.Nz, g.Nt, g.Ny)

ReSolverFlowsBase.weights(g::Layout1) = g.ws
ReSolverFlowsBase.weights(g::Layout2) = g.ws
ReSolverFlowsBase.weights(g::Layout3) = g.ws
ReSolverFlowsBase.weights(g::Layout4) = g.ws

ReSolverFlowsBase.wavenumber_scale(::Layout1, ::Int) = 1.0
ReSolverFlowsBase.wavenumber_scale(::Layout2, ::Int) = 1.0
ReSolverFlowsBase.wavenumber_scale(::Layout3, ::Int) = 1.0
ReSolverFlowsBase.wavenumber_scale(::Layout4, ::Int) = 1.0

# Construct a layout from physical sizes (Ny, Nx, Nz, Nt)
make_grid(::Type{Layout1}, Ny, Nx, Nz, Nt) = Layout1(Ny, Nx, Nz, Nt, ones(Ny))
make_grid(::Type{Layout2}, Ny, Nx, Nz, Nt) = Layout2(Nx, Ny, Nz, Nt, ones(Ny))
make_grid(::Type{Layout3}, Ny, Nx, Nz, Nt) = Layout3(Nx, Nz, Ny, Nt, ones(Ny))
make_grid(::Type{Layout4}, Ny, Nx, Nz, Nt) = Layout4(Nx, Nz, Nt, Ny, ones(Ny))

n_elements(Ny, Nx, Nz, Nt) = Ny * ((Nx >> 1) + 1) * Nz * Nt

# ------------------------------------------------------------------ #
# Grid sizes (5 points, ~3 orders of magnitude in total elements)    #
# ------------------------------------------------------------------ #

sizes = [
    (Ny=4,  Nx=6,  Nz=5,  Nt=4),
    (Ny=6,  Nx=10, Nz=8,  Nt=6),
    (Ny=8,  Nx=16, Nz=12, Nt=8),
    (Ny=12, Nx=24, Nz=16, Nt=12),
    (Ny=16, Nx=32, Nz=24, Nt=16),
]

LAYOUTS       = [Layout1, Layout2, Layout3, Layout4]
LAYOUT_LABELS = ["inh=dim1 (y,x,z,t)", "inh=dim2 (x,y,z,t)",
                 "inh=dim3 (x,z,y,t)", "inh=dim4 (x,z,t,y)"]
LAYOUT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

const Nv     = 3
const Nm     = 8
const SHIFTS = (0.15, 0.30, 0.45)

# ------------------------------------------------------------------ #
# Benchmark helper                                                    #
# ------------------------------------------------------------------ #

function bench_ratio(f_cart, f_gen; samples=500)
    f_cart(); f_gen()   # warm-up
    bc = @benchmark $f_cart() samples=samples evals=1
    bg = @benchmark $f_gen()  samples=samples evals=1
    return median(bc).time / median(bg).time
end

# ------------------------------------------------------------------ #
# Main benchmark loop                                                 #
# ------------------------------------------------------------------ #

const NFN = 15
ratios = [[Float64[] for _ in LAYOUTS] for _ in 1:NFN]
nelems = [[Int[]     for _ in LAYOUTS] for _ in 1:NFN]

println("Running benchmarks…")

for (li, LT) in enumerate(LAYOUTS)
    println("  Layout $li / $(length(LAYOUTS)): $(LAYOUT_LABELS[li])")

    for s in sizes
        Ny, Nx, Nz, Nt = s.Ny, s.Nx, s.Nz, s.Nt
        ne = n_elements(Ny, Nx, Nz, Nt)
        g  = make_grid(LT, Ny, Nx, Nz, Nt)

        u   = FTField(g); parent(u) .= randn(ComplexF64, size(parent(u)))
        v   = FTField(g); parent(v) .= randn(ComplexF64, size(parent(v)))
        out = FTField(g)
        uc  = copy(u)

        fd = fft_dims(g)           # (rfft_dim, sFFT_dim1, sFFT_dim2)
        vd = Val.(fd)

        rfft_sz = (Nx >> 1) + 1
        ms  = ntuple(_ -> randn(ComplexF64, Nm, Ny, rfft_sz, Nz, Nt), Nv)
        a   = ProjectedField(g, ms)
        b   = ProjectedField(g, ms); parent(b) .= randn(ComplexF64, size(parent(b)))
        ac  = copy(a)
        q   = VectorField(ntuple(_ -> FTField(g), Nv)...)
        for n in 1:Nv; parent(q[n]) .= randn(ComplexF64, size(parent(u))); end
        out_q = VectorField(ntuple(_ -> zero(FTField(g)), Nv)...)

        @printf("    Ny=%2d Nx=%2d Nz=%2d Nt=%2d  (%8d elems)\n", Ny, Nx, Nz, Nt, ne)

        push_ratio!(fi, fcart, fgen) = (
            push!(nelems[fi][li], ne);
            push!(ratios[fi][li], bench_ratio(fcart, fgen)))

        # 1. ddx! rfft dim
        push_ratio!(1, () -> ddx!(out, u, vd[1]), () -> _ddx_gen!(out, u, vd[1]))

        # 2. ddx! 1st signed-FFT dim
        push_ratio!(2, () -> ddx!(out, u, vd[2]), () -> _ddx_gen!(out, u, vd[2]))

        # 3. ddx! 2nd signed-FFT dim
        push_ratio!(3, () -> ddx!(out, u, vd[3]), () -> _ddx_gen!(out, u, vd[3]))

        # 4. dot(FTField)
        push_ratio!(4, () -> dot(u, v), () -> _dot_ft_gen(u, v))

        # 5. normdiff(FTField)
        push_ratio!(5, () -> normdiff(u, v), () -> _normdiff_ft_gen(u, v))

        # 6. shift!(FTField)
        uc .= u
        push_ratio!(6, () -> shift!(uc, SHIFTS), () -> _shift_ft_gen!(uc, SHIFTS))

        # 7. project! LoopGalerkin
        push_ratio!(7,
            () -> project!(a, q, LoopGalerkin()),
            () -> _project_gen!(a, q))

        # 8. expand! LoopGalerkin
        push_ratio!(8,
            () -> expand!(out_q, a, LoopGalerkin()),
            () -> _expand_gen!(out_q, a))

        # 9. project! GemmGalerkin (CartesianIndices vs LoopGalerkin)
        push_ratio!(9,
            () -> project!(a, q, GemmGalerkin()),
            () -> project!(copy(a), q, LoopGalerkin()))

        # 10. expand! GemmGalerkin (CartesianIndices vs LoopGalerkin)
        out_q_loop = copy(out_q)
        push_ratio!(10,
            () -> expand!(out_q, a, GemmGalerkin()),
            () -> expand!(out_q_loop, a, LoopGalerkin()))

        # 11. dot(ProjectedField)
        push_ratio!(11, () -> dot(a, b), () -> _dot_pf_gen(a, b))

        # 12. normdiff(ProjectedField)
        push_ratio!(12, () -> normdiff(a, b), () -> _normdiff_pf_gen(a, b))

        # 13. shift!(ProjectedField)
        ac .= a
        push_ratio!(13, () -> shift!(ac, SHIFTS), () -> _shift_pf_gen!(ac, SHIFTS))

        # 14. apply_symmetry! FTField  (cart = DC-plane only; gen = @generated all kH)
        parent(uc) .= randn(ComplexF64, size(parent(u)))
        push_ratio!(14,
            () -> ReSolverFlowsBase.apply_symmetry!(uc),
            () -> _apply_symmetry_gen!(parent(uc), Val(ReSolverFlowsBase.fft_dims(ReSolverFlowsBase.grid(uc)))))

        # 15. apply_symmetry! ProjectedField (cart = DC-plane only; gen = CartesianIndices all kH)
        parent(ac) .= randn(ComplexF64, size(parent(a)))
        push_ratio!(15,
            () -> ReSolverFlowsBase.apply_symmetry!(ac),
            () -> _apply_sym_pf_gen!(ac))
    end
end

println("Done. Plotting…")

# ------------------------------------------------------------------ #
# Figure                                                              #
# ------------------------------------------------------------------ #

fn_labels = [
    "ddx!  rfft dim (x)",
    "ddx!  signed-FFT dim 1 (z)",
    "ddx!  signed-FFT dim 2 (t)",
    "dot(FTField, FTField)",
    "normdiff(FTField)",
    "shift!(FTField)",
    "project!  LoopGalerkin",
    "expand!   LoopGalerkin",
    "project!  GemmGalerkin",
    "expand!   GemmGalerkin",
    "dot(ProjectedField, ProjectedField)",
    "normdiff(ProjectedField)",
    "shift!(ProjectedField)",
    "apply_symmetry!(FTField)  [cart=DC only · gen=all kH]",
    "apply_symmetry!(ProjectedField)  [cart=DC only · gen=all kH]",
]

fig, axs = plt.subplots(NFN, 1, figsize=(9, 5 * NFN))
fig.suptitle(
    "CartesianIndices (this branch) vs @generated / for_each_* (dev branch)\n" *
    "ratio  t_cart / t_gen   (< 1 = CartesianIndices faster)\n" *
    "4-D grid: x rfft, y inh, z+t signed-FFT  |  Nv=$Nv  Nm=$Nm",
    fontsize=12, y=1.001)

for (fi, ax) in enumerate(axs)
    for (li, _) in enumerate(LAYOUTS)
        ax.semilogx(nelems[fi][li], ratios[fi][li],
                    "o-", color=LAYOUT_COLORS[li],
                    label=LAYOUT_LABELS[li], linewidth=1.5, markersize=5)
    end
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_title(fn_labels[fi], fontsize=10, loc="left", pad=4)
    ax.set_xlabel("FTField elements  (Ny × ⌈Nx/2⌉ × Nz × Nt)", fontsize=9)
    ax.set_ylabel("t_cart / t_gen", fontsize=9)
    ax.set_ylim(bottom=0.0, top=2.0)
    ax.grid(true, which="both", linestyle=":", alpha=0.4)
    ax.tick_params(labelsize=8)
    fi == 1 && ax.legend(fontsize=8, loc="upper right")
end

fig.tight_layout()
outpath = joinpath(@__DIR__, "output", "bench_vs_dev.png")
mkpath(dirname(outpath))
fig.savefig(outpath, dpi=150, bbox_inches="tight")
println("Figure saved to $outpath")
