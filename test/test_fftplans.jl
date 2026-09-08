import FFTW
using FFTW: ESTIMATE

struct FFTDataWrapper{T, N, A<:AbstractArray{T, N}} <: AbstractArray{T, N}
    data::A
end

Base.IndexStyle(::Type{<:FFTDataWrapper}) = IndexLinear()
Base.size(a::FFTDataWrapper) = size(a.data)
Base.getindex(a::FFTDataWrapper, i::Int) = a.data[i]
Base.setindex!(a::FFTDataWrapper, v, i::Int) = (a.data[i] = v)

ReSolverFlowsBase._fft_data(a::FFTDataWrapper) = a.data

@testset "FFTPlans constructor                                                " begin
    for T in (Float32, Float64)
        # 1D, no dealiasing: cache is the standard spectral shape, norm = 1/N
        f = FFTPlans((8,), (1,), T, dealias=false, flags=ESTIMATE)
        @test f.norm ≈ T(1/8)
        @test size(f.cache) == (5,)   # (8 >> 1) + 1

        # 1D, with dealiasing: physical grid pads to 13, cache matches padded transform
        f = FFTPlans((8,), (1,), T, dealias=true, flags=ESTIMATE)
        @test f.norm ≈ T(1/13)
        @test size(f.cache) == (7,)   # (13 >> 1) + 1

        # 2D, rfft dim only: dim-2 is untransformed, norm = 1/M
        f = FFTPlans((8, 6), (1,), T, dealias=false, flags=ESTIMATE)
        @test f.norm ≈ T(1/8)
        @test size(f.cache) == (5, 6)

        # 2D, both dims transformed: norm = 1/(M*N)
        f = FFTPlans((8, 6), (1, 2), T, dealias=false, flags=ESTIMATE)
        @test f.norm ≈ T(1/48)
        @test size(f.cache) == (5, 6)

        # 2D, dealiased: cache matches padded transform shape
        f = FFTPlans((8, 6), (1, 2), T, dealias=true, flags=ESTIMATE)
        pad = ReSolverFlowsBase.get_padded_size((8, 6), (1, 2))
        @test f.norm ≈ T(1/prod(pad))
        @test size(f.cache) == ((pad[1] >> 1) + 1, pad[2])

        # 3D, all dims transformed
        f = FFTPlans((4, 6, 8), (1, 2, 3), T, dealias=false, flags=ESTIMATE)
        @test f.norm ≈ T(1/192)
        @test size(f.cache) == (3, 6, 8)

        # padded_size overrides dealias: explicit size used instead of 3/2 rule
        f = FFTPlans((8,), (1,), T, padded_size=(16,), flags=ESTIMATE)
        @test f.norm ≈ T(1/16)
        @test size(f.cache) == (9,)   # (16 >> 1) + 1

        # padded_size == shape: no padding, so DEALIAS=false
        f = FFTPlans((8,), (1,), T, padded_size=(8,), flags=ESTIMATE)
        @test f.norm ≈ T(1/8)
        @test size(f.cache) == (5,)
        @test !isa(f, FFTPlans{true})

        # padded_size in 2D, only rfft dim padded
        f = FFTPlans((8, 6), (1,), T, padded_size=(12, 6), flags=ESTIMATE)
        @test f.norm ≈ T(1/12)
        @test size(f.cache) == (7, 6)  # (12 >> 1) + 1

        # argument errors
        @test_throws ArgumentError FFTPlans((8,), (1,), T, padded_size=(6,),   flags=ESTIMATE)  # smaller than shape
        @test_throws ArgumentError FFTPlans((8,), (1,), T, padded_size=(8, 6), flags=ESTIMATE)  # wrong length
        @test_throws ArgumentError FFTPlans((8,), (1,), T, padded_size=(16,), dealias=false, flags=ESTIMATE)  # contradictory
    end
end

@testset "Transform utilities                                                 " begin
    @testset "get_padded_size                " begin
        # single transformed dimension
        @test ReSolverFlowsBase.get_padded_size((4,), (1,)) == (7,)
        @test ReSolverFlowsBase.get_padded_size((6,), (1,)) == (9,)
        @test ReSolverFlowsBase.get_padded_size((8,), (1,)) == (13,)

        # untransformed dimensions are unchanged
        @test ReSolverFlowsBase.get_padded_size((4, 5), (1,)) == (7, 5)
        @test ReSolverFlowsBase.get_padded_size((4, 5), (2,)) == (4, 9)

        # multiple transformed dimensions
        @test ReSolverFlowsBase.get_padded_size((4, 6), (1, 2))       == (7, 9)
        @test ReSolverFlowsBase.get_padded_size((4, 6, 8), (1, 2, 3)) == (7, 9, 13)
        @test ReSolverFlowsBase.get_padded_size((4, 5, 6), (1, 3))    == (7, 5, 9)

        # padded size must satisfy the 3/2 dealiasing rule for all transformed dimensions
        for s in 1:32, ndim in 1:3
            size   = ntuple(_ -> s, ndim)
            order  = ntuple(identity, ndim)
            padded = ReSolverFlowsBase.get_padded_size(size, order)
            for d in order
                @test padded[d] >= 3s/2
            end
        end
    end

    @testset "_get_transform_size               " begin
        # 1D: first (and only) dimension is halved + 1
        @test ReSolverFlowsBase._get_transform_size((4,), 1) == (3,)
        @test ReSolverFlowsBase._get_transform_size((6,), 1) == (4,)
        @test ReSolverFlowsBase._get_transform_size((8,), 1) == (5,)

        # 2D: only the selected dimension changes
        @test ReSolverFlowsBase._get_transform_size((4, 6), 1) == (3, 6)
        @test ReSolverFlowsBase._get_transform_size((4, 6), 2) == (4, 4)

        # 3D
        @test ReSolverFlowsBase._get_transform_size((4, 6, 8), 1) == (3, 6, 8)
        @test ReSolverFlowsBase._get_transform_size((4, 6, 8), 2) == (4, 4, 8)
        @test ReSolverFlowsBase._get_transform_size((4, 6, 8), 3) == (4, 6, 5)
    end

    @testset "_apply_mask!                      " begin
        for sz in ((5,), (5, 6), (3, 4, 5))
            cache = randn(ComplexF64, sz...)
            ReSolverFlowsBase._apply_mask!(cache)
            @test all(iszero, cache)
        end
    end

    @testset "_copy_to/from_padded!, 1D         " begin
        # NTuple{1}: only the rfft dim is transformed
        M, M_pad = 8, ReSolverFlowsBase.get_padded_size((8,), (1,))[1]
        M_spec, M_spec_pad = M ÷ 2 + 1, M_pad ÷ 2 + 1

        u     = randn(ComplexF64, M_spec)
        cache = zeros(ComplexF64, M_spec_pad)

        ReSolverFlowsBase._apply_mask!(cache)
        ReSolverFlowsBase._copy_to_padded!(cache, u, (1,))

        # resolved range is embedded at the low end; padding zone stays zero
        @test cache[1:M_spec] ≈ u
        @test all(iszero, cache[M_spec+1:end])

        # round-trip
        u2 = zeros(ComplexF64, M_spec)
        ReSolverFlowsBase._copy_from_padded!(u2, cache, (1,))
        @test u2 ≈ u
    end

    @testset "_copy_to/from_padded!, 2D         " begin
        # NTuple{2}: rfft dim + one full-spectrum dim
        M, N   = 8, 6
        M_pad, N_pad = ReSolverFlowsBase.get_padded_size((M, N), (1, 2))
        M_spec, M_spec_pad = M ÷ 2 + 1, M_pad ÷ 2 + 1

        u     = randn(ComplexF64, M_spec, N)
        cache = zeros(ComplexF64, M_spec_pad, N_pad)

        ReSolverFlowsBase._apply_mask!(cache)
        ReSolverFlowsBase._copy_to_padded!(cache, u, (1, 2))

        # round-trip
        u2 = zeros(ComplexF64, M_spec, N)
        ReSolverFlowsBase._copy_from_padded!(u2, cache, (1, 2))
        @test u2 ≈ u

        # positive-frequency block is at the low end of each dim
        N_pos = (N >> 1) + 1
        @test cache[1:M_spec, 1:N_pos] ≈ u[:, 1:N_pos]

        # negative-frequency block is at the high end of dim 2
        N_neg = N - N_pos
        if N_neg > 0
            @test cache[1:M_spec, N_pad-N_neg+1:N_pad] ≈ u[:, N_pos+1:end]
        end
    end

    @testset "_copy_to/from_padded!, 3D         " begin
        # NTuple{3}: rfft dim + two full-spectrum dims
        L, M, N  = 4, 6, 8
        L_pad, M_pad, N_pad = ReSolverFlowsBase.get_padded_size((L, M, N), (1, 2, 3))
        L_spec, L_spec_pad  = L ÷ 2 + 1, L_pad ÷ 2 + 1

        u     = randn(ComplexF64, L_spec, M, N)
        cache = zeros(ComplexF64, L_spec_pad, M_pad, N_pad)

        ReSolverFlowsBase._apply_mask!(cache)
        ReSolverFlowsBase._copy_to_padded!(cache, u, (1, 2, 3))

        u2 = zeros(ComplexF64, L_spec, M, N)
        ReSolverFlowsBase._copy_from_padded!(u2, cache, (1, 2, 3))
        @test u2 ≈ u
    end

    @testset "_add_from_padded!, 1D             " begin
        M, M_pad = 8, ReSolverFlowsBase.get_padded_size((8,), (1,))[1]
        M_spec, M_spec_pad = M ÷ 2 + 1, M_pad ÷ 2 + 1

        cache = randn(ComplexF64, M_spec_pad)
        accum = randn(ComplexF64, M_spec)
        accum0 = copy(accum)

        ReSolverFlowsBase._add_from_padded!(accum, cache, (1,))
        @test accum ≈ accum0 .+ cache[1:M_spec]
    end

    @testset "_add_from_padded!, 2D             " begin
        M, N   = 8, 6
        M_pad, N_pad = ReSolverFlowsBase.get_padded_size((M, N), (1, 2))
        M_spec = M ÷ 2 + 1

        cache = randn(ComplexF64, (M_pad ÷ 2 + 1), N_pad)
        accum = randn(ComplexF64, M_spec, N)
        accum0 = copy(accum)

        ReSolverFlowsBase._add_from_padded!(accum, cache, (1, 2))

        # positive-frequency block
        N_pos = (N >> 1) + 1
        @test accum[:, 1:N_pos] ≈ accum0[:, 1:N_pos] .+ cache[1:M_spec, 1:N_pos]

        # negative-frequency block
        N_neg = N - N_pos
        if N_neg > 0
            @test accum[:, N_pos+1:end] ≈ accum0[:, N_pos+1:end] .+ cache[1:M_spec, N_pad-N_neg+1:N_pad]
        end
    end
end

@testset "FFTPlans execution                                                  " begin
    @testset "forward transform, 1D             " begin
        N = 16
        u = randn(Float64, N)
        û = zeros(ComplexF64, N ÷ 2 + 1)
        f = FFTPlans((N,), (1,), Float64, dealias=false, flags=ESTIMATE)

        # correctness: matches FFTW.rfft / N
        f(û, u, false, false)
        @test û ≈ FFTW.rfft(u) ./ N

        # use_cache=true routes through f.cache but gives the same result
        û2 = zeros(ComplexF64, N ÷ 2 + 1)
        f(û2, u, false, true)
        @test û2 ≈ û

        # add=true accumulates rather than overwrites
        û3 = copy(û)
        f(û3, u, true, false)
        @test û3 ≈ 2 .* û

        # add + use_cache
        û4 = copy(û)
        f(û4, u, true, true)
        @test û4 ≈ 2 .* û
    end

    @testset "forward transform, 2D, order=(1,) " begin
        M, N = 8, 6
        u = randn(Float64, M, N)
        û = zeros(ComplexF64, M ÷ 2 + 1, N)
        f = FFTPlans((M, N), (1,), Float64, dealias=false, flags=ESTIMATE)

        f(û, u, false, false)
        # rfft along dim 1 only; dim 2 is untransformed and its entries appear as-is
        @test û ≈ FFTW.rfft(u, [1]) ./ M
    end

    @testset "forward transform, 2D, order=(1,2)" begin
        M, N = 8, 6
        u = randn(Float64, M, N)
        û = zeros(ComplexF64, M ÷ 2 + 1, N)
        f = FFTPlans((M, N), (1, 2), Float64, dealias=false, flags=ESTIMATE)

        f(û, u, false, false)
        @test û ≈ FFTW.rfft(u, [1, 2]) ./ (M * N)

        # add
        û2 = copy(û)
        f(û2, u, true, false)
        @test û2 ≈ 2 .* û
    end

    @testset "forward transform, 3D             " begin
        L, M, N = 4, 6, 8
        u = randn(Float64, L, M, N)
        û = zeros(ComplexF64, L ÷ 2 + 1, M, N)
        f = FFTPlans((L, M, N), (1, 2, 3), Float64, dealias=false, flags=ESTIMATE)

        f(û, u, false, false)
        @test û ≈ FFTW.rfft(u, [1, 2, 3]) ./ (L * M * N)
    end

    @testset "forward transform, 1D, dealiased  " begin
        N = 8
        N_pad = ReSolverFlowsBase.get_padded_size((N,), (1,))[1]   # 13
        u_pad = randn(Float64, N_pad)
        û = zeros(ComplexF64, N ÷ 2 + 1)
        f = FFTPlans((N,), (1,), Float64, dealias=true, flags=ESTIMATE)

        f(û, u_pad, false, false)

        # the dealiased forward transform is rfft on the padded grid, truncated to
        # the resolved spectral size and normalised by the padded physical size
        ref = FFTW.rfft(u_pad) ./ N_pad
        @test û ≈ ref[1:N ÷ 2 + 1]
    end

    @testset "forward transform, 2D, dealiased  " begin
        M, N = 8, 6
        M_pad, N_pad = ReSolverFlowsBase.get_padded_size((M, N), (1, 2))
        u_pad = randn(Float64, M_pad, N_pad)
        û = zeros(ComplexF64, M ÷ 2 + 1, N)
        f = FFTPlans((M, N), (1, 2), Float64, dealias=true, flags=ESTIMATE)

        f(û, u_pad, false, false)

        ref   = FFTW.rfft(u_pad, [1, 2]) ./ (M_pad * N_pad)
        N_pos = (N >> 1) + 1
        N_neg = N - N_pos
        @test û ≈ hcat(ref[1:M÷2+1, 1:N_pos], ref[1:M÷2+1, N_pad-N_neg+1:N_pad])
    end

    @testset "backward transform round-trip, 1D " begin
        for N in (8, 16)
            u  = randn(Float64, N)
            û  = zeros(ComplexF64, N ÷ 2 + 1)
            u2 = zeros(Float64, N)
            f  = FFTPlans((N,), (1,), Float64, dealias=false, flags=ESTIMATE)
            f(û, u, false, false)
            f(u2, û, true, false)
            @test u2 ≈ u
        end
    end

    @testset "backward transform round-trip, 2D " begin
        M, N = 8, 6
        u  = randn(Float64, M, N)
        û  = zeros(ComplexF64, M ÷ 2 + 1, N)
        u2 = zeros(Float64, M, N)
        f  = FFTPlans((M, N), (1, 2), Float64, dealias=false, flags=ESTIMATE)
        f(û, u, false, false)
        f(u2, û, true, false)
        @test u2 ≈ u
    end

    @testset "backward transform round-trip, 3D                                 " begin
        L, M, N = 4, 6, 8
        u  = randn(Float64, L, M, N)
        û  = zeros(ComplexF64, L ÷ 2 + 1, M, N)
        u2 = zeros(Float64, L, M, N)
        f  = FFTPlans((L, M, N), (1, 2, 3), Float64, dealias=false, flags=ESTIMATE)
        f(û, u, false, false)
        f(u2, û, true, false)
        @test u2 ≈ u
    end

    @testset "backward round-trip, 1D dealiased " begin
        # Only band-limited inputs survive the forward truncation.  Start from a
        # resolved spectral array, backward-transform to get the padded physical
        # field, then forward-transform back and verify we recover the original.
        N = 8
        N_pad = ReSolverFlowsBase.get_padded_size((N,), (1,))[1]
        f = FFTPlans((N,), (1,), Float64, dealias=true, flags=ESTIMATE)

        # û_init must represent a real-valued field: DC (mode 0) and Nyquist (last
        # mode when N is even) must be purely real, as brfft discards their imaginary parts.
        û_init      = randn(ComplexF64, N ÷ 2 + 1)
        û_init[1]   = real(û_init[1])
        û_init[end] = real(û_init[end])
        u_pad  = zeros(Float64, N_pad)
        û_out  = zeros(ComplexF64, N ÷ 2 + 1)

        f(u_pad, û_init, true, false)
        f(û_out, u_pad, false, false)
        @test û_out ≈ û_init
    end

    @testset "safe backward preserves û         " begin
        N = 16
        u  = randn(Float64, N)
        û  = zeros(ComplexF64, N ÷ 2 + 1)
        f  = FFTPlans((N,), (1,), Float64, dealias=false, flags=ESTIMATE)
        f(û, u, false, false)

        û_before = copy(û)
        u2 = zeros(Float64, N)
        f(u2, û, true, false)           # safe=true
        @test û ≈ û_before              # input unchanged
        @test u2 ≈ u
    end

    @testset "unsafe+use_cache preserves û      " begin
        N = 16
        u  = randn(Float64, N)
        û  = zeros(ComplexF64, N ÷ 2 + 1)
        f  = FFTPlans((N,), (1,), Float64, dealias=false, flags=ESTIMATE)
        f(û, u, false, false)

        û_before = copy(û)
        u2 = zeros(Float64, N)
        f(u2, û, false, true)           # safe=false, use_cache=true
        @test û ≈ û_before              # use_cache routes through f.cache, not û
        @test u2 ≈ u
    end

    @testset "unsafe backward gives same result " begin
        N = 16
        u  = randn(Float64, N)
        û  = zeros(ComplexF64, N ÷ 2 + 1)
        f  = FFTPlans((N,), (1,), Float64, dealias=false, flags=ESTIMATE)
        f(û, u, false, false)

        u_safe   = zeros(Float64, N)
        u_unsafe = zeros(Float64, N)
        f(u_safe,   copy(û), true,  false)
        f(u_unsafe, copy(û), false, false)  # may corrupt its û copy, but output matches
        @test u_unsafe ≈ u_safe
    end

    @testset "dealiasing eliminates aliasing    " begin
        # u = sin(3x), N=8 resolved modes, padded to 13.
        # u² = (1 - cos(6x))/2 has mode 6, which lies beyond the resolved range
        # (modes 0..4 for N=8).  With dealiasing the padded transform captures mode
        # 6 accurately and then truncates it, so the output has only the DC component.
        # Without dealiasing, mode 6 aliases to mode 2 (6 mod 8 = 6, 8-6 = 2),
        # corrupting û_sq[3].

        N     = 8
        N_pad = ReSolverFlowsBase.get_padded_size((N,), (1,))[1]   # 13

        # ---- dealiased ----
        f_de = FFTPlans((N,), (1,), Float64, dealias=true,  flags=ESTIMATE)
        û_de = zeros(ComplexF64, N ÷ 2 + 1)
        û_de[4] = -im/2                  # mode 3 → sin(3x) in physical space

        u_pad = zeros(Float64, N_pad)
        f_de(u_pad, û_de, true, false)   # backward: padded physical representation

        û_sq_de = zeros(ComplexF64, N ÷ 2 + 1)
        f_de(û_sq_de, u_pad .^ 2, false, false)

        @test real(û_sq_de[1]) ≈  0.5  atol=1e-12   # DC = 1/2 ✓
        @test û_sq_de[2:end]   ≈  zeros(ComplexF64, N ÷ 2)  atol=1e-12   # mode 6 truncated, not aliased

        # ---- non-dealiased ----
        f_no = FFTPlans((N,), (1,), Float64, dealias=false, flags=ESTIMATE)
        û_no = zeros(ComplexF64, N ÷ 2 + 1)
        û_no[4] = -im/2

        u_no = zeros(Float64, N)
        f_no(u_no, û_no, true, false)   # backward: 8-point physical grid

        û_sq_no = zeros(ComplexF64, N ÷ 2 + 1)
        f_no(û_sq_no, u_no .^ 2, false, false)

        @test real(û_sq_no[1]) ≈  0.5  atol=1e-12   # DC still correct
        @test abs(û_sq_no[3])  >  0.1               # mode 2 contaminated by aliased mode 6
    end

    @testset "top-level execution on fake   " begin
        # construct grid
        Nx = 16; Ny = 11
        L = 10*rand()
        g = FakeGrid(rand(Float64, Nx), Ny, L)

        # create plans
        plans  = FFTPlans(g, dealias=false, flags=FFTW.ESTIMATE)
        plansd = FFTPlans(g, dealias=true,  flags=FFTW.ESTIMATE)

        # random signal
        U  = FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1))
        u  = Field(g, dealias=false)
        ud = Field(g, dealias=true)
        Uv = VectorField(FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1)),
                        FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1)),
                        FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1)))
        uv = VectorField(g, Field)

        # test transforms
        @test  plans(    FTField(g),  plans(u,  U))           ≈   U
        @test plansd(    FTField(g), plansd(ud, U))           ≈   U
        @test plansd(       copy(U), plansd(ud, U), add=true) ≈ 2*U
        @test  plans(VectorField(g),  plans(uv, Uv))           ≈ Uv

        # test allocation
        funf(plan, A, B) = @allocated plan(A, B)
        funb(plan, B, A) = @allocated plan(B, A, add=false)
        # funf(plansd, ud, U) # run first to avoid initial allocation
        # funb(plansd, U, ud)
        @test funf(plans,  u,  U) == 0
        # @test funf(plansd, ud, U) == 0
        @test funb(plans,  U,  u) == 0
        # @test funb(plansd, U, ud) == 0

        # test constructing transforms
        @test FFT(plans(similar(u), U)) ≈ U
        @test IFFT(U) == plans(similar(u), U)
        @test FFT(plans(similar(uv), Uv)) ≈ Uv
        @test IFFT(Uv) == plans(similar(uv), Uv)
    end

    @testset "field transforms use _fft_data " begin
        Nx = 5
        Ny = 8
        g = FakeGrid(rand(Float64, Nx), Ny, 2π)
        raw = randn(Float64, size(g))

        u = Field(g, FFTDataWrapper(copy(raw)))
        U = FTField(g, FFTDataWrapper(zeros(ComplexF64, transform_size(g))))
        v = Field(g, FFTDataWrapper(zeros(Float64, size(g))))
        plans = FFTPlans(g, dealias=false, flags=FFTW.ESTIMATE)

        plans(U, u)
        @test ReSolverFlowsBase._fft_data(U) ≈ FFTW.rfft(raw, ReSolverFlowsBase.fft_storage_dims(g)) ./ prod(ReSolverFlowsBase.fft_norm(g))

        plans(v, U)
        @test ReSolverFlowsBase._fft_data(v) ≈ raw

        @test parent(FFT(u)) ≈ ReSolverFlowsBase._fft_data(U)
        @test parent(IFFT(U)) ≈ raw
    end
end
