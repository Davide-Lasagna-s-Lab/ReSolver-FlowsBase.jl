@testset "WaveNumberVector conversion                                         " begin
    @testset "indexing                      " begin
        k = WaveNumberVector(2, -1, 0)
        @test k[1] == 2
        @test k[2] == -1
        @test k[1:3] == (2, -1, 0)
        @test length(k) == 3
    end

    @testset "1D                            " begin
        N1 = 23 # N1 can be even or odd
        g = SpectralTestGrid{(16, N1, 16, 16), 4, (1, 2, 3, 4), (2,)}()

        n1s = collect(0:(N1 >> 1) + 1)

        # Non-conjugate half: n1 ≥ 0, forward ordering
        for _n1 in 1:(N1 >> 1) + 1
            k = WaveNumberVector(n1s[_n1])
            @test ReSolverFlowsBase.to_homogeneous_indices(g, k) == (_n1, false)
            @test ReSolverFlowsBase.to_wavenumber_vector(g, (_n1,)) == k
        end

        n1s = collect(0:-1:-(N1 >> 1))

        # Conjugate half: n1 < 0, reversed ordering
        for _n1 in 2:(N1 >> 1) + 1
            k = WaveNumberVector(n1s[_n1])
            @test ReSolverFlowsBase.to_homogeneous_indices(g, k) == (_n1, true)
        end
    end

    @testset "2D                            " begin
        N1 = 23 # N1 can be even or odd
        N2 = 23 # N2 has to be odd
        g = SpectralTestGrid{(16, N1, N2, 16), 4, (1, 2, 3, 4), (2, 3)}()

        n1s = collect(0:(N1 >> 1) + 1)
        n2s = [collect(0:(N2 >> 1)); collect(-(N2 >> 1):-1)]

        # Non-conjugate half: n1 ≥ 0, forward ordering
        for _n1 in 1:(N1 >> 1) + 1, _n2 in 1:N2
            k = WaveNumberVector(n1s[_n1], n2s[_n2])
            @test ReSolverFlowsBase.to_homogeneous_indices(g, k) == (_n1, _n2, false)
            @test ReSolverFlowsBase.to_wavenumber_vector(g, (_n1, _n2)) == k
        end

        n1s = collect(0:-1:-(N1 >> 1))
        n2s = [[0]; collect(-1:-1:-(N2 >> 1)); collect((N2 >> 1):-1:1)]

        # Conjugate half: n1 < 0, reversed ordering
        for _n1 in 2:(N1 >> 1) + 1, _n2 in 1:N2
            k = WaveNumberVector(n1s[_n1], n2s[_n2])
            @test ReSolverFlowsBase.to_homogeneous_indices(g, k) == (_n1, _n2, true)
        end
    end

    @testset "3D                            " begin
        N1 = 23 # N1 can be even or odd
        N2 = 23 # Nz has to be odd
        N3 = 23 # N3 has to be odd
        g = SpectralTestGrid{(16, N1, N2, N3), 4, (1, 2, 3, 4), (2, 3, 4)}()

        n1s = collect(0:(N1 >> 1) + 1)
        n2s = [collect(0:(N2 >> 1)); collect(-(N3 >> 1):-1)]
        n3s = [collect(0:(N3 >> 1)); collect(-(N3 >> 1):-1)]

        # Non-conjugate half: n1 ≥ 0, forward ordering
        for _n1 in 1:(N1 >> 1) + 1, _n2 in 1:N2, _n3 in 1:N3
            k = WaveNumberVector(n1s[_n1], n2s[_n2], n3s[_n3])
            @test ReSolverFlowsBase.to_homogeneous_indices(g, k) == (_n1, _n2, _n3, false)
            @test ReSolverFlowsBase.to_wavenumber_vector(g, (_n1, _n2, _n3)) == k
        end

        n1s = collect(0:-1:-(N1 >> 1))
        n2s = [[0]; collect(-1:-1:-(N2 >> 1)); collect((N2 >> 1:-1:1))]
        n3s = [[0]; collect(-1:-1:-(N3 >> 1)); collect((N3 >> 1:-1:1))]

        # Conjugate half: n1 < 0, reversed ordering
        for _n1 in 2:(N1 >> 1) + 1, _n2 in 1:N2, _n3 in 1:N3
            k = WaveNumberVector(n1s[_n1], n2s[_n2], n3s[_n3])
            @test ReSolverFlowsBase.to_homogeneous_indices(g, k) == (_n1, _n2, _n3, true)
        end
    end

    @testset "combined indexing         " begin
        struct MockGrid{D, AXES, FFT_DIMS_ORDER} <: AbstractGrid{Float64, D, AXES, FFT_DIMS_ORDER} end
        CI = CartesianIndex

        # ── FFT_DIMS_ORDER = () : no constrained dims, all free ────────────────────────────
        @test ReSolverFlowsBase.combine_indices(MockGrid{3, nothing, ()}(), CI(10, 20, 30), CI()) == (10, 20, 30)

        # ── FFT_DIMS_ORDER length 1 ────────────────────────────────────────────────────────
        # FFT_DIMS_ORDER = (1,): dim 1 → K, dims 2-3 → I
        @test ReSolverFlowsBase.combine_indices(MockGrid{3, nothing, (1,)}(), CI(20, 30), CI(10)) == (10, 20, 30)

        # FFT_DIMS_ORDER = (2,): dims 1,3 → I; dim 2 → K
        @test ReSolverFlowsBase.combine_indices(MockGrid{3, nothing, (2,)}(), CI(10, 30), CI(20)) == (10, 20, 30)

        # FFT_DIMS_ORDER = (3,): dims 1,2 → I; dim 3 → K
        @test ReSolverFlowsBase.combine_indices(MockGrid{3, nothing, (3,)}(), CI(10, 20), CI(30)) == (10, 20, 30)

        # FFT_DIMS_ORDER = (1,): dims () → I; dim 1 → K
        @test ReSolverFlowsBase.combine_indices(MockGrid{1, nothing, (1,)}(), CI(), CI(42)) == (42,)

        # ── FFT_DIMS_ORDER length 2 ────────────────────────────────────────────────────────
        # FFT_DIMS_ORDER = (1,2): dims 1,2 → K; dim 3 → I
        @test ReSolverFlowsBase.combine_indices(MockGrid{3, nothing, (1, 2)}(), CI(30), CI(10, 20)) == (10, 20, 30)

        # FFT_DIMS_ORDER = (2,3): dim 1 → I; dims 2,3 → K
        @test ReSolverFlowsBase.combine_indices(MockGrid{3, nothing, (2, 3)}(), CI(10), CI(20, 30)) == (10, 20, 30)

        # FFT_DIMS_ORDER = (1,3): dims 1,3 → K; dim 2 → I
        @test ReSolverFlowsBase.combine_indices(MockGrid{3, nothing, (1, 3)}(), CI(20), CI(10, 30)) == (10, 20, 30)

        # FFT_DIMS_ORDER = (1,2): both dims → K, no free dims
        @test ReSolverFlowsBase.combine_indices(MockGrid{2, nothing, (1, 2)}(), CI(), CI(10, 20)) == (10, 20)

        # ── FFT_DIMS_ORDER length 3 ────────────────────────────────────────────────────────
        # FFT_DIMS_ORDER = (1,2,3): every dim → K, no free dims
        @test ReSolverFlowsBase.combine_indices(MockGrid{3, nothing, (1, 2, 3)}(), CI(), CI(10, 20, 30)) == (10, 20, 30)

        # FFT_DIMS_ORDER = (1,3,5): dims 1,3,5 → K; dims 2,4 → I
        @test ReSolverFlowsBase.combine_indices(MockGrid{5, nothing, (1, 3, 5)}(), CI(20, 40), CI(10, 30, 50)) == (10, 20, 30, 40, 50)

        # FFT_DIMS_ORDER = (3,4,5): dims 1,2 → I; dims 3,4,5 → K
        @test ReSolverFlowsBase.combine_indices(MockGrid{5, nothing, (3, 4, 5)}(), CI(10, 20), CI(30, 40, 50)) == (10, 20, 30, 40, 50)

        # FFT_DIMS_ORDER = (2,4,6): even dims → K; odd dims → I
        @test ReSolverFlowsBase.combine_indices(MockGrid{6, nothing, (2, 4, 6)}(), CI(10, 30, 50), CI(20, 40, 60)) == (10, 20, 30, 40, 50, 60)
    end
end
