# Tests for `dot`, `norm`, `normdiff` and `minnormdiff` on FTField,
# VectorField and ProjectedField.
#
# Contract (from src/norms.jl):
#
#   dot(u, v) = Σₖ c_{k₁} · Σⱼ ws[j] · Re(conj(u_kj) · v_kj)
#
#   where c_{k₁} accounts for rfft Hermitian symmetry (1 for k₁ = 0, 2 else).
#   Combined with the 1/V FFT normalization, this gives the domain-averaged
#   inner product: (1/V) ∫ w(j) u(j,z) v(j,z) dj dz.
#
#   - dot is real and symmetric in (u, v).
#   - norm(u) = √dot(u, u) and is non-negative.
#   - normdiff(u, v, shifts) = norm(u - shift(v, shifts)) — pure norm of the
#     difference when shifts is zero, otherwise the shifted-difference norm.
#   - minnormdiff scans a small box of candidate shifts and returns the
#     pair (min_diff, best_shifts) that minimises normdiff.

@testset "Norms                                                               " begin

    @testset "FTField: dot symmetry, positivity, linearity" begin
        # On FakeGrid the parent array is (Nx, (Ny>>1)+1) and weights are
        # ones(Nx), so dot reduces to the simple Hermitian-weighted sum.
        Nx, Ny = 8, 12
        g = FakeGrid(rand(Float64, Nx), Ny, 2π)
        u = FTField(g, randn(ComplexF64, Nx, (Ny>>1)+1))
        v = FTField(g, randn(ComplexF64, Nx, (Ny>>1)+1))

        # Symmetry: dot is real, so dot(u,v) == dot(v,u).
        @test dot(u, v) ≈ dot(v, u)

        # Positivity: dot(u, u) ≥ 0, and = 0 iff u is zero.
        @test dot(u, u) ≥ 0
        @test dot(zero(u), zero(u)) == 0

        # Linearity in the second argument: dot(u, αv + βw) = α dot(u,v) + β dot(u,w)
        # for real α, β.
        w = FTField(g, randn(ComplexF64, Nx, (Ny>>1)+1))
        α, β = 0.3, -1.7
        αv = FTField(g, α .* parent(v))
        βw = FTField(g, β .* parent(w))
        αvβw = FTField(g, α .* parent(v) .+ β .* parent(w))
        # Relative tolerance: dot scales with the (random) data magnitude, so an
        # absolute atol would be flaky across RNG seeds.
        @test dot(u, αvβw) ≈ α*dot(u, v) + β*dot(u, w) rtol=1e-12
    end

    @testset "FTField: norm and norm-difference identities" begin
        # ‖u‖² == dot(u, u);  ‖u - v‖² == ‖u‖² + ‖v‖² − 2·dot(u, v) (real dot).
        Nx, Ny = 8, 12
        g = FakeGrid(rand(Float64, Nx), Ny, 2π)
        u = FTField(g, randn(ComplexF64, Nx, (Ny>>1)+1))
        v = FTField(g, randn(ComplexF64, Nx, (Ny>>1)+1))

        @test norm(u)^2 ≈ dot(u, u)

        # u - v as an FTField (broadcasting).
        d = FTField(g, parent(u) .- parent(v))
        @test norm(d)^2 ≈ dot(u, u) + dot(v, v) - 2*dot(u, v) atol=1e-12

        # normdiff(u, v) with no shifts == ‖u - v‖.
        @test normdiff(u, v) ≈ norm(d) atol=1e-12

        # normdiff(u, u) == 0.
        @test normdiff(u, u) == 0
    end

    @testset "VectorField: dot decomposes component-wise" begin
        # dot(u, v) = Σₙ dot(u[n], v[n]) — the public docstring contract.
        Nx, Ny = 8, 12
        g = FakeGrid(rand(Float64, Nx), Ny, 2π)
        Nc = 3
        u = VectorField([FTField(g, randn(ComplexF64, Nx, (Ny>>1)+1)) for _ in 1:Nc]...)
        v = VectorField([FTField(g, randn(ComplexF64, Nx, (Ny>>1)+1)) for _ in 1:Nc]...)

        @test dot(u, v) ≈ sum(dot(u[n], v[n]) for n in 1:Nc) atol=1e-14
        @test norm(u)^2 ≈ dot(u, u)
        @test normdiff(u, u) == 0

        # Shift tuples must have one entry per homogeneous dimension, even
        # when every requested shift is zero.
        @test_throws DimensionMismatch normdiff(u, v, (0.0, 0.0))
        @test_throws DimensionMismatch normdiff(u, v, (0.0, 0.0), zero(v[1]))
    end

    @testset "normdiff with shifts == norm of explicitly shifted diff" begin
        # `normdiff(u, v, shifts)` is documented as `norm(u - shift(v, shifts))`.
        # Verify by computing both sides on a TripleGrid (multi-dim shifts).
        # Use *odd* signed-FFT sizes so there is no Nyquist mode — a shift
        # breaks the Nyquist-mode self-conjugacy, which `FTField`'s
        # constructor silently re-imposes via `apply_symmetry!`.  Without the
        # Nyquist the two sides agree exactly.
        Ny, Nx, Nz = 3, 9, 5
        g = TripleGrid(Ny, Nx, Nz)
        u = FTField(g, randn(ComplexF64, Ny, (Nx>>1)+1, Nz))
        v = FTField(g, randn(ComplexF64, Ny, (Nx>>1)+1, Nz))

        s = (0.27, -0.41)
        # LHS: built-in normdiff with shifts.
        lhs = normdiff(u, v, s)

        # RHS: explicit shift-then-diff-then-norm.
        v_shifted = shift!(copy(v), s)
        diff = FTField(g, parent(u) .- parent(v_shifted))
        rhs = norm(diff)

        @test lhs ≈ rhs atol=1e-13

        # TripleGrid has two homogeneous dimensions.  Reject short and long
        # shift tuples before any zero-shift fast path can hide the mismatch.
        @test_throws DimensionMismatch normdiff(u, v, (0.0,))
        @test_throws DimensionMismatch normdiff(u, v, (0.0, 0.0, 0.0), zero(v))
    end

    @testset "minnormdiff follows the number of transform dimensions" begin
        # FakeGrid has one transform dimension, so N and the returned shift
        # tuple both have length 1.
        Nx, Ny = 5, 8
        g1 = FakeGrid(rand(Float64, Nx), Ny, 2π)
        u1 = FTField(g1, randn(ComplexF64, Nx, (Ny>>1)+1))
        diff1, shifts1 = minnormdiff(u1, u1, (4,))
        @test diff1 ≈ 0 atol=1e-14
        @test shifts1 == (0.0,)

        # TripleGrid has two transform dimensions, so the candidate grid and
        # returned shift tuple both have length 2.
        Ny2, Nx2, Nz2 = 3, 9, 5
        g2 = TripleGrid(Ny2, Nx2, Nz2)
        u2 = FTField(g2, randn(ComplexF64, Ny2, (Nx2>>1)+1, Nz2))
        diff2, shifts2 = minnormdiff(u2, u2, (4, 5))
        @test diff2 ≈ 0 atol=1e-14
        @test shifts2 == (0.0, 0.0)

        @test_throws DimensionMismatch minnormdiff(u2, u2, (4,))
    end

    @testset "ProjectedField: dot is real and matches the explicit formula      " begin
        # ProjectedField.dot is the spectral inner-product weighted by
        # Hermitian multiplicity c_{k₁} on the rfft axis, summed over (m, k).
        Nx, Ny = 8, 12
        g = FakeGrid(rand(Float64, Nx), Ny, 2π)
        Nm = 4
        Ψ = ntuple(_ -> randn(ComplexF64, Nx, Nm, (Ny>>1)+1), 3)
        a = ProjectedField(g, randn(ComplexF64, Nm, (Ny>>1)+1), Ψ)
        b = ProjectedField(g, randn(ComplexF64, Nm, (Ny>>1)+1), Ψ)

        # Explicit reference: sum over (m, k1) with the rfft Hermitian factor.
        pa, pb = parent(a), parent(b)
        ref = 0.0
        for k1 in 1:(Ny>>1)+1
            c = (k1 == 1) ? 1 : 2
            for m in 1:Nm
                ref += c * real(conj(pa[m, k1]) * pb[m, k1])
            end
        end

        @test dot(a, b) ≈ ref atol=1e-13
        @test norm(a)^2 ≈ dot(a, a)
        @test normdiff(a, a) == 0
        @test_throws DimensionMismatch normdiff(a, b, (0.0, 0.0), zero(b))
    end

    @testset "FTField: dot against HCubature (TripleGrid, generic dot)" begin
        # TripleGrid has no custom dot override — it uses the generic dot from
        # norms.jl.  With flat ws=ones(Ny) and a function independent of y, the
        # formula reduces to:
        #
        #   dot(û, û) = Ny × (1/(Lx·Lz)) × ∫₀^{Lx}∫₀^{Lz} u(x, z)² dx dz
        #
        # We verify this against HCubature for a smooth periodic function.
        Ny, Nx, Nz = 3, 64, 64
        g  = TripleGrid(Ny, Nx, Nz; α=1.0, β=1.0)
        Lx = 2π / g.α
        Lz = 2π / g.β
        û = ReSolverFlowsBase.FFT(ReSolverFlowsBase.Field(g, (y, x, z) -> exp(sin(x)) * cos(z)))
        expected, _ = HCubature.hcubature(xz -> exp(2sin(xz[1])) * cos(xz[2])^2,
                                          [0.0, 0.0], [Lx, Lz])
        @test dot(û, û) ≈ Ny * expected / (Lx * Lz) rtol=1e-5
    end
end
