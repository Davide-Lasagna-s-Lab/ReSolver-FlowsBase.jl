# Tests for the spectral-derivative wrappers `ddx!`, `ddy!`, `ddz!`, `ddt!`,
# `_add_homogeneous_laplacian!`, and `laplacian!`.
#
# Contract (from src/derivatives.jl):
#
#   - `ddx!(out, u)` / `ddy!` / `ddz!` / `ddt!` resolve to `Val{STORAGE_DIM}` and
#     compute the spectral derivative along the named physical direction.
#     For homogeneous (FFT) directions the derivative is multiplication by
#     `i · k · wavenumber_scale`.
#
#   - `dd!` on an absent direction (`:z` on a 2D grid) is a compile-time no-op.
#
#   - `_add_homogeneous_laplacian!(out, u)`  adds `-Σ_{d∈ORDER} (k_d · σ_d)² · u`
#     into `out` (note the `+=`, not `=`).
#
#   - `laplacian!(out, u)` = `_inhomogeneous_laplacian!(out, u)` followed by
#     `_add_homogeneous_laplacian!(out, u)`.  On a fully-homogeneous grid the
#     inhomogeneous part is identity-equivalent if the user defines it that
#     way; FakeGrid does so (see fake.jl).

@testset "Derivatives                                                         " begin

    @testset "analytic field derivatives on a mixed inhomogeneous/FFT grid" begin
        # This is the ReSolverFlowsBase version of the analytic derivative tests that
        # used to live naturally in ChannelFlow.  The grid has one polynomial
        # collocation direction y and one periodic rfft direction x.  The
        # y-derivative is supplied by the test grid extension, while x is the
        # generic ReSolverFlowsBase spectral derivative.  The field is a low-degree
        # polynomial in y times resolved Fourier modes in x, so both reference
        # derivatives are known analytically at the collocation points.
        Ny, Nx = 7, 16
        g = PolynomialGrid(range(-1, 1, length=Ny), Nx, 2π)

        u_fun(y, x)      = (1 + y + y^2 - 0.5y^3) * cos(2x) +
                           (0.25 - y^2) * sin(3x)
        dudx_fun(y, x)   = -2 * (1 + y + y^2 - 0.5y^3) * sin(2x) +
                            3 * (0.25 - y^2) * cos(3x)
        dudy_fun(y, x)   = (1 + 2y - 1.5y^2) * cos(2x) -
                            2y * sin(3x)
        lapl_fun(y, x)   = ((2 - 3y) - 4 * (1 + y + y^2 - 0.5y^3)) * cos(2x) +
                           (-2 - 9 * (0.25 - y^2)) * sin(3x)

        u = FFT(Field(g, u_fun))

        # Physical x is logical coordinate 1 (spectral).  Physical y is logical
        # coordinate 2 (inhomogeneous, dispatches to PolynomialGrid extension).
        @test parent(ReSolverFlowsBase.ddx!(FTField(g), u)) ≈
              parent(FFT(Field(g, dudx_fun))) atol=1e-12
        @test parent(ReSolverFlowsBase.ddy!(FTField(g), u)) ≈
              parent(FFT(Field(g, dudy_fun))) atol=1e-12

        # The Laplacian contract is the sum of the inhomogeneous second
        # derivative and the homogeneous spectral contribution.
        @test parent(ReSolverFlowsBase.laplacian!(FTField(g), u)) ≈
              parent(FFT(Field(g, lapl_fun))) atol=1e-11
    end

    @testset "ddy! is multiplication by i·k·σ on the homogeneous axis" begin
        # FakeGrid is 2-D: AXES = (1, 2, nothing, nothing), ORDER = (2,).
        # Therefore:
        #   ddx! (AXES[1] = 1): inhomogeneous (custom, defined in fake.jl)
        #   ddy! (AXES[2] = 2): rfft direction; multiplies by i·k·σ.
        #   ddz!, ddt!: absent coordinates → compile-time no-ops.
        Nx, Ny = 8, 12
        L = 2π
        g = FakeGrid(rand(Float64, Nx), Ny, L)
        σ = ReSolverFlowsBase.wavenumber_scale(g, 2)            # 2π / L = 1.0 here

        # Build u with random coefficients so we can probe individual
        # wavenumbers after applying ddy!.
        data = randn(ComplexF64, Nx, (Ny >> 1) + 1)
        u = FTField(g, data)
        out = FTField(g)
        ReSolverFlowsBase.ddy!(out, u)

        # Each storage column k (1-based) corresponds to signed wavenumber
        # k-1 on the rfft axis.  Expected: out[:, k] == i · (k-1) · σ · u[:, k].
        pu, po = parent(u), parent(out)
        for k in 1:(Ny >> 1) + 1
            n_signed = k - 1
            expected = (im * n_signed * σ) .* pu[:, k]
            @test po[:, k] ≈ expected atol=1e-13
        end

        # ddz!, ddt! are no-ops on FakeGrid (absent coordinates).
        out2 = FTField(g, copy(data))
        @test parent(ReSolverFlowsBase.ddz!(out2, u)) == data
        out2 .= data
        @test parent(ReSolverFlowsBase.ddt!(out2, u)) == data
    end

    @testset "_add_homogeneous_laplacian! adds −(k σ)² u (not =, +=)" begin
        # The function accumulates into `out`, it does not overwrite.  The
        # added value at wavenumber k is `-(k·σ)² · u[k]` on FakeGrid (only
        # one FFT dim).
        Nx, Ny = 8, 12
        L = 2π
        g = FakeGrid(rand(Float64, Nx), Ny, L)
        σ = ReSolverFlowsBase.wavenumber_scale(g, 2)

        u = FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1))

        # Seed `out` with some initial value to verify that
        # _add_homogeneous_laplacian! accumulates.  Read the seed back from
        # `parent(out)` *after* FTField has applied its Hermitian-symmetry
        # invariant (so the comparison sees the same starting state).
        out  = FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1))
        seed = copy(parent(out))

        ReSolverFlowsBase._add_homogeneous_laplacian!(out, u)

        pu, po = parent(u), parent(out)
        for k in 1:(Ny >> 1) + 1
            n_signed = k - 1
            expected = seed[:, k] .- ((n_signed * σ)^2) .* pu[:, k]
            @test po[:, k] ≈ expected atol=1e-13
        end
    end

    @testset "laplacian! = _inhomogeneous_laplacian! + _add_homogeneous_laplacian!" begin
        # On FakeGrid, `_inhomogeneous_laplacian!(out, u)` is defined in
        # fake.jl as `out .= u`, so `laplacian!(out, u)` = u − (kσ)² u
        # at every wavenumber k.
        Nx, Ny = 8, 12
        L = 2π
        g = FakeGrid(rand(Float64, Nx), Ny, L)
        σ = ReSolverFlowsBase.wavenumber_scale(g, 2)

        u = FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1))
        out = ReSolverFlowsBase.laplacian!(FTField(g), u)

        pu, po = parent(u), parent(out)
        for k in 1:(Ny >> 1) + 1
            n_signed = k - 1
            expected = pu[:, k] .- ((n_signed * σ)^2) .* pu[:, k]
            @test po[:, k] ≈ expected atol=1e-13
        end
    end

    @testset "VectorField wrappers apply component-wise" begin
        # Every derivative has a VectorField method that loops over
        # components and calls the single-field method.  Confirm the loop
        # produces the same result as per-component application.
        Nx, Ny = 8, 12
        g = FakeGrid(rand(Float64, Nx), Ny, 2π)

        u = VectorField([FTField(g, randn(ComplexF64, Nx, (Ny>>1)+1)) for _ in 1:3]...)
        out_vec = VectorField(g; N=3)
        ReSolverFlowsBase.ddy!(out_vec, u)

        for n in 1:3
            expected = FTField(g)
            ReSolverFlowsBase.ddy!(expected, u[n])
            @test parent(out_vec[n]) ≈ parent(expected) atol=1e-14
        end
    end
end
