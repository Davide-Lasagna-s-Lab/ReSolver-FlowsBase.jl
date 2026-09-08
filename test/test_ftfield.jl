# Tests for src/ftfield.jl.
#
# Contract:
#   - Construction enforces the real-field Fourier invariants at the mean and
#     Hermitian-symmetric storage locations.
#   - `u[k::WaveNumberVector, I...]` reads signed wavenumbers in FFT order.
#   - Setting coefficients through a `WaveNumberVector` preserves the same
#     Hermitian invariants.

@testset "FTField contract                                                    " begin

    @testset "constructor normalises mean and Hermitian mirror pairs            " begin
        Ny, Nx, Nz = 3, 8, 5
        g = TripleGrid(Ny, Nx, Nz)

        raw = randn(ComplexF64, Ny, (Nx >> 1) + 1, Nz)
        u = FTField(g, copy(raw))

        # The full zero-wavenumber line is real (normalise_mean! invariant).
        @test all(iszero, imag.(parent(u)[:, 1, 1]))

        # On the rfft zero plane, signed z modes are conjugate pairs
        # (apply_symmetry! invariant).
        for j in 1:Ny, kz in 1:(Nz >> 1)
            @test parent(u)[j, 1, kz + 1] ≈ conj(parent(u)[j, 1, Nz - kz + 1])
        end
    end

    @testset "apply_symmetry! only modifies the DC (rfft=0) plane            " begin
        Ny, Nx, Nz = 3, 8, 5
        g   = TripleGrid(Ny, Nx, Nz)
        raw = randn(ComplexF64, Ny, (Nx >> 1) + 1, Nz)
        u   = FTField(g, copy(raw))  # constructor already called apply_symmetry!

        # Scramble the data, then call apply_symmetry! explicitly.
        parent(u) .= randn(ComplexF64, size(parent(u)))
        ReSolverFlowsBase.apply_symmetry!(u)

        # DC plane (rfft index 1): signed z modes are conjugate pairs.
        for j in 1:Ny, kz in 1:(Nz >> 1)
            @test parent(u)[j, 1, kz + 1] ≈ conj(parent(u)[j, 1, Nz - kz + 1])
        end

        # Non-DC plane (rfft index > 1): entries must be untouched.
        # Re-run with a copy to compare.
        raw2 = randn(ComplexF64, Ny, (Nx >> 1) + 1, Nz)
        u2   = FTField(g, copy(raw2))
        parent(u2) .= copy(raw2)     # bypass constructor, load un-symmetrized data
        snapshot = copy(parent(u2))
        ReSolverFlowsBase.apply_symmetry!(u2)
        # All non-DC-plane entries are unchanged.
        for kx in 2:(Nx >> 1) + 1
            @test parent(u2)[:, kx, :] == snapshot[:, kx, :]
        end
    end

    @testset "apply_symmetry! is idempotent                                  " begin
        Ny, Nx, Nz = 3, 8, 5
        g = TripleGrid(Ny, Nx, Nz)
        u = FTField(g)
        parent(u) .= randn(ComplexF64, size(parent(u)))

        ReSolverFlowsBase.apply_symmetry!(u)
        after_first = copy(parent(u))
        ReSolverFlowsBase.apply_symmetry!(u)

        @test parent(u) == after_first
    end

    @testset "growto copies source coefficients to the same wavenumbers" begin
        Ny, Nx, Nz = 3, 8, 5
        g  = TripleGrid(Ny, Nx, Nz)
        u  = FTField(g)

        # Deterministic coefficients for the whole stored source spectrum.  On
        # the rfft-zero plane this formula respects Hermitian symmetry:
        # coeff(0, -kz, j) == conj(coeff(0, kz, j)).
        coeff(k, j) = complex(100j + 10k[1] + abs(k[2]),
                              k[1] == 0 ? 3k[2] : 7k[1] + 3k[2])

        # Populate every stored source wavenumber and every inhomogeneous point.
        for Ih in CartesianIndices(ReSolverFlowsBase.homogeneous_axes(u)), j in 1:Ny
            k = ReSolverFlowsBase.to_wavenumber_vector(g, Ih)
            u[k, j] = coeff(k, j)
        end

        # Grow to a larger grid: (Nx, Nz) → (16, 11)
        v = ReSolverFlowsBase.growto(u, (16, 11))

        @test size(ReSolverFlowsBase.grid(v)) == (Ny, 16, 11)
        @test size(parent(v)) == (Ny, 9, 11)   # rfft: (16>>1)+1=9, Nz stays 11

        # Exactness: every source wavenumber is copied to the same signed
        # wavenumber in the target, including the negative signed-FFT block.
        for Ih in CartesianIndices(ReSolverFlowsBase.homogeneous_axes(u)), j in 1:Ny
            k = ReSolverFlowsBase.to_wavenumber_vector(g, Ih)
            @test u[k, j] == coeff(k, j)
            @test v[k, j] == u[k, j]
        end

        # Every target wavenumber outside the source resolution stays exactly
        # zero.  This also catches the old bug where source k=(1,-1) was copied
        # to target k=(1,4) by reusing storage indices instead of wavenumbers.
        for Ih in CartesianIndices(ReSolverFlowsBase.homogeneous_axes(v)), j in 1:Ny
            k = ReSolverFlowsBase.to_wavenumber_vector(ReSolverFlowsBase.grid(v), Ih)
            in_source = 0 <= k[1] <= (Nx >> 1) && abs(k[2]) <= (Nz >> 1)
            @test in_source ? v[k, j] == u[k, j] : iszero(v[k, j])
        end
    end

    @testset "homogeneous_axes and inhomogeneous_axes                  " begin
        Ny, Nx, Nz = 3, 8, 5
        g  = TripleGrid(Ny, Nx, Nz)          # FFT_DIMS_ORDER = (2, 3)
        u  = FTField(g)                       # parent shape: (Ny, (Nx>>1)+1, Nz)
        pu = parent(u)

        # FTField dispatch (extracts Val from the grid).
        @test ReSolverFlowsBase.homogeneous_axes(u)   == (axes(pu, 2), axes(pu, 3))
        @test ReSolverFlowsBase.inhomogeneous_axes(u) == (axes(pu, 1),)

        # AbstractArray + Val dispatch: same result from the raw parent array.
        vFFT = Val((2, 3))
        @test ReSolverFlowsBase.homogeneous_axes(pu, vFFT)   == (axes(pu, 2), axes(pu, 3))
        @test ReSolverFlowsBase.inhomogeneous_axes(pu, vFFT) == (axes(pu, 1),)

        # Sanity: CartesianIndices over homogeneous axes covers the full kH
        # storage (rfft half-spectrum × signed-FFT dimension).
        @test length(CartesianIndices(ReSolverFlowsBase.homogeneous_axes(u))) ==
              ((Nx >> 1) + 1) * Nz
    end

    @testset "WaveNumberVector getindex and setindex preserve symmetry" begin
        Ny, Nx, Nz = 2, 8, 5
        g = TripleGrid(Ny, Nx, Nz)
        u = FTField(g)

        # Writing at kx=0 must also write the signed-FFT mirror coefficient.
        u[WaveNumberVector(0, 1), 2] = 3 + 4im
        @test u[WaveNumberVector(0,  1), 2] == 3 + 4im
        @test u[WaveNumberVector(0, -1), 2] == 3 - 4im

        # The fully-zero mode is forced to be real.
        u[WaveNumberVector(0, 0), 1] = 2 + 9im
        @test u[WaveNumberVector(0, 0), 1] == 2 + 0im

        # Negative rfft wavenumbers are represented through conjugate storage.
        u[WaveNumberVector(-2, 1), 1] = 5 - 7im
        @test u[WaveNumberVector(-2,  1), 1] == 5 - 7im
        @test u[WaveNumberVector( 2, -1), 1] == 5 + 7im
    end
end
