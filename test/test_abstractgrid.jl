# Tests for the generic
# `AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}` interface.
#
# These tests pin the contract documented in `src/abstractgrid.jl`:
#
#   - `size(grid)` and `size(grid, dim)` agree.
#   - `fft_storage_dims(grid)` returns the `FFT_DIMS_ORDER` type parameter verbatim.
#   - `spatial_fft_storage_dims(grid)` returns transformed spatial dimensions,
#     omitting a transformed logical time dimension.
#   - `inhomogeneous_storage_dims(grid)` returns the complement of
#     `fft_storage_dims` within `1:D`, in ascending order.
#   - `spatial_inhomogeneous_storage_dims(grid)` omits the logical time axis.
#   - `transform_size(grid)` halves the rfft axis to `(N÷2)+1` and leaves all
#     other axes unchanged.
#   - `fft_norm(grid)` is the product of the FFT-dim sizes, used to normalise
#     forward transforms.
#   - `to_storage_order(values, grid)` permutes the four logical-coordinate
#     slots `(X,Y,Z,T)` into storage order using `AXES`.
#   - Required-interface fall-throughs (`points`, `weights`,
#     `wavenumber_scale`, `growto`, `convert`) throw `NotImplementedError`.

@testset "AbstractGrid interface                                              " begin

    # A minimal struct with no concrete implementations.  Used to verify the
    # fall-through methods throw `NotImplementedError`.
    struct BareGrid <: AbstractGrid{Float64, 3, (2, 1, 3, nothing), (2, 3)} end
    Base.size(::BareGrid) = (4, 8, 6)


    @testset "type-parameter accessors              " begin
        g = BareGrid()

        # fft_storage_dims is a thin generated alias for the `FFT_DIMS_ORDER` parameter.
        @test fft_storage_dims(g) === (2, 3)

        # A steady grid has no logical time coordinate, so every transformed
        # direction is spatial.
        @test spatial_fft_storage_dims(g) === (2, 3)

        # If logical time is transformed, spatial_fft_storage_dims omits that array
        # dimension while preserving the remaining transform order.
        struct TimeSpectralGrid <: AbstractGrid{Float64, 4, (2, 1, 3, 4), (2, 3, 4)} end
        Base.size(::TimeSpectralGrid) = (4, 8, 6, 10)
        @test spatial_fft_storage_dims(TimeSpectralGrid()) === (2, 3)

        # inhomogeneous_storage_dims is the complement of fft_storage_dims within 1:D.
        # For our BareGrid with D=3, ORDER=(2,3), only dim 1 remains.
        @test inhomogeneous_storage_dims(g) === (1,)
        @test spatial_inhomogeneous_storage_dims(g) === (1,)

        # Logical time can be inhomogeneous. It is omitted from the spatial
        # subset used by finite-difference or collocation operators.
        struct SpaceTimeInhomogeneousGrid <: AbstractGrid{Float64, 4, (1, 2, 3, 4), (1, 2)} end
        @test inhomogeneous_storage_dims(SpaceTimeInhomogeneousGrid()) === (3, 4)
        @test spatial_inhomogeneous_storage_dims(SpaceTimeInhomogeneousGrid()) === (3,)

        # size(g, dim) must index size(g) — verify on every dimension.
        for d in 1:3
            @test size(g, d) === size(g)[d]
        end
    end

    @testset "transform_size: rfft halves only ORDER[1]                         " begin
        # Grid: size (Ny=4, Nx=8, Nz=6); rfft on dim 2, signed FFT on dim 3.
        # Expected transform_size:
        #   dim 1 (inhomogeneous)  : unchanged → 4
        #   dim 2 (rfft, ORDER[1]) : halved   → (8 >> 1) + 1 = 5
        #   dim 3 (signed FFT)     : unchanged → 6
        g = BareGrid()
        @test ReSolverFlowsBase.transform_size(g) === (4, 5, 6)
    end

    @testset "fft_norm: product over FFT dims" begin
        # fft_norm returns the *physical* sizes of every transformed dimension
        # in `FFT_DIMS_ORDER` order — used to scale forward transforms.
        # For BareGrid with ORDER=(2,3): (size(g,2), size(g,3)) = (8, 6).
        g = BareGrid()
        @test ReSolverFlowsBase.fft_norm(g) === (8, 6)
    end

    @testset "to_storage_order: AXES storage order" begin
        # Build a grid whose AXES = (2, 1, 3, 4) — i.e. coordinate 1 lives at
        # array dim 2, coordinate 2 at array dim 1, coordinate 3 at array dim 3,
        # coordinate 4 at array dim 4.  Then `to_storage_order((A,B,C,D), g)`
        # should return `(B, A, C, D)`.
        struct PermGrid <: AbstractGrid{Float64, 4, (2, 1, 3, 4), (2, 3, 4)} end
        Base.size(::PermGrid) = (4, 8, 6, 10)

        @test ReSolverFlowsBase.to_storage_order((:A, :B, :C, :D), PermGrid()) ===
              (:B, :A, :C, :D)

        # A grid without time drops the fourth logical coordinate while keeping
        # the stored dimensions in array order.
        @test ReSolverFlowsBase.to_storage_order((:X, :Y, :Z, :T), BareGrid()) ===
              (:Y, :X, :Z)

        # Missing coordinates may appear before the end of AXES.  This grid has
        # no z coordinate, but does have time stored in array dim 3, so the
        # storage-order tuple is (y, x, t).
        struct SpaceTimeGrid <: AbstractGrid{Float64, 3, (2, 1, nothing, 3), (2, 3)} end
        @test ReSolverFlowsBase.to_storage_order((:X, :Y, :Z, :T), SpaceTimeGrid()) ===
              (:Y, :X, :T)

        # Argument count must match the four Cartesian coordinate slots.
        @test_throws MethodError ReSolverFlowsBase.to_storage_order((:A, :B), PermGrid())
    end

    @testset "index utility functions                " begin
        # BareGrid: D=3, FFT_DIMS_ORDER=(2,3), inhomogeneous dim 1.
        g = BareGrid()
        I = CartesianIndex(2, 3, 4)

        # rfft_storage_dim: first entry of FFT_DIMS_ORDER.
        @test ReSolverFlowsBase.rfft_storage_dim(g) === 2

        # one_or_two: 1 when the rfft-dim slot equals 1, 2 otherwise.
        # Both the grid form and the Val form must agree.
        @test ReSolverFlowsBase.one_or_two(CartesianIndex(2, 1, 4), g) === 1
        @test ReSolverFlowsBase.one_or_two(CartesianIndex(2, 3, 4), g) === 2
        @test ReSolverFlowsBase.one_or_two(CartesianIndex(2, 1, 4), Val((2, 3))) === 1
        @test ReSolverFlowsBase.one_or_two(CartesianIndex(2, 3, 4), Val((2, 3))) === 2

        # inhomogeneous_indices: dims NOT in FFT_DIMS_ORDER — only dim 1 for BareGrid.
        @test ReSolverFlowsBase.inhomogeneous_indices(I, g) === (2,)
        @test ReSolverFlowsBase.inhomogeneous_indices(I, Val((2, 3))) === (2,)

        # homogeneous_indices: dims IN FFT_DIMS_ORDER — dims 2 and 3.
        @test ReSolverFlowsBase.homogeneous_indices(I, g) === (3, 4)
    end

    @testset "required interface fallthroughs throw" begin
        # Methods that have no default implementation must throw
        # `NotImplementedError` when called on a grid that doesn't override
        # them — this is the explicit contract from `abstractgrid.jl`.
        g = BareGrid()

        @test_throws ReSolverFlowsBase.NotImplementedError points(g)
        @test_throws ReSolverFlowsBase.NotImplementedError weights(g)
        @test_throws ReSolverFlowsBase.NotImplementedError ReSolverFlowsBase.wavenumber_scale(g, 1)
        @test_throws ReSolverFlowsBase.NotImplementedError growto(g, (16, 16, 12))

        # `convert` to a different scalar type has no default either.
        @test_throws ReSolverFlowsBase.NotImplementedError convert(Float32, g)
    end

    @testset "FakeGrid satisfies the interface" begin
        # Round-trip every interface method on the FakeGrid defined in fake.jl
        # to ensure a *correctly implemented* downstream grid wires up cleanly.
        # FakeGrid is 2-D with AXES=(1,2,nothing,nothing), ORDER=(2,) — i.e.
        # array dim 1 inhomogeneous (x), array dim 2 rfft (y).
        Nx, Ny = 8, 12
        L = 2π
        g = FakeGrid(rand(Float64, Nx), Ny, L)

        @test fft_storage_dims(g) === (2,)
        @test inhomogeneous_storage_dims(g) === (1,)
        @test size(g) === (Nx, Ny)
        @test size(g, 1) === Nx
        @test size(g, 2) === Ny
        @test ReSolverFlowsBase.transform_size(g) === (Nx, (Ny >> 1) + 1)
        @test ReSolverFlowsBase.fft_norm(g) === (Ny,)

        # weights returns a Vector of length equal to the inhomogeneous size.
        @test length(weights(g)) == Nx

        # wavenumber_scale on the FFT direction must agree with 2π/L.
        @test ReSolverFlowsBase.wavenumber_scale(g, 2) ≈ 2π / L

        # points returns a tuple of broadcastable arrays sized for storage:
        x, y = points(g; dealias=false)
        @test size(x) == (Nx, 1)
        @test size(y) == (1, Ny)
    end
end
