# Allocation contracts, organised to mirror `src/`.
#
# The rule of thumb in this file is:
#   - functions ending in `!`, cheap accessors, and hot scalar/loop utilities
#     should allocate zero bytes after warmup;
#   - constructors and explicitly allocating APIs (`copy`, `zero`, `FFT`,
#     `project`, `save_*`, ...) are smoke-tested as allocating operations.
#
# All measurements are taken inside helper functions. Top-level `@allocated`
# can measure allocations from global captures in the test itself, which is not
# the package behaviour we want to pin down.

function allocs_after_warmup(f)
    f()
    f()
    return @allocated f()
end

function alloc_fixture()
    g = TripleGrid(3, 8, 6; α=1.5, β=2.0)
    u = FTField(g)
    v = FTField(g)
    parent(u) .= randn(ComplexF64, size(parent(u)))
    parent(v) .= randn(ComplexF64, size(parent(v)))

    q = VectorField(u, copy(u), zero(u))
    p = copy(q)

    Nm = 4
    modes = ntuple(_ -> randn(ComplexF64, Nm, 3, (8 >> 1) + 1, 6), 3)
    a = ProjectedField(g, randn(ComplexF64, Nm, (8 >> 1) + 1, 6), modes)
    b = copy(a)

    return (; g, u, v, q, p, modes, a, b)
end

function alloc_polynomial_fixture()
    g = PolynomialGrid([-1.0, -0.3, 0.2, 0.7, 1.0], 8)
    u = FTField(g)
    v = FTField(g)
    parent(u) .= randn(ComplexF64, size(parent(u)))
    parent(v) .= randn(ComplexF64, size(parent(v)))
    q = VectorField(u, copy(u))
    p = copy(q)
    return (; g, u, v, q, p)
end

function alloc_projected_2d_fixture()
    g = PolynomialGrid([-1.0, -0.3, 0.2, 0.7, 1.0], 8)
    Nm = 3
    modes = ntuple(_ -> randn(ComplexF64, Nm, length(g.y), (g.Nx >> 1) + 1), 2)
    a = ProjectedField(g, randn(ComplexF64, Nm, (g.Nx >> 1) + 1), modes)
    b = copy(a)
    q = VectorField(FTField(g), FTField(g))
    parent(q[1]) .= randn(ComplexF64, size(parent(q[1])))
    parent(q[2]) .= randn(ComplexF64, size(parent(q[2])))
    return (; g, q, modes, a, b)
end

function alloc_plans_fixture(; dealias=false)
    g = PolynomialGrid([-1.0, -0.3, 0.2, 0.7, 1.0], 8)
    plans = FFTPlans(g; dealias=dealias, flags=FFTW.ESTIMATE)
    u = Field(g, (y, x) -> sin(x) + y)
    u_dealiased = Field(g, (y, x) -> sin(x) + y; dealias=dealias)
    uhat = FTField(g)
    parent(uhat) .= randn(ComplexF64, size(parent(uhat)))
    return (; g, plans, u, u_dealiased, uhat)
end

function alloc_noop_force!(out, _, _)
    return out
end

@testset "Allocation contracts                                                " begin

    @testset "src/ReSolverFlowsBase.jl" begin
        @test isdefined(ReSolverFlowsBase, :AbstractGrid)
        @test isdefined(ReSolverFlowsBase, :FTField)
    end

    @testset "src/notimplementederror.jl" begin
        g = SpectralTestGrid{(4,), 1, (1, nothing, nothing, nothing), (1,)}()
        err = try
            ReSolverFlowsBase.points(g)
        catch e
            e
        end

        @test err isa ReSolverFlowsBase.NotImplementedError
        @test allocs_after_warmup(() -> sprint(showerror, err)) > 0
    end

    @testset "src/abstractgrid.jl" begin
        (; g) = alloc_fixture()
        values = (:x1, :x2, :x3, :t)

        @test allocs_after_warmup(() -> ReSolverFlowsBase.fft_storage_dims(g)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.inhomogeneous_storage_dims(g)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.to_storage_order(values, g)) == 0
        @test allocs_after_warmup(() -> size(g)) == 0
        @test allocs_after_warmup(() -> size(g, 2)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.transform_size(g)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.fft_norm(g)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.weights(g)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.wavenumber_scale(g, 2)) == 0
        @test allocs_after_warmup(() -> convert(Float64, g)) == 0

        # Coordinate arrays are newly constructed by this fixture.
        @test allocs_after_warmup(() -> ReSolverFlowsBase.points(g)) > 0
        grown = ReSolverFlowsBase.growto(g, (16, 12))
        @test size(grown) == (size(g, 1), 16, 12)
        @test allocs_after_warmup(() -> ReSolverFlowsBase.growto(g, (16, 12))) == 0
    end

    @testset "src/wavenumbervector.jl" begin
        (; g) = alloc_fixture()
        k = WaveNumberVector(1, -2)
        storage_indices = (2, 6)

        @test allocs_after_warmup(() -> WaveNumberVector(1, -2)) == 0
        @test allocs_after_warmup(() -> length(k)) == 0
        @test allocs_after_warmup(() -> k[2]) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase._fftw_index(-1, 6)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase._fftw_sym_index(3, 6)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.to_homogeneous_indices(g, k)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.to_wavenumber_vector(g, storage_indices)) == 0
    end

    @testset "src/ftfield.jl" begin
        (; g, u) = alloc_fixture()
        k = WaveNumberVector(1, -1)

        @test allocs_after_warmup(() -> FTField(g)) > 0
        @test allocs_after_warmup(() -> FTField(g, randn(ComplexF64, size(parent(u))))) > 0
        @test allocs_after_warmup(() -> parent(u)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.grid(u)) == 0
        @test allocs_after_warmup(() -> size(u)) == 0
        @test allocs_after_warmup(() -> eltype(u)) == 0
        @test allocs_after_warmup(() -> u[1]) == 0
        @test allocs_after_warmup(() -> (u[1] = 1 + 0im)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.combine_indices(g, CartesianIndex(2), CartesianIndex(3, 4))) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase._average_complex(1 + 2im, 3 + 4im)) == 0
        @test allocs_after_warmup(() -> (u[k, 2] = 2 - 1im)) == 0
        @test allocs_after_warmup(() -> u[k, 2]) == 0

        data = randn(ComplexF64, size(parent(u)))
        @test allocs_after_warmup(() -> ReSolverFlowsBase.apply_symmetry!(data, Val(ReSolverFlowsBase.fft_storage_dims(g)))) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.normalise_mean!(data, Val(ReSolverFlowsBase.fft_storage_dims(g)))) == 0

        @test allocs_after_warmup(() -> similar(u)) > 0
        @test allocs_after_warmup(() -> copy(u)) > 0
        @test allocs_after_warmup(() -> zero(u)) > 0
        @test allocs_after_warmup(() -> u[k]) == 0
    end

    @testset "src/field.jl" begin
        (; g) = alloc_polynomial_fixture()
        u = Field(g, (y, x) -> y + sin(x))

        @test allocs_after_warmup(() -> Field(g)) > 0
        @test allocs_after_warmup(() -> Field(g, (y, x) -> y + sin(x))) > 0
        @test allocs_after_warmup(() -> Field(g, parent(u))) == 0
        @test allocs_after_warmup(() -> parent(u)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.grid(u)) == 0
        @test allocs_after_warmup(() -> size(u)) == 0
        @test allocs_after_warmup(() -> eltype(u)) == 0
        @test allocs_after_warmup(() -> u[1]) == 0
        @test allocs_after_warmup(() -> (u[1] = 0.5)) == 0
        @test allocs_after_warmup(() -> similar(u)) > 0
        @test allocs_after_warmup(() -> copy(u)) > 0
        @test allocs_after_warmup(() -> zero(u)) > 0
    end

    @testset "src/vectorfield.jl" begin
        (; g, q) = alloc_polynomial_fixture()
        base = (range(-1.0, 1.0, length=size(g, 1)) |> collect, nothing)

        @test allocs_after_warmup(() -> VectorField(g, FTField; N=2)) > 0
        @test allocs_after_warmup(() -> VectorField(g, (y, x) -> y, (y, x) -> sin(x))) > 0
        @test allocs_after_warmup(() -> parent(q)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.grid(q)) == 0
        @test allocs_after_warmup(() -> q[1]) == 0
        @test allocs_after_warmup(() -> size(q)) == 0
        @test allocs_after_warmup(() -> eltype(q)) == 0
        @test allocs_after_warmup(() -> (q[1] = q[2])) == 0
        @test allocs_after_warmup(() -> add_base_flow!(q, base)) == 0
        @test allocs_after_warmup(() -> similar(q)) > 0
        @test allocs_after_warmup(() -> copy(q)) > 0
        @test allocs_after_warmup(() -> zero(q)) > 0
        @test_throws ReSolverFlowsBase.NotImplementedError ReSolverFlowsBase.growto(q, (16,))
    end

    @testset "src/fft.jl" begin
        (; plans, u, u_dealiased, uhat) = alloc_plans_fixture(dealias=false)
        dealiased = alloc_plans_fixture(dealias=true)
        dealias_plans = dealiased.plans
        physical_dealiased = dealiased.u_dealiased
        uhat_dealiased = dealiased.uhat
        cache = similar(dealias_plans.cache)
        compact = similar(parent(uhat))

        @test allocs_after_warmup(() -> FFTPlans(size(u), ReSolverFlowsBase.fft_storage_dims(ReSolverFlowsBase.grid(u)), Float64; dealias=false, flags=FFTW.ESTIMATE)) > 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.get_padded_size((5, 8), (2,))) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase._get_transform_size((5, 8), 2)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase._loopblk!(compact, axes(compact), compact, axes(compact), Val(false))) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase._apply_mask!(cache)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase._copy_to_padded!(cache, compact, (2,))) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase._copy_from_padded!(compact, cache, (2,))) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase._add_from_padded!(compact, cache, (2,))) == 0
        @test allocs_after_warmup(() -> plans(uhat, u; add=false, use_cache=false)) == 0
        @test allocs_after_warmup(() -> plans(parent(uhat), parent(u), false, false)) == 0
        @test allocs_after_warmup(() -> dealias_plans(uhat_dealiased, physical_dealiased; add=false)) == 0
        @test allocs_after_warmup(() -> plans(u, uhat; preserve_input=false, use_cache=false)) == 0
        @test allocs_after_warmup(() -> plans(parent(u), parent(uhat), false, false)) == 0
        @test allocs_after_warmup(() -> FFT(u)) > 0
        @test allocs_after_warmup(() -> IFFT(uhat)) > 0
    end

    @testset "src/projectedfield.jl" begin
        (; g, modes, a) = alloc_fixture()
        k = WaveNumberVector(1, -1)

        @test allocs_after_warmup(() -> ProjectedField(g, modes)) > 0
        @test allocs_after_warmup(() -> ProjectedField(g, parent(a), modes)) > 0
        @test allocs_after_warmup(() -> ProjectedField(g, modes[1])) > 0
        @test allocs_after_warmup(() -> ProjectedField(FTField(g), modes)) > 0
        @test allocs_after_warmup(() -> parent(a)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.grid(a)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.modes(a)) == 0
        @test allocs_after_warmup(() -> size(a)) == 0
        @test allocs_after_warmup(() -> eltype(a)) == 0
        @test allocs_after_warmup(() -> a[1]) == 0
        @test allocs_after_warmup(() -> (a[1] = 0.5 + 0im)) == 0
        @test allocs_after_warmup(() -> a[1, 2, 3]) == 0
        @test allocs_after_warmup(() -> (a[1, 2, 3] = 0.25 - 0.5im)) == 0
        @test allocs_after_warmup(() -> a[1, k]) == 0
        @test allocs_after_warmup(() -> (a[1, k] = 1 + 2im)) == 0
        @test allocs_after_warmup(() -> similar(a)) > 0
        @test allocs_after_warmup(() -> copy(a)) > 0
        @test allocs_after_warmup(() -> zero(a)) > 0
        @test allocs_after_warmup(() -> abs(a)) > 0
    end

    @testset "src/galerkin.jl" begin
        (; q, modes, a) = alloc_projected_2d_fixture()
        out = zero(q)

        @test allocs_after_warmup(() -> LoopGalerkin()) == 0
        @test allocs_after_warmup(() -> GemmGalerkin()) == 0
        @test allocs_after_warmup(() -> project!(a, q, LoopGalerkin())) == 0
        @test allocs_after_warmup(() -> expand!(out, a, LoopGalerkin())) == 0
        @test allocs_after_warmup(() -> project(q, modes, LoopGalerkin())) > 0
        a3 = alloc_fixture().a
        @test allocs_after_warmup(() -> expand(a3, LoopGalerkin())) > 0
        @test allocs_after_warmup(() -> project!(a, q, GemmGalerkin())) >= 0
        @test allocs_after_warmup(() -> expand!(out, a, GemmGalerkin())) >= 0
    end

    @testset "src/shifts.jl" begin
        (; u, q, a) = alloc_fixture()

        @test allocs_after_warmup(() -> shift!(u, (0.13, -0.21))) == 0
        @test allocs_after_warmup(() -> shift!(q, (0.13, -0.21))) == 0
        @test allocs_after_warmup(() -> shift!(a, (0.13, -0.21))) == 0
        @test allocs_after_warmup(() -> shift(u, (0.13, -0.21))) > 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase._shift_phase(ReSolverFlowsBase.grid(u), (0.13, -0.21), WaveNumberVector(1, -1))) == 0
    end

    @testset "src/norms.jl" begin
        (; u, v, q, p, a, b) = alloc_fixture()
        tmp_ft = zero(v)
        tmp_vec1 = zero(p)
        tmp_a = zero(b)

        # These routines use one Ref accumulator internally.  BenchmarkTools
        # sees this as one 16-byte allocation; through `allocs_after_warmup`
        # the closure wrapper reports 32 bytes.
        small_accumulator_alloc = 32
        @test allocs_after_warmup(() -> dot(u, v)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> norm(u)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> normdiff(u, v)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> normdiff(u, v, (0.13, -0.21), tmp_ft)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> dot(q, p)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> norm(q)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> normdiff(q, p)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> normdiff(q, p, (0.13, -0.21), tmp_ft)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> minnormdiff(q, p, (2, 2), tmp_vec1)) <= 48
        @test allocs_after_warmup(() -> dot(a, b)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> norm(a)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> normdiff(a, b)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> normdiff(a, b, (0.13, -0.21), tmp_a)) <= small_accumulator_alloc
    end

    @testset "src/weighting.jl" begin
        (; g, a, b) = alloc_fixture()
        A = FarazmandWeight(g)
        k = WaveNumberVector(1, -1)

        @test allocs_after_warmup(() -> FarazmandWeight(g)) == 0
        @test allocs_after_warmup(() -> FarazmandWeight(1.0, 2.0)) == 0
        @test allocs_after_warmup(() -> A[k]) == 0
        @test allocs_after_warmup(() -> lmul!(A, a)) == 0
        @test allocs_after_warmup(() -> dot(a, A, b)) <= 32
    end

    @testset "src/broadcasting.jl" begin
        (; u, v, q, p) = alloc_fixture()
        out = zero(u)
        qout = zero(q)

        @test allocs_after_warmup(() -> (out .= 2 .* u .- v)) == 0
        @test allocs_after_warmup(() -> (qout .= 2 .* q .- p)) == 0
        @test allocs_after_warmup(() -> (qout .= 0)) == 0
        @test allocs_after_warmup(() -> u .+ v) > 0
        @test allocs_after_warmup(() -> q .+ p) > 0
    end

    @testset "src/derivatives.jl" begin
        (; u, q) = alloc_polynomial_fixture()
        out = zero(u)
        qout = zero(q)

        @test allocs_after_warmup(() -> ddx1!(out, u)) == 0
        # ddx2!/dd! on dim 1 dispatches to PolynomialGrid's LinearAlgebra.mul! extension.
        # On Julia < 1.11 mul! allocates when --check-bounds=yes is active; skip there.
        if VERSION >= v"1.11"
            @test allocs_after_warmup(() -> ddx2!(out, u)) == 0
        end
        @test allocs_after_warmup(() -> ddx3!(out, u)) == 0
        @test allocs_after_warmup(() -> ddt!(out, u)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.dd!(out, u, Val(2))) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase.dd!(qout, q, Val(2))) == 0
        # _inhomogeneous_laplacian! and laplacian! also go through mul! on PolynomialGrid.
        if VERSION >= v"1.11"
            @test allocs_after_warmup(() -> ReSolverFlowsBase._inhomogeneous_laplacian!(out, u)) == 0
        end
        @test allocs_after_warmup(() -> ReSolverFlowsBase._add_homogeneous_laplacian!(out, u)) == 0
        @test allocs_after_warmup(() -> ReSolverFlowsBase._add_homogeneous_laplacian!(qout, q)) == 0
        if VERSION >= v"1.11"
            @test allocs_after_warmup(() -> ReSolverFlowsBase.laplacian!(out, u)) == 0
            @test allocs_after_warmup(() -> ReSolverFlowsBase.laplacian!(qout, q)) == 0
        end
    end

    @testset "src/io.jl" begin
        (; g, a, modes) = alloc_fixture()
        mktempdir() do dir
            grid_path = joinpath(dir, "grid.jld2")
            field_path = joinpath(dir, "field.jld2")
            @test allocs_after_warmup(() -> save_grid(g; path=grid_path)) > 0
            @test allocs_after_warmup(() -> load_grid(grid_path)) > 0
            @test allocs_after_warmup(() -> save_field(a; path=field_path)) > 0
            @test allocs_after_warmup(() -> load_field(g, modes, field_path)) > 0
        end
    end

end
