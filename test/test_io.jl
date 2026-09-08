# Tests for the JLD2 persistence helpers in `src/io.jl`.
#
# Contract:
#   - `save_grid(g)` writes `g` to disk so that `load_grid(path)` returns an
#     object equal in type and contents to `g`.
#   - `save_field(a)` persists the *coefficient array* of a ProjectedField,
#     not its modes or its grid; `load_field(g, modes, path)` rebuilds the
#     ProjectedField from those three components.

@testset "IO                                                                  " begin

    # Use a unique temp directory so parallel test runs do not race on the
    # same JLD2 file names.
    tmpdir = mktempdir()

    @testset "save_grid / load_grid round-trip" begin
        # The reconstructed grid must have the same type and the same field
        # values as the original.
        Nx, Ny = 8, 12
        g = FakeGrid(collect(range(-1.0, 1.0, length=Nx)), Ny, 3π)

        path = joinpath(tmpdir, "grid.jld2")
        save_grid(g; path=path)
        g2 = load_grid(path)

        @test typeof(g2) === typeof(g)
        @test g2.x  == g.x
        @test g2.N  == g.N
        @test g2.L  == g.L
    end

    @testset "save_field / load_field round-trip                                " begin
        # A ProjectedField's coefficient array must be recovered bit-exact
        # after a save/load cycle.
        Nx, Ny = 8, 12
        g = FakeGrid(collect(range(-1.0, 1.0, length=Nx)), Ny, 2π)

        M = 4
        Ψ = ntuple(_ -> randn(ComplexF64, Nx, M, (Ny>>1)+1), 3)
        a = ProjectedField(g, randn(ComplexF64, M, (Ny>>1)+1), Ψ)

        path = joinpath(tmpdir, "a.jld2")
        save_field(a; path=path)

        b = load_field(g, Ψ, path)

        # `load_field` returns a ProjectedField of the same shape with the
        # original coefficient array.
        @test b isa ProjectedField
        @test parent(b) == parent(a)
        @test modes(b) === Ψ
        @test ReSolverFlowsBase.grid(b)  === g
    end
end
