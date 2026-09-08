# Tests for src/vectorfield.jl.
#
# Contract:
#   - A VectorField is an ordered collection of scalar fields on a common grid.
#   - `similar`, `copy`, and `zero` operate component-wise.
#   - `add_base_flow!` adds each supplied base profile only to the zero
#     homogeneous-wavenumber slice of a spectral vector field.

@testset "VectorField contract                                                " begin

    @testset "array-like wrapper semantics are component-wise" begin
        Nx, Ny = 5, 8
        g = FakeGrid(range(-1, 1, length=Nx) |> collect, Ny, 2π)
        u = VectorField(g, Field, N=2)

        @test length(u) == 2
        @test ReSolverFlowsBase.grid(u) === g
        @test u[1] isa Field
        @test similar(u) isa VectorField{2, <:Field}

        parent(u[1]) .= 1
        parent(u[2]) .= 2
        v = copy(u)
        @test v !== u
        @test parent(v[1]) == parent(u[1])
        @test parent(v[2]) == parent(u[2])

        z = zero(u)
        @test all(iszero, parent(z[1]))
        @test all(iszero, parent(z[2]))
    end

    @testset "add_base_flow! touches only the homogeneous DC slice              " begin
        Ny, Nx, Nz = 4, 8, 5
        g = TripleGrid(Ny, Nx, Nz)
        u = VectorField(g; N=3)

        U = collect(range(-1, 1, length=Ny))
        W = collect(range(2, 3, length=Ny))
        add_base_flow!(u, (U, nothing, W))

        # The zero homogeneous slice is (:, 1, 1) for TripleGrid because
        # fft_storage_dims(g) == (2, 3).  Components with `nothing` are skipped.
        @test parent(u[1])[:, 1, 1] == U
        @test parent(u[2])[:, 1, 1] == zeros(Ny)
        @test parent(u[3])[:, 1, 1] == W

        # Every non-DC coefficient remains untouched.
        @test all(iszero, parent(u[1])[:, 2:end, :])
        @test all(iszero, parent(u[1])[:, :, 2:end])
        @test all(iszero, parent(u[3])[:, 2:end, :])
        @test all(iszero, parent(u[3])[:, :, 2:end])
    end
end
