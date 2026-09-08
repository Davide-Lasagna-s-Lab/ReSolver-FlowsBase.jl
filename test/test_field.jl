# Tests for src/field.jl.
#
# Contract:
#   - `Field(grid, fun)` evaluates `fun` on `points(grid)` in storage order.
#   - `Field(grid)` allocates the zero field on the physical grid.
#   - `copy`, `zero`, and `similar` preserve the field wrapper and grid.

@testset "Field contract                                                      " begin
    Nx, Ny = 5, 7
    g = FakeGrid(range(-1, 1, length=Nx) |> collect, Ny, 2π)

    @testset "function constructor uses grid points in storage order            " begin
        xpts, ypts = points(g)
        fun(x, y) = 1 + 2x - cos(y)
        u = Field(g, fun)

        @test ReSolverFlowsBase.grid(u) === g
        @test size(u) == size(g)
        @test parent(u) ≈ fun.(xpts, ypts)
    end

    @testset "zero, copy, and similar preserve wrapper semantics" begin
        u = Field(g, (x, y) -> x + sin(y))

        @test Field(g) isa Field
        @test all(iszero, parent(Field(g)))

        v = copy(u)
        @test v isa Field
        @test v !== u
        @test ReSolverFlowsBase.grid(v) === g
        @test parent(v) == parent(u)

        z = zero(u)
        @test z isa Field
        @test ReSolverFlowsBase.grid(z) === g
        @test all(iszero, parent(z))

        s = similar(u)
        @test s isa Field
        @test ReSolverFlowsBase.grid(s) === g
        @test size(s) == size(u)
    end
end
