# `construct_equations`: the three projected Navier–Stokes operators on a
# single shared workspace.

"""
    construct_equations(grid, Re, base_flow, geometry=Cartesian(3);
                        form=Convective(), force=NoForce(), mode=AdjointDiscrete(),
                        flags=FFTW.EXHAUSTIVE, dealias=true) -> (nl, lin, adj)

Build the projected nonlinear, linearised and adjoint [`NavierStokes`](@ref)
operators, each a [`ProjectedEquation`](@ref):

```julia
nl, lin, adj = construct_equations(grid, Re, (U, nothing, nothing))

nl(out, a)                # P N(u₀ + E a)
linearise_about!(lin, a)  # point u₀ + E a, shared by lin and adj
lin(out, b)               # P L E b
adj(out, b)               # P L* E b
```

`geometry` is e.g. `Cartesian(3)` or `Cartesian(2)`; `form` the advection form,
[`Convective`](@ref) or [`Rotational`](@ref).
`base_flow` holds one entry per velocity component, either a vector of values
at the inhomogeneous grid points or `nothing`. `mode` selects the adjoint:
[`AdjointDiscrete`](@ref) (exact transpose of the discrete operator) or
[`AdjointContinuous`](@ref). `force` is called as `force(out, u, mode)`.
`dealias=true` uses the 3/2-rule padded grid for products. All three operators
share one [`Workspace`](@ref) and one pair of projection caches.
"""
function construct_equations(grid::AbstractGrid,
                             Re,
                             base_flow,
                             geometry=Cartesian(3);
                             form=Convective(),
                             force=NoForce(),
                             mode=AdjointDiscrete(),
                             flags=FFTW.EXHAUSTIVE,
                             dealias::Bool=true)

    # ---- input checks ----
    mode isa Union{AdjointDiscrete, AdjointContinuous} ||
        throw(ArgumentError("mode must be AdjointDiscrete or AdjointContinuous"))

    # ---- shared workspace and projection caches ----
    N = ncomp(geometry)
    nstate, nspectral, nphysical = _workspace_sizes(geometry, form)

    work  = Workspace(grid, N; nstate, nspectral, nphysical, flags, dealias)
    cache = (VectorField(grid, FTField, N=N), VectorField(grid, FTField, N=N))

    # ---- the three operators ----
    nl  = NavierStokes(Nonlinear(),  geometry, form, work, Re; force)
    lin = NavierStokes(Linearised(), geometry, form, work, Re; force)
    adj = NavierStokes(mode,         geometry, form, work, Re; force)

    # ---- projected wrappers ----
    return (nl  = ProjectedEquation(nl,  base_flow, cache),
            lin = ProjectedEquation(lin, base_flow, cache),
            adj = ProjectedEquation(adj, base_flow, cache))
end
