# Body forces, called as `force(out, u, mode)` and adding their contribution
# to `out`.
#
# `NoForce` is the default body-force callable: it receives `(out, u, mode)` and
# returns `out` unchanged, imposing no additional forcing.
#
# `CompoundForcing` chains multiple body-force callables in sequence.  The call
# is unrolled at compile time via `@nexprs` so that dynamic dispatch is avoided
# even when the individual force types differ.

"""
    NoForce

Default body-force callable that applies no forcing.  Its call signature is
`(out, u, mode) -> out`; it returns `out` unchanged.

Pass `NoForce()` to any NSE constructor that accepts a `force` keyword when no
body force is needed.
"""
struct NoForce end
(::NoForce)(out, _, _) = out


# ------------------------------------- #
# compound force: sequence of forces    #
# ------------------------------------- #
"""
    CompoundForcing(forces...)

A body force that applies each of `forces` in sequence. Use this to combine
multiple body-force terms, e.g.:

```julia
CompoundForcing(ConstantForcing(), CoriolisForce(Ro))
```
"""
struct CompoundForcing{N, F<:NTuple{N, Any}}
    forces::F
end
CompoundForcing(forces...) = CompoundForcing{length(forces), typeof(forces)}(forces)

# A plain `for f in cf.forces` loop iterates via `iterate(::Tuple, ::Int)`, which
# returns a Union of all element types and causes dynamic dispatch for each call.
# N is a type parameter so @nexprs can unroll the calls into N statically-typed
# statements at compile time, keeping dispatch fully specialised.
@generated function (cf::CompoundForcing{N})(out, u, mode) where {N}
    return quote
        Base.Cartesian.@nexprs $N i -> cf.forces[i](out, u, mode)
        return out
    end
end