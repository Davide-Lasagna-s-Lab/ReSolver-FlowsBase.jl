# ProjectedEquation: a thin layer mapping a full-field operator to modal coefficients.
#
# With coefficients `a`, expansion `E`, projection `P` and base flow `u₀`:
#
#     nl(out, a)                P N(u₀ + E a)     operator in Nonlinear mode
#     lin(out, b)               P L E b           Linearised mode
#     adj(out, b)               P L* E b          adjoint modes
#     linearise_about!(lin, a)  sets the point u₀ + E a in the shared workspace
#
# Nothing here depends on the formulation: the call dispatches on the mode of
# the wrapped operator, a type parameter of ProjectedEquation.

"""
    ProjectedEquation(op, base_flow, cache)

Wrap a full-field operator `op` so that it acts on [`ProjectedField`](@ref)
coefficients: `eq(out, a)` is `P N(u₀ + E a)` in `Nonlinear` mode, and `P L E a`
(or `P L* E a`) in the linearised modes, about the point set by
[`linearise_about!`](@ref). `cache` holds two spectral `VectorField`s, used
within one call and shareable between wrappers.
"""
struct ProjectedEquation{MODE, OP, B, V<:VectorField}
           op::OP           # full-field operator
    base_flow::B            # base flow, added to every expanded field
        cache::NTuple{2, V} # expanded field and operator output

    # MODE is taken from the wrapped operator
    ProjectedEquation(       op::OP,
                      base_flow::B,
                          cache::NTuple{2, V}) where {OP, B, V<:VectorField} =
        new{typeof(op.mode), OP, B, V}(op, base_flow, cache)
end


# ---------------------------------------------------------------------------- #
# nonlinear: P N(u₀ + E a)                                                     #
# ---------------------------------------------------------------------------- #
function (eq::ProjectedEquation{Nonlinear})(out::ProjectedField, a::ProjectedField)
    u, N_u = eq.cache

    # ---- total field u₀ + E a ----
    expand!(u, a)
    add_base_flow!(u, eq.base_flow)

    # ---- operator action and projection ----
    eq.op(0, u, N_u)
    project!(out, N_u)

    return out
end


# ---------------------------------------------------------------------------- #
# linearised: P L E b  or  P L* E b                                            #
# ---------------------------------------------------------------------------- #
function (eq::ProjectedEquation{<:AnyLinear})(out::ProjectedField, b::ProjectedField)
    v, L_v = eq.cache

    # ---- perturbation E b, no base flow ----
    expand!(v, b)

    # ---- operator action and projection ----
    eq.op(0, v, L_v)
    project!(out, L_v)

    return out
end


# ---------------------------------------------------------------------------- #
# linearisation point                                                          #
# ---------------------------------------------------------------------------- #
"""
    linearise_about!(eq::ProjectedEquation, a) -> eq

Set the linearisation point to `u₀ + E a`, for every operator sharing the
workspace of `eq.op`.
"""
function linearise_about!(eq::ProjectedEquation, a::ProjectedField)
    u = eq.cache[1]

    expand!(u, a)
    add_base_flow!(u, eq.base_flow)

    linearise_about!(eq.op, u)

    return eq
end
