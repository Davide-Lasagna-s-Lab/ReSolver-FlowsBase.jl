# Workspace shared by the operators of one formulation.
#
# A formulation is an operator type parametrised by its mode,
#
#     F{Nonlinear}           N(u)
#     F{Linearised}          L v     about the linearisation point
#     F{AdjointDiscrete}     L* w    about the linearisation point
#     F{AdjointContinuous}   L* w    about the linearisation point
#
# and all operators built together share one Workspace: FFT plans, scratch used
# within a single call, and the linearisation state. Only `linearise_about!`
# writes the state, and no call relies on scratch left by a previous one, so
# sharing is safe and no call moves the linearisation point.
#
# A formulation `F` provides:
#
#     ncomp(F)                       number of velocity components
#     cache_length(F, FTField)       spectral scratch VectorFields
#     cache_length(F, Field)         physical scratch VectorFields
#     state_length(F)                physical VectorFields in the state
#     F(mode, work, Re; force)       constructor
#     (op::F{MODE})(t, u, out)       one method per mode
#     linearise_about!(op::F, u)     fill work.state from the spectral field u


# ============================================================================ #
# Formulation interface                                                        #
# ============================================================================ #

"""
    ncomp(formulation) -> Int

Number of velocity components of a formulation type or operator.
"""
function ncomp end

"""
    cache_length(formulation, FTField) -> Int
    cache_length(formulation, Field)   -> Int

Number of spectral and physical scratch `VectorField`s a formulation needs.
"""
function cache_length end

"""
    state_length(formulation) -> Int

Number of physical `VectorField`s in the linearisation state of a formulation.
"""
function state_length end

"""
    linearise_about!(op, u) -> op

Set the linearisation point of `op` to the spectral field `u`. The state lives
in the shared [`Workspace`](@ref), so this sets the point of every operator
built on it.
"""
function linearise_about! end


# ============================================================================ #
# Workspace                                                                    #
# ============================================================================ #

"""
    Workspace(formulation, grid; flags=FFTW.EXHAUSTIVE, dealias=true)

FFT plans, scratch and linearisation state shared by the operators of
`formulation` (a formulation type, e.g. `CartesianPrimitive3D`).
"""
struct Workspace{PL, S, P}
     plans::PL        # FFT plans
     state::Vector{P} # linearisation point, written only by linearise_about!
    scache::Vector{S} # spectral scratch, used within one call
    pcache::Vector{P} # physical scratch, used within one call
end

function Workspace(formulation::Type, grid::AbstractGrid;
                   flags=FFTW.EXHAUSTIVE,
                   dealias::Bool=true)
    N = ncomp(formulation)

    # ---- allocators ----
    spectral() = VectorField([FTField(grid)        for _ in 1:N]...)
    physical() = VectorField([Field(grid; dealias) for _ in 1:N]...)

    # ---- plans, state and scratch ----
    plans  = FFTPlans(grid; flags)
    state  = [physical() for _ in 1:state_length(formulation)]
    scache = [spectral() for _ in 1:cache_length(formulation, FTField)]
    pcache = [physical() for _ in 1:cache_length(formulation, Field)]

    return Workspace(plans, state, scache, pcache)
end

# real scalar type of the fields in a workspace
_realtype(work::Workspace) = eltype(work.pcache[1][1])
