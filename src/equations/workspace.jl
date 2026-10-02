# Workspace shared by the operators built together on one grid.
#
# It holds the FFT plans, the scratch used within a single call, and the
# linearised state. Only `linearise_about!` writes the linearised state, and no
# call relies on scratch left by a previous one, so sharing is safe and no call
# moves the linearisation point. The workspace knows nothing about the system:
# its sizes come from the operator that uses it.

"""
    Workspace(grid, ncomp; nstate, nspectral, nphysical, flags=FFTW.EXHAUSTIVE, dealias=true)

FFT plans, scratch and linearised state shared by operators on `grid`, made of
`VectorField`s with `ncomp` components: `nstate` physical fields for the
linearised state, `nspectral` spectral and `nphysical` physical scratch fields.
"""
struct Workspace{PL, S, P}
               plans::PL        # FFT plans
    linearised_state::Vector{P} # linearisation point, written only by linearise_about!
              scache::Vector{S} # spectral scratch, used within one call
              pcache::Vector{P} # physical scratch, used within one call
end

function Workspace(grid::AbstractGrid, ncomp::Integer;
                   nstate::Integer,
                   nspectral::Integer,
                   nphysical::Integer,
                   flags=FFTW.EXHAUSTIVE,
                   dealias::Bool=true)

    # ---- allocators ----
    spectral() = VectorField([FTField(grid)        for _ in 1:ncomp]...)
    physical() = VectorField([Field(grid; dealias) for _ in 1:ncomp]...)

    # ---- plans, linearised state and scratch ----
    plans            = FFTPlans(grid; flags)
    linearised_state = [physical() for _ in 1:nstate]
    scache           = [spectral() for _ in 1:nspectral]
    pcache           = [physical() for _ in 1:nphysical]

    return Workspace(plans, linearised_state, scache, pcache)
end

"""
    ncomp(op) -> Int

Number of components of the state an operator or geometry acts on.
"""
function ncomp end

"""
    linearise_about!(op, u) -> op

Set the linearisation point of `op` to the spectral field `u`. The linearised
state lives in the shared [`Workspace`](@ref), so this sets the point of every
operator built on it.
"""
function linearise_about! end

# real scalar type of the fields in a workspace
_realtype(work::Workspace) = eltype(work.pcache[1][1])
