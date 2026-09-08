module MPIIntermediateExt

try
    # try loading the required packages for the full extension
    Base.require(Main, :HaloArrays)
catch
    # otherwise give up with the extension and throw a warning
    @warn """Failed to load HaloArrays.jl required to use MPIExt.
             Either add the missing packages or MPI functionality 
             for ReSolverFlowsBase will not be available."""
end

end
