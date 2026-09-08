# Sentinel exception type for unimplemented required interface methods.
#
# ReSolverFlowsBase defines many abstract methods whose concrete implementations must be
# provided by downstream packages (e.g. `size`, `points`, `wavenumber_scale`,
# `weights`).  Rather than leaving such methods undefined (which would produce
# Julia's cryptic "no matching method" message), each stub explicitly throws
# `NotImplementedError`.  The exception captures the calling method's name from
# the runtime stack trace, so error messages identify exactly which interface
# method is missing.

"""
    NotImplementedError <: Exception

Thrown by abstract interface stubs in ReSolverFlowsBase when a required method has not
been implemented by a downstream package.

The exception records the name of the unimplemented method (taken from the
call-stack via `stacktrace()`) and the types of the arguments that were
passed, so the error message pinpoints exactly which method signature is
missing.

# Example

```julia
struct MyGrid <: AbstractGrid{Float64, 3, (2,1,3,nothing), (1,2)} end
# Base.size not yet defined for MyGrid.
size(MyGrid())
# → NotImplementedError: size(MyGrid) is missing a concrete implementation!
```
"""
struct NotImplementedError <: Exception
    method::Symbol
    signature

    NotImplementedError(signature...) = new(stacktrace()[2].func, typeof.(signature))
end

"""
    Base.showerror(io::IO, e::NotImplementedError)

Print a human-readable description of the missing method.  Single-argument
signatures are printed as `method(Type)`; multi-argument signatures as
`method(Type1, Type2, …)`.
"""
function Base.showerror(io::IO, e::NotImplementedError)
    if length(e.signature) == 1
        print(io, e.method, "(", e.signature[1], ") is missing a concrete implementation!")
    else
        print(io, e.method, e.signature, " is missing a concrete implementation!")
    end
end
