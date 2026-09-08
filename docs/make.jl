using Documenter, ReSolverFlowsBase

makedocs(
    sitename = "ReSolverFlowsBase.jl",
    modules  = [ReSolverFlowsBase],
    authors  = "Davide Lasagna",
    format   = Documenter.HTML(
        prettyurls  = get(ENV, "CI", nothing) == "true",
        canonical   = "https://Davide-Lasagna-s-Lab.github.io/ReSolverFlowsBase.jl/stable",
    ),
    pages = [
        "Home"                    => "index.md",
        "Concepts & Conventions"  => "guide.md",
        "API Reference"           => "api.md",
    ],
    checkdocs = :exports,
    warnonly  = false,
)

deploydocs(
    repo   = "github.com/Davide-Lasagna-s-Lab/ReSolverFlowsBase.jl.git",
    target = "build",
)
