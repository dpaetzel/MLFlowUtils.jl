# MLFlowUtils


A few personal quality-of-life functions (config file–based authentication,
Slurm job–based run selection, …) for using
[MLFlowClient](https://github.com/JuliaAI/MLFlowClient.jl).


Reexports `MLFlowClient`; it therefore suffices to do

```
julia> using Pkg; Pkg.add(url="https://github.com/dpaetzel/MLFlowUtils.jl")
julia> using MLFlowUtils
```
