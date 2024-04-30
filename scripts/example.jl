using MLFlowClient
using MLFlowUtils

mlf = getmlf(; url="http://localhost:5001")

exps = searchexperiments(mlf)
println([exp.name for exp in exps])

df = loadruns(mlf, "runbest")

# Note that we can also pass SSH aliases such as `"c3d"` defined in
# `.ssh/config` for `userhost`.
df_fitness = readcsvartifact("c3d", df.artifact_uri[1], "log_fitness.csv")
