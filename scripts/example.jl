using MLFlowClient
using MLFlowUtils

mlf = getmlf()
runmlf(
    mlf,
    "example";
    check_gitdirty=false,
    rerun=true,
    param1=100,
    param2=200,
) do mlf, mlfrun, params...
    params = (; params...)
    result = params.param1 + params.param2
    logmetric(mlf, mlfrun, "result", result; step=1)
    return nothing
end

df = sort(loadruns(mlf, "example"))

# Note that we can also pass SSH aliases such as `"c3d"` defined in
# `.ssh/config` for `userhost`.
# df_fitness = readcsvartifact("c3d", df.artifact_uri[1], "log_fitness.csv")
