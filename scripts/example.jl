using MLFlowClient
using MLFlowUtils

# Without parent runs.
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

# With parent run.
mlf = getmlf()
params_parent = (; a=42, b=15)
for i in 1:5
    runmlf(
        mlf,
        "example";
        check_gitdirty=false,
        rerun=true,
        params_parent=params_parent,
        param1=100 + i,
        param2=200,
    ) do mlf, mlfrun, params...
        params = (; params...)
        result = params.param1 + params.param2
        logmetric(mlf, mlfrun, "result", result; step=1)
        return nothing
    end
end

mlfexp = getexperiment(mlf, "example")
runs =
    searchruns(mlf, mlfexp; filter="params.a = \"42\" and params.b = \"15\"")
df = MLFlowUtils.runs_to_df(runs)
mlfrun = df[end, :]
runid_parent = mlfrun.run_id
runs = searchruns(
    mlf,
    mlfexp;
    filter="tags.$(MLFlowUtils.MLFLOW_PARENT_RUN_ID) = \"$runid_parent\"",
)
df = MLFlowUtils.runs_to_df(runs)

# Note that we can also pass SSH aliases such as `"c3d"` defined in
# `.ssh/config` for `userhost`.
# df_fitness = readcsvartifact("c3d", df.artifact_uri[1], "log_fitness.csv")
