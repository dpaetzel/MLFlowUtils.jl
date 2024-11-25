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
) do mlf, mlfrun, params
    result = params[:param1] + params[:param2]
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
    ) do mlf, mlfrun, params
        result = params[:param1] + params[:param2]
        logmetric(mlf, mlfrun, "result", result; step=1)
        return nothing
    end
end

mlfexp = getexperiment(mlf, "example")
runs =
    searchruns(mlf, mlfexp; filter="params.a = \"42\" and params.b = \"15\"")
df = MLFlowUtils.runs2df(runs)
mlfrun = df[end, :]
runid_parent = mlfrun.run_id
runs = searchruns(
    mlf,
    mlfexp;
    filter="tags.$(MLFlowUtils.MLFLOW_PARENT_RUN_ID) = \"$runid_parent\"",
)
df = MLFlowUtils.runs2df(runs)

# We can also nest `runmlf` to achieve the parent run hierarchy.
mlf = getmlf()
name_exp = "example_nested"
mlfexp = getorcreateexperiment(mlf, name_exp)
runmlf(
    mlf,
    name_exp;
    check_gitdirty=false,
    rerun=true,
    param1=100,
    # (; param1=100)...,
) do mlf, mlfrun, params_parent
    for (idx, data) in enumerate(rand(5))
        runmlf(
            mlf,
            name_exp;
            check_gitdirty=false,
            rerun=true,
            # Note that we must not do `NamedTuple(params_parent)`
            # before giving this to `runmlf(; params_parent=…)`
            # because the the hashes do not match (`NamedTuple`
            # hashes differ from `NTuple` hashes which is what
            # `params_parent...` creates).
            params_parent=params_parent,
            param3=idx,
            param4=data,
        ) do mlf, mlfrun, params
            return logmetric(
                mlf,
                mlfrun,
                "result",
                params_parent[:param1] + params[:param3] + params[:param4],
            )
        end
    end
end

# Note that we can also pass SSH aliases such as `"c3d"` defined in
# `.ssh/config` for `userhost`.
# df_fitness = readcsvartifact("c3d", df.artifact_uri[1], "log_fitness.csv")
