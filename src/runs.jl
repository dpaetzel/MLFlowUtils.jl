# Extracted from mlflow Python source (could do `PyCall` but don't wanna).
const MLFLOW_PARENT_RUN_ID = "mlflow.parentRunId"

struct ReproducibilityError <: Exception
    msg::String
end

# function logtag(mlf, mlfrun, key, value)
#     return MLFlowClient.mlfpost(
#         mlf,
#         "runs/set-tag";
#         run_id=mlfrun.info.run_id,
#         key=key,
#         value=value,
#     )
# end

# https://discourse.julialang.org/t/persistent-hash/110001/6
function sha_serialize(obj)
    b = IOBuffer()
    Serialization.serialize(b, obj)
    return bytes2hex(SHA.sha256(take!(b)))
end

"""
    runmlf(f, mlf, name_exp; <keyword arguments>, <arguments for f>)

Set up an Mlflow scaffold for the callable `f` and run it unless it has already
been run with the provided parameters on the current Git commit.

Scaffolding roughly consists of (check code, might still change)

- checking whether there exists a run with `params.hash_params == hash(params)`
  and `tags.gitcommit == current_commit` and, if so, aborting
- creating the experiment with name `name_exp` if it does not exist
- creating a run
- running `f(mlf, mlfrun, params)`
- doing an epilogue to correctly log the run's final state (`FINISHED` or
  `FAILED` to mlflow)
- if the run failed, set the `failreason` tag to contain the stringyfied error

Note that due to how Julia handles `do` blocks, we provide `f` with `params` as
a named tuple to make it possible (and as a result actually requiring) to
provide keyword arguments.

# Arguments

- `f`: Function to call.
- `mlf::MLFlow`
- `name_exp::String`: Name of MLflow experiment.
- `rerun::Bool=false`: Whether to rerun the configuration even if it already
  exists in MLflow (checked by hashing the arguments for `f`).
- `check_gitdirty::Bool=true`: Whether to block execution if Git is dirty.
- `params_parent=missing`:: `missing` or if this run should be assigned to a
  parent run then a `NamedTuple` of that parent run's parameters so that they
  can be hashed.
- `params...`: Any other keyword arguments are passed to `f` by calling `f(mlf,
  mlfrun, params...)`. See the example below.

# Example

```
julia> mlf = getmlf()
julia> runmlf(mlf, "expname"; a=1.0, b=2.0) do mlf, mlfrun, params...
  params = NamedTuple(params)
  println(params.a)
  println(params.b)
  logmetric(mlf, mlfrun, "sum", params.a + params.b)
end
```
"""
function runmlf(
    f,
    mlf::MLFlow,
    name_exp::String;
    name_run::Union{String,Missing}=missing,
    rerun::Bool=false,
    check_gitdirty::Bool=true,
    params_parent=missing,
    params...,
)
    params = Dict(params)

    git_commit = LibGit2.head(".")
    @info "Git commit is $git_commit."
    git_dirty = LibGit2.isdirty(GitRepo("."))
    if check_gitdirty && git_dirty
        throw(
            ReproducibilityError(
                "Git is dirty, create a WIP commit and try again or " *
                "set `check_gitdirty=false`",
            ),
        )
    end

    # TODO Reduce duplication with run.jl
    mlfexp = getorcreateexperiment(mlf, name_exp)

    if !ismissing(params_parent)
        hash_params_parent = sha_serialize(params_parent)
        @info "Parent run parameter SHA starts with $(hash_params_parent[1:10])."

        # Check whether the parent run already exists to register this run with
        # it.
        runs_existing = searchruns(
            mlf,
            mlfexp;
            filter="params.hash_params = '$hash_params_parent' and " *
                   "tags.gitcommit = '$git_commit'",
        )
        if isempty(runs_existing)
            @info "Creating parent run in mlflow …"
            # Variable assignments can get out of `if`s in Julia.
            mlfrun_parent = createrun(
                mlf,
                mlfexp;
                run_name=name_run,
                tags=reduce(
                    vcat,
                    [
                        [
                            Dict("key" => "gitcommit", "value" => git_commit),
                            Dict(
                                "key" => "gitdirty",
                                "value" => string(git_dirty),
                            ),
                        ],
                    ],
                ),
            )
            logparam(mlf, mlfrun_parent, "hash_params", hash_params_parent)
            logparam(mlf, mlfrun_parent, pairs(params_parent))
        else # !isempty(runs_existing)
            @info "Found existing parent run in mlflow."
            mlfrun_parent =
                sort(runs_existing; by=(run -> run.info.start_time))[end]
            @info "Sanity checking parameters stored in parent run …"
            for (key, val) in pairs(params_parent)
                try
                    # Note that `key` is a `Symbol` but the keys in
                    # `mlfrun.data.params` are `Symbol`s converted to `String`.
                    #
                    # Further, the values of the `mlfrun.data.params` dictionary
                    # are `MLFlowRunDataParam` objects which contain both the
                    # key and the value.
                    val_existing = mlfrun_parent.data.params[string(key)].value
                    # Note that also values are converted to string when we log
                    # them.
                    if val_existing != string(val)
                        throw(
                            ReproducibilityError(
                                "Value $val_existing for parameter $key in " *
                                "parent run does not match current run's " *
                                "value ($val).",
                            ),
                        )
                    end
                catch e
                    if isa(e, KeyError)
                        @warn "Parameter $key was not yet logged in parent " *
                              "run, adding it now."
                        logparam(mlf, mlfrun_parent, key, val)
                    else
                        rethrow()
                    end
                end
            end
        end

        run_id_parent = mlfrun_parent.info.run_id
        tag_parentrun = if !ismissing(run_id_parent)
            [Dict("key" => MLFLOW_PARENT_RUN_ID, "value" => run_id_parent)]
        else # ismissing(run_id_parent)
            []
        end
    else # ismissing(params_parent)
        tag_parentrun = []
    end

    hash_params = sha_serialize(params)
    @info "Run parameter SHA starts with $(hash_params[1:10])."

    runs_existing = searchruns(
        mlf,
        mlfexp;
        filter="params.hash_params = '$hash_params' and " *
               "tags.gitcommit = '$git_commit' and " *
               "attributes.status = 'FINISHED'",
    )
    if !rerun && !isempty(runs_existing)
        @info "Given config was already run within this " *
              "experiment on the current Git commit at least once, " *
              "skipping and reusing results of most recently started …"
        mlfrun = sort(runs_existing; by=(run -> run.info.start_time))[end]
        return mlfrun
    else
        if rerun && !isempty(runs_existing)
            @info "Given config was already run within this " *
                  "experiment on the current Git commit at least once, " *
                  "rerunning because `rerun==True` …"
        end
        mlfrun = createrun(
            mlf,
            mlfexp;
            run_name=name_run,
            # Each tag has to be a dictionary right now. See
            # https://github.com/JuliaAI/MLFlowClient.jl/issues/30#issue-1855543109
            tags=reduce(
                vcat,
                [
                    [
                        Dict("key" => "gitcommit", "value" => git_commit),
                        Dict(
                            "key" => "gitdirty",
                            "value" => string(git_dirty),
                        ),
                    ],
                    tag_parentrun,
                ],
            ),
        )
        # Log config hash so that later above hash check will result in not
        # rerunning the same config on the same Git commit later.
        logparam(mlf, mlfrun, "hash_params", hash_params)
        logparam(mlf, mlfrun, params)
        name_run_final = mlfrun.info.run_name
        @info "Started run $name_run_final with id $(mlfrun.info.run_id)."

        # Perform the run.
        try
            f(mlf, mlfrun, params)
        catch e
            @warn "Run $name_run_final failed with $(string(e))."
            @warn "Writing exception information to MLflow …"
            settag(mlf, mlfrun, "failreason", string(e))
            io = IOBuffer()
            Base.show_backtrace(io, catch_backtrace())
            closewrite(io)
            seekstart(io)
            logartifact(mlf, mlfrun, "exceptions.txt", io)
            close(io)
            @warn "Marking run $name_run_final as failed …"
            updaterun(mlf, mlfrun, "FAILED")
            @warn "Marked run $name_run_final as failed."
            @warn "Rethrowing exception now …"
            rethrow()
        end

        @info "Finishing run $name_run_final …"
        updaterun(mlf, mlfrun, "FINISHED")
        @info "Finished run $name_run_final."

        return mlfrun
    end
end

# Required until MLFlowClient version goes up.
"""
    settag(mlf::MLFlow, run, key, value)
    settag(mlf::MLFlow, run, kv)

Associates a tag (a key and a value) to the particular run.

Refer to [the official MLflow REST API
docs](https://mlflow.org/docs/latest/rest-api.html#set-tag) for restrictions on
`key` and `value`.

# Arguments
- `mlf`: [`MLFlow`](@ref) configuration.
- `run`: one of [`MLFlowRun`](@ref), [`MLFlowRunInfo`](@ref), or `String`.
- `key`: tag key (name). Automatically converted to string before sending to MLFlow because this is the only type that MLFlow supports.
- `value`: parameter value. Automatically converted to string before sending to MLFlow because this is the only type that MLFlow supports.

One could also specify `kv::Dict` instead of separate `key` and `value` arguments.
"""
function settag(mlf::MLFlow, run_id::String, key, value)
    endpoint = "runs/set-tag"
    return MLFlowClient.mlfpost(
        mlf,
        endpoint;
        run_id=run_id,
        key=string(key),
        value=string(value),
    )
end
settag(mlf::MLFlow, run_info::MLFlowRunInfo, key, value) =
    settag(mlf, run_info.run_id, key, value)
settag(mlf::MLFlow, run::MLFlowRun, key, value) =
    settag(mlf, run.info, key, value)
function settag(mlf::MLFlow, run::Union{String,MLFlowRun,MLFlowRunInfo}, kv)
    for (k, v) in kv
        logparam(mlf, run, k, v)
    end
end
