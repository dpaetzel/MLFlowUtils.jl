module MLFlowUtils

using Base64
using LibGit2
using CSV
using DataFrames
using Dates
using MLFlowClient
using NPZ
using Serialization
using SHA
using TOML

export getmlf, loadruns, readcsvartifact, try_npzread, sha_serialize, runmlf

include("runs.jl")

"""
    getmlf(; url="http://localhost:5000/api", fname_config="config.toml")

Get an MLFlow client object which uses HTTP authentication as described in [this
GitHub
comment](https://github.com/JuliaAI/MLFlowClient.jl/issues/33#issuecomment-1830541659).

Authentication credentials are read from the given TOML file.
"""
function getmlf(; url="http://localhost:5000/api", fname_config="config.toml")
    config = try
        open(fname_config, "r") do file
            return TOML.parse(read(file, String))
        end
    catch
        println("Cannot open mlflow authentication config file, ensure it exists")
        rethrow()
    end

    username = config["database"]["user"]
    password = config["database"]["password"]
    encoded_credentials = base64encode("$username:$password")
    headers = Dict("Authorization" => "Basic $encoded_credentials")
    return MLFlow(url; headers=headers)
end

# https://stackoverflow.com/a/68853482
function struct_to_dict(s, S)
    return Dict(key => getfield(s, key) for key in fieldnames(S))
end

function run_to_dict(r)
    infodict = struct_to_dict(r.info, MLFlowRunInfo)
    datadict = struct_to_dict(r.data, MLFlowRunData)
    paramsdict =
        Dict("params.$key" => val.value for (key, val) in datadict[:params])
    metricsdict =
        Dict("metrics.$key" => val.value for (key, val) in datadict[:metrics])
    # Tags are stored as weird dicts as of 2024-07-11.
    tagsdict = Dict(
        ("tags." * keyval["key"]) => keyval["value"] for
        keyval in datadict[:tags]
    )
    # Transform infodict keys to String as well (params and metrics contain
    # dots which make it hard to use Symbols everywhere, but we want uniformity).
    infodict = Dict(string(key) => val for (key, val) in infodict)
    # Note that we ignore tags for now.
    return rundict = merge(infodict, paramsdict, metricsdict, tagsdict)
end

"""
    runs_to_df(runs::Vector{MLFlowRun})

Transforms the vector into a `DataFrame`.
"""
function runs_to_df(runs::Vector{MLFlowRun})
    dicts = run_to_dict.(runs)
    add_missing_keys!(dicts)
    return df = DataFrame(dicts)
end

"""
For each key `k` in each `dict` of the dictionaries `dicts`, add in-place an
entry `k => missing` if `!in(k, keys(dict))`. This allows us to pass the
resulting vector of dicts to `DataFrame`.
"""
function add_missing_keys!(dicts)
    if !isempty(dicts)
        keys_all = union(keys.(dicts)...)
        for key in keys_all
            for dict in dicts
                get!(dict, key, missing)
            end
        end
    end
    return dicts
end

function todatetime(mlflowtime)
    return Dates.unix2datetime(round(mlflowtime / 1000)) +
           Millisecond(mlflowtime % 1000) +
           # TODO Consider to use TimeZones.jl
           # Add my timezone.
           Hour(2)
end

"""
    loadruns([MLFlow("http://localhost:5000/api")], expname; max_results=5000)

Load all available runs for the given experiment from the mlflow tracking server
given by `mlf`.
"""
function loadruns(mlf::MLFlow, expname::String; max_results=5000)
    # TODO Consider to serialize-cache this as well (see the `jid` variant of
    # `loadruns`)
    url = mlf.apiroot

    @info "Searching for experiment $expname at $url …"
    mlfexp = getexperiment(mlf, expname)

    @info "Loading runs for experiment \"$(mlfexp.name)\" from $url …"
    runs = searchruns(mlf, mlfexp; max_results=max_results)
    @info "Finished loading $(length(runs)) runs for experiment " *
          "\"$(mlfexp.name)\" from $url."

    @info "Converting mlflow data to Julia representations …"
    df = runs_to_df(runs)
    df[!, "start_time"] .= todatetime.(df.start_time)
    df[!, "end_time"] .= passmissing(todatetime).(df.end_time)

    @info "Adding helpful additional columns …"
    df[!, "duration"] = df.end_time .- df.start_time
    df[!, "duration_min"] =
        Missings.replace(df.end_time, now()) .- df.start_time

    return df
end

loadruns(expname; kw...) =
    loadruns(MLFlow("http://localhost:5000/api"), expname; kw...)

const nameformat = r"^([^-]+)-(([^-]*)-)?(\d+)_(\d+)$"

function parse_rname(rname)
    tag, _, subtag, jid, tid = match(nameformat, rname)
    return (; tag=tag, subtag=subtag, jid=jid, tid=tid)
end

"""
    loadruns([MLFlow("http://localhost:5000/api")], expname, jids, n_runs; max_results=5000)

Load, from the mlflow tracking server given by `mlf`, all available runs whose
name contains (see `nameformat`) one of the given job IDs. Check whether
`n_runs` runs were loaded and warn if not.
"""
function loadruns(
    mlf::MLFlow,
    expname::String,
    jids,
    n_runs::Int;
    max_results::Int=5000,
)
    df = loadruns(mlf, expname; max_results=max_results)

    @info "Filtering `run_name` format …"
    subset!(df, "run_name" => (n -> occursin.(nameformat, n)))

    @info "Parsing `run_name` format to job ID, tag etc. …"
    transform!(df, "run_name" => ByRow(parse_rname) => AsTable)

    @info "Filtering job IDs …"
    subset!(df, "jid" => (jid -> jid .∈ Ref(string.(jids))))

    if nrow(df) != n_runs
        @warn "Found only $(nrow(df)) runs instead of the desired $n_runs."
    end

    return df
end

loadruns(expname::String, jids, n_runs::Int; max_results::Int=5000) = loadruns(
    MLFlow("http://localhost:5000/api"),
    expname,
    jids,
    n_runs;
    max_results=5000,
)

"""
    readcsvartifact(userhost, artifact_uri, fname)

Given an artifact URI and SSH login information (supplied to the `ssh` as the
`user@host` info), try to read the given file from it using SSH, interpret it as
a CSV file and convert it to a `DataFrame`.

# Important security notice

We put `userhost` directly into the `ssh` command where `user@hostname` is
typically put. While Julia's handling of backtick stuff leads to some
sanitization, this is probably unsafe nevertheless, so don't expose this.
"""
function readcsvartifact(userhost, artifact_uri, fname)
    fpath = "$artifact_uri/$fname"

    # Check whether the file exist.
    process = run(`ssh $userhost "test -f $fpath"`; wait=true)

    # Check the exit code of the process
    if process.exitcode != 0
        println("File does not exist on the remote server.")
        return missing
    else
        io = open(
            pipeline(
                `ssh $userhost "tar -cvj -C $artifact_uri $fname"`,
                `tar -xvj --to-stdout`,
            ),
        )
        df = DataFrame(CSV.File(io))
        close(io)
        return df
    end
end

"""
    try_nzpread(fname_npz)

Try to read the given file as an NPZ file. If any exception is thrown, return
`missing`.
"""
function try_npzread(fname_npz)
    try
        return npzread(fname_npz)
    catch
        return missing
    end
end

"""
Append to the artifact with the given file name (assumed to be at the top level
of the artifacts folder of the given run) the given string and a new line
character.

Useful if you want to iteratively build up artifacts (e.g. additional logs that
cannot be written to mlflow directly).
"""
function appendtoartifact(mlf, mlfrun, name, str)
    # If the artifact does not yet exist, create an empty file first.
    lsarts = listartifacts(mlf, mlfrun)
    fpaths = getproperty.(lsarts, :filepath)
    fnamemap = Dict([basename(fpath) => fpath for fpath in fpaths])
    # If `name` is not yet an existing artifact's file name, create
    # that file and return its path.
    fpath = get(fnamemap, name) do
        # Create empty artifact file if not yet existing.
        return logartifact(mlf, mlfrun, name, "")
    end

    # Append to the file.
    open(fpath, "a") do file
        return println(file, str)
    end
end

end
