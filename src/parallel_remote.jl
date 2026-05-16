# PARALLEL REMOTE: Maximum Parallelism SSH/SFTP/Tramp Equivalent
#
# The greatest parallelism in parenthesized languages for remote access:
#
# ┌────────────────────────────────────────────────────────────────────────┐
# │  HIERARCHY OF PARALLEL REMOTE ACCESS (Lisp-family)                    │
# │                                                                        │
# │  1. BABASHKA + bbssh pod     — Native GraalVM, pmap over SSH sessions │
# │  2. Clojure + clj-ssh        — JVM, pmap/future/core.async channels   │
# │  3. Clojure + claypoole      — Threadpool-backed parallel SSH         │
# │  4. Racket Places            — Distributed Racket over SSH tunnels    │
# │  5. Emacs Tramp + async.el   — Async subprocess with callbacks        │
# │  6. Hy + asyncio             — Python asyncio.gather over paramiko    │
# │                                                                        │
# │  CHROMATIC GNOSIS: Each remote session gets a deterministic color     │
# │  from its (host, user, seed) triple. Intermediate hues track state.   │
# │                                                                        │
# │  ABLATIVE SEMANTICS: Resources are "carried away from" the remote,    │
# │  leaving chromatic traces. Future perfect: "will have been fetched"   │
# └────────────────────────────────────────────────────────────────────────┘

module ParallelRemote

using Base.Threads

export RemoteSession, RemoteHost, RemoteCommand
export SSHConfig, SFTPTransfer, TrAMPPath
export parallel_ssh, parallel_sftp, parallel_exec
export chromatic_session, ablative_fetch, ablative_send
export MagnetResource, magnet_parse, magnet_color
export ParallelRemotePool, create_pool, with_sessions
export world_parallel_remote

const GAY_SEED = UInt64(0x6761795f636f6c6f)

# ═══════════════════════════════════════════════════════════════════════════
# CHROMATIC SESSION IDENTITY
# ═══════════════════════════════════════════════════════════════════════════

"""
Deterministic color for a remote session.
f(host, user, seed) → Okhsl color
"""
struct ChromaticSession
    host::String
    user::String
    seed::UInt64
    hue::Float64        # 0-360
    saturation::Float64 # 0-1
    lightness::Float64  # 0-1
end

function ChromaticSession(host::String, user::String; seed::UInt64=GAY_SEED)
    # Hash the triple
    h = hash((host, user, seed))
    
    # Okhsl-safe ranges
    hue = (h % 360)
    saturation = 0.5 + 0.4 * ((h >> 12) % 100) / 100
    lightness = 0.35 + 0.4 * ((h >> 24) % 100) / 100
    
    ChromaticSession(host, user, seed, Float64(hue), saturation, lightness)
end

function intermediate_hue(cs::ChromaticSession, progress::Float64)
    # Interpolate hue as work progresses (0.0 → 1.0)
    # Creates "gnosis" trail through color space
    base_hue = cs.hue
    target_hue = mod(base_hue + 120, 360)  # Triadic shift
    mod(base_hue + progress * (target_hue - base_hue), 360)
end

# ═══════════════════════════════════════════════════════════════════════════
# REMOTE HOST & SESSION
# ═══════════════════════════════════════════════════════════════════════════

"""
SSH configuration for a remote host.
"""
struct SSHConfig
    host::String
    port::Int
    user::String
    identity_file::Union{String, Nothing}
    jump_host::Union{String, Nothing}  # ProxyJump
    options::Dict{String, String}
end

function SSHConfig(host::String; 
                   port::Int=22, 
                   user::String=ENV["USER"],
                   identity_file=nothing,
                   jump_host=nothing,
                   options=Dict{String,String}())
    SSHConfig(host, port, user, identity_file, jump_host, options)
end

"""
A remote host with chromatic identity.
"""
struct RemoteHost
    config::SSHConfig
    chromatic::ChromaticSession
    capabilities::Set{Symbol}  # :ssh, :sftp, :scp, :rsync, :mosh
end

function RemoteHost(host::String; user::String=ENV["USER"], kwargs...)
    config = SSHConfig(host; user=user, kwargs...)
    chromatic = ChromaticSession(host, user)
    caps = Set([:ssh, :sftp, :scp])
    RemoteHost(config, chromatic, caps)
end

"""
An active remote session (connection).
"""
mutable struct RemoteSession
    host::RemoteHost
    pid::Union{Int, Nothing}  # Local SSH process PID
    state::Symbol             # :disconnected, :connecting, :connected, :executing, :closing
    progress::Float64         # 0.0-1.0 for chromatic interpolation
    started_at::Float64
    commands_run::Int
end

function RemoteSession(host::RemoteHost)
    RemoteSession(host, nothing, :disconnected, 0.0, time(), 0)
end

function current_hue(session::RemoteSession)
    intermediate_hue(session.host.chromatic, session.progress)
end

# ═══════════════════════════════════════════════════════════════════════════
# REMOTE COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

"""
A command to execute remotely.
"""
struct RemoteCommand
    cmd::String
    stdin::Union{String, Nothing}
    env::Dict{String, String}
    timeout::Float64  # seconds
    capture_output::Bool
end

function RemoteCommand(cmd::String; 
                       stdin=nothing, 
                       env=Dict{String,String}(),
                       timeout=60.0,
                       capture_output=true)
    RemoteCommand(cmd, stdin, env, timeout, capture_output)
end

"""
Result of a remote command execution.
"""
struct RemoteResult
    session::RemoteSession
    command::RemoteCommand
    exit_code::Int
    stdout::String
    stderr::String
    duration::Float64
    final_hue::Float64
end

# ═══════════════════════════════════════════════════════════════════════════
# PARALLEL EXECUTION PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

"""
    parallel_ssh(hosts, cmd; max_concurrent=Threads.nthreads())

Execute command on multiple hosts in parallel.
Returns results as they complete (not necessarily in order).

This is the Tramp equivalent with maximum parallelism:
- Each host gets its own SSH connection
- Connections are made concurrently
- Results stream back as they complete
"""
function parallel_ssh(hosts::Vector{RemoteHost}, cmd::RemoteCommand; 
                      max_concurrent::Int=Threads.nthreads())
    n = length(hosts)
    results = Vector{RemoteResult}(undef, n)
    
    # Use a semaphore for concurrency limiting
    sem = Base.Semaphore(max_concurrent)
    
    @sync for (i, host) in enumerate(hosts)
        @async begin
            Base.acquire(sem)
            try
                session = RemoteSession(host)
                results[i] = execute_ssh(session, cmd)
            finally
                Base.release(sem)
            end
        end
    end
    
    results
end

"""
Execute SSH command (simulated - would use bbssh or Process in real impl).
"""
function execute_ssh(session::RemoteSession, cmd::RemoteCommand)
    session.state = :connecting
    session.progress = 0.1
    
    start_time = time()
    
    # Build SSH command
    config = session.host.config
    ssh_args = String[]
    push!(ssh_args, "-p", string(config.port))
    
    if !isnothing(config.identity_file)
        push!(ssh_args, "-i", config.identity_file)
    end
    
    if !isnothing(config.jump_host)
        push!(ssh_args, "-J", config.jump_host)
    end
    
    for (k, v) in config.options
        push!(ssh_args, "-o", "$k=$v")
    end
    
    target = "$(config.user)@$(config.host)"
    full_cmd = `ssh $ssh_args $target $(cmd.cmd)`
    
    session.state = :executing
    session.progress = 0.5
    
    # Execute
    stdout_buf = IOBuffer()
    stderr_buf = IOBuffer()
    
    try
        proc = run(pipeline(full_cmd, stdout=stdout_buf, stderr=stderr_buf), wait=false)
        session.pid = getpid(proc)
        
        # Wait with timeout
        timeout_task = @async begin
            sleep(cmd.timeout)
            if process_running(proc)
                kill(proc)
            end
        end
        
        wait(proc)
        
        session.state = :connected
        session.progress = 1.0
        session.commands_run += 1
        
        duration = time() - start_time
        
        RemoteResult(
            session, cmd,
            proc.exitcode,
            String(take!(stdout_buf)),
            String(take!(stderr_buf)),
            duration,
            current_hue(session)
        )
    catch e
        session.state = :disconnected
        session.progress = 0.0
        
        RemoteResult(
            session, cmd,
            -1,
            "",
            string(e),
            time() - start_time,
            current_hue(session)
        )
    end
end

"""
    parallel_exec(hosts, cmds; strategy=:all_to_all)

Execute multiple commands on multiple hosts.

Strategies:
- :all_to_all — Every command on every host (n×m operations)
- :zip — Pair hosts with commands 1:1
- :broadcast — Same command to all hosts
- :scatter — Distribute commands round-robin
"""
function parallel_exec(hosts::Vector{RemoteHost}, cmds::Vector{RemoteCommand};
                       strategy::Symbol=:all_to_all,
                       max_concurrent::Int=Threads.nthreads())
    
    pairs = if strategy == :all_to_all
        [(h, c) for h in hosts for c in cmds]
    elseif strategy == :zip
        collect(zip(hosts, cmds))
    elseif strategy == :broadcast
        [(h, cmds[1]) for h in hosts]
    elseif strategy == :scatter
        [(hosts[mod1(i, length(hosts))], c) for (i, c) in enumerate(cmds)]
    else
        error("Unknown strategy: $strategy")
    end
    
    n = length(pairs)
    results = Vector{RemoteResult}(undef, n)
    sem = Base.Semaphore(max_concurrent)
    
    @sync for (i, (host, cmd)) in enumerate(pairs)
        @async begin
            Base.acquire(sem)
            try
                session = RemoteSession(host)
                results[i] = execute_ssh(session, cmd)
            finally
                Base.release(sem)
            end
        end
    end
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════
# SFTP TRANSFERS
# ═══════════════════════════════════════════════════════════════════════════

"""
SFTP transfer specification.
"""
struct SFTPTransfer
    direction::Symbol  # :get or :put
    local_path::String
    remote_path::String
    recursive::Bool
    preserve_times::Bool
end

function SFTPGet(remote, local_path; recursive=false)
    SFTPTransfer(:get, local_path, remote, recursive, true)
end

function SFTPPut(local_path, remote; recursive=false)
    SFTPTransfer(:put, local_path, remote, recursive, true)
end

"""
    parallel_sftp(hosts, transfers)

Execute SFTP transfers in parallel.
"""
function parallel_sftp(hosts::Vector{RemoteHost}, transfers::Vector{SFTPTransfer};
                       max_concurrent::Int=Threads.nthreads())
    # Build sftp commands from transfers
    cmds = RemoteCommand[]
    
    for t in transfers
        if t.direction == :get
            # Would use bbssh/sftp in real implementation
            push!(cmds, RemoteCommand("cat $(t.remote_path)"))
        else
            push!(cmds, RemoteCommand("cat > $(t.remote_path)"; stdin=read(t.local_path, String)))
        end
    end
    
    parallel_exec(hosts, cmds; strategy=:all_to_all, max_concurrent=max_concurrent)
end

# ═══════════════════════════════════════════════════════════════════════════
# TRAMP-STYLE PATHS
# ═══════════════════════════════════════════════════════════════════════════

"""
Tramp-style path: /ssh:user@host:/path/to/file
Extended with chromatic identity.
"""
struct TrAMPPath
    method::Symbol      # :ssh, :scp, :sftp, :sudo, :docker
    user::String
    host::String
    port::Int
    path::String
    hops::Vector{Tuple{Symbol, String, String}}  # Multi-hop: [(method, user, host), ...]
end

function parse_tramp_path(s::String)
    # Parse: /method:user@host#port:/path
    # Or multi-hop: /method:user@host|method:user@host2:/path
    
    m = match(r"^/(\w+):(?:(\w+)@)?([^:#|]+)(?:#(\d+))?(?:\|(.+))?:(.*)$", s)
    isnothing(m) && error("Invalid Tramp path: $s")
    
    method = Symbol(m.captures[1])
    user = something(m.captures[2], ENV["USER"])
    host = m.captures[3]
    port = isnothing(m.captures[4]) ? 22 : parse(Int, m.captures[4])
    
    # Parse hops
    hops = Tuple{Symbol, String, String}[]
    if !isnothing(m.captures[5])
        for hop in split(m.captures[5], '|')
            hm = match(r"(\w+):(?:(\w+)@)?(.+)", hop)
            if !isnothing(hm)
                push!(hops, (Symbol(hm.captures[1]), 
                            something(hm.captures[2], ENV["USER"]),
                            hm.captures[3]))
            end
        end
    end
    
    path = m.captures[6]
    
    TrAMPPath(method, user, host, port, path, hops)
end

function to_remote_host(tp::TrAMPPath)
    jump = isempty(tp.hops) ? nothing : 
           join(["$(h[2])@$(h[3])" for h in tp.hops], ",")
    RemoteHost(tp.host; user=tp.user, port=tp.port, jump_host=jump)
end

# ═══════════════════════════════════════════════════════════════════════════
# MAGNET:// RESOURCES
# ═══════════════════════════════════════════════════════════════════════════

"""
A magnet:// URI resource with chromatic identity.

magnet:?xt=urn:btih:<infohash>&dn=<name>&tr=<tracker>

The infohash becomes the chromatic seed — content-addressed color.
"""
struct MagnetResource
    infohash::Vector{UInt8}  # 20 bytes (SHA-1) or 32 bytes (SHA-256)
    display_name::String
    trackers::Vector{String}
    exact_length::Union{Int, Nothing}
    chromatic::ChromaticSession  # Color from infohash
end

function magnet_parse(uri::String)
    # Parse magnet:?xt=urn:btih:...
    startswith(uri, "magnet:?") || error("Not a magnet URI: $uri")
    
    params = Dict{String, Vector{String}}()
    for part in split(uri[9:end], '&')
        kv = split(part, '=', limit=2)
        length(kv) == 2 || continue
        k, v = kv
        v = replace(v, '+' => ' ')
        v = URIdecode(v)
        
        if haskey(params, k)
            push!(params[k], v)
        else
            params[k] = [v]
        end
    end
    
    # Extract infohash
    xt = get(params, "xt", String[])
    infohash = UInt8[]
    for x in xt
        m = match(r"urn:btih:([a-fA-F0-9]+)", x)
        if !isnothing(m)
            infohash = hex2bytes(m.captures[1])
            break
        end
    end
    
    isempty(infohash) && error("No infohash in magnet URI")
    
    dn = get(params, "dn", ["unknown"])[1]
    tr = get(params, "tr", String[])
    xl = let v = get(params, "xl", String[])
        isempty(v) ? nothing : parse(Int, v[1])
    end
    
    # Chromatic identity from infohash
    seed = reinterpret(UInt64, infohash[1:min(8, length(infohash))])[1]
    chromatic = ChromaticSession("magnet", dn; seed=seed)
    
    MagnetResource(infohash, dn, tr, xl, chromatic)
end

# Simple URI decode
function URIdecode(s::String)
    replace(s, r"%([0-9a-fA-F]{2})" => m -> Char(parse(UInt8, m[2:3], base=16)))
end

"""
Get the chromatic hue of a magnet resource.
Content-addressed color.
"""
function magnet_color(m::MagnetResource)
    m.chromatic.hue
end

# ═══════════════════════════════════════════════════════════════════════════
# ABLATIVE SEMANTICS
# ═══════════════════════════════════════════════════════════════════════════

"""
Ablative case: motion away from, separation.
"will have been fetched" — future perfect tense.

The resource is carried away from the remote, leaving a chromatic trace.
"""
struct AblativeTransfer
    source::TrAMPPath
    destination::String
    started_at::Float64
    completed_at::Union{Float64, Nothing}
    chromatic_trace::Vector{Float64}  # Hue at each progress point
end

"""
    ablative_fetch(remote_path, local_path)

Fetch from remote with ablative semantics.
Returns a future that "will have been" completed.
"""
function ablative_fetch(remote::String, local_dest::String)
    tp = parse_tramp_path(remote)
    host = to_remote_host(tp)
    session = RemoteSession(host)
    
    transfer = AblativeTransfer(tp, local_dest, time(), nothing, Float64[])
    
    @async begin
        # Track chromatic progress
        for progress in 0.1:0.1:1.0
            session.progress = progress
            push!(transfer.chromatic_trace, current_hue(session))
            yield()
        end
        
        # Execute fetch
        cmd = RemoteCommand("cat $(tp.path)")
        result = execute_ssh(session, cmd)
        
        if result.exit_code == 0
            write(local_dest, result.stdout)
        end
        
        transfer = AblativeTransfer(tp, local_dest, transfer.started_at, time(), transfer.chromatic_trace)
        transfer
    end
end

"""
    ablative_send(local_path, remote_path)

Send to remote with ablative semantics.
The local resource "will have been" transferred.
"""
function ablative_send(local_src::String, remote::String)
    tp = parse_tramp_path(remote)
    host = to_remote_host(tp)
    session = RemoteSession(host)
    
    content = read(local_src, String)
    transfer = AblativeTransfer(
        TrAMPPath(:local, ENV["USER"], "localhost", 0, local_src, []),
        remote, time(), nothing, Float64[]
    )
    
    @async begin
        for progress in 0.1:0.1:1.0
            session.progress = progress
            push!(transfer.chromatic_trace, current_hue(session))
            yield()
        end
        
        cmd = RemoteCommand("cat > $(tp.path)"; stdin=content)
        result = execute_ssh(session, cmd)
        
        transfer = AblativeTransfer(
            TrAMPPath(:local, ENV["USER"], "localhost", 0, local_src, []),
            remote, transfer.started_at, time(), transfer.chromatic_trace
        )
        transfer
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# SESSION POOL (Connection Reuse)
# ═══════════════════════════════════════════════════════════════════════════

"""
Pool of reusable SSH sessions for maximum parallelism.
Like Tramp's connection caching, but explicit and parallel.
"""
mutable struct ParallelRemotePool
    sessions::Dict{String, Vector{RemoteSession}}  # host → sessions
    max_per_host::Int
    lock::ReentrantLock
end

function create_pool(; max_per_host::Int=4)
    ParallelRemotePool(Dict{String, Vector{RemoteSession}}(), max_per_host, ReentrantLock())
end

function acquire_session!(pool::ParallelRemotePool, host::RemoteHost)
    key = "$(host.config.user)@$(host.config.host):$(host.config.port)"
    
    lock(pool.lock) do
        sessions = get!(pool.sessions, key, RemoteSession[])
        
        # Find idle session
        for session in sessions
            if session.state == :connected
                return session
            end
        end
        
        # Create new if under limit
        if length(sessions) < pool.max_per_host
            session = RemoteSession(host)
            push!(sessions, session)
            return session
        end
        
        # Wait for one to become available
        nothing
    end
end

function release_session!(pool::ParallelRemotePool, session::RemoteSession)
    # Session returns to pool, ready for reuse
    session.state = :connected
    session.progress = 0.0
end

"""
Execute with session pool.
"""
function with_sessions(f::Function, pool::ParallelRemotePool, hosts::Vector{RemoteHost})
    sessions = RemoteSession[]
    
    try
        for host in hosts
            session = acquire_session!(pool, host)
            while isnothing(session)
                sleep(0.01)
                session = acquire_session!(pool, host)
            end
            push!(sessions, session)
        end
        
        f(sessions)
    finally
        for session in sessions
            release_session!(pool, session)
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON: PARALLEL REMOTE IN PARENTHESIZED LANGUAGES
# ═══════════════════════════════════════════════════════════════════════════

"""
Ranking of parallel SSH implementations in Lisp-family languages.
"""
const PARALLEL_SSH_RANKINGS = [
    # (Language, Library, Parallelism Mechanism, Max Concurrency, Notes)
    (:Babashka, :bbssh, "pmap + futures", :unbounded, "Native GraalVM, fastest startup"),
    (:Clojure, Symbol("clj-ssh"), "pmap/future/core.async", :unbounded, "JVM, rich ecosystem"),
    (:Clojure, :claypoole, "threadpool pmap", :configurable, "Ordered/unordered pfor"),
    (:Racket, :places, "distributed places over SSH", :per_machine, "Built-in distributed computing"),
    (:Emacs, :Tramp, "async.el + process sentinels", :limited, "Single-threaded event loop"),
    (:Hy, :asyncio, "asyncio.gather + paramiko", :event_loop, "Python 3 async"),
    (:Fennel, :luasocket, "coroutines", :cooperative, "Lua coroutines, not true parallel"),
    (:Janet, :ev, "fibers + event loop", :fiber_pool, "Green threads"),
]

function print_rankings()
    println("╔════════════════════════════════════════════════════════════════════════╗")
    println("║  PARALLEL SSH IN PARENTHESIZED LANGUAGES                               ║")
    println("╠════════════════════════════════════════════════════════════════════════╣")
    
    for (lang, lib, mechanism, concurrency, notes) in PARALLEL_SSH_RANKINGS
        println("║  $(rpad(lang, 10)) │ $(rpad(lib, 12)) │ $(rpad(mechanism, 25)) ║")
        println("║            │ $(rpad(concurrency, 12)) │ $(rpad(notes, 25)) ║")
        println("╟────────────┼──────────────┼───────────────────────────╢")
    end
    
    println("╚════════════════════════════════════════════════════════════════════════╝")
end

# ═══════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════

function world_parallel_remote()
    println("═══════════════════════════════════════════════════════════════")
    println("  PARALLEL REMOTE: Maximum Parallelism SSH/SFTP/Tramp")
    println("═══════════════════════════════════════════════════════════════")
    println()
    
    # Chromatic sessions
    println("CHROMATIC SESSION IDENTITY:")
    hosts = ["server1.example.com", "server2.example.com", "server3.example.com"]
    for host in hosts
        cs = ChromaticSession(host, "admin")
        println("  $(host) → hue=$(round(cs.hue, digits=1))° s=$(round(cs.saturation, digits=2)) l=$(round(cs.lightness, digits=2))")
    end
    println()
    
    # Tramp paths
    println("TRAMP-STYLE PATHS:")
    paths = [
        "/ssh:admin@server1:/var/log/syslog",
        "/ssh:root@gateway|ssh:admin@internal:/etc/hosts",
        "/sftp:deploy@prod#2222:/app/config.yml",
    ]
    for p in paths
        tp = parse_tramp_path(p)
        println("  $p")
        println("    → method=$(tp.method) user=$(tp.user) host=$(tp.host) path=$(tp.path)")
        if !isempty(tp.hops)
            println("    → hops=$(tp.hops)")
        end
    end
    println()
    
    # Magnet resources
    println("MAGNET:// CHROMATIC IDENTITY:")
    # Example magnet (fake hash)
    magnet = "magnet:?xt=urn:btih:0123456789abcdef0123456789abcdef01234567&dn=example.torrent&tr=udp://tracker.example.com:6969"
    mr = magnet_parse(magnet)
    println("  Display name: $(mr.display_name)")
    println("  Infohash: $(bytes2hex(mr.infohash))")
    println("  Chromatic hue: $(round(magnet_color(mr), digits=1))°")
    println()
    
    # Rankings
    println("PARALLEL SSH RANKINGS (parenthesized languages):")
    println()
    println("  #1 BABASHKA + bbssh")
    println("     • Native GraalVM binary, ~5ms startup")
    println("     • pmap over SSH sessions: (pmap #(bbssh/exec % cmd) sessions)")
    println("     • core.async channels for streaming output")
    println()
    println("  #2 Clojure + clj-ssh + claypoole")  
    println("     • JVM with true OS threads")
    println("     • (cp/pmap pool #(ssh session %) commands)")
    println("     • Configurable thread pools, ordered/unordered")
    println()
    println("  #3 Racket Places")
    println("     • (place-channel-put ch (ssh-exec host cmd))")
    println("     • Distributed across machines via SSH tunnels")
    println("     • Message-passing parallelism")
    println()
    println("  #4 Emacs Tramp + async.el")
    println("     • Single event loop, but async process sentinels")
    println("     • (async-start-process \"ssh\" \"ssh\" callback host)")
    println("     • Connection caching via ControlMaster")
    println()
    
    println("ABLATIVE SEMANTICS (future perfect):")
    println("  ablative_fetch(\"/ssh:user@host:/file\", \"./local\")")
    println("    → The file 'will have been' fetched, leaving chromatic trace")
    println("  Chromatic trace: [hue₀=0°] → [hue₅=60°] → [hue₁₀=120°]")
    println()
    
    println("═══════════════════════════════════════════════════════════════")
    println("  \"The greatest parallelism is the one you don't have to manage.\"")
    println("  — Babashka: pmap + bbssh + pods")
    println("═══════════════════════════════════════════════════════════════")
end

end # module ParallelRemote
