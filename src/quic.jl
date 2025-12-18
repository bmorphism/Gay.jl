# Gay.jl QUIC Path Probe Coloring
# ================================
# Parallel SPI color assignment for QUIC path validation frames
# 
# QUIC (RFC 9000) uses PATH_CHALLENGE/PATH_RESPONSE frames to validate paths.
# Each path probe gets a deterministic color based on:
# - Connection ID
# - Path ID (for multipath QUIC)
# - Challenge nonce (8 bytes)
#
# This enables visual debugging of path validation in multipath scenarios.

using Colors: RGB
using Printf

export QUICPathProbe, QUICConnection, QUICPath
export path_probe_color, connection_color, path_color
export probe_challenge!, probe_response!, validate_path!
export visualize_probes, probe_timeline
export ka_probe_colors!, parallel_probe_colors

# ═══════════════════════════════════════════════════════════════════════════
# QUIC Path Probe Types
# ═══════════════════════════════════════════════════════════════════════════

"""
    QUICPathProbe

A QUIC PATH_CHALLENGE or PATH_RESPONSE frame with SPI-colored identity.

Fields:
- `nonce`: 8-byte challenge data (unpredictable)
- `connection_id`: QUIC connection identifier
- `path_id`: Path identifier (0 for single-path, >0 for multipath)
- `is_challenge`: true for PATH_CHALLENGE, false for PATH_RESPONSE
- `timestamp_ns`: Nanosecond timestamp
- `color`: Deterministic SPI color
"""
struct QUICPathProbe
    nonce::NTuple{8, UInt8}
    connection_id::UInt64
    path_id::UInt32
    is_challenge::Bool
    timestamp_ns::UInt64
    color::RGB{Float64}
end

"""
    QUICPath

A QUIC network path with SPI coloring.

A path is identified by a 4-tuple: (src_ip, dst_ip, src_port, dst_port).
For simplicity, we hash this to a path_id.
"""
mutable struct QUICPath
    path_id::UInt32
    connection_id::UInt64
    validated::Bool
    active::Bool
    rtt_us::Float64
    probes_sent::Int
    probes_received::Int
    color::RGB{Float64}
    pending_challenges::Vector{QUICPathProbe}
end

"""
    QUICConnection

A QUIC connection with multiple paths and SPI coloring.
"""
mutable struct QUICConnection
    connection_id::UInt64
    seed::UInt64
    paths::Dict{UInt32, QUICPath}
    probe_history::Vector{QUICPathProbe}
    color::RGB{Float64}
end

# ═══════════════════════════════════════════════════════════════════════════
# Color Generation (SPI-based)
# ═══════════════════════════════════════════════════════════════════════════

"""
    connection_color(connection_id, seed=GAY_SEED) -> RGB

Generate deterministic color for a QUIC connection.
Same connection_id + seed always produces same color.
"""
function connection_color(connection_id::UInt64, seed::UInt64=GAY_SEED)
    r, g, b = hash_color(connection_id, seed)
    RGB(Float64(r), Float64(g), Float64(b))
end

"""
    path_color(connection_id, path_id, seed=GAY_SEED) -> RGB

Generate deterministic color for a QUIC path within a connection.
Uses XOR mixing to derive path-specific color from connection color.
"""
function path_color(connection_id::UInt64, path_id::UInt32, seed::UInt64=GAY_SEED)
    # Mix path_id into connection seed
    path_seed = splitmix64(seed ⊻ UInt64(path_id) ⊻ connection_id)
    r, g, b = hash_color(path_seed, seed)
    RGB(Float64(r), Float64(g), Float64(b))
end

"""
    path_probe_color(nonce, connection_id, path_id, seed=GAY_SEED) -> RGB

Generate deterministic color for a specific path probe.
The 8-byte nonce provides uniqueness within a path.
"""
function path_probe_color(
    nonce::NTuple{8, UInt8},
    connection_id::UInt64,
    path_id::UInt32,
    seed::UInt64=GAY_SEED
)
    # Convert nonce to UInt64
    nonce_u64 = reinterpret(UInt64, collect(nonce))[1]
    
    # Triple XOR mixing for unique probe color
    probe_seed = splitmix64(seed ⊻ connection_id ⊻ UInt64(path_id) ⊻ nonce_u64)
    r, g, b = hash_color(probe_seed, seed)
    RGB(Float64(r), Float64(g), Float64(b))
end

# Convenience method for Vector{UInt8}
function path_probe_color(
    nonce::Vector{UInt8},
    connection_id::UInt64,
    path_id::UInt32,
    seed::UInt64=GAY_SEED
)
    length(nonce) >= 8 || error("QUIC nonce must be 8 bytes")
    path_probe_color(NTuple{8, UInt8}(nonce[1:8]), connection_id, path_id, seed)
end

# ═══════════════════════════════════════════════════════════════════════════
# Connection and Path Management
# ═══════════════════════════════════════════════════════════════════════════

"""
    QUICConnection(connection_id; seed=GAY_SEED) -> QUICConnection

Create a new QUIC connection with SPI coloring.
"""
function QUICConnection(connection_id::UInt64; seed::UInt64=GAY_SEED)
    QUICConnection(
        connection_id,
        seed,
        Dict{UInt32, QUICPath}(),
        QUICPathProbe[],
        connection_color(connection_id, seed)
    )
end

"""
    add_path!(conn, path_id) -> QUICPath

Add a new path to a QUIC connection.
"""
function add_path!(conn::QUICConnection, path_id::UInt32)
    if haskey(conn.paths, path_id)
        return conn.paths[path_id]
    end
    
    path = QUICPath(
        path_id,
        conn.connection_id,
        false,  # not validated yet
        true,   # active
        0.0,    # no RTT yet
        0,      # no probes sent
        0,      # no probes received
        path_color(conn.connection_id, path_id, conn.seed),
        QUICPathProbe[]
    )
    
    conn.paths[path_id] = path
    return path
end

"""
    generate_nonce() -> NTuple{8, UInt8}

Generate a random 8-byte nonce for PATH_CHALLENGE.
"""
function generate_nonce()
    NTuple{8, UInt8}(rand(UInt8, 8))
end

"""
    probe_challenge!(conn, path_id) -> QUICPathProbe

Send a PATH_CHALLENGE on the specified path.
Returns the probe with its SPI color.
"""
function probe_challenge!(conn::QUICConnection, path_id::UInt32)
    # Ensure path exists
    path = get!(conn.paths, path_id) do
        add_path!(conn, path_id)
    end
    
    nonce = generate_nonce()
    timestamp = time_ns()
    
    probe = QUICPathProbe(
        nonce,
        conn.connection_id,
        path_id,
        true,  # is_challenge
        timestamp,
        path_probe_color(nonce, conn.connection_id, path_id, conn.seed)
    )
    
    push!(path.pending_challenges, probe)
    push!(conn.probe_history, probe)
    path.probes_sent += 1
    
    return probe
end

"""
    probe_response!(conn, path_id, nonce, timestamp_ns) -> Union{QUICPathProbe, Nothing}

Receive a PATH_RESPONSE matching a pending challenge.
Returns the response probe if valid, nothing if no matching challenge.
"""
function probe_response!(
    conn::QUICConnection,
    path_id::UInt32,
    nonce::NTuple{8, UInt8},
    timestamp_ns::UInt64
)
    if !haskey(conn.paths, path_id)
        return nothing
    end
    
    path = conn.paths[path_id]
    
    # Find matching challenge
    idx = findfirst(p -> p.nonce == nonce, path.pending_challenges)
    if isnothing(idx)
        return nothing
    end
    
    challenge = path.pending_challenges[idx]
    deleteat!(path.pending_challenges, idx)
    
    # Calculate RTT
    rtt_ns = timestamp_ns - challenge.timestamp_ns
    path.rtt_us = rtt_ns / 1000.0
    path.probes_received += 1
    
    # Create response probe with same color as challenge
    response = QUICPathProbe(
        nonce,
        conn.connection_id,
        path_id,
        false,  # is_response
        timestamp_ns,
        challenge.color  # Same color for matched pair
    )
    
    push!(conn.probe_history, response)
    
    return response
end

"""
    validate_path!(conn, path_id) -> Bool

Mark a path as validated if it has received at least one valid response.
"""
function validate_path!(conn::QUICConnection, path_id::UInt32)
    if !haskey(conn.paths, path_id)
        return false
    end
    
    path = conn.paths[path_id]
    if path.probes_received > 0
        path.validated = true
        return true
    end
    
    return false
end

# ═══════════════════════════════════════════════════════════════════════════
# Parallel SPI Color Generation for Path Probes
# ═══════════════════════════════════════════════════════════════════════════

"""
    ka_probe_colors!(output, connection_id, path_ids, nonces, seed; backend=CPU())

Generate colors for multiple path probes in parallel using KernelAbstractions.
Output is (n, 3) Float32 array.
"""
function ka_probe_colors!(
    output::AbstractMatrix{Float32},
    connection_id::UInt64,
    path_ids::AbstractVector{UInt32},
    nonces::AbstractVector{UInt64},
    seed::UInt64;
    backend = get_backend(),
    workgroup::Int = 256
)
    n = size(output, 1)
    @assert length(path_ids) == n
    @assert length(nonces) == n
    @assert size(output, 2) == 3
    
    # Use KA kernel for parallel color generation
    kernel! = probe_color_kernel!(backend, workgroup)
    kernel!(output, connection_id, path_ids, nonces, seed, ndrange=n)
    
    KernelAbstractions.synchronize(backend)
    return output
end

@kernel function probe_color_kernel!(
    output,
    connection_id::UInt64,
    path_ids,
    nonces,
    seed::UInt64
)
    i = @index(Global)
    
    # Mix all identifiers
    path_id = path_ids[i]
    nonce = nonces[i]
    probe_seed = splitmix64(seed ⊻ connection_id ⊻ UInt64(path_id) ⊻ nonce)
    
    r, g, b = hash_color(probe_seed, seed)
    
    output[i, 1] = r
    output[i, 2] = g
    output[i, 3] = b
end

"""
    parallel_probe_colors(n, connection_id, seed=GAY_SEED) -> Matrix{Float32}

Generate n probe colors for a connection in parallel.
Simulates multipath probing with random path IDs and nonces.
"""
function parallel_probe_colors(n::Integer, connection_id::UInt64, seed::UInt64=GAY_SEED)
    output = zeros(Float32, n, 3)
    path_ids = rand(UInt32(0):UInt32(7), n)  # Up to 8 paths
    nonces = rand(UInt64, n)
    
    ka_probe_colors!(output, connection_id, path_ids, nonces, seed)
    return output
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

"""
    visualize_probes(conn; width=4) -> String

Render ANSI-colored timeline of path probes.
"""
function visualize_probes(conn::QUICConnection; width::Int=4)
    buf = IOBuffer()
    
    println(buf, "╔═══════════════════════════════════════════════════════════════╗")
    println(buf, "║  QUIC Connection $(conn.connection_id) - Path Probes          ║")
    println(buf, "╚═══════════════════════════════════════════════════════════════╝")
    println(buf)
    
    # Connection color
    c = conn.color
    r, g, b = round(Int, c.r*255), round(Int, c.g*255), round(Int, c.b*255)
    print(buf, "  Connection: \e[48;2;$(r);$(g);$(b)m", " "^width, "\e[0m")
    println(buf, " #", string(conn.connection_id, base=16, pad=16))
    println(buf)
    
    # Paths
    println(buf, "  Paths:")
    for (path_id, path) in sort(collect(conn.paths), by=first)
        c = path.color
        r, g, b = round(Int, c.r*255), round(Int, c.g*255), round(Int, c.b*255)
        status = path.validated ? "✓" : "?"
        print(buf, "    [$path_id] \e[48;2;$(r);$(g);$(b)m", " "^width, "\e[0m")
        println(buf, " $(status) RTT=$(round(path.rtt_us, digits=1))μs sent=$(path.probes_sent) recv=$(path.probes_received)")
    end
    println(buf)
    
    # Probe timeline
    if !isempty(conn.probe_history)
        println(buf, "  Probe Timeline:")
        for probe in conn.probe_history
            c = probe.color
            r, g, b = round(Int, c.r*255), round(Int, c.g*255), round(Int, c.b*255)
            type_str = probe.is_challenge ? "CHALLENGE" : "RESPONSE "
            nonce_hex = bytes2hex(collect(probe.nonce))[1:8]
            print(buf, "    \e[48;2;$(r);$(g);$(b)m", " "^2, "\e[0m")
            println(buf, " $type_str path=$(probe.path_id) nonce=$nonce_hex...")
        end
    end
    
    return String(take!(buf))
end

"""
    probe_timeline(conn) -> Vector{NamedTuple}

Get structured timeline of probes for analysis.
"""
function probe_timeline(conn::QUICConnection)
    [(
        timestamp_ns = p.timestamp_ns,
        type = p.is_challenge ? :challenge : :response,
        path_id = p.path_id,
        nonce = p.nonce,
        color = p.color
    ) for p in conn.probe_history]
end

# ═══════════════════════════════════════════════════════════════════════════
# Benchmarking
# ═══════════════════════════════════════════════════════════════════════════

"""
    benchmark_quic_probes(; n=10000, paths=4) -> NamedTuple

Benchmark QUIC path probe color generation.
"""
function benchmark_quic_probes(; n::Int=10000, paths::Int=4)
    connection_id = rand(UInt64)
    seed = GAY_SEED
    
    # Single probe coloring
    start = time_ns()
    for i in 1:n
        path_id = UInt32(i % paths)
        nonce = generate_nonce()
        _ = path_probe_color(nonce, connection_id, path_id, seed)
    end
    single_ns = (time_ns() - start) / n
    
    # Parallel probe coloring
    start = time_ns()
    _ = parallel_probe_colors(n, connection_id, seed)
    parallel_ns = (time_ns() - start) / n
    
    return (
        n = n,
        paths = paths,
        single_ns = single_ns,
        parallel_ns = parallel_ns,
        speedup = single_ns / parallel_ns,
        single_rate = 1e9 / single_ns,
        parallel_rate = 1e9 / parallel_ns
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# REPL Demo
# ═══════════════════════════════════════════════════════════════════════════

"""
    world_quic_paths(; n_paths=4, probes_per_path=3, seed=nothing)

Build composable QUIC path probe coloring state.
"""
function world_quic_paths(; n_paths::Int=4, probes_per_path::Int=3, seed::Union{Nothing, UInt64}=nothing)
    conn_seed = isnothing(seed) ? rand(UInt64) : seed
    conn = QUICConnection(conn_seed)

    for path_id in UInt32(0):UInt32(n_paths-1)
        add_path!(conn, path_id)
        for _ in 1:probes_per_path
            challenge = probe_challenge!(conn, path_id)
            response_time = time_ns()
            probe_response!(conn, path_id, challenge.nonce, response_time)
        end
        validate_path!(conn, path_id)
    end

    bench = benchmark_quic_probes()

    (
        connection = conn,
        n_paths = n_paths,
        probes_per_path = probes_per_path,
        benchmark = bench,
        seed = conn_seed,
    )
end
