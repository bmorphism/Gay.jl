# drand.jl - Distributed Randomness Beacon integration for Gay.jl
#
# Gay.jl is deterministic given a seed (SPI). But where does the seed come from?
# If you pick `gay_seed!(42)`, anyone can reproduce your sequence.
#
# drand (https://drand.love) provides:
# - Unpredictable: No one can know the value before it's published
# - Unbiasable: No single party can influence the output
# - Verifiable: Anyone can verify the randomness was produced correctly
# - Public: Available to everyone simultaneously
#
# Use case: When you need colors that NO ONE could have predicted in advance
# (lotteries, cryptographic commitments, unbiased selection, fair games)

module Drand

using HTTP
using JSON3
using Dates
using SHA

export DrandBeacon, DrandRound, fetch_latest, fetch_round, round_at_time
export drand_seed, to_seed
export verify_round, quicknet, mainnet, cross_verify
export schedule_seed, await_future_round, ScheduledSeed

# ═══════════════════════════════════════════════════════════════════════════
# Beacon Configuration
# ═══════════════════════════════════════════════════════════════════════════

"""
    DrandBeacon

Configuration for a drand randomness beacon network.

# Fields
- `name`: Human-readable name
- `urls`: HTTP API endpoints (multiple for redundancy)
- `chain_hash`: Unique identifier for the chain (root of trust)
- `public_key`: BLS public key for signature verification
- `period`: Seconds between rounds
- `genesis_time`: Unix timestamp of round 1
"""
struct DrandBeacon
    name::String
    urls::Vector{String}
    chain_hash::String
    public_key::String
    period::Int
    genesis_time::Int
end

"""
League of Entropy mainnet (default chain) - 30 second rounds, chained mode.
"""
const MAINNET = DrandBeacon(
    "mainnet",
    [
        "https://api.drand.sh",
        "https://api2.drand.sh", 
        "https://api3.drand.sh",
        "https://drand.cloudflare.com"
    ],
    "8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce",
    "868f005eb8e6e4ca0a47c8a77ceaa5309a47978a7c71bc5cce96366b5d7a569937c529eeda66c7293784a9402801af31",
    30,
    1595431050
)

"""
League of Entropy quicknet - 3 second rounds, unchained mode.
Faster but different cryptographic properties.
"""
const QUICKNET = DrandBeacon(
    "quicknet",
    [
        "https://api.drand.sh",
        "https://api2.drand.sh",
        "https://api3.drand.sh",
        "https://drand.cloudflare.com"
    ],
    "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971",
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a",
    3,
    1692803367
)

# Convenience aliases
mainnet() = MAINNET
quicknet() = QUICKNET

# ═══════════════════════════════════════════════════════════════════════════
# Round Data
# ═══════════════════════════════════════════════════════════════════════════

"""
    DrandRound

A single round of randomness from a drand beacon.

# Fields
- `round`: Monotonically increasing round number
- `randomness`: 32-byte SHA-256 hash (the actual random value)
- `signature`: BLS threshold signature (for verification)
- `previous_signature`: Previous round's signature (for chained beacons)
- `beacon`: Which beacon this came from
"""
struct DrandRound
    round::Int
    randomness::Vector{UInt8}
    signature::Vector{UInt8}
    previous_signature::Union{Vector{UInt8}, Nothing}
    beacon::DrandBeacon
end

"""
Convert a DrandRound's randomness to a UInt64 seed for Gay.jl.
Uses first 8 bytes of the 32-byte randomness.
"""
function to_seed(dr::DrandRound)::UInt64
    reinterpret(UInt64, dr.randomness[1:8])[1]
end

# ═══════════════════════════════════════════════════════════════════════════
# HTTP API
# ═══════════════════════════════════════════════════════════════════════════

"""
    fetch_latest(beacon::DrandBeacon=MAINNET) -> DrandRound

Fetch the latest randomness round from the beacon.
Tries multiple endpoints for redundancy.
"""
function fetch_latest(beacon::DrandBeacon=MAINNET)
    for url in beacon.urls
        try
            endpoint = "$(url)/$(beacon.chain_hash)/public/latest"
            response = HTTP.get(endpoint; connect_timeout=5, readtimeout=10)
            return parse_round(String(response.body), beacon)
        catch e
            @debug "Failed to fetch from $url: $e"
            continue
        end
    end
    error("Failed to fetch from all drand endpoints")
end

"""
    fetch_round(round::Int, beacon::DrandBeacon=MAINNET) -> DrandRound

Fetch a specific historical round.
"""
function fetch_round(round::Int, beacon::DrandBeacon=MAINNET)
    for url in beacon.urls
        try
            endpoint = "$(url)/$(beacon.chain_hash)/public/$(round)"
            response = HTTP.get(endpoint; connect_timeout=5, readtimeout=10)
            return parse_round(String(response.body), beacon)
        catch e
            @debug "Failed to fetch round $round from $url: $e"
            continue
        end
    end
    error("Failed to fetch round $round from all drand endpoints")
end

"""
Parse JSON response into DrandRound.
"""
function parse_round(json_str::String, beacon::DrandBeacon)
    data = JSON3.read(json_str)
    
    randomness = hex2bytes(String(data.randomness))
    signature = hex2bytes(String(data.signature))
    prev_sig = haskey(data, :previous_signature) ? 
               hex2bytes(String(data.previous_signature)) : nothing
    
    DrandRound(
        Int(data.round),
        randomness,
        signature,
        prev_sig,
        beacon
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Time-based Round Calculation
# ═══════════════════════════════════════════════════════════════════════════

"""
    round_at_time(t::DateTime, beacon::DrandBeacon=MAINNET) -> Int

Calculate which round number corresponds to a given time.
"""
function round_at_time(t::DateTime, beacon::DrandBeacon=MAINNET)
    unix_t = round(Int, datetime2unix(t))
    return round_at_time(unix_t, beacon)
end

function round_at_time(unix_time::Int, beacon::DrandBeacon=MAINNET)
    if unix_time < beacon.genesis_time
        error("Time is before beacon genesis")
    end
    return div(unix_time - beacon.genesis_time, beacon.period) + 1
end

"""
    time_of_round(round::Int, beacon::DrandBeacon=MAINNET) -> DateTime

Calculate when a specific round will be (or was) published.
"""
function time_of_round(round::Int, beacon::DrandBeacon=MAINNET)
    unix_time = beacon.genesis_time + (round - 1) * beacon.period
    return unix2datetime(unix_time)
end

"""
    next_round_time(beacon::DrandBeacon=MAINNET) -> (round, DateTime)

Get the next round number and when it will be available.
"""
function next_round_time(beacon::DrandBeacon=MAINNET)
    now_unix = round(Int, datetime2unix(now(UTC)))
    current_round = round_at_time(now_unix, beacon)
    next_round = current_round + 1
    return (next_round, time_of_round(next_round, beacon))
end

# ═══════════════════════════════════════════════════════════════════════════
# Gay.jl Integration Helpers
# ═══════════════════════════════════════════════════════════════════════════

"""
    drand_seed(beacon::DrandBeacon=MAINNET) -> (UInt64, DrandRound)

Fetch the latest drand round and convert it to a seed.
Returns both the seed and the round info.

Note: This function does NOT set the global RNG. Use `gay_seed!` from Gay.jl for that.
"""
function drand_seed(beacon::DrandBeacon=MAINNET)
    dr = fetch_latest(beacon)
    return (to_seed(dr), dr)
end

"""
    drand_seed(round::Int, beacon::DrandBeacon=MAINNET) -> (UInt64, DrandRound)

Fetch a specific round and convert it to a seed.
"""
function drand_seed(round::Int, beacon::DrandBeacon=MAINNET)
    dr = fetch_round(round, beacon)
    return (to_seed(dr), dr)
end

# ═══════════════════════════════════════════════════════════════════════════
# Future Round Scheduling (Commitment Schemes)
# ═══════════════════════════════════════════════════════════════════════════

"""
    ScheduledSeed

A commitment to use a future drand round as a seed.
The round doesn't exist yet, so no one can know the seed in advance.

# Commitment Scheme Pattern
1. Announce: "We will use drand round 12345678 for the lottery"
2. Wait: Round hasn't happened yet, seed is unknowable
3. Reveal: Round is published, everyone can verify the seed
4. Use: gay_seed!(to_seed(fetch_round(12345678)))
"""
struct ScheduledSeed
    round::Int
    beacon::DrandBeacon
    scheduled_time::DateTime
    description::String
end

"""
    schedule_seed(; delay_seconds=60, beacon=MAINNET, description="") -> ScheduledSeed

Schedule a seed from a future drand round.
The round is chosen to occur `delay_seconds` in the future.

# Example: Fair Lottery
```julia
# Step 1: Commit to a future round (announce this publicly)
scheduled = schedule_seed(delay_seconds=300, description="Color lottery 2024-01")
println("Lottery will use drand round \$(scheduled.round)")
println("Available at: \$(scheduled.scheduled_time)")

# Step 2: Wait for the round to be published
# ... time passes ...

# Step 3: Use the seed (everyone can verify)
dr = await_future_round(scheduled)
gay_seed!(to_seed(dr))
winner_color = next_color()
```
"""
function schedule_seed(; 
    delay_seconds::Int=60, 
    beacon::DrandBeacon=MAINNET,
    description::String=""
)
    future_time = now(UTC) + Second(delay_seconds)
    future_round = round_at_time(future_time, beacon) + 1  # +1 to ensure it's in the future
    actual_time = time_of_round(future_round, beacon)
    
    ScheduledSeed(future_round, beacon, actual_time, description)
end

"""
    await_future_round(scheduled::ScheduledSeed; timeout_seconds=300) -> DrandRound

Wait for a scheduled future round to become available.
Polls the beacon until the round is published or timeout.
Uses exponential backoff for polling.
"""
function await_future_round(scheduled::ScheduledSeed; timeout_seconds::Int=300)
    start_time = now(UTC)
    poll_interval = 1.0
    
    while true
        # Check timeout
        elapsed = Dates.value(now(UTC) - start_time) / 1000
        if elapsed > timeout_seconds
            error("Timeout waiting for drand round $(scheduled.round)")
        end
        
        # Check if round is available
        try
            return fetch_round(scheduled.round, scheduled.beacon)
        catch
            # Round not yet available
            # Calculate time until scheduled round
            remaining = scheduled.scheduled_time - now(UTC)
            remaining_sec = Dates.value(remaining) / 1000
            
            if remaining_sec > 5
                # Sleep most of the way there, but wake up early
                sleep(max(1.0, remaining_sec - 2.0))
                poll_interval = 0.5  # Fast poll when close
            else
                # Exponential backoff up to period/2
                sleep(poll_interval)
                poll_interval = min(poll_interval * 1.5, scheduled.beacon.period / 2.0)
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Verification (Stub - Full BLS verification requires additional dependencies)
# ═══════════════════════════════════════════════════════════════════════════

"""
    verify_round(dr::DrandRound) -> Bool

Verify the cryptographic signature of a drand round.

Note: Full BLS signature verification requires pairing-friendly curve libraries.
This stub checks basic structural validity. For production use, implement
full verification using a BLS12-381 library.
"""
function verify_round(dr::DrandRound)
    # Basic structural checks
    length(dr.randomness) == 32 || return false
    length(dr.signature) > 0 || return false
    dr.round > 0 || return false
    
    # Verify randomness = SHA256(signature)
    # (This is how drand derives randomness from the BLS signature)
    expected_randomness = sha256(dr.signature)
    
    if dr.randomness != expected_randomness
        @warn "Randomness does not match SHA256(signature) - possible tampering"
        return false
    end
    
    # Full BLS signature verification would go here
    # Requires: BLS12-381 pairing, public key deserialization, etc.
    @warn "Full BLS signature verification not implemented - verify against multiple endpoints"
    
    return true
end

"""
    cross_verify(round::Int, beacon::DrandBeacon=MAINNET) -> Bool

Fetch a round from multiple endpoints and verify they all agree.
Provides practical verification without BLS crypto.
"""
function cross_verify(round::Int, beacon::DrandBeacon=MAINNET)
    results = DrandRound[]
    
    for url in beacon.urls
        try
            endpoint = "$(url)/$(beacon.chain_hash)/public/$(round)"
            response = HTTP.get(endpoint; connect_timeout=5, readtimeout=10)
            push!(results, parse_round(String(response.body), beacon))
        catch
            continue
        end
    end
    
    if length(results) < 2
        @warn "Could only fetch from $(length(results)) endpoint(s)"
        return length(results) == 1
    end
    
    # All results should have identical randomness
    reference = results[1].randomness
    all_match = all(r -> r.randomness == reference, results)
    
    if !all_match
        @error "CRITICAL: Endpoints returned different randomness values!"
    end
    
    return all_match
end


end # module Drand
