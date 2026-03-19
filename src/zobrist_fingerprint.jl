module ZobristFingerprint

export ZobristTable, zobrist_hash, incremental_update, verify_zobrist
export ColorZobrist, init_table!, world_zobrist_fingerprint, set_color!, remove_color!

const TABLE_SIZE = 256  # Index positions
const COLOR_BUCKETS = 24  # Hue buckets (15° each)

# Zobrist table: random 64-bit values for each (index, color_bucket) pair
struct ZobristTable
    table::Matrix{UInt64}  # TABLE_SIZE × COLOR_BUCKETS
    seed::UInt64
end

function ZobristTable(seed::UInt64=UInt64(1069))
    table = Matrix{UInt64}(undef, TABLE_SIZE, COLOR_BUCKETS)
    state = seed
    for i in 1:TABLE_SIZE
        for j in 1:COLOR_BUCKETS
            state = sm64(state)
            table[i, j] = state
        end
    end
    ZobristTable(table, seed)
end

@inline sm64(s::UInt64) = begin
    z = s + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

# Map hue [0,360) to bucket [1,24]
hue_to_bucket(hue::Float64) = clamp(floor(Int, hue / 15) + 1, 1, COLOR_BUCKETS)

# Full hash from scratch
function zobrist_hash(zt::ZobristTable, colors::Vector{<:Tuple})
    fp = UInt64(0)
    for (idx, hue) in colors
        bucket = hue_to_bucket(hue)
        fp ⊻= zt.table[mod1(idx, TABLE_SIZE), bucket]
    end
    fp
end

# Incremental update: O(1) add/remove
# XOR is self-inverse: adding same (idx, color) twice cancels
function incremental_update(fp::UInt64, zt::ZobristTable, idx::Int, hue::Float64)
    bucket = hue_to_bucket(hue)
    fp ⊻ zt.table[mod1(idx, TABLE_SIZE), bucket]
end

# Verify incremental matches full recompute
function verify_zobrist(zt::ZobristTable, colors::Vector{<:Tuple})
    full = zobrist_hash(zt, colors)
    incremental = UInt64(0)
    for (idx, hue) in colors
        incremental = incremental_update(incremental, zt, idx, hue)
    end
    full == incremental
end

# Higher-level: track colors and fingerprint together
mutable struct ColorZobrist
    table::ZobristTable
    fingerprint::UInt64
    colors::Dict{Int, Float64}  # idx => hue
end

ColorZobrist(seed::UInt64=UInt64(1069)) = ColorZobrist(ZobristTable(seed), UInt64(0), Dict{Int,Float64}())

function set_color!(cz::ColorZobrist, idx::Int, hue::Float64)
    if haskey(cz.colors, idx)
        old_hue = cz.colors[idx]
        cz.fingerprint = incremental_update(cz.fingerprint, cz.table, idx, old_hue)
    end
    cz.colors[idx] = hue
    cz.fingerprint = incremental_update(cz.fingerprint, cz.table, idx, hue)
    cz
end

function remove_color!(cz::ColorZobrist, idx::Int)
    if haskey(cz.colors, idx)
        hue = cz.colors[idx]
        cz.fingerprint = incremental_update(cz.fingerprint, cz.table, idx, hue)
        delete!(cz.colors, idx)
    end
    cz
end

function init_table!(cz::ColorZobrist, colors::Vector{<:Tuple})
    empty!(cz.colors)
    cz.fingerprint = UInt64(0)
    for (idx, hue) in colors
        set_color!(cz, idx, hue)
    end
    cz
end

function world_zobrist_fingerprint()
    cz = ColorZobrist()
    
    set_color!(cz, 1, 0.0)    # Red
    set_color!(cz, 2, 120.0)  # Green
    set_color!(cz, 3, 240.0)  # Blue
    
    fp_after_rgb = cz.fingerprint
    
    set_color!(cz, 4, 60.0)   # Yellow
    fp_after_yellow = cz.fingerprint
    
    remove_color!(cz, 4)
    fp_after_remove = cz.fingerprint
    
    incremental_ok = fp_after_rgb == fp_after_remove
    
    full_check = verify_zobrist(cz.table, [(idx, hue) for (idx, hue) in cz.colors])
    
    (fingerprint=cz.fingerprint, incremental_ok=incremental_ok, verified=full_check)
end

end
