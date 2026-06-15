# GAY LATTICE EXPANSION: 3x3x3 → 23x23x23 Bucket Stability with Metal.jl
# ============================================================================
#
# "Launch 3 GayRandomWalks for each of the 3 most relevant groupings of 3x3x3
#  repos in a way that can anticipate 23x23x23 expansion with bucket stability
#  giving us -/0/+ direction based on color affinities."
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  LATTICE STRUCTURE                                                          │
# │                                                                             │
# │  3x3x3 = 27 cells per world (initial)                                       │
# │  23x23x23 = 12,167 cells per world (expansion target)                      │
# │  3 worlds × 12,167 = 36,501 total cells at maximum expansion               │
# │                                                                             │
# │  BUCKET STABILITY DIRECTIONS:                                               │
# │    - (minus):  Contracting, fewer colors, focused exploration              │
# │    0 (zero):   Stable, balanced, equilibrium state                         │
# │    + (plus):   Expanding, more colors, divergent exploration               │
# │                                                                             │
# │  Direction determined by color affinity:                                    │
# │    affinity < -threshold  →  -                                              │
# │    -threshold ≤ affinity ≤ +threshold  →  0                                │
# │    affinity > +threshold  →  +                                              │
# │                                                                             │
# │  WORLD ASSIGNMENT (top groupings by repo count):                            │
# │    Zahn (🔴):   bmorphism (390), hdresearch (64), baez (20+)               │
# │    Jules (🟢):  plurigrid (557), InverterNetwork (86), kubeflow (47)       │
# │    Fabriz (🔵): TeglonLabs (63), Tritwies (13), tanchain (6)               │
# └─────────────────────────────────────────────────────────────────────────────┘

module GayLatticeExpansion

using KernelAbstractions
using Base.Threads: @threads, @spawn, nthreads
using Printf

export
    # Core Types
    LatticeCell, LatticeBucket, LatticeWorld, GayLattice,
    BucketDirection, MINUS, ZERO, PLUS,
    
    # Worlds
    GayWorld, ZAHN, JULES, FABRIZ, WORLD_EMOJI, assign_world,
    
    # 3x3x3 Groupings
    Grouping3x3x3, create_initial_groupings,
    
    # Expansion
    expand_to_23x23x23!, bucket_stability_direction,
    
    # Metal.jl Integration
    MetalColorKernel, metal_parallel_next_color!, metal_available,
    
    # Random Walks
    GayRandomWalkLattice, launch_walk!, step_walk!, complete_walk!,
    parallel_walks!, world_fingerprint,
    
    # Affinity
    color_affinity, affinity_direction, affinity_matrix,
    
    # Demo
    world_lattice_expansion, launch_3x3_walks

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(0x6761795f636f6c6f)  # "gay_colo"
const ZAHN_SEED = UInt64(0x5A41484E)
const JULES_SEED = UInt64(0x4A554C4553)
const FABRIZ_SEED = UInt64(0x464142524947)

const LATTICE_3 = 3
const LATTICE_23 = 23
const AFFINITY_THRESHOLD = 0.33

# ═══════════════════════════════════════════════════════════════════════════════
# SPLITMIX64 (Core PRNG)
# ═══════════════════════════════════════════════════════════════════════════════

@inline function splitmix64(state::UInt64)::Tuple{UInt64, UInt64}
    z = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    (z ⊻ (z >> 31), (state + 1) & 0xFFFFFFFFFFFFFFFF)
end

@inline function next_color(seed::UInt64)::Tuple{UInt64, Tuple{Float64,Float64,Float64}}
    r, s1 = splitmix64(seed)
    g, s2 = splitmix64(s1)
    b, s3 = splitmix64(s2)
    (s3, ((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0))
end

# ═══════════════════════════════════════════════════════════════════════════════
# BUCKET DIRECTION
# ═══════════════════════════════════════════════════════════════════════════════

@enum BucketDirection begin
    MINUS = -1  # Contracting
    ZERO = 0    # Stable
    PLUS = 1    # Expanding
end

const DIRECTION_SYMBOLS = Dict(MINUS => "-", ZERO => "0", PLUS => "+")
const DIRECTION_COLORS = Dict(
    MINUS => (0.8, 0.2, 0.2),   # Red-ish
    ZERO => (0.5, 0.5, 0.5),    # Gray
    PLUS => (0.2, 0.8, 0.2)     # Green-ish
)

function affinity_direction(affinity::Float64; threshold::Float64=AFFINITY_THRESHOLD)::BucketDirection
    if affinity < -threshold
        MINUS
    elseif affinity > threshold
        PLUS
    else
        ZERO
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAY WORLD ENUM
# ═══════════════════════════════════════════════════════════════════════════════

@enum GayWorld begin
    ZAHN = 1    # 🔴 A-H, order matters
    JULES = 2   # 🟢 I-P, order agnostic
    FABRIZ = 3  # 🔵 Q-Z, order entangled
end

const WORLD_EMOJI = Dict(ZAHN => "🔴", JULES => "🟢", FABRIZ => "🔵")
const WORLD_SEED = Dict(ZAHN => ZAHN_SEED, JULES => JULES_SEED, FABRIZ => FABRIZ_SEED)
const WORLD_OPERATOR = Dict(ZAHN => :⊗, JULES => :⊕, FABRIZ => :⊛)

function assign_world(name::String)::GayWorld
    first_char = uppercase(first(name))
    if 'A' <= first_char <= 'H'
        ZAHN
    elseif 'I' <= first_char <= 'P'
        JULES
    else
        FABRIZ
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# LATTICE CELL
# ═══════════════════════════════════════════════════════════════════════════════

struct LatticeCell
    i::Int
    j::Int
    k::Int
    seed::UInt64
    color::Tuple{Float64,Float64,Float64}
    repos::Vector{String}
    direction::BucketDirection
end

function LatticeCell(i::Int, j::Int, k::Int; 
                     base_seed::UInt64=GAY_SEED, 
                     repos::Vector{String}=String[])
    cell_seed = base_seed ⊻ UInt64(i * 529 + j * 23 + k)
    _, color = next_color(cell_seed)
    
    affinity = (color[1] - color[3]) / max(0.01, color[1] + color[3])
    direction = affinity_direction(affinity)
    
    LatticeCell(i, j, k, cell_seed, color, repos, direction)
end

function cell_fingerprint(cell::LatticeCell)::UInt64
    fp = cell.seed
    for repo in cell.repos
        fp = fp ⊻ hash(repo)
    end
    fp
end

# ═══════════════════════════════════════════════════════════════════════════════
# LATTICE BUCKET (Collection of cells)
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct LatticeBucket
    id::String
    cells::Vector{LatticeCell}
    size::Int  # Current size (3 or 23)
    direction::BucketDirection
    fingerprint::UInt64
    stable::Bool
end

function LatticeBucket(id::String, cells::Vector{LatticeCell})
    size = isqrt(length(cells))
    
    fps = UInt64[]
    for cell in cells
        push!(fps, cell_fingerprint(cell))
    end
    fingerprint = reduce(⊻, fps; init=GAY_SEED)
    
    directions = [cell.direction for cell in cells]
    dir_counts = Dict(MINUS => 0, ZERO => 0, PLUS => 0)
    for d in directions
        dir_counts[d] += 1
    end
    dominant = argmax(dir_counts)
    
    stable = dir_counts[dominant] > length(cells) * 0.6
    
    LatticeBucket(id, cells, size, dominant, fingerprint, stable)
end

# ═══════════════════════════════════════════════════════════════════════════════
# LATTICE WORLD (27 or 12167 cells per world)
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct LatticeWorld
    world::GayWorld
    seed::UInt64
    size::Int  # 3 or 23
    cells::Array{LatticeCell, 3}
    buckets::Dict{Tuple{Int,Int,Int}, LatticeBucket}
    fingerprint::UInt64
    orgs::Vector{String}
end

function LatticeWorld(world::GayWorld, size::Int, orgs::Vector{String})
    seed = WORLD_SEED[world]
    cells = Array{LatticeCell, 3}(undef, size, size, size)
    
    org_index = 1
    for i in 1:size, j in 1:size, k in 1:size
        cell_repos = String[]
        if org_index <= length(orgs)
            push!(cell_repos, orgs[org_index])
            org_index += 1
        end
        cells[i, j, k] = LatticeCell(i, j, k; base_seed=seed, repos=cell_repos)
    end
    
    buckets = Dict{Tuple{Int,Int,Int}, LatticeBucket}()
    fingerprint = seed
    for cell in cells
        fingerprint = fingerprint ⊻ cell_fingerprint(cell)
    end
    
    LatticeWorld(world, seed, size, cells, buckets, fingerprint, orgs)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAY LATTICE (3 worlds)
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct GayLattice
    worlds::Dict{GayWorld, LatticeWorld}
    size::Int
    total_cells::Int
    fingerprint::UInt64
    step_count::Int
end

function GayLattice(size::Int, world_orgs::Dict{GayWorld, Vector{String}})
    worlds = Dict{GayWorld, LatticeWorld}()
    
    for world in [ZAHN, JULES, FABRIZ]
        orgs = get(world_orgs, world, String[])
        worlds[world] = LatticeWorld(world, size, orgs)
    end
    
    total_cells = size^3 * 3
    fingerprint = reduce(⊻, [w.fingerprint for w in values(worlds)]; init=GAY_SEED)
    
    GayLattice(worlds, size, total_cells, fingerprint, 0)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3x3x3 GROUPINGS (Initial state)
# ═══════════════════════════════════════════════════════════════════════════════

struct Grouping3x3x3
    world::GayWorld
    orgs::Vector{String}
    top_3::Vector{String}  # Top 3 orgs by repo count
    repo_counts::Dict{String, Int}
    lattice::LatticeWorld
end

function create_initial_groupings()
    zahn_orgs = Dict(
        "bmorphism" => 390,
        "hdresearch" => 64,
        "baez" => 22,
        "awesomeDAO" => 15,
        "DMLAI" => 12,
        "A-F-X-M" => 8,
        "Continuum-Corporation" => 6,
        "a-tractor" => 4,
    )
    
    jules_orgs = Dict(
        "plurigrid" => 557,
        "InverterNetwork" => 86,
        "kubeflow" => 47,
        "MintedMosaic" => 12,
        "ogb-interchain" => 8,
        "m8astable" => 5,
    )
    
    fabriz_orgs = Dict(
        "TeglonLabs" => 63,
        "Tritwies" => 13,
        "tanchain" => 6,
        "TheNumarati" => 3,
        "the-interlace" => 2,
    )
    
    groupings = Grouping3x3x3[]
    
    for (world, orgs) in [(ZAHN, zahn_orgs), (JULES, jules_orgs), (FABRIZ, fabriz_orgs)]
        sorted_orgs = sort(collect(keys(orgs)); by=k -> orgs[k], rev=true)
        top_3 = sorted_orgs[1:min(3, length(sorted_orgs))]
        lattice = LatticeWorld(world, LATTICE_3, sorted_orgs)
        push!(groupings, Grouping3x3x3(world, sorted_orgs, top_3, orgs, lattice))
    end
    
    groupings
end

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR AFFINITY
# ═══════════════════════════════════════════════════════════════════════════════

function color_affinity(c1::Tuple{Float64,Float64,Float64}, 
                        c2::Tuple{Float64,Float64,Float64})::Float64
    dr = c2[1] - c1[1]
    dg = c2[2] - c1[2]
    db = c2[3] - c1[3]
    
    sqrt(dr^2 + dg^2 + db^2) / sqrt(3.0)
end

function affinity_matrix(cells::Array{LatticeCell, 3})::Array{Float64, 3}
    size = size(cells, 1)
    affinities = zeros(Float64, size, size, size)
    
    center_color = cells[div(size+1,2), div(size+1,2), div(size+1,2)].color
    
    for i in 1:size, j in 1:size, k in 1:size
        affinities[i,j,k] = color_affinity(center_color, cells[i,j,k].color)
    end
    
    affinities
end

# ═══════════════════════════════════════════════════════════════════════════════
# METAL.jl INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

const METAL_AVAILABLE = Ref{Union{Bool, Nothing}}(nothing)

function metal_available()::Bool
    if METAL_AVAILABLE[] !== nothing
        return METAL_AVAILABLE[]
    end
    
    available = try
        if Base.find_package("Metal") !== nothing
            Metal = Base.require(Base.PkgId(Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal"))
            Metal.functional()
        else
            false
        end
    catch
        false
    end
    
    METAL_AVAILABLE[] = available
    available
end

@kernel function _lattice_color_kernel!(colors, @Const(base_seed::UInt64), @Const(size::Int))
    idx = @index(Global)
    
    i = ((idx - 1) ÷ (size * size)) % size + 1
    j = ((idx - 1) ÷ size) % size + 1
    k = (idx - 1) % size + 1
    
    cell_seed = base_seed ⊻ UInt64(i * 529 + j * 23 + k)
    
    z = (cell_seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    r_val = z ⊻ (z >> 31)
    
    z = (r_val + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    g_val = z ⊻ (z >> 31)
    
    z = (g_val + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    b_val = z ⊻ (z >> 31)
    
    colors[idx, 1] = Float32(r_val >> 56) / 255.0f0
    colors[idx, 2] = Float32(g_val >> 56) / 255.0f0
    colors[idx, 3] = Float32(b_val >> 56) / 255.0f0
end

function metal_parallel_next_color!(world::LatticeWorld)
    n = world.size^3
    
    if metal_available()
        Metal = Base.require(Base.PkgId(Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal"))
        backend = Metal.MetalBackend()
        colors = Metal.MtlArray{Float32}(undef, n, 3)
        kernel! = _lattice_color_kernel!(backend)
        kernel!(colors, world.seed, world.size; ndrange=n)
        KernelAbstractions.synchronize(backend)
        return Array(colors)
    else
        colors = zeros(Float32, n, 3)
        @threads for idx in 1:n
            i = ((idx - 1) ÷ (world.size * world.size)) % world.size + 1
            j = ((idx - 1) ÷ world.size) % world.size + 1
            k = (idx - 1) % world.size + 1
            
            cell_seed = world.seed ⊻ UInt64(i * 529 + j * 23 + k)
            _, color = next_color(cell_seed)
            colors[idx, 1] = Float32(color[1])
            colors[idx, 2] = Float32(color[2])
            colors[idx, 3] = Float32(color[3])
        end
        return colors
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAY RANDOM WALK (Lattice version)
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct GayRandomWalkLattice
    id::String
    lattice::GayLattice
    position::Dict{GayWorld, Tuple{Int,Int,Int}}  # Current position per world
    history::Dict{GayWorld, Vector{Tuple{Int,Int,Int}}}
    colors_seen::Dict{GayWorld, Vector{Tuple{Float64,Float64,Float64}}}
    step_count::Int
    fingerprint::UInt64
    started_at::Float64
    completed::Bool
end

function GayRandomWalkLattice(id::String, lattice::GayLattice)
    position = Dict{GayWorld, Tuple{Int,Int,Int}}()
    history = Dict{GayWorld, Vector{Tuple{Int,Int,Int}}}()
    colors_seen = Dict{GayWorld, Vector{Tuple{Float64,Float64,Float64}}}()
    
    center = div(lattice.size + 1, 2)
    for world in [ZAHN, JULES, FABRIZ]
        position[world] = (center, center, center)
        history[world] = [(center, center, center)]
        colors_seen[world] = [lattice.worlds[world].cells[center, center, center].color]
    end
    
    GayRandomWalkLattice(id, lattice, position, history, colors_seen, 0, GAY_SEED, time(), false)
end

function step_walk!(walk::GayRandomWalkLattice, world::GayWorld)
    lw = walk.lattice.worlds[world]
    pos = walk.position[world]
    
    step_seed = lw.seed ⊻ UInt64(walk.step_count * 1069)
    val, _ = splitmix64(step_seed)
    
    di = Int((val % 3)) - 1
    dj = Int(((val >> 16) % 3)) - 1
    dk = Int(((val >> 32) % 3)) - 1
    
    new_i = clamp(pos[1] + di, 1, lw.size)
    new_j = clamp(pos[2] + dj, 1, lw.size)
    new_k = clamp(pos[3] + dk, 1, lw.size)
    
    new_pos = (new_i, new_j, new_k)
    walk.position[world] = new_pos
    push!(walk.history[world], new_pos)
    
    cell = lw.cells[new_i, new_j, new_k]
    push!(walk.colors_seen[world], cell.color)
    
    walk.fingerprint = walk.fingerprint ⊻ cell_fingerprint(cell)
    
    new_pos
end

function launch_walk!(walk::GayRandomWalkLattice, steps::Int; parallel::Bool=true)
    walk.step_count = 0
    
    if parallel
        @sync for world in [ZAHN, JULES, FABRIZ]
            @spawn begin
                for _ in 1:steps
                    step_walk!(walk, world)
                end
            end
        end
    else
        for _ in 1:steps
            for world in [ZAHN, JULES, FABRIZ]
                step_walk!(walk, world)
            end
        end
    end
    
    walk.step_count = steps
    walk
end

function complete_walk!(walk::GayRandomWalkLattice)
    walk.completed = true
    
    for world in [ZAHN, JULES, FABRIZ]
        for pos in walk.history[world]
            cell = walk.lattice.worlds[world].cells[pos...]
            walk.fingerprint = walk.fingerprint ⊻ cell_fingerprint(cell)
        end
    end
    
    walk
end

function world_fingerprint(walk::GayRandomWalkLattice, world::GayWorld)::UInt64
    fp = WORLD_SEED[world]
    for pos in walk.history[world]
        cell = walk.lattice.worlds[world].cells[pos...]
        fp = fp ⊻ cell_fingerprint(cell)
    end
    fp
end

# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL WALKS (3 walks across 3 worlds)
# ═══════════════════════════════════════════════════════════════════════════════

function parallel_walks!(lattice::GayLattice, n_walks::Int, steps_per_walk::Int)
    walks = [GayRandomWalkLattice("walk_$i", lattice) for i in 1:n_walks]
    
    @sync for walk in walks
        @spawn launch_walk!(walk, steps_per_walk; parallel=true)
    end
    
    for walk in walks
        complete_walk!(walk)
    end
    
    combined_fp = reduce(⊻, [w.fingerprint for w in walks]; init=GAY_SEED)
    
    (walks=walks, fingerprint=combined_fp)
end

# ═══════════════════════════════════════════════════════════════════════════════
# EXPANSION: 3x3x3 → 23x23x23
# ═══════════════════════════════════════════════════════════════════════════════

function expand_to_23x23x23!(lattice::GayLattice; use_metal::Bool=true)
    if lattice.size == LATTICE_23
        return lattice  # Already expanded
    end
    
    new_worlds = Dict{GayWorld, LatticeWorld}()
    
    for (world, lw) in lattice.worlds
        if use_metal && metal_available()
            colors = metal_parallel_next_color!(
                LatticeWorld(world, LATTICE_23, lw.orgs)
            )
        end
        
        new_worlds[world] = LatticeWorld(world, LATTICE_23, lw.orgs)
    end
    
    lattice.worlds = new_worlds
    lattice.size = LATTICE_23
    lattice.total_cells = LATTICE_23^3 * 3
    lattice.fingerprint = reduce(⊻, [w.fingerprint for w in values(new_worlds)]; init=GAY_SEED)
    
    lattice
end

function bucket_stability_direction(lattice::GayLattice)::Dict{GayWorld, Dict{Tuple{Int,Int,Int}, BucketDirection}}
    result = Dict{GayWorld, Dict{Tuple{Int,Int,Int}, BucketDirection}}()
    
    for (world, lw) in lattice.worlds
        world_dirs = Dict{Tuple{Int,Int,Int}, BucketDirection}()
        
        for i in 1:lw.size, j in 1:lw.size, k in 1:lw.size
            cell = lw.cells[i, j, k]
            world_dirs[(i, j, k)] = cell.direction
        end
        
        result[world] = world_dirs
    end
    
    result
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO: Launch 3x3 walks
# ═══════════════════════════════════════════════════════════════════════════════

function launch_3x3_walks()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  GAY LATTICE EXPANSION: 3x3x3 → 23x23x23 with Metal.jl Parallelism        ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # Create initial groupings
    println("─── Creating Initial 3x3x3 Groupings ───")
    println()
    
    groupings = create_initial_groupings()
    
    for g in groupings
        emoji = WORLD_EMOJI[g.world]
        println("  $emoji $(g.world):")
        println("     Top 3: $(join(g.top_3, ", "))")
        println("     Total orgs: $(length(g.orgs))")
        println()
    end
    
    # Create lattice
    world_orgs = Dict(g.world => g.orgs for g in groupings)
    lattice = GayLattice(LATTICE_3, world_orgs)
    
    println("─── Lattice Created ───")
    println("  Size: $(lattice.size)x$(lattice.size)x$(lattice.size) per world")
    println("  Total cells: $(lattice.total_cells)")
    println("  Initial fingerprint: 0x$(string(lattice.fingerprint, base=16))")
    println()
    
    # Launch 3 random walks
    println("─── Launching 3 GayRandomWalks (parallel) ───")
    println()
    
    t0 = time()
    result = parallel_walks!(lattice, 3, 100)  # 3 walks, 100 steps each
    duration = time() - t0
    
    println("  Completed in $(round(duration * 1000, digits=2))ms")
    println("  Combined fingerprint: 0x$(string(result.fingerprint, base=16))")
    println()
    
    for (i, walk) in enumerate(result.walks)
        println("  Walk $i:")
        for world in [ZAHN, JULES, FABRIZ]
            fp = world_fingerprint(walk, world)
            emoji = WORLD_EMOJI[world]
            println("    $emoji $(world): fp=0x$(string(fp, base=16, pad=8)[1:8])... steps=$(length(walk.history[world]))")
        end
    end
    println()
    
    # Bucket stability directions
    println("─── Bucket Stability Directions ───")
    println()
    
    directions = bucket_stability_direction(lattice)
    
    for world in [ZAHN, JULES, FABRIZ]
        emoji = WORLD_EMOJI[world]
        world_dirs = directions[world]
        
        minus_count = count(d -> d == MINUS, values(world_dirs))
        zero_count = count(d -> d == ZERO, values(world_dirs))
        plus_count = count(d -> d == PLUS, values(world_dirs))
        
        println("  $emoji $(world): - $(minus_count) | 0 $(zero_count) | + $(plus_count)")
    end
    println()
    
    # Metal.jl status
    println("─── Metal.jl Status ───")
    println("  Available: $(metal_available())")
    println("  Threads: $(nthreads())")
    println()
    
    # Return for further use
    (lattice=lattice, walks=result.walks, fingerprint=result.fingerprint, groupings=groupings)
end

function world_lattice_expansion()
    result = launch_3x3_walks()
    
    println("─── Ready for 23x23x23 Expansion ───")
    println()
    println("  To expand: expand_to_23x23x23!(result.lattice)")
    println("  This will create $(23^3 * 3) = 36,501 cells")
    println("  With Metal.jl: GPU-accelerated color generation")
    println("  Without Metal.jl: $(nthreads())-threaded CPU fallback")
    println()
    
    result
end

end # module
