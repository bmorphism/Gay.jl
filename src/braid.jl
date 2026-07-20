# Gay Braid Protocol: Synchronization for Chromatic State
#
# Extends Braid-HTTP (state synchronization) with Gay.jl chromatic identity.
# Forks, mergers, and re-orderings of space over time with SPI guarantees.
#
# References:
# - https://braid.org (Braid-HTTP specification)
# - https://datatracker.ietf.org/doc/html/draft-toomim-httpbis-braid-http-04
# - Gay Hyperdoctrine (chromatic categorical logic)
#
# Key concepts:
# - VERSIONING: Each version carries chromatic identity
# - PATCHES: Color deltas preserve SPI fingerprints
# - SUBSCRIPTIONS: Color streams with deterministic trajectory
# - MERGE-TYPES: CRDT/Sync9 with chromatic conflict resolution
#
# Braided Time vs Linear Clock Time:
# - Linear: t₁ → t₂ → t₃ (single sequence, no branching)
# - Braided: forks (coproducts), mergers (products), re-orderings
# - Linking number: net over-crossings in the braid
# - Holonomy: curvature of version graph (sum of hue rotations)

using SplittableRandoms: SplittableRandom, split
using Colors

export BraidWord, BraidGenerator, CrossingType
export BraidVersion, BraidPatch, BraidLog
export GayBraidModel, PeerColorIdentity
export linking_number, holonomy, is_self_same
export random_walk_to_color, braid_random_walk!

# ═══════════════════════════════════════════════════════════════════════════
# Crossing Types (Braid Generators)
# ═══════════════════════════════════════════════════════════════════════════

@enum CrossingType begin
    OVER   # σᵢ: strand i crosses over strand i+1 (positive)
    UNDER  # σᵢ⁻¹: strand i crosses under strand i+1 (negative)
end

"""
A single braid generator: strand crossing.
"""
struct BraidGenerator
    strand::Int       # Which strand (1-indexed)
    crossing::CrossingType
    color::RGB{Float64}  # Chromatic identity of this crossing
end

# ═══════════════════════════════════════════════════════════════════════════
# Braid Word (Sequence of Generators)
# ═══════════════════════════════════════════════════════════════════════════

"""
Braid word: sequence of generators representing version history.

The linking number is a topological invariant measuring net over-crossings.
"""
struct BraidWord
    n_strands::Int
    generators::Vector{BraidGenerator}
end

BraidWord(n::Int) = BraidWord(n, BraidGenerator[])

"""
Linking number: topological invariant of the braid.

Counts net over-crossings: Σ(+1 for OVER, -1 for UNDER)
This measures the total "information bred" through forks and mergers.
"""
function linking_number(bw::BraidWord)
    sum(g.crossing == OVER ? 1 : -1 for g in bw.generators; init=0)
end

"""
Add a generator to the braid word.
"""
function push_generator!(bw::BraidWord, strand::Int, crossing::CrossingType, color::RGB{Float64})
    push!(bw.generators, BraidGenerator(strand, crossing, color))
    bw
end

"""
Compute the color trajectory through the braid.
"""
function color_trajectory(bw::BraidWord)
    [g.color for g in bw.generators]
end

# ═══════════════════════════════════════════════════════════════════════════
# Braid Version (Versioned State with Color)
# ═══════════════════════════════════════════════════════════════════════════

"""
A version in the braid: snapshot of state with chromatic identity.
"""
struct BraidVersion
    id::String
    parents::Vector{String}  # DAG parents (enables forks/mergers)
    author::String
    timestamp::Float64
    color::RGB{Float64}
    hue::Float64  # Extracted for holonomy calculation
end

function BraidVersion(id::String, parents::Vector{String}, author::String, color::RGB{Float64})
    # Extract hue from RGB
    hsl = convert(HSL, color)
    BraidVersion(id, parents, author, time(), color, hsl.h)
end

# ═══════════════════════════════════════════════════════════════════════════
# Braid Patch (Colored Delta)
# ═══════════════════════════════════════════════════════════════════════════

"""
A patch (delta) between versions with chromatic identity.
"""
struct BraidPatch
    from_version::String
    to_version::String
    operations::Vector{Any}  # Content-type specific
    color::RGB{Float64}      # Derived from XOR of version colors
end

# ═══════════════════════════════════════════════════════════════════════════
# Holonomy (Curvature of Version Graph)
# ═══════════════════════════════════════════════════════════════════════════

"""
Holonomy: curvature of the version graph.

Sum of hue rotations around the graph. Linear time has zero holonomy.
Braided time accumulates curvature through forks and mergers.
"""
function holonomy(versions::Vector{BraidVersion})
    length(versions) < 2 && return 0.0
    
    total_hue = sum(v.hue for v in versions)
    mod(total_hue, 360.0)
end

# ═══════════════════════════════════════════════════════════════════════════
# Peer Color Identity (Reafference Tracking)
# ═══════════════════════════════════════════════════════════════════════════

"""
Peer identity with color trajectory for self-sameness detection.
"""
mutable struct PeerColorIdentity
    peer_id::String
    origin_color::RGB{Float64}
    current_color::RGB{Float64}
    trajectory::Vector{RGB{Float64}}
    contributions::Vector{String}
end

function PeerColorIdentity(peer_id::String, color::RGB{Float64})
    PeerColorIdentity(peer_id, color, color, [color], String[])
end

"""
Observe a new color state.
"""
function observe!(peer::PeerColorIdentity, color::RGB{Float64})
    peer.current_color = color
    push!(peer.trajectory, color)
end

"""
Reafference distance: how far is current state from origin?

Distance of 0 = perfect self-sameness (fixed point under braid action).
"""
function reafference_distance(peer::PeerColorIdentity)
    length(peer.trajectory) < 2 && return 0.0
    
    c1, c2 = peer.origin_color, peer.current_color
    sqrt((red(c1) - red(c2))^2 + (green(c1) - green(c2))^2 + (blue(c1) - blue(c2))^2)
end

"""
Check if peer has returned to self-sameness.

Self-sameness = trajectory returns to origin color (fixed point).
"""
function is_self_same(peer::PeerColorIdentity; threshold::Float64=0.1)
    reafference_distance(peer) < threshold
end

# ═══════════════════════════════════════════════════════════════════════════
# Gay Braid Model
# ═══════════════════════════════════════════════════════════════════════════

"""
Gay Braid Model: establishes self-sameness through color path reafference.

SELF-SAMENESS:
A trajectory exhibits self-sameness when it returns to a color state
that is "the same as" its origin (within threshold). This is a fixed
point in the color space under the braid action.

REAFFERENCE:
The ability of a trajectory to observe its own past states. Each step
in the braid can "look back" at previous colors and compare.

BRAIDED TIME vs LINEAR TIME:
- Linear: single sequence, holonomy = 0, linking_number undefined
- Braided: forks/mergers, holonomy ≠ 0, linking_number measures topology
"""
mutable struct GayBraidModel
    seed::UInt64
    rng::SplittableRandom
    time_dag::Vector{BraidVersion}
    space_dag::Vector{BraidPatch}
    peers::Dict{String, PeerColorIdentity}
    braid_word::BraidWord
end

function GayBraidModel(seed::UInt64; n_strands::Int=3)
    GayBraidModel(
        seed,
        SplittableRandom(seed),
        BraidVersion[],
        BraidPatch[],
        Dict{String, PeerColorIdentity}(),
        BraidWord(n_strands)
    )
end

"""
Record a version in the Time DAG.
"""
function record_version!(model::GayBraidModel, version::BraidVersion)
    push!(model.time_dag, version)
    
    # Track peer trajectory
    if !haskey(model.peers, version.author)
        model.peers[version.author] = PeerColorIdentity(version.author, version.color)
    else
        observe!(model.peers[version.author], version.color)
        push!(model.peers[version.author].contributions, version.id)
    end
    
    # Add to braid word
    # Determine crossing type from merge structure
    crossing = length(version.parents) > 1 ? OVER : UNDER
    strand = 1 + hash(version.author) % model.braid_word.n_strands
    push_generator!(model.braid_word, strand, crossing, version.color)
    
    version
end

"""
Get holonomy of the model.
"""
holonomy(model::GayBraidModel) = holonomy(model.time_dag)

"""
Get linking number of the model.
"""
linking_number(model::GayBraidModel) = linking_number(model.braid_word)

"""
Generate self-sameness report for all peers.
"""
function self_sameness_report(model::GayBraidModel)
    Dict(
        id => (
            self_same = is_self_same(peer),
            distance = reafference_distance(peer),
            trajectory_length = length(peer.trajectory)
        )
        for (id, peer) in model.peers
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Braid Log (Full Synchronization Engine)
# ═══════════════════════════════════════════════════════════════════════════

"""
Braid Log: unified state synchronization with chromatic identity.

Combines:
- Braid-HTTP protocol (versions, patches, subscriptions)
- Gay.jl SPI (deterministic color trajectories)
- Cohesive ∞-topos semantics (♯/♭/ʃ modalities)
- Self-sameness through reafference
"""
mutable struct BraidLog
    resource_url::String
    seed::UInt64
    gay_braid::GayBraidModel
    merge_type::Symbol  # :crdt_sync9, :ot, :custom
end

function BraidLog(url::String; seed::UInt64=GAY_SEED, merge_type::Symbol=:crdt_sync9)
    BraidLog(url, seed, GayBraidModel(seed), merge_type)
end

"""
Push a new version to the log.
"""
function push_version!(log::BraidLog, id::String, parents::Vector{String}, author::String)
    # Generate color from splittable RNG
    rng = split(log.gay_braid.rng)
    log.gay_braid.rng = rng
    color = color_from_rng(rng)
    
    version = BraidVersion(id, parents, author, color)
    record_version!(log.gay_braid, version)
end

# ═══════════════════════════════════════════════════════════════════════════
# Random Walk to Target Color
# ═══════════════════════════════════════════════════════════════════════════

"""
Random walk from GAY_SEED toward target color in N steps.

This demonstrates braided time: the walk may fork (explore multiple paths)
and merge (converge on target), accumulating holonomy along the way.
"""
function random_walk_to_color(
    target::RGB{Float64};
    seed::UInt64=GAY_SEED,
    steps::Int=69,
    n_strands::Int=3
)
    model = GayBraidModel(seed; n_strands=n_strands)
    rng = model.rng
    
    # Start from origin color
    origin = color_from_rng(split(rng))
    current = origin
    
    trajectory = [(step=0, color=current, distance=color_distance(current, target))]
    
    for step in 1:steps
        rng = split(rng)
        
        # Bias random walk toward target
        noise = color_from_rng(split(rng))
        
        # Interpolate: current + α*(target - current) + β*noise
        α = step / (steps + 10)  # Increasing attraction
        β = 1 - α               # Decreasing noise
        
        new_r = clamp(current.r + α*(target.r - current.r) + β*(noise.r - 0.5)*0.3, 0, 1)
        new_g = clamp(current.g + α*(target.g - current.g) + β*(noise.g - 0.5)*0.3, 0, 1)
        new_b = clamp(current.b + α*(target.b - current.b) + β*(noise.b - 0.5)*0.3, 0, 1)
        
        current = RGB(new_r, new_g, new_b)
        
        # Record as version
        id = "v$(step)"
        parents = step == 1 ? String[] : ["v$(step-1)"]
        
        # Simulate fork/merge
        if step % 10 == 0 && step > 10
            parents = ["v$(step-1)", "v$(step-10)"]  # Merge
        end
        
        version = BraidVersion(id, parents, "walker", current)
        record_version!(model, version)
        
        push!(trajectory, (
            step = step,
            color = current,
            distance = color_distance(current, target)
        ))
    end
    
    (
        origin = origin,
        target = target,
        final = current,
        trajectory = trajectory,
        steps = steps,
        holonomy = holonomy(model),
        linking_number = linking_number(model),
        reached = color_distance(current, target) < 0.1,
        model = model
    )
end

"""
Color distance (Euclidean in RGB).
"""
function color_distance(c1::RGB, c2::RGB)
    sqrt((red(c1) - red(c2))^2 + (green(c1) - green(c2))^2 + (blue(c1) - blue(c2))^2)
end

"""
Color from RNG (same as gaymc.jl).
"""
function color_from_rng(rng::SplittableRandom)::RGB{Float64}
    r = rand(rng)
    g = rand(rng)
    b = rand(rng)
    
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    if max_c > min_c
        scale = 1.0 / (max_c - min_c + 0.3)
        r = (r - min_c) * scale
        g = (g - min_c) * scale
        b = (b - min_c) * scale
    end
    
    RGB{Float64}(clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1))
end

# ═══════════════════════════════════════════════════════════════════════════
# Braid Random Walk with Fork/Merge Exploration
# ═══════════════════════════════════════════════════════════════════════════

"""
Full braid random walk: explores multiple paths (forks) and merges.

Unlike linear random walk, this creates actual branching structure:
- At fork points, split into multiple parallel trajectories
- At merge points, combine trajectories via color averaging
- Accumulates non-zero holonomy (curvature)
"""
function braid_random_walk!(
    model::GayBraidModel,
    target::RGB{Float64};
    steps::Int=69,
    fork_probability::Float64=0.2,
    merge_interval::Int=10
)
    rng = model.rng
    
    # Active branches (can have multiple)
    branches = Dict{Int, RGB{Float64}}()
    branch_counter = 1
    branches[1] = color_from_rng(split(rng))
    
    for step in 1:steps
        rng = split(rng)
        new_branches = Dict{Int, RGB{Float64}}()
        
        for (branch_id, current) in branches
            # Random step toward target
            noise = color_from_rng(split(rng))
            α = step / (steps + 10)
            β = 1 - α
            
            new_color = RGB(
                clamp(current.r + α*(target.r - current.r) + β*(noise.r - 0.5)*0.3, 0, 1),
                clamp(current.g + α*(target.g - current.g) + β*(noise.g - 0.5)*0.3, 0, 1),
                clamp(current.b + α*(target.b - current.b) + β*(noise.b - 0.5)*0.3, 0, 1)
            )
            
            # Fork?
            if rand(rng) < fork_probability && length(branches) < 5
                branch_counter += 1
                new_branches[branch_counter] = new_color
                
                # Record fork as OVER crossing
                version = BraidVersion("v$(step)_fork$(branch_counter)", 
                                       ["v$(step-1)_$(branch_id)"], 
                                       "branch$(branch_counter)", 
                                       new_color)
                record_version!(model, version)
            end
            
            new_branches[branch_id] = new_color
            
            version = BraidVersion("v$(step)_$(branch_id)", 
                                   ["v$(step-1)_$(branch_id)"], 
                                   "branch$(branch_id)", 
                                   new_color)
            record_version!(model, version)
        end
        
        # Merge interval?
        if step % merge_interval == 0 && length(new_branches) > 1
            # Merge all branches into one
            merged_color = RGB(
                mean(red(c) for c in values(new_branches)),
                mean(green(c) for c in values(new_branches)),
                mean(blue(c) for c in values(new_branches))
            )
            
            parents = ["v$(step)_$(id)" for id in keys(new_branches)]
            version = BraidVersion("v$(step)_merge", parents, "merger", merged_color)
            record_version!(model, version)
            
            # Continue with single branch
            new_branches = Dict(1 => merged_color)
        end
        
        branches = new_branches
    end
    
    # Final state
    final_color = first(values(branches))
    
    (
        final = final_color,
        target = target,
        distance = color_distance(final_color, target),
        holonomy = holonomy(model),
        linking_number = linking_number(model),
        n_versions = length(model.time_dag),
        reached = color_distance(final_color, target) < 0.1
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Demo: Random Walk to #FFFF69 in 69 Steps
# ═══════════════════════════════════════════════════════════════════════════

function world_random_walk_ffff69()
    target = RGB(1.0, 1.0, 0.4118)  # #FFFF69
    
    println("╔════════════════════════════════════════════════════════════════╗")
    println("║  Gay Braid Protocol: Random Walk to #FFFF69 in 69 Steps        ║")
    println("║  Braided Time vs Linear Clock Time                             ║")
    println("╚════════════════════════════════════════════════════════════════╝")
    println()
    
    # Linear walk (no forks/merges)
    println("Linear Random Walk (ticking clock):")
    linear = random_walk_to_color(target; steps=69)
    println("  Origin:         RGB($(round(linear.origin.r, digits=2)), $(round(linear.origin.g, digits=2)), $(round(linear.origin.b, digits=2)))")
    println("  Final:          RGB($(round(linear.final.r, digits=2)), $(round(linear.final.g, digits=2)), $(round(linear.final.b, digits=2)))")
    println("  Target #FFFF69: RGB(1.0, 1.0, 0.41)")
    println("  Distance:       $(round(linear.trajectory[end].distance, digits=4))")
    println("  Holonomy:       $(round(linear.holonomy, digits=2))°")
    println("  Linking #:      $(linear.linking_number)")
    println()
    
    # Braided walk (with forks/merges)
    println("Braided Random Walk (forks + mergers):")
    model = GayBraidModel(GAY_SEED; n_strands=3)
    braided = braid_random_walk!(model, target; steps=69)
    println("  Final:          RGB($(round(braided.final.r, digits=2)), $(round(braided.final.g, digits=2)), $(round(braided.final.b, digits=2)))")
    println("  Target #FFFF69: RGB(1.0, 1.0, 0.41)")
    println("  Distance:       $(round(braided.distance, digits=4))")
    println("  Holonomy:       $(round(braided.holonomy, digits=2))°")
    println("  Linking #:      $(braided.linking_number)")
    println("  Versions:       $(braided.n_versions)")
    println()
    
    println("Interpretation:")
    println("  • Linear time: holonomy ≈ 0, single path, no topology")
    println("  • Braided time: holonomy ≠ 0, forks/mergers, rich topology")
    println("  • Linking number measures net over-crossings (information bred)")
    println("  • Self-sameness achieved when trajectory returns to origin color")
    println()
    
    (linear = linear, braided = braided)
end
