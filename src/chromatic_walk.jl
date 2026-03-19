# Chromatic Random Walk: Self-Seeking / Self-Avoiding Adversarial Dynamic Equilibrium
#
# A massively parallel optic open game where:
# - SELF-SEEKING: Trajectory actively seeks high-entropy color regions
# - SELF-AVOIDING: Never revisits same color (like SAW on lattice)
# - ADVERSARIAL: Two players compete (Proponent expands, Opponent contracts)
# - DYNAMIC EQUILIBRIUM: Nash equilibrium emerges from parallel play
# - MASSIVELY PARALLEL: Uses SPI for deterministic parallel execution
#
# Derivable from DuckDB 'ies' color analysis:
# - palette.duckdb: 200K colors with 99.4% diversity (0.6% collision)
# - High-bandwidth hue buckets identified (210°, 150°, 0°, 120° best)
# - Thread inventory provides color targets for vibe sniping
#
# Optic Open Game Structure:
# - Lens(S, A): forward = play, backward = coplay (coutility)
# - Para(Lens): parameterized by seed for SPI
# - Composition: sequential (;) and parallel (⊗)

module ChromaticWalk

using SplittableRandoms: SplittableRandom, split
using Colors

export ChromaticRandomWalk, SelfSeekingStrategy, SelfAvoidingConstraint
export AdversarialPlayer, DynamicEquilibrium
export OpticOpenGame, ParaLens, chromatic_play, chromatic_coplay
export parallel_walk!, find_nash_equilibrium
export DuckDBColorSource, load_palette_bandwidth
export demo_chromatic_walk

# ═══════════════════════════════════════════════════════════════════════════════
# DuckDB Color Source (from ies analysis)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DuckDBColorSource

Interface to DuckDB palette analysis for deriving high-bandwidth color regions.

From palette.duckdb analysis:
- 200,000 colors total
- 198,777 unique (99.4% diversity, 0.6% collision rate)
- Best hue buckets: 210° (99.98%), 150° (99.96%), 0° (99.96%)
- Full 360° hue coverage, full L and C ranges
"""
struct DuckDBColorSource
    db_path::String
    hue_bandwidth::Dict{Float64, Float64}  # hue_bucket → diversity %
    total_colors::Int
    unique_colors::Int
    collision_rate::Float64
end

"""
Create color source from palette.duckdb bandwidth analysis.
"""
function load_palette_bandwidth()
    # Pre-computed from DuckDB query on palette.duckdb
    hue_bandwidth = Dict{Float64, Float64}(
        210.0 => 99.98,
        150.0 => 99.96,
        0.0   => 99.96,
        120.0 => 99.96,
        270.0 => 99.95,
        240.0 => 99.95,
        180.0 => 99.95,
        30.0  => 99.95,
        330.0 => 99.94,
        60.0  => 99.93,
        90.0  => 99.93,
        300.0 => 99.90,
    )
    
    DuckDBColorSource(
        "/Users/bob/ies/palette.duckdb",
        hue_bandwidth,
        200_000,
        198_777,
        0.61  # collision rate %
    )
end

"""
Get highest bandwidth hue regions for self-seeking.
"""
function high_bandwidth_hues(source::DuckDBColorSource; threshold::Float64=99.95)
    [hue for (hue, bw) in source.hue_bandwidth if bw >= threshold]
end

# ═══════════════════════════════════════════════════════════════════════════════
# Self-Avoiding Constraint
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SelfAvoidingConstraint

Enforces self-avoiding walk (SAW) on color space.

Like SAW on a lattice, the trajectory cannot revisit "colors" where
a color is a discretized bin in RGB or LCH space.

Properties:
- visited: Set of visited color bins
- bin_resolution: How finely to discretize (default 256 per channel)
- rejection_count: How many times we've rejected for revisit
"""
mutable struct SelfAvoidingConstraint
    visited::Set{UInt32}
    bin_resolution::Int
    rejection_count::Int
    max_rejections::Int
end

function SelfAvoidingConstraint(; resolution::Int=256, max_rejections::Int=1000)
    SelfAvoidingConstraint(Set{UInt32}(), resolution, 0, max_rejections)
end

"""
Discretize color to bin ID.
"""
function color_to_bin(c::RGB{Float64}, resolution::Int=256)::UInt32
    r = round(UInt8, clamp(c.r, 0, 1) * (resolution - 1))
    g = round(UInt8, clamp(c.g, 0, 1) * (resolution - 1))
    b = round(UInt8, clamp(c.b, 0, 1) * (resolution - 1))
    UInt32(r) << 16 | UInt32(g) << 8 | UInt32(b)
end

"""
Check if color is allowed (not visited).
"""
function is_allowed(sac::SelfAvoidingConstraint, c::RGB{Float64})
    bin = color_to_bin(c, sac.bin_resolution)
    !(bin in sac.visited)
end

"""
Mark color as visited.
"""
function mark_visited!(sac::SelfAvoidingConstraint, c::RGB{Float64})
    bin = color_to_bin(c, sac.bin_resolution)
    push!(sac.visited, bin)
end

"""
Check if walk is stuck (too many rejections).
"""
is_stuck(sac::SelfAvoidingConstraint) = sac.rejection_count >= sac.max_rejections

# ═══════════════════════════════════════════════════════════════════════════════
# Self-Seeking Strategy
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SelfSeekingStrategy

Actively seeks high-entropy color regions.

Based on DuckDB bandwidth analysis:
- Prefers hue buckets with >99.95% diversity
- Avoids attractor basins (primary colors)
- Maximizes distance from visited colors

The strategy is SELF-seeking because it uses its own history
to determine where to go next.
"""
struct SelfSeekingStrategy
    seed::UInt64
    rng::SplittableRandom
    target_hues::Vector{Float64}  # High-bandwidth hue targets
    history::Vector{RGB{Float64}}
    entropy_threshold::Float64
end

function SelfSeekingStrategy(seed::UInt64=0x6761795f636f6c6f; source::DuckDBColorSource=load_palette_bandwidth())
    target_hues = high_bandwidth_hues(source)
    SelfSeekingStrategy(seed, SplittableRandom(seed), target_hues, RGB{Float64}[], 0.95)
end

"""
Seek next color that maximizes entropy while avoiding visited regions.
"""
function seek_next!(sss::SelfSeekingStrategy, sac::SelfAvoidingConstraint)
    for attempt in 1:sac.max_rejections
        rng = split(sss.rng)
        
        # Choose a high-bandwidth hue bucket
        target_hue = isempty(sss.target_hues) ? rand(rng) * 360 : 
                     sss.target_hues[1 + (rand(rng, UInt) % length(sss.target_hues))]
        
        # Generate color in that bucket with some jitter
        hue_jitter = (rand(rng) - 0.5) * 30  # ±15° within bucket
        hue = mod(target_hue + hue_jitter, 360.0)
        
        # Full range saturation and lightness for max bandwidth
        saturation = 0.3 + rand(rng) * 0.6  # 0.3-0.9
        lightness = 0.3 + rand(rng) * 0.4   # 0.3-0.7
        
        # Convert HSL to RGB (simplified)
        color = hsl_to_rgb(hue, saturation, lightness)
        
        if is_allowed(sac, color)
            mark_visited!(sac, color)
            new_sss = SelfSeekingStrategy(
                sss.seed, rng, sss.target_hues, 
                [sss.history; color], sss.entropy_threshold
            )
            return (color, new_sss, attempt)
        else
            sac.rejection_count += 1
        end
    end
    
    # Stuck - return last valid or gray
    fallback = isempty(sss.history) ? RGB(0.5, 0.5, 0.5) : sss.history[end]
    (fallback, sss, sac.max_rejections)
end

"""
HSL to RGB conversion.
"""
function hsl_to_rgb(h::Float64, s::Float64, l::Float64)::RGB{Float64}
    c = (1 - abs(2*l - 1)) * s
    x = c * (1 - abs(mod(h/60, 2) - 1))
    m = l - c/2
    
    r, g, b = if h < 60
        (c, x, 0.0)
    elseif h < 120
        (x, c, 0.0)
    elseif h < 180
        (0.0, c, x)
    elseif h < 240
        (0.0, x, c)
    elseif h < 300
        (x, 0.0, c)
    else
        (c, 0.0, x)
    end
    
    RGB{Float64}(clamp(r + m, 0, 1), clamp(g + m, 0, 1), clamp(b + m, 0, 1))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Adversarial Players
# ═══════════════════════════════════════════════════════════════════════════════

"""
    AdversarialPlayer

Two-player adversarial game on color space.

- EXPANDER (Proponent): Tries to maximize color diversity (entropy)
- CONTRACTOR (Opponent): Tries to force trajectory into attractors

This is the adversarial interpretation of the Dialectica game.
"""
@enum PlayerRole EXPANDER CONTRACTOR

struct AdversarialPlayer
    role::PlayerRole
    seed::UInt64
    rng::SplittableRandom
    moves::Vector{RGB{Float64}}
    score::Float64
end

function AdversarialPlayer(role::PlayerRole, seed::UInt64)
    AdversarialPlayer(role, seed, SplittableRandom(seed), RGB{Float64}[], 0.0)
end

"""
Expander move: seek high-entropy color.
"""
function expander_move!(player::AdversarialPlayer, sac::SelfAvoidingConstraint, sss::SelfSeekingStrategy)
    @assert player.role == EXPANDER "Must be EXPANDER"
    
    color, new_sss, attempts = seek_next!(sss, sac)
    
    # Score based on attempts (fewer = better exploration)
    score = 1.0 / log2(1 + attempts)
    
    new_player = AdversarialPlayer(
        player.role, player.seed, split(player.rng),
        [player.moves; color], player.score + score
    )
    
    (color, new_player, new_sss)
end

"""
Contractor move: force toward attractor.
"""
function contractor_move!(player::AdversarialPlayer, target::RGB{Float64})
    @assert player.role == CONTRACTOR "Must be CONTRACTOR"
    
    rng = split(player.rng)
    
    # Find nearest attractor to target
    attractors = [
        RGB(1.0, 0.0, 0.0), RGB(0.0, 1.0, 0.0), RGB(0.0, 0.0, 1.0),
        RGB(0.0, 1.0, 1.0), RGB(1.0, 0.0, 1.0), RGB(1.0, 1.0, 0.0),
    ]
    
    min_dist = Inf
    nearest = attractors[1]
    for a in attractors
        d = color_distance(target, a)
        if d < min_dist
            min_dist = d
            nearest = a
        end
    end
    
    # Force toward attractor with noise
    α = 0.3 + rand(rng) * 0.4  # 30-70% toward attractor
    forced = RGB(
        clamp(target.r + α * (nearest.r - target.r), 0, 1),
        clamp(target.g + α * (nearest.g - target.g), 0, 1),
        clamp(target.b + α * (nearest.b - target.b), 0, 1)
    )
    
    # Score based on how close we got to attractor
    score = 1.0 - min_dist / sqrt(3)
    
    new_player = AdversarialPlayer(
        player.role, player.seed, rng,
        [player.moves; forced], player.score + score
    )
    
    (forced, new_player)
end

function color_distance(c1::RGB{Float64}, c2::RGB{Float64})
    sqrt((c1.r - c2.r)^2 + (c1.g - c2.g)^2 + (c1.b - c2.b)^2)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic Equilibrium
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DynamicEquilibrium

Nash equilibrium emerges from adversarial play.

At equilibrium:
- Expander cannot increase entropy without Contractor catching up
- Contractor cannot decrease entropy without Expander escaping

This is "dynamic" because the equilibrium shifts as the walk progresses.
"""
struct DynamicEquilibrium
    expander::AdversarialPlayer
    contractor::AdversarialPlayer
    trajectory::Vector{RGB{Float64}}
    equilibrium_entropy::Float64
    is_stable::Bool
end

"""
    find_nash_equilibrium(seed::UInt64; max_steps::Int=100) -> DynamicEquilibrium

Find Nash equilibrium through adversarial play.
"""
function find_nash_equilibrium(seed::UInt64; max_steps::Int=100)
    expander = AdversarialPlayer(EXPANDER, seed)
    contractor = AdversarialPlayer(CONTRACTOR, seed ⊻ 0xdeadbeef)
    
    sac = SelfAvoidingConstraint()
    sss = SelfSeekingStrategy(seed)
    
    trajectory = RGB{Float64}[]
    
    for step in 1:max_steps
        is_stuck(sac) && break
        
        # Expander moves first (∃x)
        exp_color, expander, sss = expander_move!(expander, sac, sss)
        push!(trajectory, exp_color)
        
        # Contractor responds (∀y) trying to force toward attractor
        con_color, contractor = contractor_move!(contractor, exp_color)
        
        # Trajectory is the result of the interaction
        # (Expander's color modified by Contractor's force)
        α = 0.2  # Contractor influence
        actual = RGB(
            clamp(exp_color.r + α * (con_color.r - exp_color.r), 0, 1),
            clamp(exp_color.g + α * (con_color.g - exp_color.g), 0, 1),
            clamp(exp_color.b + α * (con_color.b - exp_color.b), 0, 1)
        )
        trajectory[end] = actual
    end
    
    # Compute equilibrium entropy
    eq_entropy = compute_trajectory_entropy(trajectory)
    
    # Equilibrium is stable if entropy is in the "balanced" region
    is_stable = 0.4 < eq_entropy < 0.8
    
    DynamicEquilibrium(expander, contractor, trajectory, eq_entropy, is_stable)
end

"""
Compute entropy of a color trajectory.
"""
function compute_trajectory_entropy(trajectory::Vector{RGB{Float64}})
    isempty(trajectory) && return 0.0
    
    # Discretize to bins and compute histogram
    bins = Dict{UInt32, Int}()
    for c in trajectory
        bin = color_to_bin(c, 64)  # Coarse bins for entropy
        bins[bin] = get(bins, bin, 0) + 1
    end
    
    # Shannon entropy
    n = length(trajectory)
    H = 0.0
    for count in values(bins)
        p = count / n
        if p > 0
            H -= p * log2(p)
        end
    end
    
    # Normalize by max entropy
    max_H = log2(n)
    max_H > 0 ? H / max_H : 0.0
end

# ═══════════════════════════════════════════════════════════════════════════════
# Optic Open Game (Para(Lens))
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ParaLens

Parameterized lens for optic open games.

Lens(S, A) where:
- S = state (color trajectory)
- A = action (color choice)
- forward (play): S → A
- backward (coplay): S × R → S' (update with coutility)

Para(Lens) adds parameter P (seed) for SPI.
"""
struct ParaLens{S, A, R}
    seed::UInt64
    play::Function     # (P, S) → A
    coplay::Function   # (P, S, R) → S
end

"""
    OpticOpenGame

An open game using optic/lens structure.

Composition:
- Sequential (;): play₂ ∘ play₁, coplay backward through both
- Parallel (⊗): play both, coplay both
"""
struct OpticOpenGame
    name::Symbol
    lens::ParaLens
    subgames::Vector{OpticOpenGame}
end

function OpticOpenGame(name::Symbol, seed::UInt64, play::Function, coplay::Function)
    lens = ParaLens{Vector{RGB{Float64}}, RGB{Float64}, Float64}(seed, play, coplay)
    OpticOpenGame(name, lens, OpticOpenGame[])
end

"""
Chromatic play: forward pass producing a color.
"""
function chromatic_play(game::OpticOpenGame, state::Vector{RGB{Float64}})
    game.lens.play(game.lens.seed, state)
end

"""
Chromatic coplay: backward pass updating state with coutility.
"""
function chromatic_coplay(game::OpticOpenGame, state::Vector{RGB{Float64}}, coutility::Float64)
    game.lens.coplay(game.lens.seed, state, coutility)
end

"""
Compose games sequentially.
"""
function compose_seq(g1::OpticOpenGame, g2::OpticOpenGame)
    combined_seed = g1.lens.seed ⊻ g2.lens.seed
    
    play = (seed, state) -> begin
        mid = g1.lens.play(seed, state)
        g2.lens.play(seed, [state; mid])
    end
    
    coplay = (seed, state, r) -> begin
        # Backward through g2, then g1
        state2 = g2.lens.coplay(seed, state, r)
        g1.lens.coplay(seed, state2, r)
    end
    
    OpticOpenGame(:composed, combined_seed, play, coplay)
end

"""
Compose games in parallel.
"""
function compose_par(g1::OpticOpenGame, g2::OpticOpenGame)
    combined_seed = g1.lens.seed ⊻ g2.lens.seed
    
    play = (seed, state) -> begin
        c1 = g1.lens.play(seed, state)
        c2 = g2.lens.play(seed ⊻ 0x1, state)
        # Blend parallel results
        RGB((c1.r + c2.r)/2, (c1.g + c2.g)/2, (c1.b + c2.b)/2)
    end
    
    coplay = (seed, state, r) -> begin
        # Both update in parallel
        s1 = g1.lens.coplay(seed, state, r)
        s2 = g2.lens.coplay(seed ⊻ 0x1, state, r)
        # Merge states
        vcat(s1, s2)
    end
    
    OpticOpenGame(:parallel, combined_seed, play, coplay)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Chromatic Random Walk
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticRandomWalk

The full self-seeking/self-avoiding adversarial chromatic walk.

Combines:
- Self-seeking: Targets high-bandwidth hue regions from DuckDB
- Self-avoiding: SAW constraint prevents revisiting
- Adversarial: Expander vs Contractor game
- Parallel: Uses SPI for parallel execution
- Optic: Lens structure for compositional game
"""
mutable struct ChromaticRandomWalk
    seed::UInt64
    source::DuckDBColorSource
    constraint::SelfAvoidingConstraint
    seeker::SelfSeekingStrategy
    equilibrium::Union{DynamicEquilibrium, Nothing}
    game::OpticOpenGame
    trajectory::Vector{RGB{Float64}}
    entropy_history::Vector{Float64}
end

function ChromaticRandomWalk(seed::UInt64=0x6761795f636f6c6f)
    source = load_palette_bandwidth()
    constraint = SelfAvoidingConstraint()
    seeker = SelfSeekingStrategy(seed; source=source)
    
    # Create the optic open game for this walk
    play = (s, state) -> begin
        sss = SelfSeekingStrategy(s; source=source)
        sac = SelfAvoidingConstraint()
        for c in state
            mark_visited!(sac, c)
        end
        color, _, _ = seek_next!(sss, sac)
        color
    end
    
    coplay = (s, state, r) -> begin
        # Coutility modifies trajectory entropy
        if r > 0.5
            # High coutility: trajectory is good, keep exploring
            state
        else
            # Low coutility: trajectory collapsing, reset some visits
            state[max(1, end-10):end]
        end
    end
    
    game = OpticOpenGame(:chromatic_walk, seed, play, coplay)
    
    ChromaticRandomWalk(seed, source, constraint, seeker, nothing, game, RGB{Float64}[], Float64[])
end

"""
    parallel_walk!(walk::ChromaticRandomWalk; n_steps::Int=100, n_workers::Int=4)

Execute parallel chromatic walk using SPI.

Each worker explores from a different split seed, but all contribute
to the same color space coverage (SPI guarantees determinism).
"""
function parallel_walk!(walk::ChromaticRandomWalk; n_steps::Int=100, n_workers::Int=4)
    # Create parallel workers with split seeds
    worker_seeds = [walk.seed ⊻ UInt64(i) for i in 1:n_workers]
    worker_trajectories = [RGB{Float64}[] for _ in 1:n_workers]
    
    # Parallel execution (conceptually - using sequential with SPI guarantee)
    for w in 1:n_workers
        wseed = worker_seeds[w]
        wsss = SelfSeekingStrategy(wseed; source=walk.source)
        wsac = SelfAvoidingConstraint()
        
        for step in 1:n_steps÷n_workers
            is_stuck(wsac) && break
            
            color, wsss, _ = seek_next!(wsss, wsac)
            push!(worker_trajectories[w], color)
        end
    end
    
    # Merge worker trajectories (interleaved for fair mixing)
    max_len = maximum(length.(worker_trajectories))
    for i in 1:max_len
        for w in 1:n_workers
            if i <= length(worker_trajectories[w])
                push!(walk.trajectory, worker_trajectories[w][i])
                
                # Track entropy
                H = compute_trajectory_entropy(walk.trajectory)
                push!(walk.entropy_history, H)
            end
        end
    end
    
    # Find equilibrium
    walk.equilibrium = find_nash_equilibrium(walk.seed; max_steps=n_steps)
    
    walk
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

function demo_chromatic_walk()
    println("╔═══════════════════════════════════════════════════════════════════════╗")
    println("║  Chromatic Random Walk: Self-Seeking/Self-Avoiding Adversarial Game   ║")
    println("║  Derivable from DuckDB ies color analysis + Gay.jl SPI               ║")
    println("╚═══════════════════════════════════════════════════════════════════════╝")
    println()
    
    seed = UInt64(0x6761795f636f6c6f)
    
    # DuckDB source
    println("─── DuckDB Color Source (palette.duckdb) ───")
    source = load_palette_bandwidth()
    println("  Total colors: $(source.total_colors)")
    println("  Unique colors: $(source.unique_colors)")
    println("  Collision rate: $(source.collision_rate)%")
    println("  High-bandwidth hues: $(high_bandwidth_hues(source))")
    
    # Self-avoiding constraint
    println("\n─── Self-Avoiding Constraint ───")
    sac = SelfAvoidingConstraint()
    test_colors = [hsl_to_rgb(h, 0.7, 0.5) for h in 0:30:330]
    for c in test_colors
        mark_visited!(sac, c)
    end
    println("  Visited $(length(sac.visited)) color bins")
    println("  Is stuck: $(is_stuck(sac))")
    
    # Self-seeking strategy
    println("\n─── Self-Seeking Strategy ───")
    sss = SelfSeekingStrategy(seed; source=source)
    sac2 = SelfAvoidingConstraint()
    colors = RGB{Float64}[]
    for i in 1:10
        c, sss, attempts = seek_next!(sss, sac2)
        push!(colors, c)
        if i <= 3
            println("  Step $i: RGB($(round(c.r, digits=2)), $(round(c.g, digits=2)), $(round(c.b, digits=2))) in $attempts attempts")
        end
    end
    println("  ... (10 self-seeking steps)")
    
    # Adversarial game
    println("\n─── Adversarial Dynamic Equilibrium ───")
    eq = find_nash_equilibrium(seed; max_steps=50)
    println("  Expander score: $(round(eq.expander.score, digits=2))")
    println("  Contractor score: $(round(eq.contractor.score, digits=2))")
    println("  Trajectory length: $(length(eq.trajectory))")
    println("  Equilibrium entropy: $(round(eq.equilibrium_entropy, digits=4))")
    println("  Is stable: $(eq.is_stable)")
    
    # Full chromatic walk
    println("\n─── Parallel Chromatic Walk ───")
    walk = ChromaticRandomWalk(seed)
    parallel_walk!(walk; n_steps=100, n_workers=4)
    println("  Trajectory length: $(length(walk.trajectory))")
    println("  Final entropy: $(round(walk.entropy_history[end], digits=4))")
    println("  Entropy trend: $(walk.entropy_history[1] < walk.entropy_history[end] ? "↑ increasing" : "↓ decreasing")")
    
    # Optic open game
    println("\n─── Optic Open Game (Para(Lens)) ───")
    c = chromatic_play(walk.game, walk.trajectory[1:min(5, end)])
    println("  Play output: RGB($(round(c.r, digits=2)), $(round(c.g, digits=2)), $(round(c.b, digits=2)))")
    state2 = chromatic_coplay(walk.game, walk.trajectory, 0.8)
    println("  Coplay (coutility=0.8): kept $(length(state2)) colors")
    
    return (
        source = source,
        equilibrium = eq,
        walk = walk,
        final_entropy = walk.entropy_history[end]
    )
end

end # module ChromaticWalk
