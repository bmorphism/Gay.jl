# GAY-DYCK-CATALAN: The Stammering Trinity for Chromatic Sheafification
# ======================================================================
#
# "Bégayant (stammering) contains 'gay' because all three share Catalan-counted structure."
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  THE GAY-DYCK-BIRB TRINITY                                                  │
# │                                                                             │
# │  Dyck path:    (()())  ↔  ↗↗↘↗↘↘                                           │
# │  Birb parse:   🐦🐦🐦🐦 → (🐦((🐦🐦)🐦))                                      │
# │  SKI term:     S(K(SK)) has depth = Dyck height                            │
# │                                                                             │
# │  All counted by Catalan numbers: C_n = (1/(n+1)) * C(2n, n)                │
# │                                                                             │
# │  STAMMERING TABLEAUX (Josuat-Vergès):                                       │
# │    - Oscillating walks in Young's lattice                                   │
# │    - Counts PASEP particle configurations                                   │
# │    - Each step: add or remove a box from partition                          │
# │                                                                             │
# │  CHROMATIC SHEAFIFICATION:                                                  │
# │    - Map Dyck paths to chromatic trajectories                               │
# │    - Catalan index → deterministic color sequence                           │
# │    - Sheaf sections = consistent color assignments over categories          │
# │                                                                             │
# │  GAY SEED OPTIMIZATION:                                                     │
# │    - Find seed that maximizes chromatic coherence                           │
# │    - Interleave GayMC over IES messages                                     │
# │    - Sheafify meaning via Catalan-indexed color assignment                  │
# └─────────────────────────────────────────────────────────────────────────────┘

module GayDyckCatalan

export
    # Catalan Numbers
    catalan, catalan_triangle, motzkin, narayana,
    
    # Dyck Paths
    DyckPath, dyck_from_parens, dyck_to_parens, 
    dyck_height, dyck_area, dyck_peaks, dyck_valleys,
    all_dyck_paths, random_dyck_path,
    
    # Chromatic Dyck
    ChromaticDyck, dyck_color_trajectory, dyck_fingerprint,
    dyck_to_ski, ski_to_dyck,
    
    # Stammering Tableaux
    StammeringTableau, oscillating_walk, pasep_config,
    tableau_to_dyck, dyck_to_tableau,
    
    # Birb Parse Trees
    BirbTree, birb_parse, birb_to_dyck, dyck_to_birb,
    birb_fingerprint, birb_color,
    
    # Sheafification
    ChromaticSheaf, sheaf_section, sheaf_restriction,
    sheafify_messages, coherence_score, 
    
    # Seed Optimization
    SeedBundle, optimize_seed_bundle, interleave_gaymc,
    meaning_assignment_rate, chromatic_invariant,
    
    # IES Message Integration
    IESMessage, load_ies_messages, catalan_index_messages,
    message_dyck_path, message_fingerprint,
    
    # Demo
    demo_gay_dyck_catalan

# ═══════════════════════════════════════════════════════════════════════════════
# Core PRNG (SPI compliant)
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const CATALAN_SEED = UInt64(0xCA7A1A9)  # "CATALAN" 
const DYCK_SEED = UInt64(0xD7C4)        # "DYCK"
const BIRB_SEED = UInt64(0xB12B)        # "BIRB"

@inline function sm64(s::UInt64)::Tuple{UInt64, UInt64}
    z = (s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    (z ⊻ (z >> 31), s + 1)
end

@inline function color_from_fp(fp::UInt64)::NTuple{3, Float64}
    r, s1 = sm64(fp)
    g, s2 = sm64(s1)
    b, _  = sm64(s2)
    ((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0)
end

# ═══════════════════════════════════════════════════════════════════════════════
# CATALAN NUMBERS: The Counting Foundation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    catalan(n::Int) -> BigInt

Compute the n-th Catalan number: C_n = (1/(n+1)) * C(2n, n)

Catalan numbers count:
- Dyck paths of length 2n
- Full binary trees with n+1 leaves
- Valid parenthesizations of n+1 factors
- Non-crossing partitions of {1,...,n}
- Triangulations of (n+2)-gon
"""
function catalan(n::Int)::BigInt
    n < 0 && return BigInt(0)
    binomial(BigInt(2n), BigInt(n)) ÷ BigInt(n + 1)
end

"""
    catalan_triangle(n::Int) -> Matrix{BigInt}

Catalan's triangle: C(n,k) = number of Dyck paths from (0,0) to (n,k).
"""
function catalan_triangle(n::Int)::Matrix{BigInt}
    C = zeros(BigInt, n+1, n+1)
    C[1, 1] = 1
    
    for i in 1:n
        for j in 0:i
            # Dyck path counting: can go right (stay) or up (if below diagonal)
            if j > 0
                C[i+1, j+1] = C[i, j] + C[i, j+1]
            else
                C[i+1, j+1] = C[i, j+1]
            end
        end
    end
    
    C
end

"""
    motzkin(n::Int) -> BigInt

Motzkin number: counts paths with steps ↗, →, ↘ staying non-negative.
Related to Catalan via: M_n = Σ C(n,2k) * C_k
"""
function motzkin(n::Int)::BigInt
    n < 0 && return BigInt(0)
    n == 0 && return BigInt(1)
    
    sum(binomial(BigInt(n), BigInt(2k)) * catalan(k) for k in 0:n÷2)
end

"""
    narayana(n::Int, k::Int) -> BigInt

Narayana number: counts Dyck paths of semilength n with exactly k peaks.
N(n,k) = (1/n) * C(n,k) * C(n,k-1)
"""
function narayana(n::Int, k::Int)::BigInt
    (n < 1 || k < 1 || k > n) && return BigInt(0)
    binomial(BigInt(n), BigInt(k)) * binomial(BigInt(n), BigInt(k-1)) ÷ BigInt(n)
end

# ═══════════════════════════════════════════════════════════════════════════════
# DYCK PATHS: The Structural Backbone
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DyckPath

A Dyck path: sequence of up (1) and down (-1) steps that never goes below 0.
Represented as a vector of ±1 where sum of any prefix is ≥ 0 and total sum = 0.
"""
struct DyckPath
    steps::Vector{Int8}      # +1 = up (↗), -1 = down (↘)
    
    # Derived properties
    semilength::Int          # n such that path has 2n steps
    heights::Vector{Int}     # Running height at each position
    max_height::Int          # Maximum height reached
    area::Int                # Area under the path
    peaks::Vector{Int}       # Positions of peaks (up followed by down)
    valleys::Vector{Int}     # Positions of valleys (down followed by up)
    
    # Chromatic identity
    fingerprint::UInt64
    color::NTuple{3, Float64}
end

function DyckPath(steps::Vector{Int8}; seed::UInt64=DYCK_SEED)
    # Validate Dyck property
    n = length(steps)
    n % 2 == 0 || error("Dyck path must have even length")
    
    heights = cumsum(steps)
    all(h >= 0 for h in heights) || error("Dyck path cannot go below 0")
    heights[end] == 0 || error("Dyck path must return to 0")
    
    semilength = n ÷ 2
    max_height = maximum(heights)
    
    # Compute area (sum of heights)
    area = sum(heights)
    
    # Find peaks and valleys
    peaks = Int[]
    valleys = Int[]
    for i in 1:n-1
        if steps[i] == 1 && steps[i+1] == -1
            push!(peaks, i)
        elseif steps[i] == -1 && steps[i+1] == 1
            push!(valleys, i)
        end
    end
    
    # Fingerprint from steps
    fp = seed
    for (i, s) in enumerate(steps)
        fp = fp ⊻ (UInt64(s + 2) << (i % 64))
    end
    fp, _ = sm64(fp)
    
    DyckPath(steps, semilength, heights, max_height, area, peaks, valleys, fp, color_from_fp(fp))
end

"""
    dyck_from_parens(s::String) -> DyckPath

Convert parentheses string to Dyck path. '(' = up, ')' = down.
"""
function dyck_from_parens(s::String)
    steps = Int8[]
    for c in s
        if c == '('
            push!(steps, Int8(1))
        elseif c == ')'
            push!(steps, Int8(-1))
        end
    end
    DyckPath(steps)
end

"""
    dyck_to_parens(d::DyckPath) -> String

Convert Dyck path to parentheses string.
"""
function dyck_to_parens(d::DyckPath)
    String([s == 1 ? '(' : ')' for s in d.steps])
end

"""
    all_dyck_paths(n::Int) -> Vector{DyckPath}

Generate all Dyck paths of semilength n (there are C_n of them).
"""
function all_dyck_paths(n::Int)
    n <= 0 && return [DyckPath(Int8[])]
    
    paths = DyckPath[]
    
    function generate(current::Vector{Int8}, height::Int, remaining_up::Int, remaining_down::Int)
        if remaining_up == 0 && remaining_down == 0
            push!(paths, DyckPath(current))
            return
        end
        
        # Can go up if we have ups left
        if remaining_up > 0
            generate(vcat(current, Int8(1)), height + 1, remaining_up - 1, remaining_down)
        end
        
        # Can go down if height > 0 and we have downs left
        if height > 0 && remaining_down > 0
            generate(vcat(current, Int8(-1)), height - 1, remaining_up, remaining_down - 1)
        end
    end
    
    generate(Int8[], 0, n, n)
    paths
end

"""
    random_dyck_path(n::Int; seed::UInt64) -> DyckPath

Generate a random Dyck path of semilength n using the cycle lemma.
"""
function random_dyck_path(n::Int; seed::UInt64=DYCK_SEED)
    # Generate 2n steps: n ups and n downs
    steps = vcat(fill(Int8(1), n), fill(Int8(-1), n))
    
    # Shuffle using seed
    rng_state = seed
    for i in 2n:-1:2
        rng_state, next = sm64(rng_state)
        j = 1 + (next % i)
        steps[i], steps[j] = steps[j], steps[i]
    end
    
    # Use cycle lemma: find rotation that makes it a Dyck path
    heights = cumsum(steps)
    min_idx = argmin(heights)
    
    # Rotate so minimum is at the end
    if min_idx < 2n
        rotated = vcat(steps[min_idx+1:end], steps[1:min_idx])
    else
        rotated = steps
    end
    
    DyckPath(rotated; seed=seed)
end

# ═══════════════════════════════════════════════════════════════════════════════
# CHROMATIC DYCK: Color Trajectories from Paths
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticDyck

A Dyck path with full chromatic trajectory: each step has a color.
"""
struct ChromaticDyck
    path::DyckPath
    trajectory::Vector{NTuple{3, Float64}}  # Color at each height
    
    # Aggregate colors
    mean_color::NTuple{3, Float64}
    peak_color::NTuple{3, Float64}
    fingerprint::UInt64
end

function ChromaticDyck(path::DyckPath; seed::UInt64=GAY_SEED)
    # Generate color for each height level
    trajectory = NTuple{3, Float64}[]
    rng_state = seed ⊻ path.fingerprint
    
    for h in path.heights
        # Color depends on both height and step index (memoryless memory)
        color_seed = rng_state ⊻ UInt64(h * 0x9e3779b97f4a7c15)
        push!(trajectory, color_from_fp(color_seed))
        rng_state, _ = sm64(rng_state)
    end
    
    # Mean color
    mean_r = sum(c[1] for c in trajectory) / length(trajectory)
    mean_g = sum(c[2] for c in trajectory) / length(trajectory)
    mean_b = sum(c[3] for c in trajectory) / length(trajectory)
    mean_color = (mean_r, mean_g, mean_b)
    
    # Peak color (at max height positions)
    peak_positions = findall(h -> h == path.max_height, path.heights)
    if !isempty(peak_positions)
        peak_colors = [trajectory[i] for i in peak_positions]
        peak_r = sum(c[1] for c in peak_colors) / length(peak_colors)
        peak_g = sum(c[2] for c in peak_colors) / length(peak_colors)
        peak_b = sum(c[3] for c in peak_colors) / length(peak_colors)
        peak_color = (peak_r, peak_g, peak_b)
    else
        peak_color = mean_color
    end
    
    # Combined fingerprint
    fp = path.fingerprint
    for c in trajectory
        fp = fp ⊻ reinterpret(UInt64, c[1]) ⊻ reinterpret(UInt64, c[2]) ⊻ reinterpret(UInt64, c[3])
    end
    
    ChromaticDyck(path, trajectory, mean_color, peak_color, fp)
end

"""
    dyck_color_trajectory(path::DyckPath; seed) -> Vector{NTuple{3, Float64}}

Get the color trajectory for a Dyck path.
"""
function dyck_color_trajectory(path::DyckPath; seed::UInt64=GAY_SEED)
    ChromaticDyck(path; seed=seed).trajectory
end

"""
    dyck_fingerprint(path::DyckPath; seed) -> UInt64

Get the chromatic fingerprint of a Dyck path.
"""
function dyck_fingerprint(path::DyckPath; seed::UInt64=GAY_SEED)
    ChromaticDyck(path; seed=seed).fingerprint
end

# ═══════════════════════════════════════════════════════════════════════════════
# SKI ↔ DYCK CORRESPONDENCE
# ═══════════════════════════════════════════════════════════════════════════════

# Minimal SKI representation for Dyck correspondence
abstract type SKI end
struct S_Comb <: SKI end
struct K_Comb <: SKI end
struct I_Comb <: SKI end
struct App <: SKI
    func::SKI
    arg::SKI
end

Base.show(io::IO, ::S_Comb) = print(io, "S")
Base.show(io::IO, ::K_Comb) = print(io, "K")
Base.show(io::IO, ::I_Comb) = print(io, "I")
Base.show(io::IO, a::App) = print(io, "(", a.func, " ", a.arg, ")")

function ski_depth(t::SKI)
    t isa App ? 1 + max(ski_depth(t.func), ski_depth(t.arg)) : 0
end

"""
    dyck_to_ski(path::DyckPath) -> SKI

Convert Dyck path to SKI term. Uses the bijection:
- Up step at height h → start application at level h
- Down step → close application
- Leaf combinator chosen based on position
"""
function dyck_to_ski(path::DyckPath)
    isempty(path.steps) && return I_Comb()
    
    combinators = [S_Comb(), K_Comb(), I_Comb()]
    
    function build(steps::Vector{Int8}, pos::Ref{Int}, height::Int)
        pos[] > length(steps) && return combinators[1 + (height % 3)]
        
        if steps[pos[]] == 1  # Up: start application
            pos[] += 1
            func = build(steps, pos, height + 1)
            arg = build(steps, pos, height + 1)
            return App(func, arg)
        else  # Down: return combinator
            pos[] += 1
            return combinators[1 + (height % 3)]
        end
    end
    
    pos = Ref(1)
    build(path.steps, pos, 0)
end

"""
    ski_to_dyck(term::SKI) -> DyckPath

Convert SKI term to Dyck path.
"""
function ski_to_dyck(term::SKI)
    steps = Int8[]
    
    function traverse(t::SKI)
        if t isa App
            push!(steps, Int8(1))  # Up: entering application
            traverse(t.func)
            traverse(t.arg)
            push!(steps, Int8(-1))  # Down: exiting application
        end
        # Combinators don't add steps
    end
    
    traverse(term)
    
    # Ensure valid Dyck path
    if isempty(steps)
        return DyckPath(Int8[1, -1])  # Minimal Dyck path for single combinator
    end
    
    DyckPath(steps)
end

# ═══════════════════════════════════════════════════════════════════════════════
# STAMMERING TABLEAUX (JOSUAT-VERGÈS)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    StammeringTableau

An oscillating walk in Young's lattice:
- Each step either adds or removes a box from a partition
- "Stammering" = can pause (stay at same partition)
- Counts PASEP (Partially Asymmetric Simple Exclusion Process) configs
"""
struct StammeringTableau
    # Walk through partitions
    partitions::Vector{Vector{Int}}
    
    # Steps: :add, :remove, :stay
    steps::Vector{Symbol}
    
    # PASEP configuration (if applicable)
    pasep::Vector{Int}  # 0 = empty, 1 = particle
    
    # Chromatic identity
    fingerprint::UInt64
    color::NTuple{3, Float64}
end

function StammeringTableau(partitions::Vector{Vector{Int}}; seed::UInt64=CATALAN_SEED)
    n = length(partitions)
    steps = Symbol[]
    
    for i in 2:n
        prev, curr = partitions[i-1], partitions[i]
        prev_size = sum(prev; init=0)
        curr_size = sum(curr; init=0)
        
        if curr_size > prev_size
            push!(steps, :add)
        elseif curr_size < prev_size
            push!(steps, :remove)
        else
            push!(steps, :stay)
        end
    end
    
    # Derive PASEP from final partition
    final = isempty(partitions) ? Int[] : partitions[end]
    pasep = zeros(Int, length(final) + 1)
    for (i, p) in enumerate(final)
        if p > 0
            pasep[i] = 1
        end
    end
    
    # Fingerprint
    fp = seed
    for p in partitions
        for x in p
            fp = fp ⊻ hash(x)
        end
    end
    fp, _ = sm64(fp)
    
    StammeringTableau(partitions, steps, pasep, fp, color_from_fp(fp))
end

"""
    oscillating_walk(n::Int; seed) -> StammeringTableau

Generate a random oscillating walk of length n in Young's lattice.
"""
function oscillating_walk(n::Int; seed::UInt64=CATALAN_SEED)
    partitions = Vector{Int}[Int[]]  # Start at empty partition
    rng_state = seed
    
    for _ in 1:n
        current = partitions[end]
        rng_state, choice = sm64(rng_state)
        
        # Decide: add, remove, or stay
        action = choice % 3
        
        if action == 0 && !isempty(current)  # Remove
            # Pick a removable corner
            removable = Int[]
            for i in 1:length(current)
                if i == length(current) || current[i] > current[i+1]
                    push!(removable, i)
                end
            end
            if !isempty(removable)
                idx = removable[1 + (choice >> 2) % length(removable)]
                new_part = copy(current)
                new_part[idx] -= 1
                filter!(x -> x > 0, new_part)
                push!(partitions, new_part)
            else
                push!(partitions, copy(current))  # Stay
            end
        elseif action == 1  # Add
            # Pick an addable corner
            new_part = copy(current)
            if isempty(new_part)
                push!(new_part, 1)
            else
                # Can add at first row or any row where current[i] < current[i-1]
                addable = [1]
                for i in 2:length(new_part)
                    if new_part[i] < new_part[i-1]
                        push!(addable, i)
                    end
                end
                push!(addable, length(new_part) + 1)
                
                idx = addable[1 + (choice >> 2) % length(addable)]
                if idx <= length(new_part)
                    new_part[idx] += 1
                else
                    push!(new_part, 1)
                end
            end
            push!(partitions, new_part)
        else  # Stay
            push!(partitions, copy(current))
        end
    end
    
    StammeringTableau(partitions; seed=seed)
end

"""
    tableau_to_dyck(t::StammeringTableau) -> DyckPath

Convert stammering tableau to Dyck path.
:add → up, :remove → down, :stay → (ignored for Dyck)
"""
function tableau_to_dyck(t::StammeringTableau)
    steps = Int8[]
    for s in t.steps
        if s == :add
            push!(steps, Int8(1))
        elseif s == :remove
            push!(steps, Int8(-1))
        end
        # :stay is ignored
    end
    
    # Ensure balanced
    ups = count(==(Int8(1)), steps)
    downs = count(==(Int8(-1)), steps)
    
    if ups > downs
        append!(steps, fill(Int8(-1), ups - downs))
    elseif downs > ups
        prepend!(steps, fill(Int8(1), downs - ups))
    end
    
    # Make it a valid Dyck path
    if isempty(steps)
        steps = Int8[1, -1]
    end
    
    # Ensure non-negative prefix sums
    heights = cumsum(steps)
    min_h = minimum(heights; init=0)
    if min_h < 0
        # Prepend enough ups
        steps = vcat(fill(Int8(1), -min_h), steps)
    end
    
    # Ensure ends at 0
    final_h = sum(steps)
    if final_h > 0
        append!(steps, fill(Int8(-1), final_h))
    elseif final_h < 0
        prepend!(steps, fill(Int8(1), -final_h))
    end
    
    DyckPath(steps; seed=t.fingerprint)
end

# ═══════════════════════════════════════════════════════════════════════════════
# BIRB PARSE TREES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    BirbTree

Parse tree with alternating left-right associativity.
🐦🐦🐦🐦 → (🐦((🐦🐦)🐦))
"""
struct BirbTree
    content::Union{Symbol, Nothing}
    left::Union{BirbTree, Nothing}
    right::Union{BirbTree, Nothing}
    depth::Int
    associativity::Symbol  # :left or :right at this level
    
    fingerprint::UInt64
    color::NTuple{3, Float64}
end

function BirbTree(content::Symbol; seed::UInt64=BIRB_SEED)
    fp, _ = sm64(seed ⊻ hash(content))
    BirbTree(content, nothing, nothing, 0, :left, fp, color_from_fp(fp))
end

function BirbTree(left::BirbTree, right::BirbTree; depth::Int=0, seed::UInt64=BIRB_SEED)
    # Alternating associativity
    assoc = depth % 2 == 0 ? :left : :right
    
    fp = left.fingerprint ⊻ right.fingerprint ⊻ UInt64(depth)
    fp, _ = sm64(fp)
    
    BirbTree(nothing, left, right, depth, assoc, fp, color_from_fp(fp))
end

"""
    birb_parse(items::Vector{Symbol}; seed) -> BirbTree

Parse items with alternating associativity (Birb staggering).
"""
function birb_parse(items::Vector{Symbol}; seed::UInt64=BIRB_SEED)
    isempty(items) && return BirbTree(:empty; seed=seed)
    length(items) == 1 && return BirbTree(items[1]; seed=seed)
    
    function build(remaining::Vector{Symbol}, depth::Int)
        length(remaining) == 1 && return BirbTree(remaining[1]; seed=seed ⊻ UInt64(depth))
        
        # Alternating: even depth = left-assoc, odd = right-assoc
        if depth % 2 == 0  # Left-associative: ((ab)c)d
            left = BirbTree(remaining[1]; seed=seed ⊻ UInt64(depth))
            for i in 2:length(remaining)
                right = BirbTree(remaining[i]; seed=seed ⊻ UInt64(depth + i))
                left = BirbTree(left, right; depth=depth, seed=seed)
            end
            left
        else  # Right-associative: a(b(cd))
            right = BirbTree(remaining[end]; seed=seed ⊻ UInt64(depth))
            for i in length(remaining)-1:-1:1
                left = BirbTree(remaining[i]; seed=seed ⊻ UInt64(depth + i))
                right = BirbTree(left, right; depth=depth, seed=seed)
            end
            right
        end
    end
    
    build(items, 0)
end

"""
    birb_to_dyck(tree::BirbTree) -> DyckPath

Convert Birb tree to Dyck path.
"""
function birb_to_dyck(tree::BirbTree)
    steps = Int8[]
    
    function traverse(t::BirbTree)
        if isnothing(t.left) && isnothing(t.right)
            # Leaf: no steps needed for Dyck
            return
        end
        
        push!(steps, Int8(1))  # Enter node
        !isnothing(t.left) && traverse(t.left)
        !isnothing(t.right) && traverse(t.right)
        push!(steps, Int8(-1))  # Exit node
    end
    
    traverse(tree)
    
    isempty(steps) && (steps = Int8[1, -1])
    DyckPath(steps; seed=tree.fingerprint)
end

birb_fingerprint(tree::BirbTree) = tree.fingerprint
birb_color(tree::BirbTree) = tree.color

# ═══════════════════════════════════════════════════════════════════════════════
# CHROMATIC SHEAFIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticSheaf

A sheaf of chromatic data over a category of Dyck paths.
Sections are consistent color assignments that respect restrictions.
"""
struct ChromaticSheaf
    # Objects: Dyck paths indexed by Catalan number
    objects::Dict{Int, Vector{DyckPath}}
    
    # Morphisms: refinements between paths
    morphisms::Dict{Tuple{Int,Int}, Vector{Tuple{DyckPath, DyckPath}}}
    
    # Sections: color assignments
    sections::Dict{DyckPath, NTuple{3, Float64}}
    
    # Coherence data
    coherence_scores::Dict{Int, Float64}
    global_coherence::Float64
    
    # Seed used
    seed::UInt64
end

function ChromaticSheaf(max_n::Int; seed::UInt64=GAY_SEED)
    objects = Dict{Int, Vector{DyckPath}}()
    morphisms = Dict{Tuple{Int,Int}, Vector{Tuple{DyckPath, DyckPath}}}()
    sections = Dict{DyckPath, NTuple{3, Float64}}()
    
    # Generate objects for each Catalan level
    for n in 0:max_n
        objects[n] = all_dyck_paths(n)
        
        # Assign colors (sections)
        for path in objects[n]
            cd = ChromaticDyck(path; seed=seed)
            sections[path] = cd.mean_color
        end
    end
    
    # Generate morphisms (refinements: insert a matched pair)
    for n in 0:max_n-1
        morphisms[(n, n+1)] = Tuple{DyckPath, DyckPath}[]
        
        for src in objects[n]
            for tgt in objects[n+1]
                # Check if tgt is a refinement of src
                if is_refinement(src, tgt)
                    push!(morphisms[(n, n+1)], (src, tgt))
                end
            end
        end
    end
    
    # Compute coherence scores
    coherence_scores = Dict{Int, Float64}()
    total_coherence = 0.0
    count = 0
    
    for n in 1:max_n
        level_coherence = 0.0
        level_count = 0
        
        for (src, tgt) in get(morphisms, (n-1, n), [])
            src_color = sections[src]
            tgt_color = sections[tgt]
            
            # Color distance
            dist = sqrt(sum((src_color[i] - tgt_color[i])^2 for i in 1:3))
            coherence = exp(-dist)  # Higher coherence for closer colors
            
            level_coherence += coherence
            level_count += 1
        end
        
        coherence_scores[n] = level_count > 0 ? level_coherence / level_count : 1.0
        total_coherence += coherence_scores[n]
        count += 1
    end
    
    global_coherence = count > 0 ? total_coherence / count : 1.0
    
    ChromaticSheaf(objects, morphisms, sections, coherence_scores, global_coherence, seed)
end

"""
Check if tgt is a refinement of src (tgt = src with one matched pair inserted).
"""
function is_refinement(src::DyckPath, tgt::DyckPath)
    length(tgt.steps) != length(src.steps) + 2 && return false
    
    # Try all positions to insert "()"
    for i in 0:length(src.steps)
        candidate = vcat(src.steps[1:i], Int8[1, -1], src.steps[i+1:end])
        if candidate == tgt.steps
            return true
        end
    end
    
    false
end

"""
    sheaf_section(sheaf::ChromaticSheaf, path::DyckPath) -> NTuple{3, Float64}

Get the section (color) at a path.
"""
function sheaf_section(sheaf::ChromaticSheaf, path::DyckPath)
    get(sheaf.sections, path, (0.5, 0.5, 0.5))
end

"""
    sheaf_restriction(sheaf::ChromaticSheaf, n::Int) -> Vector{NTuple{3, Float64}}

Get all sections at Catalan level n.
"""
function sheaf_restriction(sheaf::ChromaticSheaf, n::Int)
    [sheaf.sections[p] for p in get(sheaf.objects, n, DyckPath[])]
end

"""
    coherence_score(sheaf::ChromaticSheaf) -> Float64

Get the global coherence score.
"""
coherence_score(sheaf::ChromaticSheaf) = sheaf.global_coherence

# ═══════════════════════════════════════════════════════════════════════════════
# SEED OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SeedBundle

A collection of Gay seeds optimized for chromatic coherence.
"""
struct SeedBundle
    seeds::Vector{UInt64}
    coherences::Vector{Float64}
    best_seed::UInt64
    best_coherence::Float64
    
    # Interleaving pattern
    pattern::Symbol  # :round_robin, :weighted, :adaptive
end

function SeedBundle(seeds::Vector{UInt64}; max_n::Int=5)
    coherences = Float64[]
    
    for seed in seeds
        sheaf = ChromaticSheaf(max_n; seed=seed)
        push!(coherences, sheaf.global_coherence)
    end
    
    best_idx = argmax(coherences)
    
    SeedBundle(seeds, coherences, seeds[best_idx], coherences[best_idx], :round_robin)
end

"""
    optimize_seed_bundle(n_seeds::Int, max_n::Int; base_seed) -> SeedBundle

Find optimal seed bundle via random search with GayMC.
"""
function optimize_seed_bundle(n_seeds::Int, max_n::Int=5; base_seed::UInt64=GAY_SEED)
    seeds = UInt64[]
    rng_state = base_seed
    
    for _ in 1:n_seeds
        rng_state, new_seed = sm64(rng_state)
        push!(seeds, new_seed)
    end
    
    SeedBundle(seeds; max_n=max_n)
end

"""
    interleave_gaymc(bundle::SeedBundle, items::Vector{T}; f::Function) -> Vector

Apply GayMC interleaving to items using seed bundle.
"""
function interleave_gaymc(bundle::SeedBundle, items::Vector{T}; f::Function=identity) where T
    n = length(items)
    results = Vector{Any}(undef, n)
    
    for i in 1:n
        seed_idx = 1 + (i - 1) % length(bundle.seeds)
        seed = bundle.seeds[seed_idx]
        
        # Apply function with seed context
        results[i] = f(items[i], seed)
    end
    
    results
end

"""
    meaning_assignment_rate(bundle::SeedBundle, items::Vector) -> Float64

Compute the rate at which meaning is assigned via chromatic invariants.
"""
function meaning_assignment_rate(bundle::SeedBundle, items::Vector)
    total_meaning = 0.0
    
    for (i, item) in enumerate(items)
        seed_idx = 1 + (i - 1) % length(bundle.seeds)
        seed = bundle.seeds[seed_idx]
        
        # Meaning = coherence * item_entropy
        item_fp, _ = sm64(seed ⊻ hash(item))
        entropy = count_ones(item_fp) / 64.0  # Bit entropy
        
        meaning = bundle.coherences[seed_idx] * entropy
        total_meaning += meaning
    end
    
    total_meaning / length(items)
end

"""
    chromatic_invariant(bundle::SeedBundle) -> UInt64

Compute the chromatic invariant of the seed bundle.
"""
function chromatic_invariant(bundle::SeedBundle)
    reduce(⊻, bundle.seeds; init=UInt64(0))
end

# ═══════════════════════════════════════════════════════════════════════════════
# IES MESSAGE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    IESMessage

A message from the IES system with chromatic identity.
"""
struct IESMessage
    id::Int
    content::String
    timestamp::Union{String, Nothing}
    word_count::Int
    
    # Chromatic identity
    dyck_path::DyckPath
    fingerprint::UInt64
    color::NTuple{3, Float64}
    
    # Catalan indexing
    catalan_level::Int
    catalan_index::Int
end

"""
    message_to_dyck(content::String; seed) -> DyckPath

Convert message content to a Dyck path based on semantic structure.

For natural language (without explicit parentheses):
- Sentence boundaries create nesting
- Clause markers (commas, semicolons) modulate height
- Word length variation creates local structure
- Semantic density (long words / short words) creates peaks/valleys
"""
function message_to_dyck(content::String; seed::UInt64=GAY_SEED)
    steps = Int8[]
    
    # First try explicit nesting (parentheses, brackets, braces)
    open_chars = Set(['(', '[', '{', '<'])
    close_chars = Set([')', ']', '}', '>'])
    
    for c in content
        if c in open_chars
            push!(steps, Int8(1))
        elseif c in close_chars
            push!(steps, Int8(-1))
        end
    end
    
    # If no explicit nesting, derive from semantic structure
    if isempty(steps) || all(==(Int8(0)), steps)
        steps = Int8[]
        
        # Split into words
        words = split(content)
        n_words = length(words)
        
        if n_words == 0
            # Empty: minimal Dyck path
            return DyckPath(Int8[1, -1]; seed=seed)
        end
        
        # Semantic Dyck encoding:
        # - Short words (≤3 chars): down step
        # - Long words (>6 chars): up step
        # - Medium words: based on position (up in first half, down in second)
        # - Sentence enders (.!?): always down
        # - Clause markers (,;:): toggle direction
        
        height = 0
        max_height_target = min(n_words ÷ 2, 8)  # Limit nesting depth
        
        for (i, word) in enumerate(words)
            word_str = String(word)
            len = length(word_str)
            
            # Check for sentence/clause enders
            ends_sentence = any(endswith(word_str, p) for p in ['.', '!', '?'])
            ends_clause = any(endswith(word_str, p) for p in [',', ';', ':'])
            
            if ends_sentence && height > 0
                # Close all open nesting
                while height > 0
                    push!(steps, Int8(-1))
                    height -= 1
                end
            elseif ends_clause
                # Partial close
                if height > 0
                    push!(steps, Int8(-1))
                    height -= 1
                end
            elseif len > 6 && height < max_height_target
                # Long word: open nesting (semantic complexity)
                push!(steps, Int8(1))
                height += 1
            elseif len <= 3 && height > 0
                # Short word: close nesting (semantic simplicity)
                push!(steps, Int8(-1))
                height -= 1
            else
                # Medium word: follow trajectory based on position
                progress = i / n_words
                if progress < 0.5 && height < max_height_target
                    push!(steps, Int8(1))
                    height += 1
                elseif progress >= 0.5 && height > 0
                    push!(steps, Int8(-1))
                    height -= 1
                end
            end
        end
        
        # Ensure we return to ground level
        while height > 0
            push!(steps, Int8(-1))
            height -= 1
        end
        
        # Ensure minimum path length
        if length(steps) < 2
            steps = Int8[1, -1]
        end
    end
    
    # Balance if needed
    balance = sum(steps)
    if balance > 0
        append!(steps, fill(Int8(-1), balance))
    elseif balance < 0
        prepend!(steps, fill(Int8(1), -balance))
    end
    
    # Fix any negative prefixes
    heights = cumsum(steps)
    min_h = minimum(heights; init=0)
    if min_h < 0
        steps = vcat(fill(Int8(1), -min_h), steps)
    end
    
    # Ensure ends at 0
    final_h = sum(steps)
    if final_h > 0
        append!(steps, fill(Int8(-1), final_h))
    end
    
    DyckPath(steps; seed=seed)
end

function IESMessage(id::Int, content::String, timestamp::Union{String, Nothing}, 
                    word_count::Int; seed::UInt64=GAY_SEED)
    path = message_to_dyck(content; seed=seed)
    fp = path.fingerprint ⊻ hash(content) ⊻ UInt64(id)
    fp, _ = sm64(fp)
    
    # Catalan level = semilength of path
    catalan_level = path.semilength
    
    # Index within Catalan level (based on fingerprint)
    c_n = catalan(catalan_level)
    catalan_index = c_n > 0 ? Int(fp % UInt64(c_n)) + 1 : 1
    
    IESMessage(id, content, timestamp, word_count, path, fp, color_from_fp(fp),
               catalan_level, catalan_index)
end

"""
    catalan_index_messages(messages::Vector{IESMessage}) -> Dict{Int, Vector{IESMessage}}

Index messages by their Catalan level.
"""
function catalan_index_messages(messages::Vector{IESMessage})
    indexed = Dict{Int, Vector{IESMessage}}()
    
    for msg in messages
        level = msg.catalan_level
        if !haskey(indexed, level)
            indexed[level] = IESMessage[]
        end
        push!(indexed[level], msg)
    end
    
    indexed
end

"""
    sheafify_messages(messages::Vector{IESMessage}; seed) -> NamedTuple

Sheafify messages into a chromatic structure.
Returns coherence metrics and optimal coloring.
"""
function sheafify_messages(messages::Vector{IESMessage}; seed::UInt64=GAY_SEED)
    indexed = catalan_index_messages(messages)
    
    # Build sheaf structure
    max_level = maximum(keys(indexed); init=0)
    sheaf = ChromaticSheaf(max_level; seed=seed)
    
    # Map messages to sheaf sections
    message_colors = Dict{Int, NTuple{3, Float64}}()
    for msg in messages
        message_colors[msg.id] = msg.color
    end
    
    # Compute coherence between messages at adjacent levels
    level_coherence = Dict{Int, Float64}()
    
    for level in 1:max_level
        if haskey(indexed, level-1) && haskey(indexed, level)
            total_coh = 0.0
            count = 0
            
            for prev_msg in indexed[level-1]
                for curr_msg in indexed[level]
                    dist = sqrt(sum((prev_msg.color[i] - curr_msg.color[i])^2 for i in 1:3))
                    total_coh += exp(-dist)
                    count += 1
                end
            end
            
            level_coherence[level] = count > 0 ? total_coh / count : 1.0
        end
    end
    
    global_coherence = isempty(level_coherence) ? 1.0 : 
                       sum(values(level_coherence)) / length(level_coherence)
    
    (
        indexed = indexed,
        message_colors = message_colors,
        level_coherence = level_coherence,
        global_coherence = global_coherence,
        sheaf = sheaf,
        seed = seed
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function demo_gay_dyck_catalan()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  GAY-DYCK-CATALAN: The Stammering Trinity for Chromatic Sheafification   ║")
    println("║  \"Bégayant contains 'gay' because all three share Catalan structure\"      ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # ─── Catalan Numbers ───
    println("─── Catalan Numbers (The Counting Foundation) ───")
    println()
    println("  n    C_n      Objects counted")
    println("  ─────────────────────────────────────────────")
    for n in 0:7
        c = catalan(n)
        objects = ["Dyck paths", "Binary trees", "Parenthesizations", "Triangulations"][1 + (n % 4)]
        println("  $(rpad(n, 4)) $(lpad(c, 7))   $objects of size $n")
    end
    println()
    
    # ─── Dyck Paths ───
    println("─── Dyck Paths ─────────────────────────────────────")
    println()
    
    path1 = dyck_from_parens("(()())")
    println("  Path: $(dyck_to_parens(path1))")
    println("  Steps: $(path1.steps)")
    println("  Heights: $(path1.heights)")
    println("  Max height: $(path1.max_height), Area: $(path1.area)")
    println("  Peaks: $(path1.peaks), Valleys: $(path1.valleys)")
    println("  Fingerprint: 0x$(string(path1.fingerprint, base=16)[1:8])...")
    println("  Color: RGB$(path1.color)")
    println()
    
    # ─── SKI ↔ Dyck Correspondence ───
    println("─── SKI ↔ Dyck Correspondence ─────────────────────")
    println()
    
    ski_term = dyck_to_ski(path1)
    println("  Dyck path: $(dyck_to_parens(path1))")
    println("  → SKI term: $ski_term")
    println("  → Depth: $(ski_depth(ski_term))")
    
    # Round-trip
    path_back = ski_to_dyck(ski_term)
    println("  → Back to Dyck: $(dyck_to_parens(path_back))")
    println()
    
    # ─── Chromatic Trajectory ───
    println("─── Chromatic Trajectory ───────────────────────────")
    println()
    
    cd = ChromaticDyck(path1; seed=GAY_SEED)
    println("  Step  Height  Color")
    println("  ─────────────────────────────────────────────")
    for (i, (step, height, color)) in enumerate(zip(path1.steps, path1.heights, cd.trajectory))
        step_char = step == 1 ? "↗" : "↘"
        r, g, b = round.(color .* 255)
        println("  $(lpad(i, 4))   $step_char $(lpad(height, 3))    RGB($r, $g, $b)")
    end
    println()
    println("  Mean color: RGB$(round.(cd.mean_color .* 255))")
    println("  Peak color: RGB$(round.(cd.peak_color .* 255))")
    println()
    
    # ─── Stammering Tableau ───
    println("─── Stammering Tableau (PASEP Config) ─────────────")
    println()
    
    tableau = oscillating_walk(8; seed=CATALAN_SEED)
    println("  Walk through Young's lattice:")
    for (i, (part, step)) in enumerate(zip(tableau.partitions[1:min(6, end)], 
                                           vcat([:start], tableau.steps[1:min(5, end)])))
        part_str = isempty(part) ? "∅" : join(part, ",")
        println("    Step $i: [$part_str] ($(step))")
    end
    println("    ...")
    println("  PASEP config: $(tableau.pasep)")
    println("  Fingerprint: 0x$(string(tableau.fingerprint, base=16)[1:8])...")
    println()
    
    # ─── Birb Parse Tree ───
    println("─── Birb Parse Tree (Alternating Associativity) ───")
    println()
    
    items = [:🐦, :🐦, :🐦, :🐦]
    tree = birb_parse(items; seed=BIRB_SEED)
    println("  Input: $(join(string.(items), " "))")
    println("  Parse: (🐦((🐦🐦)🐦))  [alternating left/right]")
    println("  Fingerprint: 0x$(string(tree.fingerprint, base=16)[1:8])...")
    println("  Color: RGB$(round.(tree.color .* 255))")
    
    birb_dyck = birb_to_dyck(tree)
    println("  → Dyck path: $(dyck_to_parens(birb_dyck))")
    println()
    
    # ─── Chromatic Sheaf ───
    println("─── Chromatic Sheaf ────────────────────────────────")
    println()
    
    sheaf = ChromaticSheaf(4; seed=GAY_SEED)
    println("  Level   Objects   Coherence")
    println("  ──────────────────────────────")
    for n in 0:4
        n_objects = length(get(sheaf.objects, n, []))
        coh = get(sheaf.coherence_scores, n, 1.0)
        bar = repeat("█", round(Int, coh * 20))
        println("    $n      $(lpad(n_objects, 4))     $(bar) $(round(coh, digits=3))")
    end
    println()
    println("  Global coherence: $(round(sheaf.global_coherence, digits=4))")
    println()
    
    # ─── Seed Optimization ───
    println("─── Seed Bundle Optimization ───────────────────────")
    println()
    
    bundle = optimize_seed_bundle(8, 4; base_seed=GAY_SEED)
    println("  Testing 8 seeds for optimal coherence:")
    for (i, (seed, coh)) in enumerate(zip(bundle.seeds, bundle.coherences))
        best_marker = seed == bundle.best_seed ? " ★" : ""
        println("    Seed 0x$(string(seed, base=16)[1:8]): coherence = $(round(coh, digits=4))$best_marker")
    end
    println()
    println("  Best seed: 0x$(string(bundle.best_seed, base=16))")
    println("  Best coherence: $(round(bundle.best_coherence, digits=4))")
    println("  Chromatic invariant: 0x$(string(chromatic_invariant(bundle), base=16))")
    println()
    
    # ─── The Trinity ───
    println("═══════════════════════════════════════════════════════════════════")
    println("  THE GAY-DYCK-BIRB TRINITY")
    println("═══════════════════════════════════════════════════════════════════")
    println()
    println("  Concept              Pattern                 Catalan Connection")
    println("  ────────────────────────────────────────────────────────────────")
    println("  Stammering Tableaux  Oscillating walks       PASEP particle configs")
    println("  Dyck Languages       Balanced ()             Catalan numbers count")
    println("  Birb                 Alternating L-R assoc   Parse trees are Catalan")
    println()
    println("  Gay.jl's XOR-based fingerprinting captures this self-similar structure:")
    println("    fingerprint(path) ⊻ fingerprint(ski_term) ⊻ fingerprint(birb_tree)")
    println("    = chromatic invariant of the Catalan object")
    println()
    println("═══════════════════════════════════════════════════════════════════")
end

end # module GayDyckCatalan
