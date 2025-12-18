# SPC REPL: Symbolic · Possible · Compositional
# Launch with SPC (Space bar) - tight ReplMaker.jl integration
#
# The SPC trichotomy:
#   S - Symbolic: Color/music chains as computable symbols
#   P - Possible: Counterfactual worlds via seed perturbation  
#   C - Compositional: Obstructions and fixed points in ⊗-chains

using ReplMaker
using Colors: RGB, HSL

export init_spc_repl, spc_eval

# ═══════════════════════════════════════════════════════════════════════════
# Core State: The SPC World
# ═══════════════════════════════════════════════════════════════════════════

mutable struct SPCWorld
    seed::UInt64
    chain::Vector{RGB{Float64}}     # Color chain
    notes::Vector{Int}               # Musical notes (pitch classes)
    intervals::Vector{Int}           # Interval chain
    fixpoints::Vector{Int}           # Indices where σ(i) = i
    obstructions::Vector{String}     # Compositionality failures
    counterfactuals::Dict{UInt64, Vector{RGB{Float64}}}  # seed → chain
end

const SPC_WORLD = Ref{SPCWorld}()

function init_world(seed::UInt64=GAY_SEED)
    chain = [color_at(i; seed=seed) for i in 1:12]
    notes = [hue_to_pc(c) for c in chain]
    intervals = [(notes[i+1] - notes[i] + 12) % 12 for i in 1:11]
    fixpoints = findall(i -> notes[i] == i - 1, 1:12)  # σ(i) = i (0-indexed)
    SPC_WORLD[] = SPCWorld(seed, chain, notes, intervals, fixpoints, String[], Dict())
end

# hue_to_pc defined in whale_world.jl - reuse it here

# NOTE_NAMES also defined in whale_world.jl
const SEMIRING_NAMES = ["min+", "max+", "min×", "max×", "∨∧", "gcd"]

# ═══════════════════════════════════════════════════════════════════════════
# ANSI Color Display
# ═══════════════════════════════════════════════════════════════════════════

function rgb_block(c::RGB)
    r, g, b = round.(Int, clamp.((c.r, c.g, c.b), 0, 1) .* 255)
    "\e[48;2;$(r);$(g);$(b)m  \e[0m"
end

function rainbow(s::String)
    colors = [(228,3,3), (255,140,0), (255,237,0), (0,128,38), (0,77,255), (117,7,135)]
    join(["\e[38;2;$(c[1]);$(c[2]);$(c[3])m$(ch)\e[0m" for (ch, c) in zip(s, Iterators.cycle(colors))])
end

# ═══════════════════════════════════════════════════════════════════════════
# SPC Commands
# ═══════════════════════════════════════════════════════════════════════════

const SPC_CMDS = Dict{String, Function}()

# --- Symbolic Commands ---

SPC_CMDS["chain"] = function(args...)
    w = SPC_WORLD[]
    println("  ╭─ Chain (seed 0x$(string(w.seed, base=16))) ─╮")
    print("  │ ")
    for c in w.chain
        r = round(Int, clamp(c.r, 0, 1) * 255)
        g = round(Int, clamp(c.g, 0, 1) * 255)
        b = round(Int, clamp(c.b, 0, 1) * 255)
        print("\e[48;2;$(r);$(g);$(b)m  \e[0m")
    end
    println(" │")
    print("  │ ")
    for n in w.notes
        print(" $(NOTE_NAMES[n+1])")
        length(NOTE_NAMES[n+1]) == 1 && print(" ")
    end
    println("│")
    println("  ╰────────────────────────────────────╯")
end

SPC_CMDS["notes"] = function(args...)
    w = SPC_WORLD[]
    println("  $(join([NOTE_NAMES[n+1] for n in w.notes], "-"))")
    println("  Intervals: $(join(w.intervals, "-"))")
    println("  Unique PCs: $(length(unique(w.notes)))/12")
end

SPC_CMDS["play"] = function(args...)
    w = SPC_WORLD[]
    
    # Add octave variation based on color lightness
    octaves = [4 + Int(floor(convert(HSL, c).l * 2)) for c in w.chain]
    midis = [60 + n + (octaves[i] - 4) * 12 for (i, n) in enumerate(w.notes)]
    freqs = [440.0 * 2^((m - 69) / 12.0) for m in midis]
    
    println("  ♪ Playing chain...")
    print("    ")
    for (i, n) in enumerate(w.notes)
        oct = octaves[i]
        print("$(NOTE_NAMES[n+1])$oct ")
    end
    println()
    
    # Python audio synthesis with envelope
    freq_str = join(["$(round(f, digits=1))" for f in freqs], ",")
    
    try
        cmd = ```python3 -c "
import wave, struct, math

freqs = [$(freq_str)]
sr, dur = 22050, 0.25
out = b''

for freq in freqs:
    for i in range(int(sr * dur)):
        t = i / sr
        # ADSR envelope
        attack = min(1.0, t / 0.02)
        decay = max(0.0, 1.0 - (t - dur + 0.05) / 0.05) if t > dur - 0.05 else 1.0
        env = attack * decay * 0.5
        # Sine with slight 5th harmonic
        s = math.sin(2 * math.pi * freq * t) + 0.15 * math.sin(4 * math.pi * freq * t)
        sample = int(32767 * env * s)
        out += struct.pack('<h', max(-32767, min(32767, sample)))

with wave.open('/tmp/spc_chain.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_chain.wav`, wait=true)
        println("    ✓ Audio complete")
    catch e
        println("    (audio unavailable: $e)")
    end
end

# --- Possible World Commands ---

SPC_CMDS["cf"] = function(args...)  # counterfactual
    delta = isempty(args) ? 1 : parse(Int, args[1])
    w = SPC_WORLD[]
    alt_seed = w.seed + UInt64(delta)
    alt_chain = [color_at(i; seed=alt_seed) for i in 1:12]
    alt_notes = [hue_to_pc(c) for c in alt_chain]
    
    w.counterfactuals[alt_seed] = alt_chain
    
    println("  ◇ Counterfactual world (Δ=$delta):")
    print("    Actual:   ")
    for c in w.chain
        r = round(Int, clamp(c.r, 0, 1) * 255)
        g = round(Int, clamp(c.g, 0, 1) * 255)
        b = round(Int, clamp(c.b, 0, 1) * 255)
        print("\e[48;2;$(r);$(g);$(b)m  \e[0m")
    end
    println()
    print("    Possible: ")
    for c in alt_chain
        r = round(Int, clamp(c.r, 0, 1) * 255)
        g = round(Int, clamp(c.g, 0, 1) * 255)
        b = round(Int, clamp(c.b, 0, 1) * 255)
        print("\e[48;2;$(r);$(g);$(b)m  \e[0m")
    end
    println()
    
    # Divergence analysis
    diverged = sum(w.notes[i] != alt_notes[i] for i in 1:12)
    common = [NOTE_NAMES[n+1] for n in intersect(Set(w.notes), Set(alt_notes))]
    println("    Divergence: $diverged/12 notes differ")
    println("    Common PCs: {$(join(common, ","))}")
end

SPC_CMDS["worlds"] = function(args...)
    w = SPC_WORLD[]
    println("  Explored worlds: $(length(w.counterfactuals) + 1)")
    println("    ★ 0x$(string(w.seed, base=16)) (actual)")
    for (s, _) in w.counterfactuals
        d = Int(s) - Int(w.seed)
        println("    ◇ 0x$(string(s, base=16)) (Δ=$d)")
    end
end

SPC_CMDS["modal"] = function(args...)
    # Kripke-style modal reasoning
    w = SPC_WORLD[]
    n_worlds = 5
    
    println("  ◇ Modal necessity analysis:")
    
    # Check which properties hold in ALL nearby worlds
    necessary_pcs = Set(w.notes)
    for delta in 1:n_worlds
        alt_seed = w.seed + UInt64(delta)
        alt_notes = [hue_to_pc(color_at(i; seed=alt_seed)) for i in 1:12]
        intersect!(necessary_pcs, Set(alt_notes))
    end
    
    println("    □ Necessary PCs (in all $n_worlds worlds): {$(join([NOTE_NAMES[n+1] for n in sort(collect(necessary_pcs))], ","))}")
    
    possible_pcs = Set(w.notes)
    for delta in 1:n_worlds
        alt_seed = w.seed + UInt64(delta)
        alt_notes = [hue_to_pc(color_at(i; seed=alt_seed)) for i in 1:12]
        union!(possible_pcs, Set(alt_notes))
    end
    println("    ◇ Possible PCs (in some world): {$(join([NOTE_NAMES[n+1] for n in sort(collect(possible_pcs))], ","))}")
end

# --- Compositional Commands ---

SPC_CMDS["fix"] = function(args...)  # fixed points
    w = SPC_WORLD[]
    
    # Find fixed points: positions where note = position (mod 12)
    fps = [(i, w.notes[i]) for i in 1:12 if w.notes[i] == (i - 1) % 12]
    
    println("  Fixed points (σ(i) = i):")
    if isempty(fps)
        println("    None - this is a derangement!")
    else
        for (i, n) in fps
            println("    Position $i → $(NOTE_NAMES[n+1]) (fixed)")
        end
    end
    
    # Lawvere fixed point
    println("  Lawvere diagonal: seed ⊕ hash(seed) = 0x$(string(w.seed ⊻ splitmix64(w.seed), base=16))")
end

SPC_CMDS["obs"] = function(args...)  # obstructions
    w = SPC_WORLD[]
    println("  Obstructions to compositionality:")
    
    # Check interval closure
    ivs = Set(w.intervals)
    closure_fail = length(ivs) < 11
    if closure_fail
        missing = setdiff(1:11, ivs)
        push!(w.obstructions, "Interval closure: missing $(collect(missing))")
        println("    ✗ Interval closure fails: missing $(collect(missing))")
    else
        println("    ✓ All 11 interval classes present")
    end
    
    # Check tritone balance
    tritones = count(==(6), w.intervals)
    if tritones > 3
        push!(w.obstructions, "Tritone saturation: $tritones")
        println("    ⚠ Tritone saturation: $tritones tritones (dissonant)")
    end
    
    # Check for chromatic clusters
    clusters = 0
    for i in 1:10
        if w.intervals[i] == 1 && w.intervals[i+1] == 1
            clusters += 1
        end
    end
    if clusters > 0
        push!(w.obstructions, "Chromatic clusters: $clusters")
        println("    ⚠ Chromatic clusters: $clusters (voice leading tension)")
    end
    
    isempty(w.obstructions) && println("    ✓ No obstructions detected")
end

SPC_CMDS["tensor"] = function(args...)  # ⊗ composition
    w = SPC_WORLD[]
    
    # Tensor two adjacent notes → third
    println("  Tensor products (⊗):")
    for i in 1:10
        a, b = w.notes[i], w.notes[i+1]
        c = (a + b) % 12  # tropical addition in Z/12
        actual = w.notes[min(i+2, 12)]
        match = c == actual ? "✓" : "✗"
        println("    $(NOTE_NAMES[a+1]) ⊗ $(NOTE_NAMES[b+1]) = $(NOTE_NAMES[c+1]) $match (actual: $(NOTE_NAMES[actual+1]))")
    end
end

# --- Seed Commands ---

SPC_CMDS["seed"] = function(args...)
    seed = isempty(args) ? GAY_SEED : parse(UInt64, args[1], base=16)
    init_world(seed)
    println("  Seed: 0x$(string(seed, base=16))")
    SPC_CMDS["chain"]()
end

SPC_CMDS["gay"] = function(args...)
    init_world(GAY_SEED)
    println("  Reset to GAY_SEED (0x6761795f636f6c6f)")
    SPC_CMDS["chain"]()
end

SPC_CMDS["walk"] = function(args...)
    n = isempty(args) ? 10 : parse(Int, args[1])
    w = SPC_WORLD[]
    
    println("  Random walk from seed:")
    best = (seed=w.seed, coverage=length(unique(w.notes)))
    
    for i in 1:n
        delta = rand(1:1000)
        test_seed = w.seed + UInt64(delta)
        test_notes = [hue_to_pc(color_at(j; seed=test_seed)) for j in 1:12]
        cov = length(unique(test_notes))
        
        if cov > best.coverage
            best = (seed=test_seed, coverage=cov)
            println("    +$delta → $cov/12 coverage ★")
        end
    end
    
    println("  Best: 0x$(string(best.seed, base=16)) ($((best.coverage))/12)")
end

# --- Distillation: Find resolutions to obstructions ---

SPC_CMDS["distill"] = function(args...)
    w = SPC_WORLD[]
    
    println("  ⚗ Distillation: Obstruction Analysis")
    println("  ─────────────────────────────────────")
    
    # 1. Interval closure obstruction
    ivs = Set(w.intervals)
    missing_ivs = setdiff(1:11, ivs)
    
    if !isempty(missing_ivs)
        println("  ✗ Interval closure fails:")
        println("    Missing: $(collect(missing_ivs))")
        
        # Find seeds that would add these intervals
        println("    Seeking resolutions...")
        for delta in 1:100
            test_seed = w.seed + UInt64(delta)
            test_notes = [hue_to_pc(color_at(i; seed=test_seed)) for i in 1:12]
            test_ivs = Set((test_notes[i+1] - test_notes[i] + 12) % 12 for i in 1:11)
            gained = intersect(missing_ivs, test_ivs)
            if !isempty(gained)
                println("    → Δ=$delta gains intervals: $(collect(gained))")
                break
            end
        end
    else
        println("  ✓ Interval closure: complete!")
    end
    
    # 2. Fixed point obstruction (for derangement)
    if !isempty(w.fixpoints)
        println("  ✗ Fixed points obstruct derangement:")
        for fp in w.fixpoints
            println("    Position $fp is fixed (σ($fp) = $fp)")
        end
        
        # Find nearby derangement
        for delta in 1:100
            test_seed = w.seed + UInt64(delta)
            test_notes = [hue_to_pc(color_at(i; seed=test_seed)) for i in 1:12]
            fps = findall(i -> test_notes[i] == i - 1, 1:12)
            if isempty(fps)
                println("    → Δ=$delta is a derangement!")
                break
            end
        end
    else
        println("  ✓ No fixed points: valid derangement")
    end
    
    # 3. Chromatic coverage obstruction
    coverage = length(unique(w.notes))
    if coverage < 12
        missing_pcs = setdiff(0:11, w.notes)
        println("  ✗ Chromatic coverage: $(coverage)/12")
        println("    Missing PCs: {$(join([NOTE_NAMES[n+1] for n in missing_pcs], ","))}")
        
        # Find seed with better coverage
        best = (delta=0, coverage=coverage)
        for delta in 1:500
            test_seed = w.seed + UInt64(delta)
            test_notes = [hue_to_pc(color_at(i; seed=test_seed)) for i in 1:12]
            cov = length(unique(test_notes))
            if cov > best.coverage
                best = (delta=delta, coverage=cov)
            end
        end
        if best.delta > 0
            println("    → Δ=$(best.delta) achieves $(best.coverage)/12 coverage")
        end
    else
        println("  ✓ Full chromatic coverage: 12/12")
    end
    
    # 4. Tensor associativity check
    println("  Tensor associativity (a⊗b)⊗c = a⊗(b⊗c):")
    assoc_fails = 0
    for i in 1:9
        a, b, c = w.notes[i], w.notes[i+1], w.notes[i+2]
        left = ((a + b) % 12 + c) % 12   # (a⊗b)⊗c
        right = (a + (b + c) % 12) % 12  # a⊗(b⊗c)
        if left != right
            assoc_fails += 1
        end
    end
    if assoc_fails == 0
        println("    ✓ Associative (mod 12 addition)")
    else
        println("    ✗ $assoc_fails failures (expected - ⊗ is associative in Z/12)")
    end
end

# --- Search for special seeds ---

SPC_CMDS["seek"] = function(args...)
    target = isempty(args) ? "chromatic" : lowercase(args[1])
    w = SPC_WORLD[]
    
    println("  🔍 Seeking: $target")
    
    if target == "chromatic" || target == "12"
        # Find 12-tone seed
        for i in 1:100000
            test_seed = w.seed + UInt64(i)
            test_notes = [hue_to_pc(color_at(j; seed=test_seed)) for j in 1:12]
            if length(unique(test_notes)) == 12
                println("    ★ Found at Δ=$i: 0x$(string(test_seed, base=16))")
                Gay.init_world(test_seed)
                SPC_CMDS["chain"]()
                return
            end
        end
        println("    Not found in 100K seeds")
        
    elseif target == "derangement" || target == "derange"
        for i in 1:10000
            test_seed = w.seed + UInt64(i)
            test_notes = [hue_to_pc(color_at(j; seed=test_seed)) for j in 1:12]
            fps = findall(j -> test_notes[j] == j - 1, 1:12)
            if isempty(fps)
                println("    ★ Derangement at Δ=$i")
                Gay.init_world(test_seed)
                SPC_CMDS["chain"]()
                return
            end
        end
        
    elseif target == "magic"
        for i in 1:10000
            test_seed = w.seed + UInt64(i)
            test_notes = [hue_to_pc(color_at(j; seed=test_seed)) for j in 1:12]
            grid = reshape(test_notes, 3, 4)
            r1 = sum(grid[1, :]) % 12
            r2 = sum(grid[2, :]) % 12
            r3 = sum(grid[3, :]) % 12
            if r1 == r2 == r3
                println("    ★ Magic grid at Δ=$i (rows sum to $r1)")
                Gay.init_world(test_seed)
                SPC_CMDS["chain"]()
                return
            end
        end
        
    elseif target == "tritone"
        best = (delta=0, tritones=0)
        for i in 1:10000
            test_seed = w.seed + UInt64(i)
            test_notes = [hue_to_pc(color_at(j; seed=test_seed)) for j in 1:12]
            ivs = [(test_notes[j+1] - test_notes[j] + 12) % 12 for j in 1:11]
            tri = count(==(6), ivs)
            if tri > best.tritones
                best = (delta=i, tritones=tri)
            end
        end
        println("    ★ Max tritones ($(best.tritones)) at Δ=$(best.delta)")
        Gay.init_world(w.seed + UInt64(best.delta))
        SPC_CMDS["chain"]()
    end
end

# --- Tropical Semiring Operations ---

SPC_CMDS["tropical"] = function(args...)
    semiring = isempty(args) ? "min+" : lowercase(args[1])
    w = SPC_WORLD[]
    
    println("  🌴 Tropical Semiring Analysis: $semiring")
    
    if semiring == "min+" || semiring == "minplus"
        # (min, +) - shortest path
        println("    Semiring: (min, +) - shortest path distances")
        total = 0
        for iv in w.intervals
            total += iv
        end
        println("    Total path: $total (tropical sum of intervals)")
        println("    Min interval: $(minimum(w.intervals))")
        
    elseif semiring == "max+" || semiring == "maxplus"
        # (max, +) - critical path
        println("    Semiring: (max, +) - critical path")
        println("    Max interval: $(maximum(w.intervals))")
        critical = sum(iv for iv in w.intervals if iv >= 6)
        println("    Critical path (intervals ≥ 6): $critical")
        
    elseif semiring == "minmax"
        # (min, max) - fuzzy logic
        println("    Semiring: (min, max) - fuzzy logic")
        # Compute fuzzy AND/OR on normalized notes
        normalized = [n / 11.0 for n in w.notes]
        fuzzy_and = minimum(normalized)
        fuzzy_or = maximum(normalized)
        println("    Fuzzy AND (min): $(round(fuzzy_and, digits=3))")
        println("    Fuzzy OR (max): $(round(fuzzy_or, digits=3))")
        
    elseif semiring == "bool" || semiring == "boolean"
        # Boolean reachability
        println("    Semiring: (∨, ∧) - boolean reachability")
        # Which pitch classes are "reachable" from first note
        reachable = Set{Int}([w.notes[1]])
        for n in w.notes[2:end]
            push!(reachable, n)
        end
        println("    Reachable PCs: {$(join([NOTE_NAMES[n+1] for n in sort(collect(reachable))], ","))}")
    end
end

# --- Lawvere Fixed Point ---

SPC_CMDS["lawvere"] = function(args...)
    w = SPC_WORLD[]
    
    println("  📐 Lawvere Fixed Point Theorem")
    println("  ─────────────────────────────────")
    
    # The diagonal σ: seed → seed ⊕ hash(seed)
    diagonal = w.seed ⊻ splitmix64(w.seed)
    println("    Diagonal σ(seed) = seed ⊕ hash(seed)")
    println("    σ(0x$(string(w.seed, base=16))) = 0x$(string(diagonal, base=16))")
    
    # Find fixed points in the note sequence
    # A fixed point is where the note equals its position (mod 12)
    println("\n    Fixed points in note sequence:")
    for (i, n) in enumerate(w.notes)
        expected = (i - 1) % 12
        if n == expected
            println("      Position $i: $(NOTE_NAMES[n+1]) = $n (fixed)")
        end
    end
    
    # Gödel-style self-reference: find seed s where hash(s) produces notes containing s's digits
    println("\n    Self-referential search:")
    seed_digits = Set(parse(Int, string(d), base=16) % 12 for d in string(w.seed, base=16) if d != 'x')
    note_set = Set(w.notes)
    overlap = intersect(seed_digits, note_set)
    println("      Seed hex digits (mod 12): {$(join(sort(collect(seed_digits)), ","))}")
    println("      Note pitch classes: {$(join(sort(collect(note_set)), ","))}")
    println("      Self-reference overlap: $(length(overlap))/$(length(seed_digits))")
    
    # Y combinator: find cycle in note sequence
    println("\n    Cycle detection (Y combinator):")
    seen = Dict{Int, Int}()
    cycle_start = nothing
    for (i, n) in enumerate(w.notes)
        if haskey(seen, n)
            cycle_start = (seen[n], i, n)
            break
        end
        seen[n] = i
    end
    if cycle_start !== nothing
        println("      First repeat: $(NOTE_NAMES[cycle_start[3]+1]) at positions $(cycle_start[1]) and $(cycle_start[2])")
    end
end

# --- Compose two chains ---

SPC_CMDS["compose"] = function(args...)
    delta = isempty(args) ? 1 : parse(Int, args[1])
    w = SPC_WORLD[]
    
    alt_seed = w.seed + UInt64(delta)
    alt_notes = [hue_to_pc(color_at(i; seed=alt_seed)) for i in 1:12]
    
    println("  ⊗ Chain Composition (seed ⊗ seed+$delta)")
    
    # Compose via addition mod 12 (tropical)
    composed = [(w.notes[i] + alt_notes[i]) % 12 for i in 1:12]
    
    print("    Chain 1: ")
    println(join([NOTE_NAMES[n+1] for n in w.notes], "-"))
    print("    Chain 2: ")
    println(join([NOTE_NAMES[n+1] for n in alt_notes], "-"))
    print("    C1 ⊗ C2: ")
    println(join([NOTE_NAMES[n+1] for n in composed], "-"))
    
    # Check if composition improves coverage
    cov1 = length(unique(w.notes))
    cov2 = length(unique(alt_notes))
    cov_composed = length(unique(composed))
    
    println("    Coverage: $cov1 ⊗ $cov2 → $cov_composed")
    
    if cov_composed > max(cov1, cov2)
        println("    ★ Composition improved coverage!")
    end
end

# --- Derangement permutation ---

SPC_CMDS["derange"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🔀 Derangement Analysis")
    
    # Generate derangement permutation from seed
    perm = collect(1:12)
    state = w.seed
    
    # Fisher-Yates with rejection for derangement
    for attempt in 1:100
        state = splitmix64(state + UInt64(attempt))
        perm = collect(1:12)
        local_state = state
        for i in 12:-1:2
            local_state = splitmix64(local_state)
            j = 1 + Int(local_state % UInt64(i))
            perm[i], perm[j] = perm[j], perm[i]
        end
        if all(perm[i] != i for i in 1:12)
            break
        end
    end
    
    is_derangement = all(perm[i] != i for i in 1:12)
    
    println("    Permutation: $(perm)")
    println("    Is derangement: $(is_derangement ? "✓ Yes" : "✗ No")")
    
    # Apply to notes
    deranged_notes = w.notes[perm]
    print("    Original:  ")
    println(join([NOTE_NAMES[n+1] for n in w.notes], "-"))
    print("    Deranged:  ")
    println(join([NOTE_NAMES[n+1] for n in deranged_notes], "-"))
    
    # Check if derangement changes coverage
    cov_orig = length(unique(w.notes))
    cov_deranged = length(unique(deranged_notes))
    println("    Coverage preserved: $(cov_orig == cov_deranged ? "✓" : "✗")")
    
    # Cycle structure
    visited = falses(12)
    cycles = Vector{Int}[]
    for i in 1:12
        if !visited[i]
            cycle = Int[]
            j = i
            while !visited[j]
                visited[j] = true
                push!(cycle, j)
                j = perm[j]
            end
            push!(cycles, cycle)
        end
    end
    println("    Cycle structure: $(join([length(c) for c in cycles], "+")) = 12")
end

# --- Chromatic Theremin ---

SPC_CMDS["theremin"] = function(args...)
    w = SPC_WORLD[]
    duration = isempty(args) ? 4.0 : parse(Float64, args[1])
    
    println("  🎻 Chromatic Theremin ($(duration)s)")
    
    # Generate smooth glissando through the chain
    octaves = [4 + Int(floor(convert(HSL, c).l * 2)) for c in w.chain]
    midis = [60 + n + (octaves[i] - 4) * 12 for (i, n) in enumerate(w.notes)]
    
    # Show the path
    print("    Path: ")
    for (i, n) in enumerate(w.notes)
        print("$(NOTE_NAMES[n+1])$(octaves[i])")
        i < 12 && print("→")
    end
    println()
    
    # Build frequency envelope for smooth glide
    midi_str = join(midis, ",")
    
    try
        cmd = ```python3 -c "
import wave, struct, math

midis = [$(midi_str)]
sr = 44100
total_dur = $(duration)
segment_dur = total_dur / (len(midis) - 1)
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

# Generate smooth glissando
t = 0
sample_idx = 0
phase = 0

while t < total_dur:
    # Find current segment
    seg = min(int(t / segment_dur), len(midis) - 2)
    seg_t = (t - seg * segment_dur) / segment_dur
    
    # Interpolate MIDI (exponential for perceptual smoothness)
    m1, m2 = midis[seg], midis[seg + 1]
    current_midi = m1 + (m2 - m1) * seg_t
    freq = midi_to_freq(current_midi)
    
    # Theremin-like waveform: sine + vibrato
    vibrato = 1.0 + 0.02 * math.sin(2 * math.pi * 5 * t)  # 5Hz vibrato
    
    # Phase-continuous synthesis
    phase += 2 * math.pi * freq * vibrato / sr
    
    # Mix sine with slight triangle for theremin timbre
    sine = math.sin(phase)
    tri = 2 * abs((phase / (2*math.pi)) % 1 - 0.5) - 0.5
    wave_val = 0.8 * sine + 0.2 * tri
    
    # Envelope
    env = min(1.0, t * 10) * min(1.0, (total_dur - t) * 5)
    
    sample = int(32767 * env * wave_val * 0.6)
    out += struct.pack('<h', max(-32767, min(32767, sample)))
    
    t += 1.0 / sr
    sample_idx += 1

with wave.open('/tmp/spc_theremin.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)

print(f'Generated {sample_idx} samples')
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_theremin.wav`, wait=true)
        println("    ✓ Theremin complete")
    catch e
        println("    (audio error: $e)")
    end
end

# --- Chromatic Walk with Audio ---

SPC_CMDS["glide"] = function(args...)
    w = SPC_WORLD[]
    steps = isempty(args) ? 24 : parse(Int, args[1])
    
    println("  🌈 Chromatic Glide ($steps steps)")
    
    # Generate chromatic walk through hue space
    hues = range(0, 360, length=steps+1)[1:end-1]
    notes = [Int(floor(h / 30.0)) % 12 for h in hues]
    
    print("    ")
    for (i, n) in enumerate(notes)
        print("$(NOTE_NAMES[n+1]) ")
        i % 12 == 0 && i < steps && println("\n    ")
    end
    println()
    
    # Ascending chromatic with smooth portamento
    try
        cmd = ```python3 -c "
import wave, struct, math

sr = 44100
steps = $(steps)
step_dur = 0.15
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

phase = 0
for step in range(steps):
    # Chromatic: C, C#, D, D#, ... wrapping
    midi = 60 + (step % 12) + 12 * (step // 12)
    freq = midi_to_freq(midi)
    
    for i in range(int(sr * step_dur)):
        t = i / sr
        
        # Portamento: slight pitch bend at start
        bend = 1.0 - 0.02 * math.exp(-t * 20)
        
        phase += 2 * math.pi * freq * bend / sr
        
        # Pure sine for chromatic clarity
        wave_val = math.sin(phase)
        
        # Quick attack/decay
        env = min(1.0, t * 30) * min(1.0, (step_dur - t) * 20)
        
        sample = int(32767 * env * wave_val * 0.5)
        out += struct.pack('<h', max(-32767, min(32767, sample)))

with wave.open('/tmp/spc_glide.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_glide.wav`, wait=true)
        println("    ✓ Glide complete")
    catch e
        println("    (audio error: $e)")
    end
end

# --- Spectral Render (visual + audio) ---

SPC_CMDS["spectrum"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🌊 Spectral Render")
    println()
    
    # ASCII spectrogram-like display
    for (i, c) in enumerate(w.chain)
        hsl = convert(HSL, c)
        n = w.notes[i]
        
        # Color block
        r = round(Int, clamp(c.r, 0, 1) * 255)
        g = round(Int, clamp(c.g, 0, 1) * 255)
        b = round(Int, clamp(c.b, 0, 1) * 255)
        
        # Frequency bar (width based on pitch class)
        bar_width = n + 3
        bar = "█" ^ bar_width
        
        # Print with color
        print("  \e[48;2;$(r);$(g);$(b)m  \e[0m ")
        print("$(lpad(NOTE_NAMES[n+1], 2)) ")
        
        # Pitch visualization
        spaces = " " ^ n
        println("$spaces$bar $(round(440.0 * 2^((60+n-69)/12.0), digits=1))Hz")
    end
    println()
    
    # Play with harmonic synthesis
    println("  Playing spectral synthesis...")
    
    octaves = [4 + Int(floor(convert(HSL, c).l * 2)) for c in w.chain]
    midis = [60 + n + (octaves[i] - 4) * 12 for (i, n) in enumerate(w.notes)]
    midi_str = join(midis, ",")
    sat_str = join([round(convert(HSL, c).s, digits=2) for c in w.chain], ",")
    
    try
        cmd = ```python3 -c "
import wave, struct, math

midis = [$(midi_str)]
sats = [$(sat_str)]  # saturation controls harmonic richness
sr = 44100
note_dur = 0.3
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

for idx, midi in enumerate(midis):
    freq = midi_to_freq(midi)
    sat = sats[idx]
    
    for i in range(int(sr * note_dur)):
        t = i / sr
        
        # Envelope
        attack = min(1.0, t / 0.01)
        decay = max(0.0, 1.0 - (t - note_dur + 0.05) / 0.05) if t > note_dur - 0.05 else 1.0
        env = attack * decay * 0.4
        
        # Harmonic synthesis based on saturation
        h1 = math.sin(2 * math.pi * freq * t)
        h2 = math.sin(4 * math.pi * freq * t) * sat * 0.3
        h3 = math.sin(6 * math.pi * freq * t) * sat * 0.15
        h4 = math.sin(8 * math.pi * freq * t) * sat * 0.08
        
        wave_val = h1 + h2 + h3 + h4
        
        sample = int(32767 * env * wave_val)
        out += struct.pack('<h', max(-32767, min(32767, sample)))

with wave.open('/tmp/spc_spectrum.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_spectrum.wav`, wait=true)
        println("  ✓ Spectral render complete")
    catch e
        println("  (audio error: $e)")
    end
end

# --- Rainbow Arpeggio ---

SPC_CMDS["arp"] = function(args...)
    pattern = isempty(args) ? "up" : lowercase(args[1])
    w = SPC_WORLD[]
    
    println("  🎹 Arpeggio: $pattern")
    
    # Get unique pitch classes in order
    unique_pcs = unique(w.notes)
    n_notes = length(unique_pcs)
    
    # Build pattern
    if pattern == "up"
        seq = sort(unique_pcs)
    elseif pattern == "down"
        seq = sort(unique_pcs, rev=true)
    elseif pattern == "updown"
        up = sort(unique_pcs)
        seq = vcat(up, reverse(up)[2:end-1])
    elseif pattern == "random"
        seq = shuffle(unique_pcs)
    else
        seq = unique_pcs
    end
    
    print("    ")
    for n in seq
        print("$(NOTE_NAMES[n+1]) ")
    end
    println()
    
    # Fast arpeggio
    midi_seq = [60 + 12 + n for n in seq]  # Octave 5
    midi_str = join(midi_seq, ",")
    
    try
        cmd = ```python3 -c "
import wave, struct, math

midis = [$(midi_str)]
sr = 44100
note_dur = 0.12
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

for midi in midis:
    freq = midi_to_freq(midi)
    
    for i in range(int(sr * note_dur)):
        t = i / sr
        env = min(1.0, t * 50) * max(0.0, 1.0 - t / note_dur) * 0.5
        wave_val = math.sin(2 * math.pi * freq * t)
        sample = int(32767 * env * wave_val)
        out += struct.pack('<h', max(-32767, min(32767, sample)))

with wave.open('/tmp/spc_arp.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_arp.wav`, wait=true)
    catch e
        println("    (audio error: $e)")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# STRANGE INSTRUMENTS: Samovar Involution Chromatic Peptides
# ═══════════════════════════════════════════════════════════════════════════

# --- Samovar: Nested layered drones like a Russian tea urn ---
SPC_CMDS["samovar"] = function(args...)
    w = SPC_WORLD[]
    layers = isempty(args) ? 4 : parse(Int, args[1])
    
    println("  🫖 Samovar ($layers layers)")
    println("    Nested drones, each layer an octave apart")
    
    # Base frequencies from chain
    base_notes = unique(w.notes)[1:min(layers, length(unique(w.notes)))]
    
    print("    Layers: ")
    for (i, n) in enumerate(base_notes)
        print("$(NOTE_NAMES[n+1])$(3+i) ")
    end
    println()
    
    midi_base = [48 + n for n in base_notes]  # Start at octave 3
    
    try
        cmd = ```python3 -c "
import wave, struct, math

base_midis = [$(join(midi_base, ","))]
layers = $(layers)
sr = 44100
dur = 6.0
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

# Samovar: each layer enters gradually, creating nested warmth
for i in range(int(sr * dur)):
    t = i / sr
    sample = 0.0
    
    for layer, base_midi in enumerate(base_midis):
        # Each layer enters at different time
        layer_start = layer * 0.5
        if t < layer_start:
            continue
            
        layer_t = t - layer_start
        
        # Envelope: slow swell, sustain, gentle release
        env = min(1.0, layer_t / 1.5) * min(1.0, (dur - t) / 1.0)
        env *= 0.3 / (layer + 1)  # Quieter for higher layers
        
        # Frequency with slight beating between layers
        freq = midi_to_freq(base_midi + layer * 12)  # Octave per layer
        detune = 1.0 + 0.002 * math.sin(2 * math.pi * 0.1 * t * (layer + 1))
        
        # Rich drone: fundamental + 5th + octave
        wave = math.sin(2 * math.pi * freq * detune * t)
        wave += 0.3 * math.sin(2 * math.pi * freq * 1.5 * t)  # 5th
        wave += 0.2 * math.sin(2 * math.pi * freq * 2 * t)    # Octave
        
        sample += env * wave
    
    out += struct.pack('<h', int(max(-32767, min(32767, sample * 32767))))

with wave.open('/tmp/spc_samovar.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_samovar.wav`, wait=true)
        println("    ✓ Samovar complete")
    catch e
        println("    (error: $e)")
    end
end

# --- Involution: f(f(x)) = x, each note triggers its chromatic inverse ---
SPC_CMDS["involution"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🔄 Involution: f(f(x)) = x")
    println("    Each note paired with its chromatic inverse (12 - n)")
    
    # Involution: n ↦ (12 - n) mod 12
    pairs = [(n, (12 - n) % 12) for n in w.notes]
    
    print("    Pairs: ")
    for (a, b) in pairs[1:6]
        print("$(NOTE_NAMES[a+1])↔$(NOTE_NAMES[b+1]) ")
    end
    println("...")
    
    # Interleave original and inverted
    interleaved = Int[]
    for (a, b) in pairs
        push!(interleaved, a, b)
    end
    
    midi_seq = [60 + n for n in interleaved]
    
    try
        cmd = ```python3 -c "
import wave, struct, math

midis = [$(join(midi_seq, ","))]
sr = 44100
note_dur = 0.15
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

phase = 0
for idx, midi in enumerate(midis):
    freq = midi_to_freq(midi)
    is_inverse = idx % 2 == 1
    
    for i in range(int(sr * note_dur)):
        t = i / sr
        
        # Inverse notes play backwards envelope
        if is_inverse:
            env = min(1.0, t * 20) * 0.4
        else:
            env = max(0.0, 1.0 - t / note_dur) * 0.5
        
        # Phase continuous
        phase += 2 * math.pi * freq / sr
        
        # Original: sine, Inverse: square-ish
        if is_inverse:
            wave_val = math.copysign(1, math.sin(phase)) * 0.7 + 0.3 * math.sin(phase)
        else:
            wave_val = math.sin(phase)
        
        sample = int(32767 * env * wave_val)
        out += struct.pack('<h', max(-32767, min(32767, sample)))

with wave.open('/tmp/spc_involution.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_involution.wav`, wait=true)
        println("    ✓ Involution complete")
    catch e
        println("    (error: $e)")
    end
end

# --- Peptide: Notes fold like amino acid chains, secondary structure ---
SPC_CMDS["peptide"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🧬 Chromatic Peptide: Protein folding sonification")
    
    # Classify notes as "amino acids" by interval properties
    # Hydrophobic (large intervals) vs Hydrophilic (small intervals)
    hydrophobic = Int[]  # Will form alpha helix (drone)
    hydrophilic = Int[]  # Will form beta sheet (staccato)
    
    for i in 1:11
        if w.intervals[i] >= 5  # Large interval = hydrophobic
            push!(hydrophobic, w.notes[i])
        else
            push!(hydrophilic, w.notes[i])
        end
    end
    push!(hydrophilic, w.notes[12])  # Last note
    
    println("    Hydrophobic core (α-helix): $(join([NOTE_NAMES[n+1] for n in hydrophobic], "-"))")
    println("    Hydrophilic surface (β-sheet): $(join([NOTE_NAMES[n+1] for n in hydrophilic], "-"))")
    
    # Folding: hydrophobic notes form sustained core, hydrophilic punctuate
    h_phobic = isempty(hydrophobic) ? [w.notes[1]] : hydrophobic
    h_philic = isempty(hydrophilic) ? [w.notes[2]] : hydrophilic
    
    try
        cmd = ```python3 -c "
import wave, struct, math

hydrophobic = [$(join([48 + n for n in h_phobic], ","))]
hydrophilic = [$(join([72 + n for n in h_philic], ","))]
sr = 44100
dur = 5.0
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

# Alpha helix: low sustained drone from hydrophobic residues
# Beta sheet: high staccato from hydrophilic residues

for i in range(int(sr * dur)):
    t = i / sr
    sample = 0.0
    
    # Alpha helix core (sustained)
    for idx, midi in enumerate(hydrophobic):
        freq = midi_to_freq(midi)
        # Slow undulation like protein breathing
        breath = 1.0 + 0.1 * math.sin(2 * math.pi * 0.2 * t + idx)
        env = min(1.0, t / 0.5) * min(1.0, (dur - t) / 0.5) * 0.2
        sample += env * math.sin(2 * math.pi * freq * breath * t)
    
    # Beta sheet punctuation (staccato bursts)
    beat_dur = 0.3
    beat_idx = int(t / beat_dur) % len(hydrophilic)
    beat_t = t % beat_dur
    
    if beat_t < 0.1:  # Short burst
        midi = hydrophilic[beat_idx]
        freq = midi_to_freq(midi)
        env = (0.1 - beat_t) / 0.1 * 0.4
        # Metallic timbre for surface
        sample += env * (math.sin(2*math.pi*freq*t) + 0.3*math.sin(3*math.pi*freq*t))
    
    out += struct.pack('<h', int(max(-32767, min(32767, sample * 32767))))

with wave.open('/tmp/spc_peptide.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_peptide.wav`, wait=true)
        println("    ✓ Peptide folded")
    catch e
        println("    (error: $e)")
    end
end

# --- Möbius: Loop with a twist, returns inverted ---
SPC_CMDS["mobius"] = function(args...)
    w = SPC_WORLD[]
    
    println("  ∞ Möbius Strip: Returns twisted")
    
    # Play forward, then play inverted (retrograde inversion)
    forward = w.notes
    # Retrograde inversion: reverse and invert each
    backward_inverted = [(12 - n) % 12 for n in reverse(w.notes)]
    
    print("    Forward:  ")
    println(join([NOTE_NAMES[n+1] for n in forward], "-"))
    print("    Twisted:  ")
    println(join([NOTE_NAMES[n+1] for n in backward_inverted], "-"))
    
    full_seq = vcat(forward, backward_inverted)
    
    try
        cmd = ```python3 -c "
import wave, struct, math

notes = [$(join([60 + n for n in full_seq], ","))]
sr = 44100
note_dur = 0.2
crossfade = 0.05
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

phase = 0
prev_freq = midi_to_freq(notes[0])

for idx, midi in enumerate(notes):
    freq = midi_to_freq(midi)
    is_twisted = idx >= 12
    
    for i in range(int(sr * note_dur)):
        t = i / sr
        
        # Crossfade between notes
        blend = min(1.0, t / crossfade) if i < sr * crossfade else 1.0
        current_freq = prev_freq + (freq - prev_freq) * blend
        
        phase += 2 * math.pi * current_freq / sr
        
        # Twisted half has different timbre (phase modulation)
        if is_twisted:
            mod = 0.5 * math.sin(2 * math.pi * 3 * t)
            wave_val = math.sin(phase + mod)
        else:
            wave_val = math.sin(phase)
        
        env = min(1.0, t * 20) * min(1.0, (note_dur - t) * 20) * 0.5
        sample = int(32767 * env * wave_val)
        out += struct.pack('<h', max(-32767, min(32767, sample)))
    
    prev_freq = freq

with wave.open('/tmp/spc_mobius.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_mobius.wav`, wait=true)
        println("    ✓ Möbius complete")
    catch e
        println("    (error: $e)")
    end
end

# --- Quine: Self-referential, plays the seed as audio ---
SPC_CMDS["quine"] = function(args...)
    w = SPC_WORLD[]
    
    println("  📜 Quine: Self-referential seed sonification")
    
    # Convert seed bytes directly to frequencies
    seed_bytes = reinterpret(UInt8, [w.seed])
    
    print("    Seed bytes: ")
    for b in seed_bytes
        print("$(string(b, base=16, pad=2)) ")
    end
    println()
    
    # Each byte becomes a frequency ratio
    ratios = [1.0 + b / 256.0 for b in seed_bytes]
    base_freq = 220.0
    
    println("    Base: $(base_freq)Hz, ratios: $(round.(ratios, digits=2))")
    
    try
        cmd = ```python3 -c "
import wave, struct, math

seed_bytes = [$(join(seed_bytes, ","))]
base_freq = 220.0
sr = 44100
dur = 4.0
out = b''

# Self-reference: the seed determines everything
for i in range(int(sr * dur)):
    t = i / sr
    sample = 0.0
    
    for idx, byte_val in enumerate(seed_bytes):
        # Frequency from byte
        ratio = 1.0 + byte_val / 256.0
        freq = base_freq * ratio
        
        # Phase offset from byte position
        phase_offset = byte_val / 256.0 * 2 * math.pi
        
        # Amplitude from byte (louder for larger values)
        amp = (byte_val / 255.0) * 0.15
        
        # Time offset (bytes enter sequentially)
        byte_start = idx * 0.3
        if t < byte_start:
            continue
        
        byte_t = t - byte_start
        env = min(1.0, byte_t / 0.2) * min(1.0, (dur - t) / 0.5)
        
        sample += amp * env * math.sin(2 * math.pi * freq * t + phase_offset)
    
    out += struct.pack('<h', int(max(-32767, min(32767, sample * 32767))))

with wave.open('/tmp/spc_quine.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_quine.wav`, wait=true)
        println("    ✓ Quine: The seed sings itself")
    catch e
        println("    (error: $e)")
    end
end

# --- Ribosome: Translates note triplets to "amino" chords ---
SPC_CMDS["ribosome"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🔬 Ribosome: Codon translation")
    println("    Triplet codons → Amino acid chords")
    
    # Group notes into triplets (codons)
    codons = [(w.notes[i], w.notes[i+1], w.notes[i+2]) for i in 1:3:10]
    
    println("    Codons: ", join(["($(NOTE_NAMES[a+1]),$(NOTE_NAMES[b+1]),$(NOTE_NAMES[c+1]))" for (a,b,c) in codons], " → "))
    
    # Each codon becomes a chord played together
    try
        cmd = ```python3 -c "
import wave, struct, math

codons = [$(join(["[$(c[1]+60),$(c[2]+60),$(c[3]+60)]" for c in codons], ","))]
sr = 44100
codon_dur = 0.8
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

for codon in codons:
    for i in range(int(sr * codon_dur)):
        t = i / sr
        sample = 0.0
        
        # Play all three notes as chord
        for note_idx, midi in enumerate(codon):
            freq = midi_to_freq(midi)
            
            # Stagger entry slightly
            note_start = note_idx * 0.05
            if t < note_start:
                continue
            
            note_t = t - note_start
            env = min(1.0, note_t / 0.1) * min(1.0, (codon_dur - t) / 0.2) * 0.25
            
            sample += env * math.sin(2 * math.pi * freq * t)
        
        out += struct.pack('<h', int(max(-32767, min(32767, sample * 32767))))

with wave.open('/tmp/spc_ribosome.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_ribosome.wav`, wait=true)
        println("    ✓ Translation complete")
    catch e
        println("    (error: $e)")
    end
end

# --- Enzyme: Catalyzes transformation, speeds up over time ---
SPC_CMDS["enzyme"] = function(args...)
    w = SPC_WORLD[]
    
    println("  ⚗️ Enzyme: Catalytic acceleration")
    
    # Start slow, accelerate exponentially
    try
        cmd = ```python3 -c "
import wave, struct, math

notes = [$(join([60 + n for n in w.notes], ","))]
sr = 44100
total_dur = 4.0
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

t = 0
note_idx = 0
tempo = 0.5  # Starting tempo (notes per second)
phase = 0

while t < total_dur and note_idx < len(notes) * 3:  # Loop 3x
    midi = notes[note_idx % len(notes)]
    freq = midi_to_freq(midi)
    
    # Note duration decreases (acceleration)
    note_dur = 1.0 / tempo
    
    for i in range(int(sr * min(note_dur, total_dur - t))):
        sample_t = i / sr
        
        env = min(1.0, sample_t * 20) * min(1.0, (note_dur - sample_t) * 10) * 0.5
        
        phase += 2 * math.pi * freq / sr
        wave_val = math.sin(phase)
        
        out += struct.pack('<h', int(max(-32767, min(32767, env * wave_val * 32767))))
    
    t += note_dur
    note_idx += 1
    tempo *= 1.15  # Accelerate 15% each note

with wave.open('/tmp/spc_enzyme.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_enzyme.wav`, wait=true)
        println("    ✓ Reaction catalyzed")
    catch e
        println("    (error: $e)")
    end
end

# --- Helicase: Unwinds double helix, separating strands ---
SPC_CMDS["helicase"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🧬 Helicase: Unwinding the double helix")
    
    # Split chain into two strands (odd/even indices)
    strand_a = [w.notes[i] for i in 1:2:12]
    strand_b = [w.notes[i] for i in 2:2:12]
    
    println("    Strand A: $(join([NOTE_NAMES[n+1] for n in strand_a], "-"))")
    println("    Strand B: $(join([NOTE_NAMES[n+1] for n in strand_b], "-"))")
    
    try
        cmd = ```python3 -c "
import wave, struct, math

strand_a = [$(join([60 + n for n in strand_a], ","))]
strand_b = [$(join([48 + n for n in strand_b], ","))]
sr = 44100
dur = 5.0
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

# Start intertwined, gradually separate in stereo space and pitch
for i in range(int(sr * dur)):
    t = i / sr
    separation = min(1.0, t / 2.0)  # Gradually separate over 2s
    
    sample = 0.0
    
    # Strand A rises, strand B falls
    for idx, midi in enumerate(strand_a):
        freq = midi_to_freq(midi + separation * 6)  # Rise by tritone
        phase = 2 * math.pi * freq * t + idx * 0.5
        note_t = (t * 2 + idx * 0.3) % (len(strand_a) * 0.4)
        note_idx = int(note_t / 0.4)
        if note_idx == idx:
            env = min(1.0, (note_t % 0.4) * 10) * 0.2
            sample += env * math.sin(phase)
    
    for idx, midi in enumerate(strand_b):
        freq = midi_to_freq(midi - separation * 6)  # Fall by tritone
        phase = 2 * math.pi * freq * t + idx * 0.5 + math.pi
        note_t = (t * 2 + idx * 0.3) % (len(strand_b) * 0.4)
        note_idx = int(note_t / 0.4)
        if note_idx == idx:
            env = min(1.0, (note_t % 0.4) * 10) * 0.2
            sample += env * math.sin(phase)
    
    # Global envelope
    env = min(1.0, t * 2) * min(1.0, (dur - t) * 2)
    out += struct.pack('<h', int(max(-32767, min(32767, sample * env * 32767))))

with wave.open('/tmp/spc_helicase.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_helicase.wav`, wait=true)
        println("    ✓ Helix unwound")
    catch e
        println("    (error: $e)")
    end
end

# --- Prion: Self-propagating misfolded pattern, infectious spread ---
SPC_CMDS["prion"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🦠 Prion: Self-propagating misfolded pattern")
    
    # The "misfolded" pattern: tritone transposition
    original = w.notes[1:4]
    misfolded = [(n + 6) % 12 for n in original]  # Tritone = corruption
    
    println("    Original: $(join([NOTE_NAMES[n+1] for n in original], "-"))")
    println("    Misfolded: $(join([NOTE_NAMES[n+1] for n in misfolded], "-"))")
    println("    Watch it spread...")
    
    try
        cmd = ```python3 -c "
import wave, struct, math

original = [$(join([60 + n for n in original], ","))]
misfolded = [$(join([60 + n for n in misfolded], ","))]
sr = 44100
dur = 6.0
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

# Prion spreads: each iteration, more notes become misfolded
iterations = 8
iter_dur = dur / iterations

for iteration in range(iterations):
    infection_rate = iteration / (iterations - 1)  # 0 to 1
    
    for note_idx in range(len(original)):
        # Probability this note is infected
        is_infected = (note_idx / len(original)) < infection_rate
        midi = misfolded[note_idx] if is_infected else original[note_idx]
        freq = midi_to_freq(midi)
        
        note_dur = iter_dur / len(original)
        for i in range(int(sr * note_dur)):
            t = i / sr
            global_t = iteration * iter_dur + note_idx * note_dur + t
            
            env = min(1.0, t * 30) * min(1.0, (note_dur - t) * 20) * 0.4
            
            # Infected notes have unstable, jittery timbre
            if is_infected:
                jitter = 1.0 + 0.03 * math.sin(2 * math.pi * 30 * t)
                wave_val = math.sin(2 * math.pi * freq * jitter * global_t)
            else:
                wave_val = math.sin(2 * math.pi * freq * global_t)
            
            out += struct.pack('<h', int(max(-32767, min(32767, env * wave_val * 32767))))

with wave.open('/tmp/spc_prion.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_prion.wav`, wait=true)
        println("    ✓ Infection complete")
    catch e
        println("    (error: $e)")
    end
end

# --- Mitosis: Chain divides and recombines ---
SPC_CMDS["mitosis"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🔬 Mitosis: Chromatic cell division")
    
    # Divide chain at midpoint
    left = w.notes[1:6]
    right = w.notes[7:12]
    
    # Recombine with crossover
    daughter_a = vcat(left[1:3], right[4:6])
    daughter_b = vcat(right[1:3], left[4:6])
    
    println("    Parent L: $(join([NOTE_NAMES[n+1] for n in left], "-"))")
    println("    Parent R: $(join([NOTE_NAMES[n+1] for n in right], "-"))")
    println("    Daughter A: $(join([NOTE_NAMES[n+1] for n in daughter_a], "-"))")
    println("    Daughter B: $(join([NOTE_NAMES[n+1] for n in daughter_b], "-"))")
    
    try
        cmd = ```python3 -c "
import wave, struct, math

parent_l = [$(join([60 + n for n in left], ","))]
parent_r = [$(join([60 + n for n in right], ","))]
daughter_a = [$(join([60 + n for n in daughter_a], ","))]
daughter_b = [$(join([60 + n for n in daughter_b], ","))]
sr = 44100
phase_dur = 1.5
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

def play_seq(seq, start_t, duration, octave_shift=0):
    samples = b''
    note_dur = duration / len(seq)
    for idx, midi in enumerate(seq):
        freq = midi_to_freq(midi + octave_shift * 12)
        for i in range(int(sr * note_dur)):
            t = i / sr
            env = min(1.0, t * 20) * min(1.0, (note_dur - t) * 10) * 0.35
            wave_val = math.sin(2 * math.pi * freq * (start_t + idx * note_dur + t))
            samples += struct.pack('<h', int(max(-32767, min(32767, env * wave_val * 32767))))
    return samples

# Phase 1: Parents together
out += play_seq(parent_l + parent_r, 0, phase_dur)
# Phase 2: Separation (silence with tension)
for i in range(int(sr * 0.3)):
    out += struct.pack('<h', 0)
# Phase 3: Daughters emerge
out += play_seq(daughter_a, phase_dur + 0.3, phase_dur, 0)
out += play_seq(daughter_b, phase_dur * 2 + 0.3, phase_dur, -1)

with wave.open('/tmp/spc_mitosis.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_mitosis.wav`, wait=true)
        println("    ✓ Division complete")
    catch e
        println("    (error: $e)")
    end
end

# --- Telomere: Shortening sequence, aging ---
SPC_CMDS["telomere"] = function(args...)
    w = SPC_WORLD[]
    
    println("  ⏳ Telomere: Chromatic aging (shortening ends)")
    
    try
        cmd = ```python3 -c "
import wave, struct, math

notes = [$(join([60 + n for n in w.notes], ","))]
sr = 44100
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

# Each generation, lose one note from each end
generations = 6
for gen in range(generations):
    start_idx = gen
    end_idx = len(notes) - gen
    if start_idx >= end_idx:
        break
    
    current_notes = notes[start_idx:end_idx]
    note_dur = 0.15 + gen * 0.05  # Slower as we age
    
    print(f'Gen {gen}: {len(current_notes)} notes')
    
    for midi in current_notes:
        freq = midi_to_freq(midi - gen * 2)  # Lower pitch = aging
        
        for i in range(int(sr * note_dur)):
            t = i / sr
            
            # Envelope gets weaker with age
            strength = 1.0 - gen * 0.15
            env = min(1.0, t * 20) * min(1.0, (note_dur - t) * 10) * 0.4 * strength
            
            # Timbre gets duller
            harmonics = max(1, 4 - gen)
            wave_val = sum(math.sin(2 * math.pi * freq * (h+1) * t) / (h+1) 
                          for h in range(harmonics))
            
            out += struct.pack('<h', int(max(-32767, min(32767, env * wave_val * 32767))))
    
    # Pause between generations
    for i in range(int(sr * 0.2)):
        out += struct.pack('<h', 0)

with wave.open('/tmp/spc_telomere.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_telomere.wav`, wait=true)
        println("    ✓ Senescence reached")
    catch e
        println("    (error: $e)")
    end
end

# --- Apoptosis: Programmed cell death, graceful fadeout ---
SPC_CMDS["apoptosis"] = function(args...)
    w = SPC_WORLD[]
    
    println("  💀 Apoptosis: Programmed chromatic death")
    println("    Notes self-destruct in controlled sequence")
    
    try
        cmd = ```python3 -c "
import wave, struct, math, random

notes = [$(join([60 + n for n in w.notes], ","))]
sr = 44100
dur = 5.0
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

# Each note has a death time
random.seed($(w.seed % 1000))
death_times = sorted([random.uniform(1.0, dur - 0.5) for _ in notes])

for i in range(int(sr * dur)):
    t = i / sr
    sample = 0.0
    
    alive_count = sum(1 for dt in death_times if t < dt)
    
    for idx, midi in enumerate(notes):
        if t >= death_times[idx]:
            continue  # This note is dead
        
        freq = midi_to_freq(midi)
        
        # Approaching death: become unstable
        time_to_death = death_times[idx] - t
        if time_to_death < 0.3:
            # Death rattle: frequency wobble
            wobble = 1.0 + 0.1 * math.sin(2 * math.pi * 20 * t) * (0.3 - time_to_death) / 0.3
            freq *= wobble
        
        env = 0.15 * min(1.0, time_to_death * 3)
        sample += env * math.sin(2 * math.pi * freq * t + idx)
    
    out += struct.pack('<h', int(max(-32767, min(32767, sample * 32767))))

with wave.open('/tmp/spc_apoptosis.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_apoptosis.wav`, wait=true)
        println("    ✓ Cellular death complete")
    catch e
        println("    (error: $e)")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# INFORMATION GEOMETRY & ACTIVE INFERENCE: 7×17 Refinement Framework
# ═══════════════════════════════════════════════════════════════════════════

# --- C. elegans Connectome: 302 neurons mapped to 12-tone chromatic space ---
SPC_CMDS["elegans"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🪱 C. elegans Connectome Sonification")
    println("    302 neurons → 12 pitch classes (mod 12 projection)")
    
    # Map seed bytes to neuron firing patterns
    # C. elegans has ~302 neurons, we simulate firing cascades
    seed_bytes = reinterpret(UInt8, [w.seed])
    
    # Connectome topology: sensory → inter → motor (3 layers)
    sensory = w.notes[1:4]      # 4 sensory neurons
    inter = w.notes[5:8]        # 4 interneurons  
    motor = w.notes[9:12]       # 4 motor neurons
    
    println("    Sensory: $(join([NOTE_NAMES[n+1] for n in sensory], "-"))")
    println("    Inter:   $(join([NOTE_NAMES[n+1] for n in inter], "-"))")
    println("    Motor:   $(join([NOTE_NAMES[n+1] for n in motor], "-"))")
    
    # Gap junctions create synchrony (electrical synapses)
    gap_freq = sum(sensory) / 4.0  # Average creates drone
    
    try
        cmd = ```python3 -c "
import wave, struct, math

sensory = [$(join([60 + n for n in sensory], ","))]
inter = [$(join([48 + n for n in inter], ","))]
motor = [$(join([36 + n for n in motor], ","))]
sr = 44100
dur = 6.0
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

# C. elegans cascade: sensory triggers inter triggers motor
# With gap junction synchronization creating phase-locked oscillation

for i in range(int(sr * dur)):
    t = i / sr
    sample = 0.0
    
    # Wave propagation delay through layers
    sensory_active = t > 0.0
    inter_active = t > 0.5
    motor_active = t > 1.0
    
    # Sensory layer: fast oscillation (chemosensory neurons)
    if sensory_active:
        phase_s = t * 2.0
        for idx, midi in enumerate(sensory):
            freq = midi_to_freq(midi)
            # Gap junctions create 0.1Hz beating
            beat = 1.0 + 0.05 * math.sin(2 * math.pi * 0.1 * t + idx)
            env = min(1.0, t * 2) * 0.15
            sample += env * math.sin(2 * math.pi * freq * beat * t)
    
    # Interneuron layer: integration (command interneurons)
    if inter_active:
        for idx, midi in enumerate(inter):
            freq = midi_to_freq(midi)
            # Interneurons sum inputs, slower modulation
            mod = math.sin(2 * math.pi * 0.3 * t + idx * math.pi/2)
            env = min(1.0, (t - 0.5) * 2) * 0.2
            sample += env * math.sin(2 * math.pi * freq * t) * (0.5 + 0.5 * mod)
    
    # Motor layer: rhythmic output (body wall muscles)
    if motor_active:
        for idx, midi in enumerate(motor):
            freq = midi_to_freq(midi)
            # Motor neurons create locomotion rhythm (~0.5 Hz undulation)
            undulate = math.sin(2 * math.pi * 0.5 * t + idx * math.pi/4)
            env = min(1.0, (t - 1.0) * 1) * min(1.0, (dur - t) * 2) * 0.25
            sample += env * math.sin(2 * math.pi * freq * t) * (0.3 + 0.7 * abs(undulate))
    
    out += struct.pack('<h', int(max(-32767, min(32767, sample * 32767))))

with wave.open('/tmp/spc_elegans.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_elegans.wav`, wait=true)
        println("    ✓ Neural cascade complete")
    catch e
        println("    (error: $e)")
    end
end

# --- Immune System: Clonal selection with affinity maturation ---
SPC_CMDS["immune"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🦠 Immune System: Clonal Selection & Affinity Maturation")
    
    # Antigen = seed-derived pattern
    antigen = w.notes[1:4]
    
    # Initial B-cell repertoire (diverse, low affinity)
    repertoire = [(n + i) % 12 for (i, n) in enumerate(w.notes[5:12])]
    
    println("    Antigen:    $(join([NOTE_NAMES[n+1] for n in antigen], "-"))")
    println("    Repertoire: $(join([NOTE_NAMES[n+1] for n in repertoire], "-"))")
    
    # Affinity = inverse Hamming distance to antigen pattern
    function affinity(cell, antigen)
        sum(abs(cell[mod1(i, length(cell))] - a) for (i, a) in enumerate(antigen))
    end
    
    try
        cmd = ```python3 -c "
import wave, struct, math, random

antigen = [$(join([60 + n for n in antigen], ","))]
repertoire = [$(join([60 + n for n in repertoire], ","))]
sr = 44100
dur = 7.0
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

random.seed($(w.seed % 10000))

# 7 rounds of selection (generations)
rounds = 7
round_dur = dur / rounds

for round_idx in range(rounds):
    round_start = round_idx * round_dur
    
    # Affinity maturation: cells mutate toward antigen
    maturation = round_idx / (rounds - 1)  # 0 to 1
    
    # Selected cells (higher affinity survive)
    selected = []
    for cell in repertoire:
        # Mutate toward antigen
        target = antigen[round_idx % len(antigen)]
        mutated = int(cell + (target - cell) * maturation * 0.3)
        selected.append(mutated)
    
    # Play selected clones
    for cell_idx, cell in enumerate(selected):
        cell_start = round_start + cell_idx * (round_dur / len(selected))
        cell_dur = round_dur / len(selected) * 0.8
        
        freq = midi_to_freq(cell)
        
        for i in range(int(sr * cell_dur)):
            t = i / sr
            global_t = cell_start + t
            
            if global_t >= dur:
                break
            
            # Higher rounds = more coherent (affinity matured)
            coherence = 0.5 + 0.5 * maturation
            noise = (1 - maturation) * 0.1 * random.random()
            
            env = min(1.0, t * 30) * min(1.0, (cell_dur - t) * 20) * 0.3 * coherence
            wave_val = math.sin(2 * math.pi * freq * (1 + noise) * global_t)
            
            out += struct.pack('<h', int(max(-32767, min(32767, env * wave_val * 32767))))

# Memory cells: final refined pattern as sustained chord
memory_dur = 1.0
print(f'Memory B-cells forming...')
for i in range(int(sr * memory_dur)):
    t = i / sr
    sample = 0.0
    
    for cell in selected[:4]:
        freq = midi_to_freq(cell)
        env = min(1.0, t * 5) * min(1.0, (memory_dur - t) * 3) * 0.2
        sample += env * math.sin(2 * math.pi * freq * (dur + t))
    
    out += struct.pack('<h', int(max(-32767, min(32767, sample * 32767))))

with wave.open('/tmp/spc_immune.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_immune.wav`, wait=true)
        println("    ✓ Immune response complete (memory formed)")
    catch e
        println("    (error: $e)")
    end
end

# --- Fisher: Information geometry metric on probability manifold ---
SPC_CMDS["fisher"] = function(args...)
    w = SPC_WORLD[]
    
    println("  📐 Fisher Information Metric")
    println("    Curvature of probability manifold over pitch classes")
    
    # Compute empirical distribution of pitch classes
    counts = zeros(12)
    for n in w.notes
        counts[n+1] += 1
    end
    probs = counts ./ sum(counts)
    
    # Fisher information: I(θ) = E[(∂logp/∂θ)²]
    # For categorical: I_ij = δ_ij/p_i
    fisher_diag = [p > 0 ? 1.0/p : 0.0 for p in probs]
    
    println("    Distribution: $(round.(probs, digits=2))")
    println("    Fisher diag:  $(round.(fisher_diag, digits=1))")
    
    # Geodesic on probability simplex (natural gradient descent)
    # Play trajectory that follows Fisher metric
    
    # High Fisher info = rare = emphasized
    emphasized = [(i-1, fisher_diag[i]) for i in 1:12 if probs[i] > 0]
    sort!(emphasized, by=x->-x[2])  # Sort by Fisher info descending
    
    println("    High curvature (rare notes): $(join([NOTE_NAMES[e[1]+1] for e in emphasized[1:min(4,length(emphasized))]], ", "))")
    
    try
        midis = [60 + e[1] for e in emphasized]
        fishers = [e[2] for e in emphasized]
        
        cmd = ```python3 -c "
import wave, struct, math

midis = [$(join(midis, ","))]
fishers = [$(join(fishers, ","))]
sr = 44100
dur = 5.0
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

# Natural gradient: emphasize high-curvature directions
max_fisher = max(fishers) if fishers else 1.0

for i in range(int(sr * dur)):
    t = i / sr
    sample = 0.0
    
    for idx, (midi, fisher) in enumerate(zip(midis, fishers)):
        freq = midi_to_freq(midi)
        
        # Fisher weight determines amplitude
        weight = fisher / max_fisher
        
        # Phase offset creates geodesic trajectory feel
        phase = 2 * math.pi * freq * t + idx * math.pi / 6
        
        # Envelope with Fisher-weighted emphasis
        note_period = dur / len(midis)
        note_t = t - idx * note_period * 0.5
        if note_t >= 0:
            env = min(1.0, note_t * 5) * math.exp(-note_t * 0.5) * weight * 0.3
            sample += env * math.sin(phase)
    
    out += struct.pack('<h', int(max(-32767, min(32767, sample * 32767))))

with wave.open('/tmp/spc_fisher.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_fisher.wav`, wait=true)
        println("    ✓ Information geodesic traced")
    catch e
        println("    (error: $e)")
    end
end

# --- Solomonoff: Universal prior via algorithmic complexity ---
SPC_CMDS["solomonoff"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🎰 Solomonoff Induction: Universal Prior")
    println("    P(x) = Σ 2^(-|p|) for all programs p that output x")
    
    # Approximate Kolmogorov complexity via run-length encoding
    # Simpler patterns = higher probability = emphasized
    
    notes = w.notes
    
    # Compute "complexity" as inverse compressibility
    # Run-length encode the note sequence
    runs = Tuple{Int, Int}[]  # (note, count)
    current = notes[1]
    count = 1
    for i in 2:length(notes)
        if notes[i] == current
            count += 1
        else
            push!(runs, (current, count))
            current = notes[i]
            count = 1
        end
    end
    push!(runs, (current, count))
    
    # Complexity ≈ number of runs (fewer runs = more compressible)
    complexity = length(runs)
    max_complexity = length(notes)  # Each note different = max complexity
    
    # Also compute interval complexity
    intervals = [(notes[i+1] - notes[i] + 12) % 12 for i in 1:11]
    interval_runs = length(unique(intervals))
    
    println("    Note runs: $complexity (max $max_complexity)")
    println("    Interval variety: $interval_runs/11")
    println("    Runs: $([(NOTE_NAMES[r[1]+1], r[2]) for r in runs])")
    
    # Universal prior weight: 2^(-complexity)
    prior_weight = 2.0^(-complexity)
    println("    Solomonoff prior: 2^(-$complexity) = $(round(prior_weight, digits=6))")
    
    # Simple = probable = consonant/sustained
    # Complex = improbable = dissonant/staccato
    
    try
        cmd = ```python3 -c "
import wave, struct, math

runs = [$(join(["($(r[1]+60), $(r[2]))" for r in runs], ","))]
complexity = $(complexity)
max_complexity = $(max_complexity)
sr = 44100
dur = 6.0
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

# Solomonoff sonification:
# - Low complexity (simple) = long sustained tones, consonant
# - High complexity (random) = short staccato, dissonant

simplicity = 1.0 - (complexity / max_complexity)  # 0 = complex, 1 = simple

t = 0
for midi, run_length in runs:
    freq = midi_to_freq(midi)
    
    # Duration proportional to run length and simplicity
    base_dur = 0.3 + simplicity * 0.4
    note_dur = base_dur * run_length
    
    # Timbre: simple = pure sine, complex = harsh harmonics
    for i in range(int(sr * min(note_dur, dur - t))):
        sample_t = i / sr
        
        # Envelope
        env = min(1.0, sample_t * 10) * min(1.0, (note_dur - sample_t) * 5) * 0.4
        
        # Waveform based on simplicity
        wave_val = math.sin(2 * math.pi * freq * (t + sample_t))
        
        # Add harmonics for complex sequences
        if simplicity < 0.5:
            dissonance = (0.5 - simplicity) * 2
            wave_val += dissonance * 0.3 * math.sin(2 * math.pi * freq * 1.5 * (t + sample_t))
            wave_val += dissonance * 0.2 * math.sin(2 * math.pi * freq * 2.5 * (t + sample_t))
        
        out += struct.pack('<h', int(max(-32767, min(32767, env * wave_val * 32767))))
    
    t += note_dur
    if t >= dur:
        break

# Pad to full duration if needed
while len(out) < int(sr * dur) * 2:
    out += struct.pack('<h', 0)

with wave.open('/tmp/spc_solomonoff.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_solomonoff.wav`, wait=true)
        println("    ✓ Universal prior sonified")
    catch e
        println("    (error: $e)")
    end
end

# --- Anticipate: 7×17 Active Inference Refinement Loop ---
SPC_CMDS["anticipate"] = function(args...)
    w = SPC_WORLD[]
    iterations = isempty(args) ? 7 : parse(Int, args[1])
    results_per = 17
    
    println("  🔮 Anticipatory Active Inference")
    println("    $iterations iterations × $results_per results = $(iterations * results_per) refinement steps")
    println("    Strategy: Simulated annealing + genetic crossover + bit surgery")
    
    # Free energy F = -log p(o|s) + D_KL(q(s)||p(s))
    # Multi-objective: coverage, interval variety, tritone count, derangement
    
    function evaluate(seed::UInt64)
        chain = [color_at(i; seed=seed) for i in 1:12]
        notes = [hue_to_pc(c) for c in chain]
        
        # Coverage (primary)
        coverage = length(unique(notes))
        
        # Interval variety
        intervals = [(notes[i+1] - notes[i] + 12) % 12 for i in 1:11]
        interval_variety = length(unique(intervals))
        
        # Tritone count (tension)
        tritones = count(i -> i == 6, intervals)
        
        # Derangement score (no fixed points)
        fixed = count(i -> notes[i] == i - 1, 1:12)
        derangement = 12 - fixed
        
        # Combined score with weights
        score = coverage * 10 + interval_variety * 2 + tritones + derangement
        return (score, coverage, notes)
    end
    
    current_seed = w.seed
    (current_score, current_cov, current_notes) = evaluate(current_seed)
    best_seed = current_seed
    best_score = current_score
    best_cov = current_cov
    
    # Population for genetic operations
    population = [(current_seed, current_score)]
    
    println()
    println("    Start: cov=$(current_cov)/12, score=$(current_score)")
    
    temperature = 1.0  # Simulated annealing
    
    for iter in 1:iterations
        print("    Iter $iter (T=$(round(temperature, digits=2))): ")
        iter_best_score = current_score
        iter_best_cov = current_cov
        
        for r in 1:results_per
            # Choose perturbation strategy based on iteration phase
            strategy = (iter + r) % 5
            
            if strategy == 0
                # Single bit flip (surgical)
                bit_pos = (hash((iter, r)) % 64)
                test_seed = xor(current_seed, UInt64(1) << bit_pos)
                
            elseif strategy == 1
                # Byte shuffle (structural)
                bytes = reinterpret(UInt8, [current_seed])
                i, j = rand(1:8), rand(1:8)
                bytes[i], bytes[j] = bytes[j], bytes[i]
                test_seed = reinterpret(UInt64, bytes)[1]
                
            elseif strategy == 2
                # Genetic crossover with population member
                if length(population) > 1
                    parent2 = population[rand(1:length(population))][1]
                    mask = UInt64(hash((iter, r, current_seed)) % typemax(UInt64))
                    test_seed = (current_seed & mask) | (parent2 & ~mask)
                else
                    test_seed = xor(current_seed, UInt64(rand(UInt16)) << (r % 4 * 16))
                end
                
            elseif strategy == 3
                # Gradient-like: flip bits that improved before
                # Use hash of best_seed as "gradient direction"
                gradient = xor(best_seed, current_seed)
                noise = UInt64(hash((iter, r))) & 0xFFFF
                test_seed = xor(current_seed, gradient) ⊻ noise
                
            else
                # Lévy flight (occasional large jump)
                if rand() < 0.2
                    test_seed = UInt64(hash((iter, r, best_seed)))
                else
                    delta = UInt64(hash((iter, r, current_seed)) % 0xFFFFFF)
                    test_seed = xor(current_seed, delta)
                end
            end
            
            (test_score, test_cov, test_notes) = evaluate(test_seed)
            
            # Metropolis-Hastings acceptance
            ΔE = test_score - current_score
            accept = ΔE > 0 || rand() < exp(ΔE / temperature)
            
            if accept
                current_seed = test_seed
                current_score = test_score
                current_cov = test_cov
                current_notes = test_notes
                
                # Update population (keep best 5)
                push!(population, (test_seed, test_score))
                sort!(population, by=x->-x[2])
                length(population) > 5 && pop!(population)
            end
            
            if test_score > best_score
                best_score = test_score
                best_seed = test_seed
                best_cov = test_cov
            end
            if test_score > iter_best_score
                iter_best_score = test_score
                iter_best_cov = test_cov
            end
        end
        
        # Cooling schedule
        temperature *= 0.7
        
        println("best_cov=$(iter_best_cov)/12, score=$(iter_best_score)")
    end
    
    println()
    println("    ═══ Final Refinement ═══")
    println("    Best seed: 0x$(string(best_seed, base=16))")
    println("    Coverage: $(best_cov)/12 pitch classes")
    println("    Score: $(best_score) (cov×10 + variety×2 + tritones + derangement)")
    
    # Apply best configuration
    init_world(best_seed)
    w = SPC_WORLD[]
    println("    Chain: $(join([NOTE_NAMES[n+1] for n in w.notes], "-"))")
    
    # Detailed analysis
    intervals = [(w.notes[i+1] - w.notes[i] + 12) % 12 for i in 1:11]
    tritones = count(i -> i == 6, intervals)
    fixed = count(i -> w.notes[i] == i - 1, 1:12)
    println("    Intervals: $(join(intervals, "-"))")
    println("    Tritones: $tritones, Fixed points: $fixed")
    println("    ✓ World updated to refined state")
    
    # Play the refined configuration with richer synthesis
    try
        midis = [60 + n for n in w.notes]
        cmd = ```python3 -c "
import wave, struct, math

midis = [$(join(midis, ","))]
sr = 44100
note_dur = 0.3
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

# Anticipatory synthesis with harmonic prediction
phase = 0
for idx, midi in enumerate(midis):
    freq = midi_to_freq(midi)
    next_midi = midis[(idx + 1) % len(midis)]
    next_freq = midi_to_freq(next_midi)
    prev_midi = midis[(idx - 1) % len(midis)]
    prev_freq = midi_to_freq(prev_midi)
    
    for i in range(int(sr * note_dur)):
        t = i / sr
        progress = t / note_dur
        
        # Main note with vibrato
        vibrato = 1.0 + 0.008 * math.sin(2 * math.pi * 5.5 * t)
        phase += 2 * math.pi * freq * vibrato / sr
        
        # Harmonic content based on interval tension
        interval = abs(midi - prev_midi) % 12
        tension = 1.0 if interval in [1, 2, 6, 10, 11] else 0.3
        
        wave_val = math.sin(phase)
        wave_val += 0.2 * math.sin(2 * phase)  # Octave
        wave_val += tension * 0.15 * math.sin(3 * phase)  # 5th harmonic for tension
        
        # Anticipatory ghost of next note (grows toward end)
        ghost_env = progress ** 3 * 0.2
        ghost_phase = 2 * math.pi * next_freq * (idx * note_dur + t)
        wave_val += ghost_env * math.sin(ghost_phase)
        
        # Memory echo of previous note (fades from start)
        echo_env = (1 - progress) ** 2 * 0.1
        echo_phase = 2 * math.pi * prev_freq * (idx * note_dur + t)
        wave_val += echo_env * math.sin(echo_phase)
        
        # Envelope
        attack = min(1.0, t * 25)
        release = min(1.0, (note_dur - t) * 12)
        env = attack * release * 0.35
        
        out += struct.pack('<h', int(max(-32767, min(32767, env * wave_val * 32767))))

with wave.open('/tmp/spc_anticipate.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_anticipate.wav`, wait=true)
        println("    ✓ Anticipatory sequence complete")
    catch e
        println("    (error: $e)")
    end
end

# --- CRISPR: Cut and splice at specific loci ---
SPC_CMDS["crispr"] = function(args...)
    w = SPC_WORLD[]
    target = isempty(args) ? 5 : parse(Int, args[1])  # Cut position
    
    println("  ✂️ CRISPR: Gene editing at position $target")
    
    # Cut chain and insert new material
    before = w.notes[1:target]
    after = w.notes[target+1:end]
    insert = [(n + 7) % 12 for n in w.notes[1:3]]  # Insert perfect 5th transposition
    
    edited = vcat(before, insert, after)
    
    println("    Before cut: $(join([NOTE_NAMES[n+1] for n in before], "-"))")
    println("    Insert: $(join([NOTE_NAMES[n+1] for n in insert], "-"))")
    println("    After cut: $(join([NOTE_NAMES[n+1] for n in after], "-"))")
    println("    Edited: $(join([NOTE_NAMES[n+1] for n in edited], "-"))")
    
    try
        cmd = ```python3 -c "
import wave, struct, math

before = [$(join([60 + n for n in before], ","))]
insert = [$(join([72 + n for n in insert], ","))]  # Higher octave for insert
after = [$(join([60 + n for n in after], ","))]
sr = 44100
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

def play_notes(notes, is_insert=False):
    result = b''
    for midi in notes:
        freq = midi_to_freq(midi)
        note_dur = 0.25 if not is_insert else 0.35
        
        for i in range(int(sr * note_dur)):
            t = i / sr
            env = min(1.0, t * 30) * min(1.0, (note_dur - t) * 15) * 0.4
            
            if is_insert:
                # Inserted material has distinct timbre
                wave_val = 0.7 * math.sin(2 * math.pi * freq * t)
                wave_val += 0.3 * math.sin(4 * math.pi * freq * t)
            else:
                wave_val = math.sin(2 * math.pi * freq * t)
            
            result += struct.pack('<h', int(max(-32767, min(32767, env * wave_val * 32767))))
    return result

# Play sequence with dramatic pause at cut
out += play_notes(before)
# Cutting sound (noise burst)
for i in range(int(sr * 0.15)):
    import random
    out += struct.pack('<h', int(random.randint(-5000, 5000) * (1 - i/(sr*0.15))))
out += play_notes(insert, is_insert=True)
out += play_notes(after)

with wave.open('/tmp/spc_crispr.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_crispr.wav`, wait=true)
        println("    ✓ Gene edited")
    catch e
        println("    (error: $e)")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# WHALE COMMUNICATION: Sperm Whale Phonetic Alphabet (Sharma et al. 2024)
# Features: Rhythm (18 types), Tempo (5 types), Rubato, Ornamentation
# ═══════════════════════════════════════════════════════════════════════════

# --- Coda: Click sequence with rhythm and tempo ---
SPC_CMDS["coda"] = function(args...)
    w = SPC_WORLD[]
    n_clicks = isempty(args) ? 5 : parse(Int, args[1])
    n_clicks = clamp(n_clicks, 3, 8)
    
    println("  🐋 Sperm Whale Coda ($n_clicks clicks)")
    
    # Generate inter-click intervals from note chain
    # ICI pattern determined by pitch class intervals
    icis = Float64[]
    for i in 1:n_clicks-1
        base_ici = 0.1 + (w.notes[mod1(i, 12)] / 12.0) * 0.3  # 100-400ms
        push!(icis, base_ici)
    end
    
    total_dur = sum(icis)
    rhythm = icis ./ total_dur  # Normalized rhythm
    
    println("    ICIs: $(round.(icis .* 1000, digits=0))ms")
    println("    Rhythm: $(round.(rhythm, digits=2))")
    println("    Total duration: $(round(total_dur * 1000, digits=0))ms")
    
    # Classify tempo type (1-5)
    tempo_type = clamp(Int(ceil(total_dur / 0.2)), 1, 5)
    println("    Tempo type: $tempo_type/5")
    
    try
        ici_str = join(icis, ",")
        cmd = ```python3 -c "
import wave, struct, math

icis = [$(ici_str)]
sr = 44100
out = b''

# Sperm whale clicks: broadband impulses
# Real clicks are ~15kHz center, we use audible ~2kHz

click_dur = 0.003  # 3ms click
click_freq = 2000  # 2kHz center

# Generate each click
t = 0
for i, ici in enumerate([0] + icis):
    t += ici
    
    # Generate click at time t
    for j in range(int(sr * click_dur)):
        sample_t = j / sr
        
        # Broadband click: damped sine burst
        damping = math.exp(-sample_t * 500)
        click = damping * math.sin(2 * math.pi * click_freq * sample_t)
        
        # Add some harmonics for richness
        click += 0.3 * damping * math.sin(4 * math.pi * click_freq * sample_t)
        click += 0.1 * damping * math.sin(6 * math.pi * click_freq * sample_t)
        
        out += struct.pack('<h', int(max(-32767, min(32767, click * 32767 * 0.7))))

# Pad to fill gaps
total_samples = int(sr * (sum(icis) + click_dur))
while len(out) < total_samples * 2:
    out += struct.pack('<h', 0)

with wave.open('/tmp/spc_coda.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_coda.wav`, wait=true)
        println("    ✓ Coda transmitted")
    catch e
        println("    (error: $e)")
    end
end

# --- Rubato: Smooth tempo modulation across coda sequence ---
SPC_CMDS["rubato"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🐋 Rubato: Context-sensitive tempo modulation")
    println("    (Whales smoothly vary coda duration while preserving rhythm)")
    
    # Generate sequence of codas with rubato
    n_codas = 6
    base_duration = 0.4  # 400ms base
    
    # Rubato curve: smooth variation
    rubato_curve = [1.0 + 0.3 * sin(2π * i / n_codas) for i in 1:n_codas]
    
    println("    Rubato curve: $(round.(rubato_curve, digits=2))")
    
    # Each coda has same rhythm but different tempo
    n_clicks = 5
    base_rhythm = [0.15, 0.2, 0.25, 0.4]  # Normalized ICIs
    
    try
        cmd = ```python3 -c "
import wave, struct, math

n_codas = $(n_codas)
rubato = [$(join(rubato_curve, ","))]
base_rhythm = [$(join(base_rhythm, ","))]
base_dur = $(base_duration)
sr = 44100
out = b''

click_dur = 0.003
click_freq = 2000

t = 0
for coda_idx in range(n_codas):
    # Apply rubato scaling
    duration = base_dur * rubato[coda_idx]
    icis = [r * duration for r in base_rhythm]
    
    # Generate clicks
    for ici_idx, ici in enumerate([0] + icis):
        t += ici
        
        for j in range(int(sr * click_dur)):
            sample_t = j / sr
            damping = math.exp(-sample_t * 500)
            click = damping * math.sin(2 * math.pi * click_freq * sample_t)
            click += 0.3 * damping * math.sin(4 * math.pi * click_freq * sample_t)
            out += struct.pack('<h', int(max(-32767, min(32767, click * 32767 * 0.6))))
    
    # Gap between codas (~4 second periodicity scaled down)
    gap = 0.8  # 800ms gap
    for j in range(int(sr * gap)):
        out += struct.pack('<h', 0)
    t += gap

with wave.open('/tmp/spc_rubato.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_rubato.wav`, wait=true)
        println("    ✓ Rubato sequence complete")
    catch e
        println("    (error: $e)")
    end
end

# --- Ornament: Extra click at phrase boundaries ---
SPC_CMDS["ornament"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🐋 Ornamentation: Extra clicks at phrase boundaries")
    println("    (4% of codas have ornaments; signal turn-taking changes)")
    
    # Generate sequence with ornament at beginning/end
    n_codas = 5
    ornament_positions = [1, 5]  # First and last
    
    try
        cmd = ```python3 -c "
import wave, struct, math

n_codas = 5
ornament_pos = [0, 4]  # 0-indexed
sr = 44100
out = b''

click_dur = 0.003
click_freq = 2000
ornament_freq = 1500  # Slightly different pitch for ornament

base_icis = [0.12, 0.15, 0.18, 0.15]  # 4 ICIs = 5 clicks

t = 0
for coda_idx in range(n_codas):
    has_ornament = coda_idx in ornament_pos
    
    # Generate base clicks
    for ici_idx, ici in enumerate([0] + base_icis):
        t += ici
        
        for j in range(int(sr * click_dur)):
            sample_t = j / sr
            damping = math.exp(-sample_t * 500)
            click = damping * math.sin(2 * math.pi * click_freq * sample_t)
            click += 0.3 * damping * math.sin(4 * math.pi * click_freq * sample_t)
            out += struct.pack('<h', int(max(-32767, min(32767, click * 32767 * 0.6))))
    
    # Add ornament if applicable
    if has_ornament:
        t += 0.25  # Longer ICI before ornament
        for j in range(int(sr * click_dur * 1.5)):  # Slightly longer click
            sample_t = j / sr
            damping = math.exp(-sample_t * 400)
            # Ornament has different timbre
            click = damping * math.sin(2 * math.pi * ornament_freq * sample_t)
            click += 0.5 * damping * math.sin(3 * math.pi * ornament_freq * sample_t)
            out += struct.pack('<h', int(max(-32767, min(32767, click * 32767 * 0.7))))
    
    # Gap between codas
    gap = 0.6
    for j in range(int(sr * gap)):
        out += struct.pack('<h', 0)
    t += gap

with wave.open('/tmp/spc_ornament.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)

print(f'Ornaments at positions 1 and 5')
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_ornament.wav`, wait=true)
        println("    ✓ Ornamented sequence complete")
    catch e
        println("    (error: $e)")
    end
end

# --- Upswell: 3 whales coordinating in tripartite pattern ---
SPC_CMDS["upswell"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🐋🐋🐋 Upswell: 3-Whale Tripartite Coordination")
    println("    Pseudo-boolean gadget: each whale pair shares a constraint")
    
    # Extract 3 distinct rhythm patterns from chain
    # Whale A: notes 1-4, Whale B: notes 5-8, Whale C: notes 9-12
    rhythm_a = [(w.notes[i+1] - w.notes[i] + 12) % 12 for i in 1:3]
    rhythm_b = [(w.notes[i+1] - w.notes[i] + 12) % 12 for i in 5:7]  
    rhythm_c = [(w.notes[i+1] - w.notes[i] + 12) % 12 for i in 9:11]
    
    # Tripartite constraint: parity matching
    # If A↔B parity matches, C responds; if B↔C matches, A responds; etc.
    parity_ab = sum(rhythm_a) % 2 == sum(rhythm_b) % 2
    parity_bc = sum(rhythm_b) % 2 == sum(rhythm_c) % 2
    parity_ca = sum(rhythm_c) % 2 == sum(rhythm_a) % 2
    
    println("    Whale A rhythm: $(rhythm_a) (Σ=$(sum(rhythm_a)))")
    println("    Whale B rhythm: $(rhythm_b) (Σ=$(sum(rhythm_b)))")
    println("    Whale C rhythm: $(rhythm_c) (Σ=$(sum(rhythm_c)))")
    println()
    println("    Tripartite constraints:")
    println("      A↔B parity: $(parity_ab ? "✓ match" : "✗ differ")")
    println("      B↔C parity: $(parity_bc ? "✓ match" : "✗ differ")")
    println("      C↔A parity: $(parity_ca ? "✓ match" : "✗ differ")")
    
    # Gadget class: count matching edges
    n_matches = sum([parity_ab, parity_bc, parity_ca])
    gadget_class = if n_matches == 3
        "SAT (all consistent)"
    elseif n_matches == 2
        "2-SAT fragment"
    elseif n_matches == 1
        "XOR gadget"
    else
        "UNSAT (parity conflict)"
    end
    println()
    println("    Gadget class: $gadget_class ($n_matches/3 edges)")
    
    try
        # Convert rhythms to ICI patterns
        ici_a = [0.08 + r * 0.02 for r in rhythm_a]
        ici_b = [0.08 + r * 0.02 for r in rhythm_b]
        ici_c = [0.08 + r * 0.02 for r in rhythm_c]
        
        cmd = ```python3 -c "
import wave, struct, math

sr = 44100
dur = 10.0
out = [0.0] * int(sr * dur)

click_dur = 0.003

def add_click(samples, start_time, freq, amp):
    start_idx = int(start_time * sr)
    for j in range(int(sr * click_dur)):
        if start_idx + j < len(samples):
            sample_t = j / sr
            damping = math.exp(-sample_t * 500)
            click = damping * math.sin(2 * math.pi * freq * sample_t)
            click += 0.3 * damping * math.sin(4 * math.pi * freq * sample_t)
            samples[start_idx + j] += click * amp

def add_coda(samples, start_time, icis, freq, amp):
    t = start_time
    for ici in [0] + icis:
        t += ici
        add_click(samples, t, freq, amp)
    return t

# Three whales with distinct frequencies
# A: 2200Hz (high), B: 1800Hz (mid), C: 1400Hz (low)
freq_a, freq_b, freq_c = 2200, 1800, 1400

ici_a = [$(join(ici_a, ","))]
ici_b = [$(join(ici_b, ","))]
ici_c = [$(join(ici_c, ","))]

# Upswell pattern: whales coordinate in rising intensity
# Phase 1: A leads
# Phase 2: B joins (if A↔B match, they sync; else offset)
# Phase 3: C completes the triad

parity_ab = $(parity_ab ? "True" : "False")
parity_bc = $(parity_bc ? "True" : "False") 
parity_ca = $(parity_ca ? "True" : "False")

# Phase 1: A solo (0-2s)
t = 0.0
for rep in range(3):
    add_coda(out, t, ici_a, freq_a, 0.4)
    t += 0.7

# Phase 2: A + B (2-5s)
t = 2.0
for rep in range(4):
    add_coda(out, t, ici_a, freq_a, 0.35)
    # B responds - synced if parity matches, offset if not
    b_offset = 0.15 if parity_ab else 0.35
    add_coda(out, t + b_offset, ici_b, freq_b, 0.35)
    t += 0.8

# Phase 3: Full triad (5-9s)
t = 5.0
for rep in range(5):
    add_coda(out, t, ici_a, freq_a, 0.3)
    
    # B timing based on A↔B constraint
    b_offset = 0.12 if parity_ab else 0.3
    add_coda(out, t + b_offset, ici_b, freq_b, 0.3)
    
    # C timing based on B↔C and C↔A constraints
    if parity_bc and parity_ca:
        c_offset = 0.24  # Fully synced
    elif parity_bc or parity_ca:
        c_offset = 0.4   # Partial sync
    else:
        c_offset = 0.55  # Out of phase
    add_coda(out, t + c_offset, ici_c, freq_c, 0.3)
    
    t += 0.9

# Convert to bytes
out_bytes = b''
for s in out:
    out_bytes += struct.pack('<h', int(max(-32767, min(32767, s * 32767))))

with wave.open('/tmp/spc_upswell.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out_bytes)

print(f'Upswell complete: A↔B={parity_ab}, B↔C={parity_bc}, C↔A={parity_ca}')
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_upswell.wav`, wait=true)
        println("    ✓ Upswell complete")
    catch e
        println("    (error: $e)")
    end
end

# --- Gadget: Identify tripartite rewriting gadget class ---
SPC_CMDS["gadget"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🔧 Pseudo-Boolean Tripartite Gadget Analysis")
    println("    Finding which 3 notes form valid rewriting gadgets")
    
    # A gadget is a triple (i,j,k) where the interval relationships
    # satisfy a pseudo-boolean constraint
    
    notes = w.notes
    gadgets = Dict{String, Vector{Tuple{Int,Int,Int}}}()
    gadgets["XOR"] = []      # a ⊕ b ⊕ c = 0 (mod 2)
    gadgets["MAJ"] = []      # majority gate
    gadgets["PARITY"] = []   # all same parity
    gadgets["CLAUSE"] = []   # at least one true (SAT clause)
    
    for i in 1:10
        for j in i+1:11
            for k in j+1:12
                a, b, c = notes[i], notes[j], notes[k]
                
                # XOR gadget: intervals sum to 0 mod 12 (tritone closure)
                if (a + b + c) % 12 == 0
                    push!(gadgets["XOR"], (i, j, k))
                end
                
                # Majority: 2 or more share same pitch class mod 4
                m4 = [a % 4, b % 4, c % 4]
                if maximum(counts(m4)) >= 2
                    push!(gadgets["MAJ"], (i, j, k))
                end
                
                # Parity: all odd or all even
                parities = [a % 2, b % 2, c % 2]
                if all(p == parities[1] for p in parities)
                    push!(gadgets["PARITY"], (i, j, k))
                end
                
                # Clause: contains a tritone relationship
                intervals = [(b - a + 12) % 12, (c - b + 12) % 12, (c - a + 12) % 12]
                if 6 in intervals
                    push!(gadgets["CLAUSE"], (i, j, k))
                end
            end
        end
    end
    
    println()
    for (name, triples) in sort(collect(gadgets), by=x->-length(x[2]))
        println("    $name gadgets: $(length(triples))")
        if !isempty(triples) && length(triples) <= 5
            for (i, j, k) in triples
                a, b, c = notes[i], notes[j], notes[k]
                println("      ($i,$j,$k): $(NOTE_NAMES[a+1])-$(NOTE_NAMES[b+1])-$(NOTE_NAMES[c+1])")
            end
        elseif !isempty(triples)
            # Show first 3
            for (i, j, k) in triples[1:min(3, length(triples))]
                a, b, c = notes[i], notes[j], notes[k]
                println("      ($i,$j,$k): $(NOTE_NAMES[a+1])-$(NOTE_NAMES[b+1])-$(NOTE_NAMES[c+1])")
            end
            println("      ... and $(length(triples) - 3) more")
        end
    end
    
    # Find the "best" gadget triple (most constraint coverage)
    best_triple = nothing
    best_score = 0
    for i in 1:10
        for j in i+1:11
            for k in j+1:12
                score = sum([(i,j,k) in g for (_, g) in gadgets])
                if score > best_score
                    best_score = score
                    best_triple = (i, j, k)
                end
            end
        end
    end
    
    if best_triple !== nothing
        i, j, k = best_triple
        a, b, c = notes[i], notes[j], notes[k]
        println()
        println("    ★ Best gadget triple: ($i,$j,$k)")
        println("      Notes: $(NOTE_NAMES[a+1])-$(NOTE_NAMES[b+1])-$(NOTE_NAMES[c+1])")
        println("      Covers $best_score/4 gadget types")
        
        # Sonify the best gadget as a chord progression
        try
            midis = [60 + a, 60 + b, 60 + c]
            cmd = ```python3 -c "
import wave, struct, math

midis = [$(join(midis, ","))]
sr = 44100
dur = 3.0
out = b''

def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12.0))

freqs = [midi_to_freq(m) for m in midis]

for i in range(int(sr * dur)):
    t = i / sr
    sample = 0.0
    
    # Play as arpeggiated then chord
    for idx, freq in enumerate(freqs):
        note_start = idx * 0.3
        if t >= note_start:
            note_t = t - note_start
            env = min(1.0, note_t * 10) * min(1.0, (dur - t) * 3) * 0.25
            sample += env * math.sin(2 * math.pi * freq * t)
    
    out += struct.pack('<h', int(max(-32767, min(32767, sample * 32767))))

with wave.open('/tmp/spc_gadget.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out)
"```
            run(cmd, wait=true)
            run(`afplay /tmp/spc_gadget.wav`, wait=true)
        catch e
            println("    (audio error: $e)")
        end
    end
end

# Helper function for gadget analysis
function counts(arr)
    d = Dict{eltype(arr), Int}()
    for x in arr
        d[x] = get(d, x, 0) + 1
    end
    return values(d)
end

# --- Chorus: Multi-whale exchange with imitation ---
SPC_CMDS["chorus"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🐋 Whale Chorus: Multi-voice exchange")
    println("    (Turn-taking and overlapping codas with duration matching)")
    
    # Two whales exchanging codas
    # Whale A: 5-click codas, Whale B: 4-click codas (different rhythm, matched duration)
    
    try
        cmd = ```python3 -c "
import wave, struct, math

sr = 44100
dur = 8.0
out = [0.0] * int(sr * dur)

click_dur = 0.003

def add_click(samples, start_time, freq, amp):
    start_idx = int(start_time * sr)
    for j in range(int(sr * click_dur)):
        if start_idx + j < len(samples):
            sample_t = j / sr
            damping = math.exp(-sample_t * 500)
            click = damping * math.sin(2 * math.pi * freq * sample_t)
            click += 0.3 * damping * math.sin(4 * math.pi * freq * sample_t)
            samples[start_idx + j] += click * amp

def add_coda(samples, start_time, icis, freq, amp):
    t = start_time
    for ici in [0] + icis:
        t += ici
        add_click(samples, t, freq, amp)

# Whale A: 5-click, rhythm type A, freq 2000Hz
# Whale B: 4-click, rhythm type B, freq 1800Hz (different whale)

# Exchange with rubato matching
whale_a_times = [0.0, 1.5, 3.0, 4.5, 6.0]
whale_b_times = [0.8, 2.3, 3.8, 5.3]  # Responds after ~0.8s

rubato_a = [1.0, 1.1, 1.2, 1.15, 1.0]  # Whale A's tempo drift
rubato_b = [1.05, 1.15, 1.18, 1.05]    # Whale B matches!

base_dur_a = 0.35
base_dur_b = 0.35

rhythm_a = [0.15, 0.2, 0.25, 0.4]  # 5 clicks
rhythm_b = [0.2, 0.35, 0.45]       # 4 clicks

for i, t in enumerate(whale_a_times):
    d = base_dur_a * rubato_a[i]
    icis = [r * d for r in rhythm_a]
    add_coda(out, t, icis, 2000, 0.5)

for i, t in enumerate(whale_b_times):
    d = base_dur_b * rubato_b[i]
    icis = [r * d for r in rhythm_b]
    add_coda(out, t, icis, 1800, 0.45)

# Convert to bytes
out_bytes = b''
for s in out:
    out_bytes += struct.pack('<h', int(max(-32767, min(32767, s * 32767))))

with wave.open('/tmp/spc_chorus.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out_bytes)

print('Two whales exchanging codas with rubato matching')
"```
        run(cmd, wait=true)
        run(`afplay /tmp/spc_chorus.wav`, wait=true)
        println("    ✓ Chorus exchange complete")
    catch e
        println("    (error: $e)")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Whale-Human Bridge Commands (Rapid World Exchange Protocol)
# ═══════════════════════════════════════════════════════════════════════════

# Global bridge instance
const WHALE_BRIDGE = Ref{Any}(nothing)

SPC_CMDS["bridge"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🌉 Whale-Human Semantic Bridge")
    println("    Initializing bidirectional interpretation space...")
    
    # Initialize bridge with current seed
    WHALE_BRIDGE[] = (
        seed = w.seed,
        observations = Vector{Any}(),
        candidate_seeds = Vector{UInt64}(),
        coupling = 0.0,
        fixpoints = Int[]
    )
    
    println("    Base seed: 0x$(string(w.seed, base=16))")
    println("    Ready for: listen, dialogue, consensus, couple")
    println()
    println("    Usage flow:")
    println("      1. listen 0.12,0.15,0.18  (observe whale ICI sequence)")
    println("      2. dialogue               (2-whale bidirectional refine)")
    println("      3. consensus              (3-whale tripartite resolve)")
    println("      4. couple                 (find Galois fixpoints)")
end

SPC_CMDS["listen"] = function(args...)
    if WHALE_BRIDGE[] === nothing
        println("  ⚠ Run 'bridge' first to initialize")
        return
    end
    
    w = SPC_WORLD[]
    
    # Parse ICI sequence (comma-separated floats)
    if isempty(args)
        # Default: generate from current chain
        intervals = [(w.notes[i+1] - w.notes[i] + 12) % 12 for i in 1:4]
        icis = [0.1 + (iv / 12.0) * 0.3 for iv in intervals]
    else
        icis = parse.(Float64, split(args[1], ","))
    end
    
    println("  👂 Listening to whale coda...")
    println("    ICIs: $(round.(icis .* 1000, digits=0))ms")
    
    total = sum(icis)
    rhythm = icis ./ total
    println("    Rhythm: $(round.(rhythm, digits=3))")
    println("    Duration: $(round(total * 1000))ms")
    
    # Rapid walk to find matching seeds
    println("    🔍 Rapid random walk (1000 steps, 4 chains)...")
    
    best_seeds = Tuple{UInt64, Float64}[]
    
    for chain in 1:4
        current = w.seed + UInt64(chain * 0x123456789)
        β = 1.0
        
        for step in 1:250
            # SplitMix64 proposal
            proposal = xor(current, UInt64(hash((step, chain))))
            proposal = xor(proposal, proposal >> 30) * 0xbf58476d1ce4e5b9
            proposal = xor(proposal, proposal >> 27) * 0x94d049bb133111eb
            proposal = xor(proposal, proposal >> 31)
            
            # Score: how well does this seed match the rhythm?
            test_notes = [hue_to_pc(color_at(i; seed=proposal)) for i in 1:length(icis)+1]
            test_ivs = [(test_notes[i+1] - test_notes[i] + 12) % 12 for i in 1:length(icis)]
            test_rhythm = test_ivs ./ max(1, sum(test_ivs))
            
            # Cosine similarity
            dot = sum(rhythm .* test_rhythm)
            norm_a = sqrt(sum(rhythm .^ 2))
            norm_b = sqrt(sum(test_rhythm .^ 2))
            score = (norm_a > 0 && norm_b > 0) ? dot / (norm_a * norm_b) : 0.0
            
            if score > 0.8
                push!(best_seeds, (proposal, score))
            end
            
            if rand() < score
                current = proposal
            end
        end
    end
    
    # Deduplicate and sort
    unique_seeds = Dict{UInt64, Float64}()
    for (s, sc) in best_seeds
        if !haskey(unique_seeds, s) || unique_seeds[s] < sc
            unique_seeds[s] = sc
        end
    end
    sorted = sort(collect(unique_seeds), by=x->-x[2])
    
    if !isempty(sorted)
        best = first(sorted)
        println("    ✓ Found $(length(sorted)) candidate worlds")
        println("    Best: 0x$(string(best[1], base=16)) (score=$(round(best[2], digits=3)))")
        
        # Update bridge
        bridge = WHALE_BRIDGE[]
        push!(bridge.observations, (icis=icis, rhythm=rhythm))
        WHALE_BRIDGE[] = (bridge..., candidate_seeds = [s for (s,_) in first(sorted, 5)])
        
        # Optionally update world
        init_world(best[1])
        println("    World updated to best matching seed")
        SPC_CMDS["chain"]()
    else
        println("    ⚠ No good matches found")
    end
end

SPC_CMDS["dialogue"] = function(args...)
    if WHALE_BRIDGE[] === nothing
        println("  ⚠ Run 'bridge' first")
        return
    end
    
    w = SPC_WORLD[]
    bridge = WHALE_BRIDGE[]
    
    println("  🐋🐋 Two-Whale Dialogue")
    println("    Bidirectional duration matching + rhythm refinement")
    
    # Generate two whale patterns from current seed
    notes = w.notes
    
    # Whale A: positions 1-5
    ivs_a = [(notes[i+1] - notes[i] + 12) % 12 for i in 1:4]
    icis_a = [0.1 + (iv / 12.0) * 0.3 for iv in ivs_a]
    
    # Whale B: positions 5-9
    ivs_b = [(notes[i+1] - notes[i] + 12) % 12 for i in 5:8]
    icis_b = [0.1 + (iv / 12.0) * 0.3 for iv in ivs_b]
    
    dur_a = sum(icis_a)
    dur_b = sum(icis_b)
    dur_match = 1.0 - abs(dur_a - dur_b) / max(dur_a, dur_b)
    
    println("    Whale A: $(round.(icis_a .* 1000, digits=0))ms ($(round(dur_a*1000))ms)")
    println("    Whale B: $(round.(icis_b .* 1000, digits=0))ms ($(round(dur_b*1000))ms)")
    println("    Duration match: $(round(dur_match * 100, digits=1))%")
    
    # Coupling strength increases with duration matching
    new_coupling = min(1.0, bridge.coupling + 0.2 * dur_match)
    WHALE_BRIDGE[] = (bridge..., coupling = new_coupling)
    
    println("    Coupling strength: $(round(new_coupling * 100, digits=1))%")
    
    # Sonify
    try
        cmd = ```python3 -c "
import wave, struct, math

sr = 44100
out = [0.0] * int(sr * 4.0)

def add_click(samples, t, freq, amp):
    idx = int(t * sr)
    for j in range(int(sr * 0.003)):
        if idx + j < len(samples):
            s = j / sr
            d = math.exp(-s * 500)
            samples[idx + j] += d * math.sin(2 * math.pi * freq * s) * amp

# Whale A (high freq)
icis_a = [$(join(icis_a, ","))]
t = 0
for ici in [0] + icis_a:
    t += ici
    add_click(out, t, 2200, 0.6)

# Whale B (low freq, offset)
icis_b = [$(join(icis_b, ","))]
t = $(dur_a + 0.3)
for ici in [0] + icis_b:
    t += ici
    add_click(out, t, 1600, 0.6)

out_bytes = b''.join(struct.pack('<h', int(max(-32767, min(32767, s * 32767)))) for s in out)

with wave.open('/tmp/dialogue.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out_bytes)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/dialogue.wav`, wait=true)
        println("    ✓ Dialogue sonified")
    catch e
        println("    (audio: $e)")
    end
end

SPC_CMDS["consensus"] = function(args...)
    if WHALE_BRIDGE[] === nothing
        println("  ⚠ Run 'bridge' first")
        return
    end
    
    w = SPC_WORLD[]
    bridge = WHALE_BRIDGE[]
    
    println("  🐋🐋🐋 Three-Whale Consensus (Upswell)")
    println("    Tripartite constraint resolution")
    
    # Extract three rhythm patterns
    notes = w.notes
    
    ivs_a = [(notes[i+1] - notes[i] + 12) % 12 for i in 1:3]
    ivs_b = [(notes[i+1] - notes[i] + 12) % 12 for i in 5:7]
    ivs_c = [(notes[i+1] - notes[i] + 12) % 12 for i in 9:11]
    
    # Constraint analysis
    sum_a, sum_b, sum_c = sum(ivs_a), sum(ivs_b), sum(ivs_c)
    
    xor_residue = (sum_a + sum_b + sum_c) % 12
    is_xor = xor_residue == 0
    
    parities = [sum_a % 2, sum_b % 2, sum_c % 2]
    is_parity = length(unique(parities)) == 1
    
    gadget = if is_xor
        "XOR (intervals cancel mod 12)"
    elseif is_parity
        "PARITY (all same parity)"
    else
        "CLAUSE (general constraint)"
    end
    
    println("    Whale A: $(ivs_a) (Σ=$sum_a)")
    println("    Whale B: $(ivs_b) (Σ=$sum_b)")
    println("    Whale C: $(ivs_c) (Σ=$sum_c)")
    println()
    println("    Gadget class: $gadget")
    println("    XOR residue: $xor_residue")
    
    # Increase coupling
    new_coupling = min(1.0, bridge.coupling + 0.3)
    WHALE_BRIDGE[] = (bridge..., coupling = new_coupling)
    
    println("    Coupling: $(round(new_coupling * 100, digits=1))%")
end

SPC_CMDS["couple"] = function(args...)
    if WHALE_BRIDGE[] === nothing
        println("  ⚠ Run 'bridge' first")
        return
    end
    
    w = SPC_WORLD[]
    bridge = WHALE_BRIDGE[]
    
    println("  🔗 Galois Coupling: α-γ Iteration")
    println("    Finding semantic fixpoints (shared meanings)")
    
    # α: Whale → Human (interval extraction)
    # γ: Human → Whale (interval → ICI)
    
    notes = w.notes
    intervals = [(notes[i+1] - notes[i] + 12) % 12 for i in 1:11]
    
    println("    Initial intervals: $intervals")
    
    # Iterate α-γ
    current = intervals
    converged = false
    iters = 0
    
    for i in 1:20
        # γ: intervals → ICIs
        icis = [0.1 + (iv / 12.0) * 0.3 for iv in current]
        
        # α: ICIs → normalized → back to intervals
        rhythm = icis ./ sum(icis)
        recovered = [round(Int, r * 12) % 12 for r in rhythm]
        
        iters = i
        if recovered == current
            converged = true
            break
        end
        current = recovered
    end
    
    if converged
        println("    ✓ Converged in $iters iterations!")
        println("    Fixpoint intervals: $current")
        
        # Count fixpoints (positions where interval is preserved)
        fixpoints = findall(i -> intervals[i] == current[i], 1:length(intervals))
        println("    $(length(fixpoints)) Galois fixpoints at positions: $fixpoints")
        
        WHALE_BRIDGE[] = (bridge..., coupling = 1.0, fixpoints = fixpoints)
        
        println()
        println("    🎵 Coupling complete - playing shared semantic space")
        SPC_CMDS["play"]()
    else
        println("    ⚠ Did not converge in $iters iterations")
        println("    Final state: $current")
    end
end

SPC_CMDS["mixing"] = function(args...)
    println("  📊 Mixing Time Analysis")
    println("    (Coupon collector bound for random walk convergence)")
    
    n = isempty(args) ? 1000 : parse(Int, args[1])
    γ = 0.5772156649  # Euler-Mascheroni
    
    expected = n * (log(n) + γ)
    ε_mixing = ceil(Int, expected * log(100))  # 99% mixing
    
    println("    Seed space size: $n")
    println("    Expected coverage: $(round(expected, digits=0)) steps")
    println("    99% mixing guarantee: $ε_mixing steps")
    println()
    println("    Recommendation: use walk $ε_mixing for full exploration")
end

# ═══════════════════════════════════════════════════════════════════════════
# Trajectory Tracking Commands (Zipf + Information Theory)
# ═══════════════════════════════════════════════════════════════════════════

# Global trajectory state
const TRAJECTORY_STATE = Ref{Any}(nothing)

SPC_CMDS["track"] = function(args...)
    w = SPC_WORLD[]
    
    println("  📈 Trajectory Tracking")
    println("    Information-theoretic seed scoring")
    
    # Initialize tracking state if needed
    if TRAJECTORY_STATE[] === nothing
        TRAJECTORY_STATE[] = (
            corpus = Vector{Vector{Int}}(),
            seeds = Dict{UInt64, NamedTuple}(),
            interactions = Vector{Any}()
        )
    end
    
    state = TRAJECTORY_STATE[]
    
    # Score current seed
    seed = w.seed
    
    # Entropy: diversity of codas from seed neighborhood
    coda_counts = Dict{Vector{Int}, Int}()
    for i in 1:100
        test_seed = seed + UInt64(i)
        test_notes = [hue_to_pc(color_at(j; seed=test_seed)) for j in 1:5]
        pattern = [(test_notes[j+1] - test_notes[j] + 12) % 12 for j in 1:4]
        coda_counts[pattern] = get(coda_counts, pattern, 0) + 1
    end
    
    total = sum(values(coda_counts))
    H = 0.0
    for count in values(coda_counts)
        p = count / total
        if p > 0
            H -= p * log2(p)
        end
    end
    
    # Zipf rank (if corpus exists)
    notes = w.notes
    current_pattern = [(notes[i+1] - notes[i] + 12) % 12 for i in 1:4]
    
    if !isempty(state.corpus)
        counts = Dict{Vector{Int}, Int}()
        for p in state.corpus
            counts[p] = get(counts, p, 0) + 1
        end
        sorted = sort(collect(counts), by=x->-x[2])
        
        rank = findfirst(x -> x[1] == current_pattern, sorted)
        if rank === nothing
            rank = length(sorted) + 1
        end
    else
        rank = 1
    end
    
    # Serial redundancy
    patterns = Vector{Vector{Int}}()
    current = seed
    for _ in 1:10
        test_notes = [hue_to_pc(color_at(j; seed=current)) for j in 1:5]
        pattern = [(test_notes[j+1] - test_notes[j] + 12) % 12 for j in 1:4]
        push!(patterns, pattern)
        # SplitMix64
        current += 0x9e3779b97f4a7c15
        current = (current ⊻ (current >> 30)) * 0xbf58476d1ce4e5b9
        current = (current ⊻ (current >> 27)) * 0x94d049bb133111eb
        current = current ⊻ (current >> 31)
    end
    
    transitions = Dict{Tuple{Vector{Int}, Vector{Int}}, Int}()
    for i in 1:length(patterns)-1
        key = (patterns[i], patterns[i+1])
        transitions[key] = get(transitions, key, 0) + 1
    end
    
    trans_total = sum(values(transitions))
    H_trans = 0.0
    for count in values(transitions)
        p = count / trans_total
        if p > 0
            H_trans -= p * log2(p)
        end
    end
    H_max = log2(max(1, length(transitions)))
    redundancy = H_max > 0 ? 1.0 - H_trans / H_max : 0.0
    
    # Store
    state.seeds[seed] = (
        entropy = H,
        zipf_rank = rank,
        redundancy = redundancy,
        pattern = current_pattern
    )
    
    # Add to corpus
    push!(state.corpus, current_pattern)
    
    println()
    println("    Current seed: 0x$(string(seed, base=16))")
    println("    Entropy H: $(round(H, digits=3)) bits")
    println("    Zipf rank: $rank")
    println("    Serial redundancy: $(round(redundancy * 100, digits=1))%")
    println("    Pattern: $current_pattern")
    println()
    println("    Corpus size: $(length(state.corpus))")
    println("    Tracked seeds: $(length(state.seeds))")
end

SPC_CMDS["zipf"] = function(args...)
    println("  📊 Zipf Distribution Analysis")
    
    if TRAJECTORY_STATE[] === nothing || isempty(TRAJECTORY_STATE[].corpus)
        println("    ⚠ No corpus yet. Use 'track' to record patterns.")
        return
    end
    
    state = TRAJECTORY_STATE[]
    corpus = state.corpus
    
    # Count frequencies
    counts = Dict{Vector{Int}, Int}()
    for p in corpus
        counts[p] = get(counts, p, 0) + 1
    end
    
    # Sort by frequency
    sorted = sort(collect(counts), by=x->-x[2])
    
    println("    Corpus size: $(length(corpus))")
    println("    Unique patterns: $(length(sorted))")
    println()
    println("    Top 10 by frequency (Zipf distribution):")
    
    for (i, (pattern, freq)) in enumerate(first(sorted, 10))
        bar = repeat("█", min(40, freq))
        println("      $i. $pattern  ($freq) $bar")
    end
    
    # Estimate Zipf exponent
    if length(sorted) >= 2
        freqs = [f for (_, f) in sorted]
        log_ranks = log.(1:length(freqs))
        log_freqs = log.(freqs)
        
        n = length(freqs)
        sum_x = sum(log_ranks)
        sum_y = sum(log_freqs)
        sum_xy = sum(log_ranks .* log_freqs)
        sum_x2 = sum(log_ranks .^ 2)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2)
        α = -slope
        
        println()
        println("    Zipf exponent α ≈ $(round(α, digits=2))")
        println("    (α ≈ 1.0 indicates Zipf's law)")
        
        if α > 0.8 && α < 1.3
            println("    ✓ Distribution follows Zipf's law!")
        end
    end
end

SPC_CMDS["promising"] = function(args...)
    println("  🌟 Promising Seeds")
    
    if TRAJECTORY_STATE[] === nothing || isempty(TRAJECTORY_STATE[].seeds)
        println("    ⚠ No seeds tracked yet. Use 'track' after interactions.")
        return
    end
    
    state = TRAJECTORY_STATE[]
    bridge = WHALE_BRIDGE[]
    
    # Score each seed
    scored = []
    for (seed, info) in state.seeds
        # Composite score
        # Prefer: high entropy, mid-range Zipf, low redundancy
        entropy_score = info.entropy / 4.0  # normalize to ~0-1
        
        # Zipf sweet spot: rank 5-15
        zipf_score = exp(-((info.zipf_rank - 10)^2) / 50.0)
        
        # Low redundancy = more informative
        redund_score = 1.0 - info.redundancy
        
        # Coupling from bridge (if available)
        coupling_score = bridge !== nothing ? bridge.coupling : 0.0
        
        composite = 0.2 * entropy_score + 0.3 * zipf_score + 
                    0.1 * redund_score + 0.4 * coupling_score
        
        push!(scored, (seed=seed, score=composite, info=info))
    end
    
    # Sort by score
    sort!(scored, by=x->-x.score)
    
    println("    Scored $(length(scored)) seeds")
    println()
    println("    Top promising seeds:")
    
    for (i, s) in enumerate(first(scored, 5))
        println("      $i. 0x$(string(s.seed, base=16)[1:8])...")
        println("         Score: $(round(s.score, digits=3))")
        println("         H=$(round(s.info.entropy, digits=2)), " *
                "Zipf=$(s.info.zipf_rank), " *
                "R=$(round(s.info.redundancy * 100))%")
    end
    
    # Offer to switch to best
    if !isempty(scored)
        best = first(scored)
        println()
        println("    💡 Use 'seed $(string(best.seed, base=16))' to switch to best")
    end
end

SPC_CMDS["corpus"] = function(args...)
    if TRAJECTORY_STATE[] === nothing
        TRAJECTORY_STATE[] = (
            corpus = Vector{Vector{Int}}(),
            seeds = Dict{UInt64, NamedTuple}(),
            interactions = Vector{Any}()
        )
    end
    
    state = TRAJECTORY_STATE[]
    
    if isempty(args)
        println("  📚 Coda Corpus")
        println("    Size: $(length(state.corpus))")
        println("    Usage: corpus add | corpus clear | corpus export")
    elseif args[1] == "add"
        # Add current pattern to corpus
        w = SPC_WORLD[]
        pattern = [(w.notes[i+1] - w.notes[i] + 12) % 12 for i in 1:4]
        push!(state.corpus, pattern)
        println("    Added: $pattern")
        println("    Corpus size: $(length(state.corpus))")
    elseif args[1] == "clear"
        state = (
            corpus = Vector{Vector{Int}}(),
            seeds = state.seeds,
            interactions = state.interactions
        )
        TRAJECTORY_STATE[] = state
        println("    Corpus cleared")
    elseif args[1] == "export"
        # Export to JSON
        json_str = "[\n"
        for (i, p) in enumerate(state.corpus)
            json_str *= "  $(p)"
            if i < length(state.corpus)
                json_str *= ","
            end
            json_str *= "\n"
        end
        json_str *= "]"
        
        path = "/tmp/coda_corpus.json"
        open(path, "w") do f
            write(f, json_str)
        end
        println("    Exported to $path")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization Commands
# ═══════════════════════════════════════════════════════════════════════════

SPC_CMDS["heatmap"] = function(args...)
    println("  🗺 Coupling Heatmap")
    
    if TRAJECTORY_STATE[] === nothing || isempty(TRAJECTORY_STATE[].seeds)
        println("    ⚠ No data. Use 'track' during interactions first.")
        return
    end
    
    state = TRAJECTORY_STATE[]
    seeds = collect(keys(state.seeds))
    n = min(10, length(seeds))
    
    println("    Seed similarity matrix (top $n seeds):")
    println()
    
    # Compute pairwise similarity
    for i in 1:n
        row = ""
        for j in 1:n
            # Compare patterns
            info_i = state.seeds[seeds[i]]
            info_j = state.seeds[seeds[j]]
            
            if i == j
                row *= "██"
            else
                # Similarity based on pattern overlap
                sim = sum(info_i.pattern .== info_j.pattern) / 4.0
                if sim > 0.75
                    row *= "▓▓"
                elseif sim > 0.5
                    row *= "▒▒"
                elseif sim > 0.25
                    row *= "░░"
                else
                    row *= "  "
                end
            end
        end
        # Short seed identifier
        seed_str = string(seeds[i], base=16)[1:min(4, length(string(seeds[i], base=16)))]
        println("    $seed_str │$row│")
    end
end

SPC_CMDS["evolution"] = function(args...)
    println("  📈 Trajectory Evolution")
    
    if TRAJECTORY_STATE[] === nothing || length(TRAJECTORY_STATE[].corpus) < 5
        println("    ⚠ Need at least 5 corpus entries. Use 'track' more.")
        return
    end
    
    state = TRAJECTORY_STATE[]
    corpus = state.corpus
    
    # Plot entropy evolution over corpus growth
    println("    Corpus entropy over time:")
    println()
    
    entropies = Float64[]
    for t in 5:length(corpus)
        # Entropy of first t patterns
        counts = Dict{Vector{Int}, Int}()
        for p in corpus[1:t]
            counts[p] = get(counts, p, 0) + 1
        end
        
        H = 0.0
        total = t
        for count in values(counts)
            prob = count / total
            if prob > 0
                H -= prob * log2(prob)
            end
        end
        push!(entropies, H)
    end
    
    # ASCII sparkline
    max_H = maximum(entropies)
    min_H = minimum(entropies)
    range_H = max_H - min_H
    
    chars = " ▁▂▃▄▅▆▇█"
    sparkline = ""
    for H in entropies
        idx = range_H > 0 ? round(Int, (H - min_H) / range_H * 8) + 1 : 5
        idx = clamp(idx, 1, 9)
        sparkline *= chars[idx]
    end
    
    println("    H: $sparkline")
    println("       $(round(min_H, digits=2))         $(round(max_H, digits=2)) bits")
    println()
    println("    Final entropy: $(round(entropies[end], digits=3)) bits")
    println("    Growth: $(length(corpus)) patterns")
end

SPC_CMDS["midi"] = function(args...)
    w = SPC_WORLD[]
    
    filename = isempty(args) ? "/tmp/whale_translation.mid" : args[1]
    
    println("  🎹 MIDI Export")
    println("    Exporting color chain to MIDI...")
    
    # Simple MIDI file generation
    # Format 0, single track
    
    notes = w.notes
    
    # MIDI header
    header = UInt8[
        0x4D, 0x54, 0x68, 0x64,  # "MThd"
        0x00, 0x00, 0x00, 0x06,  # Header length = 6
        0x00, 0x00,              # Format 0
        0x00, 0x01,              # 1 track
        0x00, 0x60               # 96 ticks per quarter
    ]
    
    # Build track data
    track_data = UInt8[]
    
    # Tempo: 120 BPM = 500000 microseconds per beat
    append!(track_data, [0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20])
    
    # Notes
    for (i, note) in enumerate(notes)
        midi_note = 60 + note  # Middle C + pitch class
        
        # Note on (delta=0 for first, 48 ticks = eighth note for rest)
        delta = i == 1 ? 0x00 : 0x30
        push!(track_data, delta)
        push!(track_data, 0x90)  # Note on, channel 0
        push!(track_data, midi_note)
        push!(track_data, 0x64)  # Velocity 100
        
        # Note off after quarter note (96 ticks)
        push!(track_data, 0x60)
        push!(track_data, 0x80)  # Note off
        push!(track_data, midi_note)
        push!(track_data, 0x00)
    end
    
    # End of track
    append!(track_data, [0x00, 0xFF, 0x2F, 0x00])
    
    # Track header
    track_len = length(track_data)
    track_header = UInt8[
        0x4D, 0x54, 0x72, 0x6B,  # "MTrk"
        UInt8((track_len >> 24) & 0xFF),
        UInt8((track_len >> 16) & 0xFF),
        UInt8((track_len >> 8) & 0xFF),
        UInt8(track_len & 0xFF)
    ]
    
    # Write file
    open(filename, "w") do f
        write(f, header)
        write(f, track_header)
        write(f, track_data)
    end
    
    println("    ✓ Exported to $filename")
    println("    Notes: $(join([NOTE_NAMES[n+1] for n in notes], "-"))")
end

SPC_CMDS["clan"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🐋 Clan Dialect Analysis")
    println("    Comparing current pattern to known EC-1 clan markers")
    
    # Known EC-1 clan patterns (from Sharma et al.)
    # These are simplified representations
    ec1_markers = [
        ([1, 1, 1, 1], "5R1 - Regular 5-click"),
        ([2, 1, 1, 1], "5R2 - Accelerating"),
        ([1, 1, 1, 2], "5R3 - Decelerating"),
        ([1, 2, 2, 1], "5R4 - Symmetric"),
        ([3, 1, 1, 1], "4R1 - Quick start")
    ]
    
    current = [(w.notes[i+1] - w.notes[i] + 12) % 12 for i in 1:4]
    
    # Normalize to rhythm pattern (relative ratios)
    total = sum(current)
    if total > 0
        rhythm = [round(Int, c * 4 / total) for c in current]
        rhythm = [max(1, r) for r in rhythm]  # Ensure no zeros
    else
        rhythm = [1, 1, 1, 1]
    end
    
    println("    Current pattern: $current")
    println("    Normalized rhythm: $rhythm")
    println()
    println("    Clan marker matches:")
    
    best_match = nothing
    best_score = 0.0
    
    for (marker, name) in ec1_markers
        # Cosine similarity
        dot = sum(rhythm .* marker)
        norm_a = sqrt(sum(rhythm .^ 2))
        norm_b = sqrt(sum(marker .^ 2))
        sim = (norm_a > 0 && norm_b > 0) ? dot / (norm_a * norm_b) : 0.0
        
        bar = repeat("█", round(Int, sim * 20))
        println("      $name: $(round(sim * 100))% $bar")
        
        if sim > best_score
            best_score = sim
            best_match = name
        end
    end
    
    println()
    if best_score > 0.8
        println("    ✓ Strong match: $best_match")
        println("    Likely EC-1 clan dialect!")
    elseif best_score > 0.5
        println("    ~ Partial match: $best_match")
    else
        println("    ? No strong clan marker detected")
        println("    Could be novel pattern or different clan")
    end
end

SPC_CMDS["translate"] = function(args...)
    w = SPC_WORLD[]
    
    println("  🌐 Bidirectional Translation")
    println()
    
    # Human → Whale
    println("    Human interpretation (musical):")
    notes_str = join([NOTE_NAMES[n+1] for n in w.notes], "-")
    println("      Notes: $notes_str")
    
    intervals = [(w.notes[i+1] - w.notes[i] + 12) % 12 for i in 1:11]
    println("      Intervals: $(join(intervals, "-"))")
    
    # Whale interpretation
    println()
    println("    Whale interpretation (acoustic):")
    
    # Convert to ICIs
    icis = [0.1 + (iv / 12.0) * 0.3 for iv in intervals[1:4]]
    println("      ICIs: $(round.(icis .* 1000, digits=0))ms")
    
    # Rhythm type
    total = sum(icis)
    rhythm = icis ./ total
    println("      Rhythm: $(round.(rhythm, digits=2))")
    println("      Duration: $(round(total * 1000))ms")
    
    # Tempo type
    tempo = clamp(ceil(Int, total / 0.2), 1, 5)
    println("      Tempo type: $tempo/5")
    
    # Ornament?
    has_ornament = any(iv == 0 for iv in intervals)
    println("      Ornament: $(has_ornament ? "yes (repeated note)" : "no")")
    
    println()
    println("    Equivalence:")
    println("      Human: $notes_str")
    println("      Whale: Coda($(length(icis)+1) clicks, R$(tempo), $(round(total*1000))ms)")
    
    # Play both
    println()
    print("    Play human version? (y/n) ")
    # Can't actually read input in this context, so just play
    println("    Playing whale version...")
    SPC_CMDS["coda"](string(length(icis)+1))
end

SPC_CMDS["simulate"] = function(args...)
    println("  🎲 Whale Coda Simulation (EC-1 Clan Data)")
    
    rhythm_type = isempty(args) ? "5R1" : args[1]
    
    # Check if we have the data
    if !isdefined(Main, :EC1_RHYTHM_TYPES) && !isdefined(@__MODULE__, :EC1_RHYTHM_TYPES)
        println("    ⚠ whale_data.jl not loaded")
        return
    end
    
    println("    Generating realistic coda...")
    println()
    
    # Use EC1 data if available, otherwise simulate
    # Available rhythm types
    available = ["3R1", "4R1", "4R2", "4R3", "5R1", "5R2", "5R3", "5R4", "5R5", "5R6",
                 "6R1", "6R2", "7R1", "7R2", "8R1", "ID1", "ID2", "ID3"]
    
    if !(rhythm_type in available)
        println("    Unknown rhythm type: $rhythm_type")
        println("    Available: $(join(available, ", "))")
        return
    end
    
    # Simplified simulation
    rhythm_data = Dict(
        "3R1" => [0.5, 0.5],
        "4R1" => [0.33, 0.33, 0.34],
        "4R2" => [0.4, 0.3, 0.3],
        "5R1" => [0.25, 0.25, 0.25, 0.25],
        "5R2" => [0.3, 0.25, 0.25, 0.2],
        "5R3" => [0.2, 0.25, 0.25, 0.3],
        "ID1" => [0.4, 0.2, 0.2, 0.2],
        "ID2" => [0.2, 0.2, 0.2, 0.4],
    )
    
    rhythm = get(rhythm_data, rhythm_type, [0.25, 0.25, 0.25, 0.25])
    tempo_ms = 600.0
    
    # Generate ICIs with small random variation
    icis = [r * tempo_ms * (1.0 + 0.05 * (rand() - 0.5)) for r in rhythm]
    
    println("    Rhythm type: $rhythm_type")
    println("    Pattern: $(round.(rhythm, digits=2))")
    println("    ICIs: $(round.(icis, digits=0))ms")
    println("    Duration: $(round(sum(icis)))ms")
    println("    Clicks: $(length(icis) + 1)")
    println()
    
    # Convert to intervals for Gay.jl
    total = sum(icis)
    intervals = [round(Int, (ici / total) * 12) % 12 for ici in icis]
    println("    → Intervals (mod 12): $intervals")
    
    # Find matching seed
    println()
    println("    🔍 Searching for matching seed...")
    
    w = SPC_WORLD[]
    best_seed = w.seed
    best_score = 0.0
    
    for delta in 0:1000
        test_seed = w.seed + UInt64(delta)
        test_notes = [hue_to_pc(color_at(j; seed=test_seed)) for j in 1:length(intervals)+1]
        test_ivs = [(test_notes[j+1] - test_notes[j] + 12) % 12 for j in 1:length(intervals)]
        
        # Score
        matches = sum(test_ivs .== intervals)
        if matches > best_score
            best_score = matches
            best_seed = test_seed
        end
        
        if matches == length(intervals)
            break
        end
    end
    
    println("    Best seed: 0x$(string(best_seed, base=16))")
    println("    Match: $(Int(best_score))/$(length(intervals)) intervals")
    
    if best_score >= length(intervals) * 0.75
        println()
        println("    ✓ Good match! Updating world...")
        init_world(best_seed)
        SPC_CMDS["chain"]()
        
        # Play as whale coda
        println()
        println("    Playing as whale coda...")
        SPC_CMDS["coda"](string(length(intervals)+1))
    end
end

SPC_CMDS["exchange"] = function(args...)
    println("  🐋🐋 Simulated Whale Exchange")
    println()
    
    n_codas = isempty(args) ? 6 : parse(Int, args[1])
    
    println("    Generating $n_codas-coda exchange (Whale A ↔ Whale B)...")
    println()
    
    # Rubato curve
    rubato = [0.2 * sin(2π * i / n_codas) for i in 1:n_codas]
    
    exchange_data = []
    
    for i in 1:n_codas
        whale = i % 2 == 1 ? "A" : "B"
        rhythm = rand() < 0.7 ? [0.25, 0.25, 0.25, 0.25] : [0.3, 0.25, 0.25, 0.2]
        tempo = 600.0 * (1.0 + rubato[i])
        
        icis = [r * tempo for r in rhythm]
        
        marker = whale == "A" ? "🐋" : "🐳"
        println("    $marker Whale $whale: $(round.(icis, digits=0))ms ($(round(sum(icis)))ms)")
        
        push!(exchange_data, (whale=whale, icis=icis, tempo=tempo))
    end
    
    println()
    println("    Duration matching analysis:")
    
    # Check duration matching
    durs = [sum(e.icis) for e in exchange_data]
    for i in 1:length(durs)-1
        diff = abs(durs[i+1] - durs[i])
        match = 100 * (1 - diff / max(durs[i], durs[i+1]))
        println("      Coda $i → $(i+1): $(round(match, digits=1))% match")
    end
    
    # Play exchange
    println()
    println("    Playing exchange...")
    
    try
        freq_a = 2200
        freq_b = 1600
        
        all_events = []
        t = 0.0
        
        for (i, e) in enumerate(exchange_data)
            freq = e.whale == "A" ? freq_a : freq_b
            for ici in [0; e.icis]
                t += ici / 1000.0
                push!(all_events, (t=t, freq=freq))
            end
            t += 0.8  # Gap between codas
        end
        
        events_str = join(["($(round(e.t, digits=3)), $(e.freq))" for e in all_events], ", ")
        
        cmd = ```python3 -c "
import wave, struct, math

events = [$(events_str)]
sr = 44100
total_dur = max(e[0] for e in events) + 0.5
out = [0.0] * int(sr * total_dur)

def add_click(samples, t, freq):
    idx = int(t * sr)
    for j in range(int(sr * 0.003)):
        if idx + j < len(samples):
            s = j / sr
            d = math.exp(-s * 500)
            samples[idx + j] += d * math.sin(2 * math.pi * freq * s) * 0.5

for t, freq in events:
    add_click(out, t, freq)

out_bytes = b''.join(struct.pack('<h', int(max(-32767, min(32767, s * 32767)))) for s in out)

with wave.open('/tmp/exchange.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out_bytes)
"```
        run(cmd, wait=true)
        run(`afplay /tmp/exchange.wav`, wait=true)
        println("    ✓ Exchange complete")
    catch e
        println("    (audio: $e)")
    end
end

SPC_CMDS["session"] = function(args...)
    println("  📋 Session Summary")
    println()
    
    w = SPC_WORLD[]
    bridge = WHALE_BRIDGE[]
    state = TRAJECTORY_STATE[]
    
    println("    Current World:")
    println("      Seed: 0x$(string(w.seed, base=16))")
    println("      Notes: $(join([NOTE_NAMES[n+1] for n in w.notes], "-"))")
    println("      Coverage: $(length(unique(w.notes)))/12")
    println()
    
    if bridge !== nothing
        println("    Whale Bridge:")
        println("      Coupling: $(round(bridge.coupling * 100, digits=1))%")
        println("      Observations: $(length(bridge.observations))")
        println("      Fixpoints: $(length(bridge.fixpoints))")
        println()
    end
    
    if state !== nothing
        println("    Trajectory Tracking:")
        println("      Corpus size: $(length(state.corpus))")
        println("      Seeds tracked: $(length(state.seeds))")
        
        if !isempty(state.seeds)
            # Best seed
            best = nothing
            best_score = -Inf
            for (seed, info) in state.seeds
                score = info.entropy - 0.1 * info.zipf_rank
                if score > best_score
                    best_score = score
                    best = (seed, info)
                end
            end
            if best !== nothing
                println("      Best seed: 0x$(string(best[1], base=16)[1:8])...")
            end
        end
        println()
    end
    
    println("    Next steps:")
    if bridge === nothing
        println("      1. bridge    - Initialize whale bridge")
    elseif bridge.coupling < 0.5
        println("      1. listen    - Observe more whale codas")
        println("      2. dialogue  - 2-whale refinement")
    elseif bridge.coupling < 1.0
        println("      1. consensus - 3-whale resolution")
        println("      2. couple    - Find Galois fixpoints")
    else
        println("      ✓ Fully coupled! Use 'translate' to verify")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# WHALE WORLD: Parallel SPI Demonstration
# ═══════════════════════════════════════════════════════════════════════════

const WHALE_WORLD = Ref{WhaleWorld}()

SPC_CMDS["world"] = function(args...)
    action = isempty(args) ? "status" : lowercase(args[1])
    
    if action == "init" || !isassigned(WHALE_WORLD)
        n = length(args) > 1 ? parse(Int, args[2]) : 6
        WHALE_WORLD[] = demo_whale_world(n_whales=n, seed=SPC_WORLD[].seed)
        println("  🌊 Whale World initialized")
        println("     $(length(WHALE_WORLD[].whales)) whales, base seed 0x$(string(WHALE_WORLD[].base_seed, base=16))")
        return
    end
    
    world = WHALE_WORLD[]
    
    if action == "status"
        println("  🌊 Whale World Status")
        println("  ═══════════════════════════════════════")
        println("     Base seed: 0x$(string(world.base_seed, base=16))")
        println("     Whales: $(length(world.whales))")
        println("     Triads: $(length(world.synergies))")
        println("     Fingerprint: 0x$(string(world_state_hash(world), base=16))")
        println()
        println("  Whales:")
        for (id, w) in sort(collect(world.whales))
            print("     $id: ")
            for c in w.chain[1:6]
                r = round(Int, clamp(c.r, 0, 1) * 255)
                g = round(Int, clamp(c.g, 0, 1) * 255)
                b = round(Int, clamp(c.b, 0, 1) * 255)
                print("\e[48;2;$(r);$(g);$(b)m  \e[0m")
            end
            println(" $(length(unique(w.notes)))/12 PCs")
        end
        
    elseif action == "add"
        id = length(args) > 1 ? args[2] : "W$(lpad(length(world.whales)+1, 3, '0'))"
        add_whale!(world, id)
        println("  + Added whale $id")
        
    elseif action == "synergy"
        println("  🔬 Computing tripartite synergies...")
        compute_all_synergies!(world)
        println("     Computed $(length(world.synergies)) triads")
        
    elseif action == "optimal"
        k = length(args) > 1 ? parse(Int, args[2]) : 5
        optimal = find_optimal_triads(world; k=k)
        println("  ⭐ Top $k synergistic triads:")
        for (key, syn) in optimal
            gadget_color = syn.gadget_class == :XOR ? "\e[32m" : 
                          syn.gadget_class == :MAJ ? "\e[33m" : "\e[36m"
            println("     $(key): $(gadget_color)$(syn.gadget_class)\e[0m coupling=$(round(syn.coupling_score, digits=3))")
        end
        
    elseif action == "spi"
        println("  🎯 Running SPI Demonstration...")
        result = spi_parallel_demo(world; verbose=true)
        
    elseif action == "verify"
        challenge = first_contact_challenge(world)
        println("  🔐 First-Contact Verification")
        println("     Challenge seeds: $(length(challenge.challenge_seeds))")
        println("     Expected fingerprint: 0x$(string(challenge.expected_fingerprint, base=16))")
        
        verification = verify_first_contact(world, challenge.expected_fingerprint)
        status = verification.verified ? "\e[32m✓ VERIFIED\e[0m" : "\e[31m✗ FAILED\e[0m"
        println("     Status: $status")
        
    elseif action == "matrix"
        result = synergy_matrix(world)
        println("  📊 Synergy Matrix")
        println("     (whale pair → average coupling when both in a triad)")
        println()
        
        ids = result.whale_ids
        print("        ")
        for id in ids
            print("  $(id[1:min(4,length(id))]) ")
        end
        println()
        
        for (i, id) in enumerate(ids)
            print("   $(id[1:min(4,length(id))]) ")
            for j in 1:length(ids)
                v = result.matrix[i, j]
                if v > 0.7
                    print("\e[32m$(round(v, digits=2)) \e[0m")
                elseif v > 0.5
                    print("\e[33m$(round(v, digits=2)) \e[0m")
                else
                    print("$(round(v, digits=2)) ")
                end
            end
            println()
        end
        
    elseif action == "help"
        println("""
  Whale World Commands:
     world           - show world status
     world init [n]  - create world with n whales
     world add [id]  - add a whale
     world synergy   - compute all tripartite synergies
     world optimal [k] - show top k synergistic triads
     world spi       - run parallel SPI demonstration
     world verify    - first-contact verification protocol
     world matrix    - show whale pair synergy matrix
""")
    else
        println("  Unknown world action: $action (try 'world help')")
    end
end

SPC_CMDS["triads"] = function(args...)
    if !isassigned(WHALE_WORLD)
        println("  Initialize whale world first: world init")
        return
    end
    
    world = WHALE_WORLD[]
    if isempty(world.synergies)
        compute_all_synergies!(world)
    end
    
    filter_gadget = isempty(args) ? nothing : Symbol(uppercase(args[1]))
    
    println("  🐋🐋🐋 Whale Triads")
    println()
    
    for (key, syn) in sort(collect(world.synergies), by=x->-x[2].coupling_score)
        if filter_gadget !== nothing && syn.gadget_class != filter_gadget
            continue
        end
        
        # Gadget class with color
        gadget_color = syn.gadget_class == :XOR ? "\e[32m" : 
                      syn.gadget_class == :MAJ ? "\e[33m" :
                      syn.gadget_class == :PARITY ? "\e[36m" : "\e[37m"
        
        # Coupling bar
        bar_len = round(Int, syn.coupling_score * 20)
        bar = repeat("█", bar_len) * repeat("░", 20 - bar_len)
        
        println("  $(key[1]) + $(key[2]) + $(key[3])")
        println("     $(gadget_color)$(syn.gadget_class)\e[0m  │$bar│ $(round(syn.coupling_score, digits=3))")
        if syn.xor_residue == 0
            println("     XOR cancellation: ✓ (residue = 0)")
        end
        println()
    end
end

SPC_CMDS["fingerprint"] = function(args...)
    w = SPC_WORLD[]
    
    seeds = if !isempty(args)
        [parse(UInt64, s, base=16) for s in args]
    else
        [w.seed]
    end
    
    fp = color_fingerprint(seeds)
    
    println("  🔐 Color Fingerprint")
    println("     Seeds: $(length(seeds))")
    for s in seeds
        println("       0x$(string(s, base=16))")
    end
    println()
    println("     Fingerprint: 0x$(string(fp, base=16))")
    println()
    println("  This fingerprint proves we computed the same")
    println("  color chains from the same seeds (SPI verification).")
end

# --- Help ---

SPC_CMDS["?"] = function(args...)
    println("""
  ╔═══════════════════════════════════════════════════════════════╗
  ║  SPC REPL: Symbolic · Possible · Compositional                ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  SYMBOLIC (S)           POSSIBLE (P)        COMPOSITIONAL (C) ║
  ║  chain  - show chain    cf [Δ] - counterfact  fix  - fixpoints║
  ║  notes  - pitch classes worlds - all worlds   obs  - obstruct ║
  ║  play   - sonify chain  modal  - □◇ analysis  tensor - ⊗ test ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  DISTILLATION                                                 ║
  ║  distill - analyze obstructions & find resolutions            ║
  ║  seek <type> - find: chromatic, derangement, magic, tritone   ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  TROPICAL & LAWVERE                                           ║
  ║  tropical [sr] - semiring: min+, max+, minmax, bool           ║
  ║  lawvere  - fixed point theorem & self-reference              ║
  ║  compose [Δ] - chain composition via ⊗                        ║
  ║  derange  - permutation derangement analysis                  ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  AUDIO SYNTHESIS                                              ║
  ║  theremin [s] - chromatic glissando (s seconds)               ║
  ║  glide [n]  - chromatic walk (n steps)                        ║
  ║  spectrum   - visual + harmonic synthesis                     ║
  ║  arp [pat]  - arpeggio: up, down, updown, random              ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  STRANGE INSTRUMENTS (Biochemical Synthesis)                  ║
  ║  samovar  - nested layered drones    involution - f(f(x))=x   ║
  ║  peptide  - protein folding          mobius - retrograde inv  ║
  ║  quine    - seed sings itself        ribosome - codon chords  ║
  ║  enzyme   - catalytic acceleration   helicase - unwinding     ║
  ║  prion    - infectious spread        mitosis - cell division  ║
  ║  telomere - chromatic aging          apoptosis - cell death   ║
  ║  crispr [pos] - gene editing at position                      ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  INFORMATION GEOMETRY & ACTIVE INFERENCE                      ║
  ║  elegans  - C.elegans 302-neuron connectome cascade           ║
  ║  immune   - clonal selection + affinity maturation (7 rounds) ║
  ║  fisher   - Fisher information metric on pitch distribution   ║
  ║  anticipate [n] - 7×17 active inference refinement loop       ║
  ║  solomonoff - Kolmogorov complexity / universal prior        ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  WHALE COMMUNICATION (Sperm Whale Phonetic Alphabet)         ║
  ║  coda [n]   - click coda with n clicks (3-8)                 ║
  ║  rubato     - smooth tempo modulation (context-sensitive)    ║
  ║  ornament   - add ornamentation clicks at phrase boundaries  ║
  ║  chorus     - multi-voice whale exchange with imitation      ║
  ║  upswell    - 3-whale tripartite coordination                ║
  ║  gadget     - pseudo-boolean constraint analysis             ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  WHALE-HUMAN BRIDGE (Rapid World Exchange)                   ║
  ║  bridge     - init bidirectional interpretation bridge       ║
  ║  listen <icis> - observe whale coda (ICI sequence)           ║
  ║  dialogue   - 2-whale bidirectional refinement               ║
  ║  consensus  - 3-whale tripartite meaning resolution          ║
  ║  couple     - iterate Galois α-γ until fixpoints found       ║
  ║  mixing     - estimate random walk mixing time               ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  TRAJECTORY TRACKING (Zipf + Information Theory)             ║
  ║  track      - score current seed (entropy, Zipf, redundancy) ║
  ║  zipf       - analyze corpus Zipf distribution               ║
  ║  promising  - show top-scoring seeds for translation         ║
  ║  corpus     - manage coda corpus (add/clear/export)          ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  ANALYSIS & EXPORT                                           ║
  ║  heatmap    - seed similarity matrix visualization           ║
  ║  evolution  - corpus entropy sparkline over time             ║
  ║  clan       - EC-1 dialect marker matching                   ║
  ║  translate  - bidirectional human↔whale display              ║
  ║  midi [file] - export color chain to MIDI file               ║
  ║  session    - full session summary + next steps              ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  WHALE WORLD (Parallel SPI Demonstration)                    ║
  ║  world           - show world status with whale colors       ║
  ║  world init [n]  - create world with n whales                ║
  ║  world synergy   - compute N-choose-3 tripartite synergies   ║
  ║  world optimal   - find most synergistic whale groupings     ║
  ║  world spi       - run parallel SPI demonstration            ║
  ║  world verify    - first-contact color fingerprint protocol  ║
  ║  triads [type]   - list all triads, filter by XOR/MAJ/PARITY ║
  ║  fingerprint     - compute color fingerprint for verification║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  WHALE CURRICULUM (Omniglot-style Hierarchical Refinement)   ║
  ║  curriculum      - run full curriculum for whale student     ║
  ║  teach <seed>    - α: teach whale with seed example          ║
  ║  examine         - γ: examine whale understanding            ║
  ║  rwalk <target>  - random walk exploration for seed          ║
  ║  prove           - whale proves SPI understanding            ║
  ║  generate <seed> - whale generates colors from seed          ║
  ║  omniglot        - multi-modal cross-domain learning         ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  NAVIGATION                                                   ║
  ║  seed <hex> - set seed    gay - reset to GAY_SEED             ║
  ║  walk [n]   - random walk seeking higher coverage             ║
  ╚═══════════════════════════════════════════════════════════════╝
""")
end

# ═══════════════════════════════════════════════════════════════════════════
# SPC Evaluator
# ═══════════════════════════════════════════════════════════════════════════

function spc_eval(input::String)
    input = strip(input)
    isempty(input) && return nothing
    
    parts = split(input)
    cmd = lowercase(parts[1])
    args = parts[2:end]
    
    if haskey(SPC_CMDS, cmd)
        try
            return SPC_CMDS[cmd](args...)
        catch e
            println("  Error: $e")
        end
    else
        # Try as Julia expression
        try
            expr = Meta.parse(input)
            return Core.eval(Main, expr)
        catch
            println("  Unknown: $cmd (try ? for help)")
        end
    end
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# REPL Initialization - SPC launches with SPACE
# ═══════════════════════════════════════════════════════════════════════════

function init_spc_repl(; sticky::Bool = true)
    # Initialize world
    init_world(GAY_SEED)
    
    # Prompt shows current pitch class set
    function spc_prompt()
        w = SPC_WORLD[]
        cov = length(unique(w.notes))
        rainbow("spc[$cov]▸ ")
    end
    
    ReplMaker.initrepl(
        spc_eval,
        repl = Base.active_repl,
        prompt_text = spc_prompt,
        prompt_color = :nothing,
        start_key = ' ',  # SPACE bar to enter SPC mode
        sticky_mode = sticky,
        mode_name = "SPC"
    )
    
    println()
    println(rainbow("  ╔═══════════════════════════════════════╗"))
    println(rainbow("  ║   SPC REPL: Symbolic·Possible·Comp    ║"))
    println(rainbow("  ╚═══════════════════════════════════════╝"))
    println("  Press SPACE to enter SPC mode. Type ? for help.")
    println()
    
    # Show initial chain
    SPC_CMDS["chain"]()
end
