# Coloring Inconsistency with Consistent Colors
# ==============================================
# Use Gay.jl's deterministic SPI colors to visualize
# the chaos of non-deterministic color generation.
#
# The meta-joke: we use reliable colors to show unreliable colors.

using Gay
using Gay: hash_color, splitmix64, GAY_SEED, xor_fingerprint
using Random

# ═══════════════════════════════════════════════════════════════════════════
# ANSI Color Helpers
# ═══════════════════════════════════════════════════════════════════════════

function ansi_rgb(r, g, b)
    ri = round(Int, clamp(r, 0, 1) * 255)
    gi = round(Int, clamp(g, 0, 1) * 255)
    bi = round(Int, clamp(b, 0, 1) * 255)
    "\e[38;2;$(ri);$(gi);$(bi)m"
end

const RESET = "\e[0m"
const DIM = "\e[2m"
const BOLD = "\e[1m"

block(r, g, b; width=2) = "$(ansi_rgb(r, g, b))$("█" ^ width)$(RESET)"
block(c::Tuple; width=2) = block(c[1], c[2], c[3]; width=width)

# ═══════════════════════════════════════════════════════════════════════════
# Generate colors: Inconsistent (thread-local RNG) vs Consistent (hash-based)
# ═══════════════════════════════════════════════════════════════════════════

"""
Generate colors using thread-local RNG (INCONSISTENT - different each run).
"""
function inconsistent_colors(n::Int)
    colors = Vector{NTuple{3, Float32}}(undef, n)
    Threads.@threads for i in 1:n
        colors[i] = (rand(Float32), rand(Float32), rand(Float32))
    end
    colors
end

"""
Generate colors using hash-based RNG (CONSISTENT - same every time).
"""
function consistent_colors(n::Int, seed::UInt64=GAY_SEED)
    [hash_color(seed, UInt64(i)) for i in 1:n]
end

"""
Compute fingerprint of color array.
"""
function fingerprint(colors::Vector{<:Tuple})
    fp = UInt32(0)
    for (r, g, b) in colors
        fp = xor(fp, reinterpret(UInt32, Float32(r)))
        fp = xor(fp, reinterpret(UInt32, Float32(g)))
        fp = xor(fp, reinterpret(UInt32, Float32(b)))
    end
    fp
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization: Show Inconsistency with Consistent Labeling
# ═══════════════════════════════════════════════════════════════════════════

"""
Assign a consistent label color to a fingerprint.
The label color is DETERMINISTIC - same fingerprint always gets same color.
"""
function label_color_for_fingerprint(fp::UInt32)
    hash_color(GAY_SEED, UInt64(fp))
end

"""
Demo: Run multiple trials and show how inconsistent colors vary,
while using consistent colors to label each unique outcome.
"""
function demo_inconsistency(; n_colors::Int=100, n_trials::Int=7)
    println()
    println("╔══════════════════════════════════════════════════════════════════════╗")
    println("║  COLORING INCONSISTENCY WITH CONSISTENT COLORS                      ║")
    println("║  Using Gay.jl SPI colors to visualize non-deterministic chaos       ║")
    println("╚══════════════════════════════════════════════════════════════════════╝")
    println()
    
    # First: Show what CONSISTENT looks like
    println("$(BOLD)1. CONSISTENT COLORS (hash-based SPI)$(RESET)")
    println("   Same seed → Same colors every time")
    println()
    
    consistent_fps = UInt32[]
    for trial in 1:n_trials
        colors = consistent_colors(n_colors)
        fp = fingerprint(colors)
        push!(consistent_fps, fp)
        
        # Show first 20 colors
        print("   Trial $trial: ")
        for i in 1:min(20, n_colors)
            print(block(colors[i]; width=1))
        end
        
        # Label with fingerprint color (meta: using consistent color to label consistent result)
        label = label_color_for_fingerprint(fp)
        println(" $(block(label)) 0x$(string(fp, base=16, pad=8))")
    end
    
    unique_consistent = length(unique(consistent_fps))
    println()
    println("   $(BOLD)Unique fingerprints: $unique_consistent$(RESET) (should be 1)")
    println()
    
    # Second: Show what INCONSISTENT looks like
    println("$(BOLD)2. INCONSISTENT COLORS (thread-local RNG)$(RESET)")
    println("   Same code → Different colors every time!")
    println()
    
    inconsistent_fps = UInt32[]
    fp_to_label = Dict{UInt32, NTuple{3, Float32}}()
    
    for trial in 1:n_trials
        colors = inconsistent_colors(n_colors)
        fp = fingerprint(colors)
        push!(inconsistent_fps, fp)
        
        # Assign consistent label color to this unique fingerprint
        if !haskey(fp_to_label, fp)
            fp_to_label[fp] = label_color_for_fingerprint(fp)
        end
        label = fp_to_label[fp]
        
        # Show first 20 colors (these will be DIFFERENT each trial)
        print("   Trial $trial: ")
        for i in 1:min(20, n_colors)
            print(block(colors[i]; width=1))
        end
        
        # But the LABEL color is CONSISTENT for each unique fingerprint
        println(" $(block(label)) 0x$(string(fp, base=16, pad=8))")
    end
    
    unique_inconsistent = length(unique(inconsistent_fps))
    println()
    println("   $(BOLD)Unique fingerprints: $unique_inconsistent$(RESET) (chaos!)")
    println()
    
    # Third: The meta-visualization
    println("$(BOLD)3. THE META: Consistent Labels for Inconsistent Results$(RESET)")
    println("   Each unique bad outcome gets a CONSISTENT color label")
    println()
    
    println("   Unique outcomes detected:")
    for (i, (fp, label)) in enumerate(sort(collect(fp_to_label), by=first))
        count = sum(inconsistent_fps .== fp)
        println("     $(block(label; width=4)) 0x$(string(fp, base=16, pad=8)) (seen $count time$(count > 1 ? "s" : ""))")
    end
    println()
    
    # Show the contrast
    println("─" ^ 72)
    println()
    println("  $(BOLD)THE LESSON:$(RESET)")
    println()
    print("  ")
    for i in 1:20
        c = hash_color(GAY_SEED, UInt64(i))
        print(block(c; width=1))
    end
    println(" ← Always the same (SPI)")
    println()
    print("  ")
    for i in 1:20
        print(block((rand(Float32), rand(Float32), rand(Float32)); width=1))
    end
    println(" ← Different every time (chaos)")
    println()
    println("  Gay.jl guarantees: same seed → same colors")
    println("  Across threads, GPUs, runs, and time.")
    println()
    println("═" ^ 72)
    
    (consistent = unique_consistent, inconsistent = unique_inconsistent)
end

# ═══════════════════════════════════════════════════════════════════════════
# Advanced: Visualize Race Conditions with Colored Timeline
# ═══════════════════════════════════════════════════════════════════════════

"""
Show a race condition timeline where each thread's writes are colored
consistently by thread ID, but the final result is chaotic.
"""
function demo_race_timeline(; n_colors::Int=50, n_trials::Int=5)
    println()
    println("╔══════════════════════════════════════════════════════════════════════╗")
    println("║  RACE CONDITION TIMELINE                                            ║")
    println("║  Thread colors are consistent, but write order is not              ║")
    println("╚══════════════════════════════════════════════════════════════════════╝")
    println()
    
    n_threads = Threads.nthreads()
    
    # Assign consistent color to each thread
    thread_colors = [hash_color(GAY_SEED, UInt64(t * 1000)) for t in 1:n_threads]
    
    println("  Thread legend:")
    for t in 1:n_threads
        println("    Thread $t: $(block(thread_colors[t]; width=4))")
    end
    println()
    
    for trial in 1:n_trials
        # Shared array - multiple threads write to it
        result = zeros(Float32, n_colors, 3)
        writer = zeros(Int, n_colors)  # Track which thread wrote each position
        
        Threads.@threads for i in 1:n_colors
            tid = Threads.threadid()
            result[i, 1] = rand(Float32)
            result[i, 2] = rand(Float32)
            result[i, 3] = rand(Float32)
            writer[i] = tid
        end
        
        # Show who wrote where (consistent thread colors)
        print("  Trial $trial writes: ")
        for i in 1:min(40, n_colors)
            tc = thread_colors[writer[i]]
            print(block(tc; width=1))
        end
        println()
        
        # Show what values ended up there (chaotic)
        print("  Trial $trial values: ")
        for i in 1:min(40, n_colors)
            print(block((result[i,1], result[i,2], result[i,3]); width=1))
        end
        fp = xor_fingerprint(result)
        label = label_color_for_fingerprint(fp)
        println(" $(block(label)) 0x$(string(fp, base=16, pad=8))")
    end
    
    println()
    println("  $(DIM)Top row: WHO wrote (consistent by thread ID)$(RESET)")
    println("  $(DIM)Bottom row: WHAT was written (chaotic values)$(RESET)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Run the demos
# ═══════════════════════════════════════════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    demo_inconsistency()
    demo_race_timeline()
end

export demo_inconsistency, demo_race_timeline
