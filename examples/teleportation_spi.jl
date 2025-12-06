# # Teleportation Between Worlds: A Strong Parallelism Invariance Story
#
# What happens when you travel to another world, compute your colors there,
# and come back? Are you still *you*?
#
# This example demonstrates SPI (Strong Parallelism Invariance) through the
# metaphor of teleportation between computational worlds:
#
# - **World A**: CPU sequential (the home world)
# - **World B**: CPU parallel (the threaded realm)
# - **World C**: Metal GPU (the silicon dimension)
#
# The fundamental question: if you derive your identity (colors) in another
# world and return, do you come back as the same person?
#
# Spoiler: Yes. That's the SPI guarantee.

using Gay
using Gay: ka_colors, ka_colors!, xor_fingerprint, hash_color, set_backend!, get_backend
using KernelAbstractions
using KernelAbstractions: CPU

# Try to load Metal if available
const HAS_METAL = try
    @eval using Metal
    true
catch
    false
end

# ═══════════════════════════════════════════════════════════════════════════════
# The Traveler: A color identity that will visit different worlds
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Traveler

A being defined by their color sequence. Their identity is the XOR fingerprint
of all their colors - a single hash that captures their entire chromatic essence.
"""
struct Traveler
    name::String
    seed::UInt64
    n_colors::Int
    fingerprint::UInt32
    origin_world::String
end

function Base.show(io::IO, t::Traveler)
    print(io, "Traveler(\"$(t.name)\", seed=0x$(string(t.seed, base=16)), ")
    print(io, "identity=0x$(string(t.fingerprint, base=16, pad=8)), ")
    print(io, "from=$(t.origin_world))")
end

"""
    create_traveler(name, seed, n; world="unknown")

Birth a traveler in a specific world, deriving their color identity there.
"""
function create_traveler(name::String, seed::Integer, n::Int; world::String="unknown")
    colors = zeros(Float32, n, 3)
    for i in 1:n
        r, g, b = hash_color(UInt64(seed), UInt64(i))
        colors[i, 1] = r
        colors[i, 2] = g
        colors[i, 3] = b
    end
    fp = xor_fingerprint(colors)
    Traveler(name, UInt64(seed), n, fp, world)
end

# ═══════════════════════════════════════════════════════════════════════════════
# World Definitions
# ═══════════════════════════════════════════════════════════════════════════════

"""
    WorldA - CPU Sequential

The home world. Colors computed one by one, in order.
The most "natural" computation - no parallelism, no magic.
"""
function visit_world_a(seed::Integer, n::Int)
    colors = zeros(Float32, n, 3)
    for i in 1:n
        r, g, b = hash_color(UInt64(seed), UInt64(i))
        colors[i, 1] = r
        colors[i, 2] = g
        colors[i, 3] = b
    end
    xor_fingerprint(colors)
end

"""
    WorldB - CPU Parallel (KernelAbstractions on CPU)

The threaded realm. Colors computed by SPMD kernel on CPU.
Multiple threads, but same results?
"""
function visit_world_b(seed::Integer, n::Int)
    colors = zeros(Float32, n, 3)
    ka_colors!(colors, seed; backend=CPU())
    xor_fingerprint(colors)
end

"""
    WorldC - Metal GPU

The silicon dimension. Colors computed on Apple GPU.
Thousands of threads, utterly different hardware. Same soul?
"""
function visit_world_c(seed::Integer, n::Int)
    if !HAS_METAL
        return nothing  # World C is inaccessible
    end
    
    backend = Metal.MetalBackend()
    gpu_colors = KernelAbstractions.zeros(backend, Float32, n, 3)
    
    kernel! = Gay._ka_colors_kernel!(backend, 256)
    kernel!(gpu_colors, UInt64(seed), ndrange=n)
    KernelAbstractions.synchronize(backend)
    
    cpu_colors = Array(gpu_colors)
    xor_fingerprint(cpu_colors)
end

# ═══════════════════════════════════════════════════════════════════════════════
# The Teleportation Journey
# ═══════════════════════════════════════════════════════════════════════════════

"""
    teleportation_journey(; seed=42069, n=100_000)

A traveler visits all accessible worlds and checks if they remain themselves.

The journey:
1. Born in World A (CPU sequential) - establish identity
2. Teleport to World B (CPU parallel) - derive colors there
3. Return to World A - verify identity preserved
4. Teleport to World C (Metal GPU) - if accessible
5. Return to World A - verify identity still preserved
6. Visit worlds in different order - identity must be path-independent

Returns true if identity is preserved across all teleportations.
"""
function teleportation_journey(; seed::Integer=42069, n::Int=100_000, verbose::Bool=true)
    if verbose
        println("═" ^ 70)
        println("  TELEPORTATION BETWEEN WORLDS: An SPI Story")
        println("═" ^ 70)
        println()
        println("  \"If you travel to another world, compute your colors there,")
        println("   and return home... are you still you?\"")
        println()
    end
    
    # Step 1: Birth in World A
    if verbose
        println("─" ^ 70)
        println("  STEP 1: Birth in World A (CPU Sequential)")
        println("─" ^ 70)
    end
    
    traveler = create_traveler("Alice", seed, n; world="World A")
    original_identity = traveler.fingerprint
    
    if verbose
        println("  Born: $traveler")
        println("  Identity established: 0x$(string(original_identity, base=16, pad=8))")
        println()
    end
    
    all_same = true
    journey_log = [(world="A (birth)", fingerprint=original_identity, match=true)]
    
    # Step 2: Teleport to World B
    if verbose
        println("─" ^ 70)
        println("  STEP 2: Teleport to World B (CPU Parallel)")
        println("─" ^ 70)
    end
    
    fp_b = visit_world_b(seed, n)
    match_b = fp_b == original_identity
    all_same &= match_b
    push!(journey_log, (world="B (parallel)", fingerprint=fp_b, match=match_b))
    
    if verbose
        status = match_b ? "✓ SAME" : "✗ DIFFERENT"
        println("  Derived identity: 0x$(string(fp_b, base=16, pad=8)) [$status]")
        println()
    end
    
    # Step 3: Return to World A
    if verbose
        println("─" ^ 70)
        println("  STEP 3: Return to World A (verify)")
        println("─" ^ 70)
    end
    
    fp_a2 = visit_world_a(seed, n)
    match_a2 = fp_a2 == original_identity
    all_same &= match_a2
    push!(journey_log, (world="A (return)", fingerprint=fp_a2, match=match_a2))
    
    if verbose
        status = match_a2 ? "✓ SAME" : "✗ DIFFERENT"
        println("  Home identity: 0x$(string(fp_a2, base=16, pad=8)) [$status]")
        println()
    end
    
    # Step 4: Teleport to World C (if accessible)
    if HAS_METAL
        if verbose
            println("─" ^ 70)
            println("  STEP 4: Teleport to World C (Metal GPU)")
            println("─" ^ 70)
        end
        
        fp_c = visit_world_c(seed, n)
        if fp_c !== nothing
            match_c = fp_c == original_identity
            all_same &= match_c
            push!(journey_log, (world="C (Metal)", fingerprint=fp_c, match=match_c))
            
            if verbose
                status = match_c ? "✓ SAME" : "✗ DIFFERENT"
                println("  Silicon identity: 0x$(string(fp_c, base=16, pad=8)) [$status]")
                println()
            end
            
            # Step 5: Return home after GPU
            fp_a3 = visit_world_a(seed, n)
            match_a3 = fp_a3 == original_identity
            all_same &= match_a3
            push!(journey_log, (world="A (post-GPU)", fingerprint=fp_a3, match=match_a3))
            
            if verbose
                println("─" ^ 70)
                println("  STEP 5: Return to World A (post-GPU)")
                println("─" ^ 70)
                status = match_a3 ? "✓ SAME" : "✗ DIFFERENT"
                println("  Home identity: 0x$(string(fp_a3, base=16, pad=8)) [$status]")
                println()
            end
        end
    else
        if verbose
            println("─" ^ 70)
            println("  STEP 4: World C (Metal GPU) - INACCESSIBLE")
            println("─" ^ 70)
            println("  (Metal.jl not loaded - GPU world cannot be reached)")
            println()
        end
    end
    
    # Step 6: Random order traversal
    if verbose
        println("─" ^ 70)
        println("  STEP 6: Path Independence Test")
        println("─" ^ 70)
        println("  Visiting worlds in different orders...")
    end
    
    orders = [
        ["B", "A", "B"],
        ["A", "B", "A", "B", "A"],
        ["B", "B", "B"],
    ]
    
    if HAS_METAL
        push!(orders, ["C", "A", "C"])
        push!(orders, ["B", "C", "A", "C", "B"])
    end
    
    for order in orders
        fps = UInt32[]
        for w in order
            if w == "A"
                push!(fps, visit_world_a(seed, n))
            elseif w == "B"
                push!(fps, visit_world_b(seed, n))
            elseif w == "C" && HAS_METAL
                fp = visit_world_c(seed, n)
                fp !== nothing && push!(fps, fp)
            end
        end
        
        path_same = all(fp -> fp == original_identity, fps)
        all_same &= path_same
        
        if verbose
            status = path_same ? "✓" : "✗"
            println("    $(join(order, " → ")): $status")
        end
    end
    
    if verbose
        println()
    end
    
    # Final verdict
    if verbose
        println("═" ^ 70)
        if all_same
            println("  VERDICT: Identity PRESERVED across all worlds ✓")
            println()
            println("  The traveler visited $(length(journey_log)) worlds.")
            println("  Every world computed the same identity: 0x$(string(original_identity, base=16, pad=8))")
            println()
            println("  This is Strong Parallelism Invariance:")
            println("  Same seed → Same colors → Same identity")
            println("  Regardless of WHERE or HOW you compute.")
        else
            println("  VERDICT: Identity CORRUPTED ✗")
            println()
            println("  The traveler lost themselves between worlds.")
            println("  This should never happen with correct SPI implementation.")
        end
        println("═" ^ 70)
    end
    
    return all_same
end

# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Traveler Test: Different Seeds, Same Guarantee
# ═══════════════════════════════════════════════════════════════════════════════

"""
    multi_traveler_test(; n_travelers=10, n_colors=10_000)

Multiple travelers with different seeds all journey between worlds.
Each must preserve their unique identity.
"""
function multi_traveler_test(; n_travelers::Int=10, n_colors::Int=10_000, verbose::Bool=true)
    if verbose
        println()
        println("═" ^ 70)
        println("  MULTI-TRAVELER TELEPORTATION TEST")
        println("═" ^ 70)
        println("  $n_travelers travelers, each with unique seed")
        println("  Each visits all worlds and must return as themselves")
        println()
    end
    
    seeds = [rand(UInt64) for _ in 1:n_travelers]
    all_pass = true
    
    for (i, seed) in enumerate(seeds)
        pass = teleportation_journey(seed=seed, n=n_colors, verbose=false)
        all_pass &= pass
        
        if verbose
            status = pass ? "✓" : "✗"
            fp = visit_world_a(seed, n_colors)
            println("  Traveler $i (0x$(string(seed, base=16, pad=16)[1:8])...): ",
                    "0x$(string(fp, base=16, pad=8)) $status")
        end
    end
    
    if verbose
        println()
        if all_pass
            println("  ALL $n_travelers TRAVELERS PRESERVED ✓")
        else
            println("  SOME TRAVELERS LOST ✗")
        end
        println("═" ^ 70)
    end
    
    return all_pass
end

# ═══════════════════════════════════════════════════════════════════════════════
# Step-by-Step Derivation: Watch the Colors Form
# ═══════════════════════════════════════════════════════════════════════════════

"""
    step_by_step_derivation(; seed=42, n=10)

Watch colors being derived step by step in each world.
Shows that each index produces the same color in every world.
"""
function step_by_step_derivation(; seed::Integer=42, n::Int=10)
    println()
    println("═" ^ 70)
    println("  STEP-BY-STEP DERIVATION")
    println("  Watching colors form in each world")
    println("═" ^ 70)
    println()
    println("  seed = $seed")
    println()
    
    # Headers
    print("  Index │ ")
    print(rpad("World A (CPU seq)", 28))
    print(" │ ")
    print(rpad("World B (CPU par)", 28))
    if HAS_METAL
        print(" │ ")
        print(rpad("World C (Metal)", 28))
    end
    println()
    println("  " * "─" ^ 6 * "┼" * "─" ^ 30 * "┼" * "─" ^ 30 * (HAS_METAL ? "┼" * "─" ^ 30 : ""))
    
    # Generate in each world
    colors_a = zeros(Float32, n, 3)
    for i in 1:n
        r, g, b = hash_color(UInt64(seed), UInt64(i))
        colors_a[i, 1] = r
        colors_a[i, 2] = g
        colors_a[i, 3] = b
    end
    
    colors_b = ka_colors(n, seed)
    
    colors_c = if HAS_METAL
        backend = Metal.MetalBackend()
        gpu = KernelAbstractions.zeros(backend, Float32, n, 3)
        kernel! = Gay._ka_colors_kernel!(backend, 256)
        kernel!(gpu, UInt64(seed), ndrange=n)
        KernelAbstractions.synchronize(backend)
        Array(gpu)
    else
        nothing
    end
    
    for i in 1:n
        ra, ga, ba = colors_a[i, 1], colors_a[i, 2], colors_a[i, 3]
        rb, gb, bb = colors_b[i, 1], colors_b[i, 2], colors_b[i, 3]
        
        # Format as hex RGB
        hex_a = "#" * string(round(Int, ra*255), base=16, pad=2) *
                      string(round(Int, ga*255), base=16, pad=2) *
                      string(round(Int, ba*255), base=16, pad=2) |> uppercase
        hex_b = "#" * string(round(Int, rb*255), base=16, pad=2) *
                      string(round(Int, gb*255), base=16, pad=2) *
                      string(round(Int, bb*255), base=16, pad=2) |> uppercase
        
        # ANSI color block
        block_a = "\e[38;2;$(round(Int, ra*255));$(round(Int, ga*255));$(round(Int, ba*255))m████\e[0m"
        block_b = "\e[38;2;$(round(Int, rb*255));$(round(Int, gb*255));$(round(Int, bb*255))m████\e[0m"
        
        match_ab = (ra, ga, ba) == (rb, gb, bb)
        
        print("  $(lpad(i, 5)) │ ")
        print("$block_a $hex_a ")
        print(match_ab ? "✓" : "✗")
        print(" │ ")
        print("$block_b $hex_b ")
        print(match_ab ? "✓" : "✗")
        
        if colors_c !== nothing
            rc, gc, bc = colors_c[i, 1], colors_c[i, 2], colors_c[i, 3]
            hex_c = "#" * string(round(Int, rc*255), base=16, pad=2) *
                          string(round(Int, gc*255), base=16, pad=2) *
                          string(round(Int, bc*255), base=16, pad=2) |> uppercase
            block_c = "\e[38;2;$(round(Int, rc*255));$(round(Int, gc*255));$(round(Int, bc*255))m████\e[0m"
            match_ac = (ra, ga, ba) == (rc, gc, bc)
            print(" │ ")
            print("$block_c $hex_c ")
            print(match_ac ? "✓" : "✗")
        end
        
        println()
    end
    
    println()
    println("  Every row shows the SAME color in EVERY world.")
    println("  This is the magic of deterministic hash-based generation.")
    println("═" ^ 70)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main Demo
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println()
    println("╔" * "═" ^ 68 * "╗")
    println("║" * " " ^ 20 * "TELEPORTATION SPI DEMO" * " " ^ 26 * "║")
    println("║" * " " ^ 68 * "║")
    println("║  \"Are you still you after traveling to another computational world?\" ║")
    println("╚" * "═" ^ 68 * "╝")
    println()
    
    if HAS_METAL
        println("  Metal.jl detected - World C (GPU) is accessible! 🚀")
    else
        println("  Metal.jl not loaded - World C (GPU) is inaccessible")
        println("  Run `using Metal` first to enable GPU teleportation")
    end
    println()
    
    # Demo 1: Step-by-step derivation
    step_by_step_derivation(seed=42, n=10)
    
    # Demo 2: Single traveler journey
    println()
    teleportation_journey(seed=42069, n=100_000)
    
    # Demo 3: Multi-traveler test
    println()
    multi_traveler_test(n_travelers=5, n_colors=10_000)
    
    println()
    println("  To run with Metal GPU:")
    println("    julia> using Metal")
    println("    julia> include(\"examples/teleportation_spi.jl\")")
    println("    julia> main()")
    println()
end

# Export for REPL use
export teleportation_journey, multi_traveler_test, step_by_step_derivation

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
