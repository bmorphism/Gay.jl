# Computation Lifetimes Demo: Mortal and Immortal Color Assignment
#
# Every computation has a lifetime:
# - MORTAL: finite steps, guaranteed termination, complete fingerprint
# - IMMORTAL: infinite epochs, productive forever, rolling fingerprint
#
# SPI (Strong Parallelism Invariance) applies to both:
# - Same seed → same colors → same fingerprint
# - Parallel execution order doesn't matter

using Pkg
Pkg.activate(dirname(dirname(@__FILE__)))

using Gay
include(joinpath(dirname(dirname(@__FILE__)), "src", "lifetimes.jl"))
using .Lifetimes
using Colors

# ═══════════════════════════════════════════════════════════════════════════════
# Mortal Computations: Finite, Terminating
# ═══════════════════════════════════════════════════════════════════════════════

function demo_mortal()
    println()
    println("═" ^ 70)
    println("  MORTAL COMPUTATIONS (finite, terminating)")
    println("═" ^ 70)
    println()
    
    # Create a mortal computation
    mc = MortalComputation(42; id=1, max_steps=50)
    println("  Created: $mc")
    println()
    
    # Take steps, each gets a color
    print("  Taking 20 steps: ")
    for i in 1:20
        c = mortal_step!(mc)
        r = round(Int, c.r * 255)
        g = round(Int, c.g * 255)
        b = round(Int, c.b * 255)
        print("\e[48;2;$(r);$(g);$(b)m \e[0m")
    end
    println()
    
    # Fingerprint captures entire history
    fp = mortal_fingerprint(mc)
    println("  Fingerprint (20 steps): 0x$(string(fp, base=16, pad=8))")
    println()
    
    # O(1) random access - no state needed
    println("  Pure O(1) random access:")
    for step in [1, 10, 100, 1000, 10000]
        c = mortal_color(42, 1, step)
        r = round(Int, c.r * 255)
        g = round(Int, c.g * 255)
        b = round(Int, c.b * 255)
        print("    step $(lpad(step, 5)): ")
        println("\e[48;2;$(r);$(g);$(b)m    \e[0m")
    end
    println()
    
    # Terminate and get final fingerprint
    final_fp = mortal_terminate!(mc)
    println("  Terminated: $mc")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Immortal Computations: Infinite, Productive
# ═══════════════════════════════════════════════════════════════════════════════

function demo_immortal()
    println()
    println("═" ^ 70)
    println("  IMMORTAL COMPUTATIONS (infinite, productive)")
    println("═" ^ 70)
    println()
    
    # Create an immortal computation with rolling window
    ic = ImmortalComputation(42; id=1, window_size=50)
    println("  Created: $ic")
    println()
    
    # Run epochs (can run forever)
    print("  First 25 epochs: ")
    for i in 1:25
        c = immortal_epoch!(ic)
        r = round(Int, c.r * 255)
        g = round(Int, c.g * 255)
        b = round(Int, c.b * 255)
        print("\e[48;2;$(r);$(g);$(b)m \e[0m")
    end
    println()
    
    fp1 = immortal_fingerprint(ic)
    println("  Rolling fingerprint (25 epochs): 0x$(string(fp1, base=16, pad=8))")
    
    # Run more epochs - fingerprint evolves
    print("  Next 25 epochs:  ")
    for i in 1:25
        c = immortal_epoch!(ic)
        r = round(Int, c.r * 255)
        g = round(Int, c.g * 255)
        b = round(Int, c.b * 255)
        print("\e[48;2;$(r);$(g);$(b)m \e[0m")
    end
    println()
    
    fp2 = immortal_fingerprint(ic)
    println("  Rolling fingerprint (50 epochs): 0x$(string(fp2, base=16, pad=8))")
    println()
    
    # The fingerprint changes as the window slides
    println("  Note: Rolling fingerprint changes as window slides")
    println("  But for same (seed, epoch_range) → same fingerprint (SPI)")
    println()
    
    # Pure O(1) access
    println("  Pure O(1) random access:")
    for epoch in [1, 100, 1000, 10000]
        c = immortal_color(42, 1, epoch)
        r = round(Int, c.r * 255)
        g = round(Int, c.g * 255)
        b = round(Int, c.b * 255)
        print("    epoch $(lpad(epoch, 5)): ")
        println("\e[48;2;$(r);$(g);$(b)m    \e[0m")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Parallel SPI Verification
# ═══════════════════════════════════════════════════════════════════════════════

function demo_parallel_spi()
    println()
    println("═" ^ 70)
    println("  PARALLEL SPI VERIFICATION")
    println("═" ^ 70)
    println()
    
    # Many mortal computations in parallel
    n_comps = 8
    n_steps = 100
    
    println("  Running $n_comps mortal computations, $n_steps steps each...")
    println("  (Parallel execution on $(Threads.nthreads()) threads)")
    println()
    
    # Run twice - should get identical results
    fps1 = parallel_mortal_colors(42, n_comps, n_steps)
    fps2 = parallel_mortal_colors(42, n_comps, n_steps)
    
    println("  Results:")
    all_match = true
    for i in 1:n_comps
        match = fps1[i] == fps2[i]
        all_match &= match
        status = match ? "◆" : "◇"
        println("    Computation $i: 0x$(string(fps1[i], base=16, pad=8)) $status")
    end
    println()
    
    if all_match
        println("  ◆ ALL FINGERPRINTS MATCH - SPI VERIFIED")
    else
        println("  ◇ MISMATCH DETECTED - SPI VIOLATED")
    end
    println()
    
    # Combined fingerprint of all computations
    combined = reduce(xor, fps1)
    println("  Combined fingerprint (XOR all): 0x$(string(combined, base=16, pad=8))")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Metatheory: Ascend and Harvest Functors
# ═══════════════════════════════════════════════════════════════════════════════

function demo_functors()
    println()
    println("═" ^ 70)
    println("  METATHEORY: Ascend and Harvest Functors")
    println("═" ^ 70)
    println()
    
    println("  Categorical structure:")
    println()
    println("    MORTAL (Commutative Monoidal Category)")
    println("    ├─ Objects: finite color sequences")
    println("    ├─ Morphisms: computation steps")
    println("    ├─ ⊗ parallel: XOR-combine colors")
    println("    └─ I identity: empty computation")
    println()
    println("    IMMORTAL (Traced Monoidal Category)")
    println("    ├─ Objects: infinite color streams")
    println("    ├─ Morphisms: epoch transitions")
    println("    └─ trace: feedback loops (fixpoints)")
    println()
    println("  Functors:")
    println("    ascend  : Mortal → Immortal (iterate forever)")
    println("    harvest : Immortal → Mortal (take n epochs)")
    println()
    
    # Create mortal computation
    mc = MortalComputation(1337; id=1)
    print("  Mortal (10 steps):   ")
    for _ in 1:10
        c = mortal_step!(mc)
        r = round(Int, c.r * 255)
        g = round(Int, c.g * 255)
        b = round(Int, c.b * 255)
        print("\e[48;2;$(r);$(g);$(b)m \e[0m")
    end
    mc.alive = false
    fp_mortal = mortal_fingerprint(mc)
    println(" fp=0x$(string(fp_mortal, base=16, pad=8))")
    
    # Ascend to immortal
    ic = ascend(mc)
    println("  Ascended to immortal: $ic")
    
    # Run more epochs on the immortal
    print("  Immortal (+10 epochs): ")
    for _ in 1:10
        c = immortal_epoch!(ic)
        r = round(Int, c.r * 255)
        g = round(Int, c.g * 255)
        b = round(Int, c.b * 255)
        print("\e[48;2;$(r);$(g);$(b)m \e[0m")
    end
    fp_immortal = immortal_fingerprint(ic)
    println(" fp=0x$(string(fp_immortal, base=16, pad=8))")
    
    # Harvest back to mortal
    mc2 = harvest(ic, 15)
    print("  Harvested (15 steps):  ")
    # Show the colors
    for i in 1:min(15, mc2.step)
        r = round(Int, mc2.colors[i, 1] * 255)
        g = round(Int, mc2.colors[i, 2] * 255)
        b = round(Int, mc2.colors[i, 3] * 255)
        print("\e[48;2;$(r);$(g);$(b)m \e[0m")
    end
    fp_harvested = mortal_fingerprint(mc2)
    println(" fp=0x$(string(fp_harvested, base=16, pad=8))")
    println()
    
    println("  The ascend-harvest roundtrip transforms the fingerprint")
    println("  because immortal adds new epochs to the sequence.")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println()
    println("╔" * "═" ^ 68 * "╗")
    println("║" * " " ^ 16 * "COMPUTATION LIFETIMES DEMO" * " " ^ 26 * "║")
    println("║" * " " ^ 68 * "║")
    println("║  Mortal: finite, terminating, complete fingerprint                 ║")
    println("║  Immortal: infinite, productive, rolling fingerprint               ║")
    println("╚" * "═" ^ 68 * "╝")
    
    demo_mortal()
    demo_immortal()
    demo_parallel_spi()
    demo_functors()
    
    println()
    println("═" ^ 70)
    println("  Every computation gets a color. Every lifetime gets a fingerprint.")
    println("  SPI guarantees reproducibility across parallel worlds. ◈")
    println("═" ^ 70)
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
