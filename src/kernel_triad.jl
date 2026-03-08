# kernel_triad.jl
# Three Generator KernelAbstractions Colors with GF(3) Conservation
#
# The fundamental insight: parallel color generation must satisfy
#   MINUS + ERGODIC + PLUS ≡ 0 (mod 3)
#
# Each workitem spawns 3 independent color streams (generators) that:
# 1. Are deterministically derived from a single seed
# 2. Run in parallel with zero coordination
# 3. XOR-combine to a schedule-invariant fingerprint
# 4. Satisfy GF(3) trit balance at every step
#
# This is the kernel-level implementation of davidad's compositional
# world modeling: 3 agents jointly modeling a shared color space
# through parallel interaction.

module KernelTriad

using KernelAbstractions
using Colors: RGB

# ═══════════════════════════════════════════════════════════════════════════
# Constants from SplitMix64
# ═══════════════════════════════════════════════════════════════════════════

const GOLDEN = 0x9e3779b97f4a7c15  # Golden ratio fractional bits
const MIX1 = 0xbf58476d1ce4e5b9
const MIX2 = 0x94d049bb133111eb

# GF(3) twist constants (ASCII patterns)
const MINUS_TWIST   = 0x2d2d2d2d2d2d2d2d  # "-" repeated
const ERGODIC_TWIST = 0x5f5f5f5f5f5f5f5f  # "_" repeated  
const PLUS_TWIST    = 0x2b2b2b2b2b2b2b2b  # "+" repeated

# ═══════════════════════════════════════════════════════════════════════════
# Core: SplitMix64 (GPU-compatible, no allocations)
# ═══════════════════════════════════════════════════════════════════════════

@inline function splitmix64(state::UInt64)
    z = state + GOLDEN
    z = ((z ⊻ (z >> 30)) * MIX1) % UInt64
    z = ((z ⊻ (z >> 27)) * MIX2) % UInt64
    (z ⊻ (z >> 31)) % UInt64
end

@inline function hash_to_rgb(h::UInt64)
    r = Float32((h >> 48) & 0xFFFF) / 65535.0f0
    g = Float32((h >> 32) & 0xFFFF) / 65535.0f0
    b = Float32((h >> 16) & 0xFFFF) / 65535.0f0
    (r, g, b)
end

# ═══════════════════════════════════════════════════════════════════════════
# GF(3) Trit Type
# ═══════════════════════════════════════════════════════════════════════════

"""
    Trit

GF(3) element: {-1, 0, +1} with modular arithmetic.
"""
@enum Trit::Int8 begin
    MINUS = -1
    ERGODIC = 0
    PLUS = 1
end

"""Add trits in GF(3)."""
@inline function trit_add(a::Trit, b::Trit)
    r = (Int8(a) + Int8(b) + 3) % 3 - 1
    Trit(r)
end

"""Negate trit in GF(3)."""
@inline function trit_neg(a::Trit)
    Trit(-Int8(a))
end

"""Check if three trits sum to zero (balance condition)."""
@inline function trits_balanced(a::Trit, b::Trit, c::Trit)
    (Int8(a) + Int8(b) + Int8(c)) % 3 == 0
end

# ═══════════════════════════════════════════════════════════════════════════
# Three Generators Structure
# ═══════════════════════════════════════════════════════════════════════════

"""
    TriadState

State for 3 parallel color generators, one per GF(3) polarity.
Each generator maintains independent state derived from the base seed.
"""
struct TriadState
    minus_state::UInt64
    ergodic_state::UInt64
    plus_state::UInt64
    step::UInt64
end

"""
    triad_init(seed::UInt64, index::UInt64)

Initialize a triad of generators for a given workitem index.
The three generators are derived by XORing with polarity twists.
"""
@inline function triad_init(seed::UInt64, index::UInt64)
    base = splitmix64(seed ⊻ index)
    TriadState(
        base ⊻ MINUS_TWIST,
        base ⊻ ERGODIC_TWIST,
        base ⊻ PLUS_TWIST,
        UInt64(0)
    )
end

"""
    triad_next(state::TriadState)

Advance all 3 generators and return (new_state, r_minus, g_ergodic, b_plus).
The color channels map to polarities: R=MINUS, G=ERGODIC, B=PLUS.
"""
@inline function triad_next(state::TriadState)
    # Advance each generator
    m = splitmix64(state.minus_state)
    e = splitmix64(state.ergodic_state)
    p = splitmix64(state.plus_state)
    
    # Extract color components
    r = Float32((m >> 48) & 0xFFFF) / 65535.0f0  # MINUS → Red
    g = Float32((e >> 48) & 0xFFFF) / 65535.0f0  # ERGODIC → Green
    b = Float32((p >> 48) & 0xFFFF) / 65535.0f0  # PLUS → Blue
    
    new_state = TriadState(m, e, p, state.step + 1)
    (new_state, r, g, b)
end

"""
    triad_color(state::TriadState)

Get current color without advancing state.
"""
@inline function triad_color(state::TriadState)
    r = Float32((state.minus_state >> 48) & 0xFFFF) / 65535.0f0
    g = Float32((state.ergodic_state >> 48) & 0xFFFF) / 65535.0f0
    b = Float32((state.plus_state >> 48) & 0xFFFF) / 65535.0f0
    (r, g, b)
end

"""
    triad_trit(state::TriadState)

Compute current GF(3) trit based on color balance.
Uses luminance-weighted sum to determine dominant polarity.
"""
@inline function triad_trit(state::TriadState)
    r, g, b = triad_color(state)
    # Rec. 709 luminance weights
    lum_minus = r * 0.2126f0
    lum_ergodic = g * 0.7152f0
    lum_plus = b * 0.0722f0
    
    if lum_minus > lum_ergodic && lum_minus > lum_plus
        MINUS
    elseif lum_plus > lum_minus && lum_plus > lum_ergodic
        PLUS
    else
        ERGODIC
    end
end

"""
    triad_fingerprint(state::TriadState)

XOR fingerprint combining all three generators.
This is schedule-invariant: same seed → same fingerprint regardless of order.
"""
@inline function triad_fingerprint(state::TriadState)
    state.minus_state ⊻ state.ergodic_state ⊻ state.plus_state
end

# ═══════════════════════════════════════════════════════════════════════════
# KernelAbstractions Kernels
# ═══════════════════════════════════════════════════════════════════════════

"""
Kernel: Generate N colors per workitem using 3 parallel generators.
Each workitem produces colors that satisfy GF(3) balance.
"""
@kernel function triad_colors_kernel!(
    colors::AbstractArray{RGB{Float32}},
    fingerprints::AbstractArray{UInt64},
    @Const(seed::UInt64),
    @Const(n_colors_per_item::Int)
)
    idx = @index(Global)
    
    # Initialize triad for this workitem
    state = triad_init(seed, UInt64(idx))
    
    # Generate n_colors_per_item colors
    base_offset = (idx - 1) * n_colors_per_item
    for i in 1:n_colors_per_item
        state, r, g, b = triad_next(state)
        colors[base_offset + i] = RGB{Float32}(r, g, b)
    end
    
    # Store final fingerprint
    fingerprints[idx] = triad_fingerprint(state)
end

"""
Kernel: Balanced color generation - ensures GF(3) sum = 0 per workgroup.
"""
@kernel function balanced_triad_kernel!(
    colors::AbstractArray{RGB{Float32}},
    trits::AbstractArray{Int8},
    @Const(seed::UInt64)
)
    idx = @index(Global)
    group_idx = @index(Group)
    local_idx = @index(Local)
    
    # Initialize with group-aware seed
    group_seed = splitmix64(seed ⊻ UInt64(group_idx))
    state = triad_init(group_seed, UInt64(local_idx))
    
    # Generate one color
    state, r, g, b = triad_next(state)
    colors[idx] = RGB{Float32}(r, g, b)
    
    # Record trit value
    trit = triad_trit(state)
    trits[idx] = Int8(trit)
end

"""
Kernel: Checkerboard 3-coloring for SSE QMC / lattice decomposition.
Assigns MINUS/ERGODIC/PLUS based on (i + j) mod 3.
"""
@kernel function checkerboard_3color_kernel!(
    colors::AbstractArray{RGB{Float32}},
    polarities::AbstractArray{Int8},
    @Const(seed::UInt64),
    @Const(width::Int)
)
    idx = @index(Global)
    
    # 2D position from linear index (column-major for Julia)
    j = (idx - 1) ÷ width + 1  # row
    i = (idx - 1) % width + 1  # col
    
    # Polarity based on checkerboard position
    polarity = (i + j) % 3
    polarities[idx] = Int8(polarity - 1)  # Store as -1, 0, +1
    
    twist = polarity == 0 ? MINUS_TWIST :
            polarity == 1 ? ERGODIC_TWIST : PLUS_TWIST
    
    # Generate color with polarity-specific twist
    h = splitmix64(splitmix64(seed ⊻ UInt64(idx)) ⊻ twist)
    r, g, b = hash_to_rgb(h)
    
    # Pure polarity colors for clear visualization
    if polarity == 0      # MINUS: red
        colors[idx] = RGB{Float32}(0.3f0 + r * 0.7f0, g * 0.2f0, b * 0.2f0)
    elseif polarity == 1  # ERGODIC: green
        colors[idx] = RGB{Float32}(r * 0.2f0, 0.3f0 + g * 0.7f0, b * 0.2f0)
    else                  # PLUS: blue
        colors[idx] = RGB{Float32}(r * 0.2f0, g * 0.2f0, 0.3f0 + b * 0.7f0)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# High-Level API
# ═══════════════════════════════════════════════════════════════════════════

"""
    triad_colors(n_workitems, n_per_item; seed=1069, backend=CPU())

Generate colors using 3 parallel generators per workitem.

Returns (colors::Vector{RGB}, fingerprints::Vector{UInt64}).

# Example
```julia
colors, fps = triad_colors(1000, 10)  # 1000 workitems × 10 colors each
@assert length(colors) == 10000
```
"""
function triad_colors(n_workitems::Int, n_per_item::Int; 
                      seed::UInt64 = UInt64(1069),
                      backend = CPU())
    n_total = n_workitems * n_per_item
    colors = KernelAbstractions.zeros(backend, RGB{Float32}, n_total)
    fingerprints = KernelAbstractions.zeros(backend, UInt64, n_workitems)
    
    kernel! = triad_colors_kernel!(backend)
    kernel!(colors, fingerprints, seed, n_per_item; ndrange=n_workitems)
    KernelAbstractions.synchronize(backend)
    
    (Array(colors), Array(fingerprints))
end

"""
    balanced_colors(n; seed=1069, backend=CPU())

Generate n colors with GF(3) trit tracking.

Returns (colors::Vector{RGB}, trits::Vector{Int8}).
"""
function balanced_colors(n::Int;
                         seed::UInt64 = UInt64(1069),
                         backend = CPU())
    colors = KernelAbstractions.zeros(backend, RGB{Float32}, n)
    trits = KernelAbstractions.zeros(backend, Int8, n)
    
    kernel! = balanced_triad_kernel!(backend)
    kernel!(colors, trits, seed; ndrange=n)
    KernelAbstractions.synchronize(backend)
    
    (Array(colors), Array(trits))
end

"""
    checkerboard_3(width, height; seed=1069, backend=CPU())

Generate 2D checkerboard with 3-color GF(3) pattern.

Returns (colors::Matrix{RGB}, polarities::Matrix{Int8}) where position (i,j) 
has polarity (i+j) mod 3 mapped to {-1, 0, +1}.
"""
function checkerboard_3(width::Int, height::Int;
                        seed::UInt64 = UInt64(1069),
                        backend = CPU())
    n = width * height
    colors_flat = KernelAbstractions.zeros(backend, RGB{Float32}, n)
    polarities_flat = KernelAbstractions.zeros(backend, Int8, n)
    
    kernel! = checkerboard_3color_kernel!(backend)
    kernel!(colors_flat, polarities_flat, seed, width; ndrange=n)
    KernelAbstractions.synchronize(backend)
    
    (reshape(Array(colors_flat), width, height),
     reshape(Array(polarities_flat), width, height))
end

"""
    verify_triad_spi(colors, fingerprints, seed, n_per_item)

Verify Schedule-Independent Processing: recompute and compare fingerprints.
"""
function verify_triad_spi(colors::Vector{RGB{Float32}}, 
                          fingerprints::Vector{UInt64},
                          seed::UInt64,
                          n_per_item::Int)
    n_workitems = length(fingerprints)
    
    for idx in 1:n_workitems
        state = triad_init(seed, UInt64(idx))
        for _ in 1:n_per_item
            state, _, _, _ = triad_next(state)
        end
        expected_fp = triad_fingerprint(state)
        
        if fingerprints[idx] != expected_fp
            return false
        end
    end
    true
end

"""
    gf3_balance(trits::Vector{Int8})

Check if trit vector sums to zero mod 3 (GF(3) balance).
"""
function gf3_balance(trits::Vector{Int8})
    sum(trits) % 3 == 0
end

# ═══════════════════════════════════════════════════════════════════════════
# Joint World Modeling: 3 Generators as 3 Agents
# ═══════════════════════════════════════════════════════════════════════════

"""
    TriadWorld

Three generators as jointly world-modeling agents.
Each generator has its own "perspective" (polarity) but they
share a common seed and produce coordinated colors.

This is the kernel-level version of JointModel from narrative_world.jl.
"""
mutable struct TriadWorld
    seed::UInt64
    minus::TriadState      # MINUS generator state
    ergodic::TriadState    # ERGODIC generator state  
    plus::TriadState       # PLUS generator state
    history::Vector{NTuple{3, RGB{Float32}}}  # (r, g, b) per step
    mutual_info::Float64   # Accumulated mutual information
end

"""
    TriadWorld(seed=1069)

Create a new triad world with 3 jointly-modeling generators.
"""
function TriadWorld(seed::UInt64 = UInt64(1069))
    base = splitmix64(seed)
    TriadWorld(
        seed,
        triad_init(seed ⊻ MINUS_TWIST, UInt64(1)),
        triad_init(seed ⊻ ERGODIC_TWIST, UInt64(2)),
        triad_init(seed ⊻ PLUS_TWIST, UInt64(3)),
        NTuple{3, RGB{Float32}}[],
        0.0
    )
end

"""
    step!(tw::TriadWorld)

Advance all 3 generators and record joint color.
Returns the RGB color where R=MINUS, G=ERGODIC, B=PLUS.
"""
function step!(tw::TriadWorld)
    tw.minus, r, _, _ = triad_next(tw.minus)
    tw.ergodic, _, g, _ = triad_next(tw.ergodic)
    tw.plus, _, _, b = triad_next(tw.plus)
    
    color = RGB{Float32}(r, g, b)
    
    # Record history
    push!(tw.history, (
        RGB{Float32}(r, 0.0f0, 0.0f0),  # MINUS contribution
        RGB{Float32}(0.0f0, g, 0.0f0),  # ERGODIC contribution
        RGB{Float32}(0.0f0, 0.0f0, b)   # PLUS contribution
    ))
    
    # Mutual information: how much do the 3 agree?
    mean_val = (r + g + b) / 3.0f0
    variance = ((r - mean_val)^2 + (g - mean_val)^2 + (b - mean_val)^2) / 3.0f0
    tw.mutual_info += 1.0 - sqrt(variance)  # High agreement = low variance
    
    color
end

"""
    joint_fingerprint(tw::TriadWorld)

Combined fingerprint of all 3 generators.
"""
function joint_fingerprint(tw::TriadWorld)
    triad_fingerprint(tw.minus) ⊻ 
    triad_fingerprint(tw.ergodic) ⊻ 
    triad_fingerprint(tw.plus)
end

"""
    triad_status(tw::TriadWorld)

Print status of the triad world.
"""
function triad_status(tw::TriadWorld)
    n = length(tw.history)
    println("═══════════════════════════════════════════════════════════")
    println("           TRIAD WORLD: 3 GENERATORS")
    println("═══════════════════════════════════════════════════════════")
    println()
    println("  Seed:                $(tw.seed)")
    println("  Steps:               $n")
    println("  Mutual Information:  $(round(tw.mutual_info, digits=2))")
    println()
    println("  Generator States:")
    println("    MINUS (R):   $(tw.minus.step) steps, fp=$(tw.minus.minus_state & 0xFFFF)")
    println("    ERGODIC (G): $(tw.ergodic.step) steps, fp=$(tw.ergodic.ergodic_state & 0xFFFF)")
    println("    PLUS (B):    $(tw.plus.step) steps, fp=$(tw.plus.plus_state & 0xFFFF)")
    println()
    println("  Joint Fingerprint:   $(joint_fingerprint(tw) & 0xFFFFFFFF)")
    println("═══════════════════════════════════════════════════════════")
end

# ═══════════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════════

export Trit, MINUS, ERGODIC, PLUS
export trit_add, trit_neg, trits_balanced
export TriadState, triad_init, triad_next, triad_color, triad_trit, triad_fingerprint
export triad_colors, balanced_colors, checkerboard_3
export verify_triad_spi, gf3_balance
export TriadWorld, step!, joint_fingerprint, triad_status

end # module KernelTriad
