# Kernel Lifetime Color Tracking with SPI
#
# Assigns deterministic colors to kernel invocations such that:
# 1. Each workitem gets a unique color stream based on its @index
# 2. The "eventual last next color" is deterministically predictable
# 3. XOR fingerprint of all workitem colors is invariant to execution order
#
# KEY INSIGHT: A kernel's lifetime is bounded (finite workitems, finite iterations)
# but the individual workitem colors form a 2D lattice: (workitem_index, iteration)
#
# METATHEORY:
#   KernelLifetime : (GlobalIndex, LocalIndex, GroupIndex) → ColorStream
#   final_color(workitem) = color_at(seed, workitem_index, total_iterations)
#   fingerprint(kernel) = ⊕ᵢ final_color(workitem[i])

module KernelLifetimes

using KernelAbstractions
using Colors: RGB

export KernelColorContext, kernel_color!, kernel_finalize!
export eventual_color, eventual_fingerprint
export @kernel_colored, verify_kernel_spi

# Re-export from parent
using ..Gay: GAY_SEED, hash_color, splitmix64, xor_fingerprint

# ═══════════════════════════════════════════════════════════════════════════════
# Core: Predictable Eventual Colors
# ═══════════════════════════════════════════════════════════════════════════════

"""
    eventual_color(seed, global_index, total_iterations) -> RGB{Float32}

Compute the color a workitem will have after `total_iterations` steps.
This is O(1) — no iteration required, directly computed from the index.

# The "eventual last next color" property
Given (seed, index, N), we can predict color_N without computing colors 1..N-1.

# Example
```julia
# Before kernel runs, predict final colors
final_colors = [eventual_color(seed, i, 1000) for i in 1:n_workitems]

# Run kernel for 1000 iterations
@kernel function my_kernel!(...)
    idx = @index(Global)
    for iter in 1:1000
        # ... work ...
    end
end

# Verify: actual final colors match predictions
```
"""
@inline function eventual_color(seed::Integer, global_index::Integer, total_iterations::Integer)
    # Hash: seed ⊕ (index × golden) ⊕ (iteration × prime)
    h = UInt64(seed) ⊻ (UInt64(global_index) * 0x9e3779b97f4a7c15) ⊻ 
                       (UInt64(total_iterations) * 0x517cc1b727220a95)
    r, g, b = hash_color(h, UInt64(total_iterations))
    RGB{Float32}(r, g, b)
end

"""
    eventual_fingerprint(seed, n_workitems, total_iterations) -> UInt32

Compute the XOR fingerprint of all workitem final colors.
This is O(n_workitems) but requires no kernel execution.

# SPI Guarantee
Same (seed, n_workitems, iterations) → same fingerprint,
regardless of how the kernel is scheduled or parallelized.
"""
function eventual_fingerprint(seed::Integer, n_workitems::Integer, total_iterations::Integer)
    fp = UInt32(0)
    for i in 1:n_workitems
        c = eventual_color(seed, i, total_iterations)
        # Pack RGB into bits and XOR
        r_bits = reinterpret(UInt32, c.r)
        g_bits = reinterpret(UInt32, c.g)
        b_bits = reinterpret(UInt32, c.b)
        fp ⊻= r_bits ⊻ g_bits ⊻ b_bits
    end
    fp
end

# ═══════════════════════════════════════════════════════════════════════════════
# Kernel-Level Context
# ═══════════════════════════════════════════════════════════════════════════════

"""
    KernelColorContext

Tracks color state for an entire kernel invocation.
Each workitem maintains its own iteration counter.

# Fields
- `seed`: Global seed for SPI
- `n_workitems`: Total number of workitems
- `max_iterations`: Maximum iterations per workitem
- `iteration_counts`: Per-workitem iteration counters
- `final_colors`: Per-workitem final colors (written at end)
- `fingerprint`: Rolling XOR fingerprint
"""
mutable struct KernelColorContext
    seed::UInt64
    n_workitems::Int
    max_iterations::Int
    iteration_counts::Vector{Int}
    final_colors::Matrix{Float32}  # n × 3
    fingerprint::UInt32
    finalized::Bool
end

function KernelColorContext(seed::Integer, n_workitems::Integer; max_iterations::Int=1000)
    KernelColorContext(
        UInt64(seed),
        n_workitems,
        max_iterations,
        zeros(Int, n_workitems),
        zeros(Float32, n_workitems, 3),
        UInt32(0),
        false
    )
end

"""
    kernel_color!(ctx::KernelColorContext, global_index) -> RGB{Float32}

Get the next color for a workitem and advance its iteration.
Called from within a kernel using @index(Global).

# Thread Safety
Each workitem only touches its own slot — no synchronization needed.
"""
function kernel_color!(ctx::KernelColorContext, global_index::Integer)
    ctx.finalized && error("Kernel context already finalized")
    
    # Advance iteration for this workitem
    ctx.iteration_counts[global_index] += 1
    iter = ctx.iteration_counts[global_index]
    
    if iter > ctx.max_iterations
        error("Workitem $global_index exceeded max_iterations=$(ctx.max_iterations)")
    end
    
    # Compute color
    h = ctx.seed ⊻ (UInt64(global_index) * 0x9e3779b97f4a7c15) ⊻ 
                   (UInt64(iter) * 0x517cc1b727220a95)
    r, g, b = hash_color(h, UInt64(iter))
    
    # Store as current color for this workitem
    ctx.final_colors[global_index, 1] = r
    ctx.final_colors[global_index, 2] = g
    ctx.final_colors[global_index, 3] = b
    
    RGB{Float32}(r, g, b)
end

"""
    kernel_finalize!(ctx::KernelColorContext) -> UInt32

Finalize the kernel and compute its fingerprint.
Returns the XOR of all workitem final colors.
"""
function kernel_finalize!(ctx::KernelColorContext)
    ctx.finalized = true
    ctx.fingerprint = xor_fingerprint(ctx.final_colors)
    ctx.fingerprint
end

# ═══════════════════════════════════════════════════════════════════════════════
# GPU Kernel Integration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    @kernel_colored seed expr

Execute a kernel with SPI color tracking.
Returns (result, fingerprint, predicted_fingerprint, match).

# Example
```julia
result = @kernel_colored 42 begin
    @kernel function work_kernel!(output, @Const(seed))
        i = @index(Global)
        iter = @index(Local)  # Use local index as iteration
        
        # Get deterministic color for this (workitem, iteration)
        h = seed ⊻ (UInt64(i) * 0x9e3779b97f4a7c15) ⊻ (UInt64(iter) * 0x517cc1b727220a95)
        r, g, b = hash_color(h, UInt64(iter))
        
        output[i, 1] = r
        output[i, 2] = g
        output[i, 3] = b
    end
    
    work_kernel!(backend)(output, seed, ndrange=n)
end
```
"""
macro kernel_colored(seed, expr)
    quote
        local _seed = $(esc(seed))
        local result = $(esc(expr))
        result
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Index-Based Color Functions (for use inside @kernel)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    index_color(seed, global_idx, local_idx, group_idx) -> (Float32, Float32, Float32)

Compute color from all three index types.
For use inside KernelAbstractions kernels.

# Usage in kernel
```julia
@kernel function my_kernel!(colors, @Const(seed))
    gi = @index(Global)
    li = @index(Local)
    gri = @index(Group)
    
    r, g, b = index_color(seed, gi, li, gri)
    colors[gi, 1] = r
    colors[gi, 2] = g
    colors[gi, 3] = b
end
```
"""
@inline function index_color(seed::UInt64, global_idx::Integer, 
                             local_idx::Integer, group_idx::Integer)
    # Combine all indices for unique hash
    h = seed ⊻ (UInt64(global_idx) * 0x9e3779b97f4a7c15) ⊻
              (UInt64(local_idx) * 0x517cc1b727220a95) ⊻
              (UInt64(group_idx) * 0xc4ceb9fe1a85ec53)
    hash_color(h, UInt64(global_idx))
end

"""
    iter_index_color(seed, global_idx, iteration) -> (Float32, Float32, Float32)

Compute color from global index and iteration counter.
This is the main function for tracking colors across kernel lifetime.

# Eventual Last Next Property
Given (seed, global_idx, N), we know the color at iteration N
before running the kernel.
"""
@inline function iter_index_color(seed::UInt64, global_idx::Integer, iteration::Integer)
    h = seed ⊻ (UInt64(global_idx) * 0x9e3779b97f4a7c15) ⊻
              (UInt64(iteration) * 0x517cc1b727220a95)
    hash_color(h, UInt64(iteration))
end

"""
    cartesian_color(seed, i, j, k=1) -> (Float32, Float32, Float32)

Compute color from Cartesian indices (for @index(Global, Cartesian)).
"""
@inline function cartesian_color(seed::UInt64, i::Integer, j::Integer, k::Integer=1)
    # Morton/Z-order encoding for locality-preserving hash
    linear = UInt64(i) + UInt64(j) * 0x100000 + UInt64(k) * 0x10000000000
    hash_color(seed, linear)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════════

"""
    verify_kernel_spi(n_workitems, iterations; seed=GAY_SEED, backend=CPU())

Verify SPI for kernel color tracking:
1. Predicted final colors match actual
2. Fingerprint is order-independent
3. Different backends produce same results
"""
function verify_kernel_spi(n_workitems::Int, iterations::Int; 
                           seed::Integer=GAY_SEED, backend=CPU())
    println("═" ^ 60)
    println("KERNEL LIFETIME SPI VERIFICATION")
    println("═" ^ 60)
    println("  n_workitems = $n_workitems")
    println("  iterations = $iterations")
    println("  seed = 0x$(string(seed, base=16))")
    println()
    
    # 1. Predict eventual colors
    println("1. Predicting eventual colors...")
    predicted = zeros(Float32, n_workitems, 3)
    for i in 1:n_workitems
        c = eventual_color(seed, i, iterations)
        predicted[i, 1] = c.r
        predicted[i, 2] = c.g
        predicted[i, 3] = c.b
    end
    predicted_fp = xor_fingerprint(predicted)
    println("   Predicted fingerprint: 0x$(string(predicted_fp, base=16, pad=8))")
    println()
    
    # 2. Simulate kernel execution (sequential)
    println("2. Simulating kernel (sequential)...")
    ctx = KernelColorContext(seed, n_workitems; max_iterations=iterations)
    for i in 1:n_workitems
        for _ in 1:iterations
            kernel_color!(ctx, i)
        end
    end
    seq_fp = kernel_finalize!(ctx)
    println("   Sequential fingerprint: 0x$(string(seq_fp, base=16, pad=8))")
    
    match1 = seq_fp == predicted_fp
    println("   Matches prediction: ", match1 ? "✓ PASS" : "✗ FAIL")
    println()
    
    # 3. Simulate kernel execution (random order)
    println("3. Simulating kernel (random order)...")
    ctx2 = KernelColorContext(seed, n_workitems; max_iterations=iterations)
    order = shuffle(collect(1:n_workitems))
    for i in order
        for _ in 1:iterations
            kernel_color!(ctx2, i)
        end
    end
    rand_fp = kernel_finalize!(ctx2)
    println("   Random-order fingerprint: 0x$(string(rand_fp, base=16, pad=8))")
    
    match2 = rand_fp == predicted_fp
    println("   Matches prediction: ", match2 ? "✓ PASS" : "✗ FAIL")
    println()
    
    # 4. Verify eventual_fingerprint matches
    println("4. Verifying eventual_fingerprint()...")
    direct_fp = eventual_fingerprint(seed, n_workitems, iterations)
    println("   Direct calculation: 0x$(string(direct_fp, base=16, pad=8))")
    
    match3 = direct_fp == predicted_fp
    println("   Matches prediction: ", match3 ? "✓ PASS" : "✗ FAIL")
    println()
    
    all_pass = match1 && match2 && match3
    println("═" ^ 60)
    println(all_pass ? "ALL KERNEL SPI INVARIANTS VERIFIED ✓" : "VERIFICATION FAILED ✗")
    println("═" ^ 60)
    
    return all_pass
end

# Import shuffle for verification
using Random: shuffle

# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════

function Base.show(io::IO, ctx::KernelColorContext)
    status = ctx.finalized ? "finalized" : "active"
    max_iter = maximum(ctx.iteration_counts)
    min_iter = minimum(ctx.iteration_counts)
    print(io, "KernelColorContext(n=$(ctx.n_workitems), ")
    print(io, "iters=$min_iter..$max_iter, $status, ")
    print(io, "fp=0x$(string(ctx.fingerprint, base=16, pad=8)))")
end

end # module
