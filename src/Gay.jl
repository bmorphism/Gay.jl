module Gay

# Re-export LispSyntax for the Lisp REPL
using LispSyntax
export sx, desx, codegen, @lisp_str, assign_reader_dispatch, include_lisp

# Color dependencies
using Colors
using ColorTypes
using Random
using SplittableRandoms

# Include wide-gamut color space support
include("colorspaces.jl")

# Include splittable RNG for deterministic color generation
include("splittable.jl")
export color_at, colors_at, palette_at, GAY_SEED

# Include custom REPL
include("repl.jl")

# Include Comrade.jl-style sky model DSL
include("comrade.jl")
export comrade_show, comrade_mring, comrade_disk, comrade_crescent

# Include KernelAbstractions SPMD kernels for portable parallel execution
include("kernels.jl")

# Include parallel color generation (OhMyThreads + Pigeons SPI + KA)
include("parallel.jl")

# Include GayMC - Colored Monte Carlo with SPI
include("gaymc.jl")
export GayMCContext, gay_sweep!, gay_measure!, gay_checkpoint, gay_restore!
export color_sweep, color_measure, color_state
export gay_exponential!, gay_cauchy!, gay_gaussian!, gay_metropolis!
export gay_workers, gay_tempering

# Include Lifetimes - Bounded/Unbounded computation traces
include("lifetimes.jl")
using .Lifetimes
export BoundedComputation, UnboundedComputation
export bounded_iter!, bounded_finalize!, bounded_color, bounded_fingerprint
export unbounded_iter!, unbounded_color, unbounded_fingerprint
export extend, project, @bounded, @unbounded

# Include KernelLifetimes - SPI colors for KernelAbstractions @index
include("kernel_lifetimes.jl")
using .KernelLifetimes
export KernelColorContext, kernel_color!, kernel_finalize!
export eventual_color, eventual_fingerprint, verify_kernel_spi
export index_color, iter_index_color, cartesian_color

# Include TensorParallel - SPI verification for distributed inference
include("tensor_parallel.jl")
using .TensorParallel
export TensorPartition, ShardedTensor, DistributedContext
export color_hidden_states!, color_logits!, color_embeddings!
export verify_allgather, verify_allreduce, verify_pipeline_handoff
export ExoPartition, create_exo_partitions, verify_exo_ring
export verify_distributed_inference

# Include ExoMLX - Exo + MLX cluster verification
include("exo_mlx.jl")
using .ExoMLX
export ExoCluster, ExoDevice, ExoVerifier
export discover_exo_cluster, verify_exo_inference
export inject_spi_colors, extract_fingerprint
export quick_verify_two_macs, model_config

# Include FaultTolerant - Jepsen-style fault injection and testing
include("fault_tolerant.jl")
using .FaultTolerant
export SimulatedCluster, DeviceState, FaultInjector
export inject!, heal!, heal_all!, run_inference!
export BidirectionalTracker, track_forward!, track_backward!, verify_consistency!
export GaloisConnection, alpha, gamma, verify_closure, verify_all_closures
export demo_fault_tolerant

# Include Abductive Testing for World Teleportation
include("abductive.jl")

# Include Chairmarks benchmarking
include("bench.jl")

# Metal GPU backend is now in ext/GayMetalExt.jl (loaded when Metal.jl is available)
# Check if Metal is available (macOS Apple Silicon only)
const HAS_METAL = Sys.isapple() && Sys.ARCH == :aarch64 && Base.find_package("Metal") !== nothing
export HAS_METAL

# Include JSON3 serialization
include("serialization.jl")

# Include QUIC path probe coloring
include("quic.jl")

# Include deterministic test tracking
include("tracking.jl")

# Include xy-pic LaTeX diagram generation
include("xypic.jl")

# Include SDF-style Propagator system with chromatic identity
include("propagator.jl")
include("propagator_lisp.jl")
export Propagator, PropagatorLisp

# Include Enzyme.jl integration for autodiff on colored S-expressions
include("enzyme.jl")

# Include Learnable Okhsl - the general class of all general classes
include("okhsl_learnable.jl")
using .OkhslLearnable
export LearnableColorSpace, LearnableOkhsl, LearnableSeedMap
export OkhslParameters, SeedProjection, EquivalenceClassObjective
export forward_color, learn_colorspace!, compute_loss
export EnzymeColorState, enzyme_color_gradient
export demo_learnable_okhsl

# ═══════════════════════════════════════════════════════════════════════════
# Lisp bindings for color operations
# ═══════════════════════════════════════════════════════════════════════════

"""
Lisp-accessible DETERMINISTIC color generation.

Usage from Gay REPL (Lisp syntax with parentheses):
  (gay-next)                  ; Next deterministic color  
  (gay-next 5)                ; Next 5 colors
  (gay-at 42)                 ; Color at index 42
  (gay-at 1 2 3)              ; Colors at indices 1,2,3
  (gay-palette 6)             ; 6 visually distinct colors
  (gay-seed 1337)             ; Set RNG seed
  (pride :rainbow)            ; Rainbow flag
  (pride :trans :rec2020)     ; Trans flag in Rec.2020
  (gay-blackhole 42)          ; Render black hole with seed
"""

# Symbol to ColorSpace mapping for Lisp interface
function sym_to_colorspace(s::Symbol)
    if s == :srgb || s == :SRGB
        return SRGB()
    elseif s == :p3 || s == :P3 || s == :displayp3
        return DisplayP3()
    elseif s == :rec2020 || s == :Rec2020 || s == :bt2020
        return Rec2020()
    else
        error("Unknown color space: $s. Use :srgb, :p3, or :rec2020")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Lisp-friendly deterministic color functions (kebab-case → snake_case)
# These are the primary API for reproducible colors from S-expressions
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_next()
    gay_next(n::Int)
    gay_next(cs::Symbol)

Generate the next deterministic color(s) from the global stream.
Equivalent to `next_color()`.
"""
gay_next() = next_color(current_colorspace())
gay_next(n::Int) = [next_color(current_colorspace()) for _ in 1:n]
gay_next(cs::Symbol) = next_color(sym_to_colorspace(cs))
gay_next(n::Int, cs::Symbol) = [next_color(sym_to_colorspace(cs)) for _ in 1:n]

"""
    gay_at(index)
    gay_at(indices...)

Get color(s) at specific invocation index/indices.
Equivalent to `color_at(index)`.
"""
gay_at(idx::Int) = color_at(idx, current_colorspace())
gay_at(idx::Int, cs::Symbol) = color_at(idx, sym_to_colorspace(cs))
gay_at(indices::Int...) = [color_at(i, current_colorspace()) for i in indices]

"""
    gay_palette(n)

Generate n visually distinct deterministic colors.
"""
gay_palette(n::Int) = next_palette(n, current_colorspace())
gay_palette(n::Int, cs::Symbol) = next_palette(n, sym_to_colorspace(cs))

"""
    gay_seed(n)

Set the global RNG seed for reproducibility.
"""
gay_seed(n::Int) = gay_seed!(n)

"""
    gay_space(cs::Symbol)

Set the current color space (:srgb, :p3, :rec2020).
"""
gay_space(cs::Symbol) = (CURRENT_COLORSPACE[] = sym_to_colorspace(cs); current_colorspace())

"""
    gay_rng_state()

Show the current RNG state (seed and invocation count).
"""
gay_rng_state() = (r = gay_rng(); (seed=r.seed, invocation=r.invocation))

"""
    gay_pride(flag::Symbol)

Get colors for a pride flag (:rainbow, :trans, :bi, :nb, :pan).
"""
gay_pride(flag::Symbol) = pride_flag(flag, current_colorspace())
gay_pride(flag::Symbol, cs::Symbol) = pride_flag(flag, sym_to_colorspace(cs))

# Legacy random (non-deterministic) wrappers
"""
    gay_random_color()

Generate a non-deterministic random color.
"""
gay_random_color() = random_color(SRGB())
gay_random_color(cs::Symbol) = random_color(sym_to_colorspace(cs))

"""
    gay_random_colors(n)

Generate n non-deterministic random colors.
"""
gay_random_colors(n::Int) = random_colors(n, SRGB())
gay_random_colors(n::Int, cs::Symbol) = random_colors(n, sym_to_colorspace(cs))

"""
    gay_random_palette(n)

Generate n visually distinct non-deterministic random colors.
"""
gay_random_palette(n::Int) = random_palette(n, SRGB())
gay_random_palette(n::Int, cs::Symbol) = random_palette(n, sym_to_colorspace(cs))

# Export all Lisp-friendly names (kebab-case maps to these)
export gay_next, gay_at, gay_palette, gay_seed, gay_space, gay_rng_state
export gay_random_color, gay_random_colors, gay_random_palette, gay_pride

# ═══════════════════════════════════════════════════════════════════════════
# Color display helpers
# ═══════════════════════════════════════════════════════════════════════════

"""
    show_colors(colors; width=2)

Display colors as ANSI true-color blocks in the terminal.
"""
function show_colors(colors::Vector; width::Int=2)
    block = "█" ^ width
    for c in colors
        rgb = convert(RGB, c)
        r = round(Int, clamp(rgb.r, 0, 1) * 255)
        g = round(Int, clamp(rgb.g, 0, 1) * 255)
        b = round(Int, clamp(rgb.b, 0, 1) * 255)
        print("\e[38;2;$(r);$(g);$(b)m$(block)\e[0m")
    end
    println()
end

"""
    show_palette(colors)

Display colors with their hex codes.
"""
function show_palette(colors::Vector)
    for c in colors
        rgb = convert(RGB, c)
        r = round(Int, clamp(rgb.r, 0, 1) * 255)
        g = round(Int, clamp(rgb.g, 0, 1) * 255)
        b = round(Int, clamp(rgb.b, 0, 1) * 255)
        hex = "#" * string(r, base=16, pad=2) * 
                    string(g, base=16, pad=2) * 
                    string(b, base=16, pad=2) |> uppercase
        print("\e[38;2;$(r);$(g);$(b)m████\e[0m $hex  ")
    end
    println()
end

export show_colors, show_palette

# ═══════════════════════════════════════════════════════════════════════════
# Main entry point (SpaceInvaders.jl style)
# ═══════════════════════════════════════════════════════════════════════════

"""
    main(; seed=42, n=6)

Launch a color palette demo, SpaceInvaders.jl style.
Displays a rainbow palette with the given seed.
"""
function main(; seed::Int=42, n::Int=6)
    gay_seed!(seed)
    
    println()
    println(rainbow_text("  ╔════════════════════════════════════════════════════════════════╗"))
    println(rainbow_text("  ║              Gay.jl - Wide Gamut Color Palettes                ║"))
    println(rainbow_text("  ╚════════════════════════════════════════════════════════════════╝"))
    println()
    
    # Show pride flags
    println("  Pride Flags:")
    print("    Rainbow:    "); show_colors(rainbow(); width=4)
    print("    Trans:      "); show_colors(transgender(); width=4)
    print("    Bi:         "); show_colors(bisexual(); width=4)
    print("    Nonbinary:  "); show_colors(nonbinary(); width=4)
    println()
    
    # Show deterministic palettes
    println("  Deterministic Palettes (seed=$seed):")
    for cs in [SRGB(), DisplayP3(), Rec2020()]
        gay_seed!(seed)
        colors = next_palette(n, cs)
        print("    $(rpad(typeof(cs), 12)): ")
        show_colors(colors; width=4)
    end
    println()
    
    # Show indexed access
    println("  Random Access (same seed = same colors):")
    print("    color_at(1,2,3,4,5,6; seed=$seed): ")
    colors = [color_at(i; seed=seed) for i in 1:6]
    show_colors(colors; width=4)
    println()
    
    println(rainbow_text("  Press SPC in REPL to enter Gay mode! 🏳️‍🌈"))
    println()
    
    return nothing
end

export main

# ═══════════════════════════════════════════════════════════════════════════
# Module initialization
# ═══════════════════════════════════════════════════════════════════════════

function __init__()
    # Initialize global splittable RNG
    gay_seed!(GAY_SEED)
    
    # Auto-initialize REPL if running interactively
    if isdefined(Base, :active_repl) && Base.active_repl !== nothing
        @async begin
            sleep(0.1)  # Let REPL finish loading
            init_gay_repl()
        end
    else
        @info "Gay.jl loaded 🏳️‍🌈 - Wide-gamut colors + splittable determinism"
        @info "In REPL: init_gay_repl() to start Gay mode (press SPC to enter)"
    end
end

end # module Gay
