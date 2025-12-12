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

# Include maximally parallel seed search (Fugue-inspired minimal syncpoints)
include("parallel_seed_search.jl")
export find_seeds_parallel, TARGET_COLORS, SearchResult, demo_parallel_search

# Include maximally parallel genetic search
include("genetic_search.jl")
export GeneticSearchConfig, GeneticSearchResult
export genetic_search_parallel, island_evolution, demo_genetic_search

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

# Include Chaos Vibing - Maximal fault injection into parallel causal chains
include("chaos_vibing.jl")
export ChaosConfig, ChaosResult, ChaosVibe
export inject_chaos!, run_chaos_campaign, chaos_vibe!
export CausalChain, break_chain!, verify_chain, chain_fingerprint
export demo_chaos_vibing

# Include Push-Pull Sequence Verification
include("push_pull_sequence.jl")
using .PushPullSequence
export SequenceColorStream, push_token!, pull_verify!
export StreamingVerifier, push_chunk!, verify_chunk!
export demo_push_pull_sequence

# Include Abductive Testing for World Teleportation
include("abductive.jl")

# Include Chairmarks benchmarking
include("bench.jl")

# Include SPI Regression Benchmarks
include("bench_spi_regression.jl")
using .SPIRegressionBench
export run_spi_regression_tests, calibrate_baselines

# Include Concept Tensor (69³ parallel interaction space)
include("concept_tensor.jl")
using .ConceptTensor
export ConceptLattice, step_parallel!, verify_monoid_laws
export interpolate_subtext!, extrapolate_superstructure!, interact!
export lattice_magnetization, lattice_fingerprint, propagate_all!
export demo_concept_tensor, demo_exponential, demo_higher_structure
export ConceptMorphism, identity_morphism, compose, eval_morphism
export concept_to_morphism, verify_exponential_laws, morphism_fingerprint
export step_as_morphism, iterate_morphism, fixed_points, orbit
export trace_morphism, verify_trace_laws, self_application

# Include Regression Tests That Don't Suck
include("regression.jl")
using .SPIRegression
export run_regression_suite, verify_splitmix64_reference
export verify_galois_closure, verify_parallel_order_independence
export verify_concept_tensor_invariants, @test_spi

# Metal GPU backend is now in ext/GayMetalExt.jl (loaded when Metal.jl is available)
# Check if Metal is available (macOS Apple Silicon only)
const HAS_METAL = Sys.isapple() && Sys.ARCH == :aarch64 && Base.find_package("Metal") !== nothing
export HAS_METAL

# Include Proof of Color Parallelism (PoCP) - Making Chia Gay
include("proof_of_color.jl")
using .ProofOfColor
export ColorPlot, ColorVDF, ProofOfColorParallelism
export create_plot, verify_plot, plot_fingerprint
export create_vdf, verify_vdf, vdf_output
export create_pocp, verify_pocp, demo_pocp
export Seed  # Universal seed wrapper

# Include Ergodic Bridge (Wall Clock ↔ Color Bandwidth ↔ Compositionality)
include("ergodic_bridge.jl")
using .ErgodicBridge
export WallClockBridge, ColorBandwidth, ErgodicMeasure, CompositionObstruction
export create_bridge, verify_bridge, measure_bandwidth, measure_ergodicity
export detect_obstructions, horizon_analysis, demo_ergodic_bridge

# Include Spectral Bridge (dgleich GenericArpack ↔ Gay.jl ↔ PyT TDL)
include("spectral_bridge.jl")
export SpectralColorBridge, ArpackSeed, HodgeLaplacian
export color_eigenvector, verify_spectral_spi, eigencolor_fingerprint
export simplicial_hodge, chromatic_spectral_clustering
export demo_spectral_bridge

# Include Gay Hyperdoctrine (Categorical Logic with Chromatic Predicates)
include("hyperdoctrine.jl")
export ChromaticType, ChromaticPredicate, GayHyperdoctrine
export substitution, existential, universal, verify_beck_chevalley
export heyting_and, heyting_or, heyting_implies, heyting_not
export predicate_color, predicate_fingerprint
export demo_hyperdoctrine

# Include Color-Logic Pullback Squares (Proper Beck-Chevalley from Hatchery)
include("color_logic_pullback.jl")
export ColorLogicSystem, LogicPullbackSquare, ColoredTheory
export fibered_product, pullback_color, beck_chevalley_proper
export theory_level, metatheory_level, logic_system_color
export LogicSystem, TheoryLevel
export INTUITIONISTIC, PARACONSISTENT, LINEAR, MODAL_S4, HOTT, CLASSICAL, METATHEORY
export OBJECT_LEVEL, META_LEVEL, HIGHER_META
export ChromaticPredicate_v2
export demo_color_logic_pullback

# Include Tropical Semirings with verification
include("tropical_semirings.jl")

# Include JSON3 serialization
include("serialization.jl")

# Include QUIC path probe coloring
include("quic.jl")

# Include deterministic test tracking
include("tracking.jl")

# Include Whale-Human Translation Bridge
include("whale_bridge.jl")

# Include Real Whale Data (EC-1 Clan from Sharma et al. 2024)
include("whale_data.jl")

# Include Whale Demo
include("whale_demo.jl")

# Include Whale World (Parallel SPI Demonstration through tripartite synergy)
# NOTE: Must come before spc_repl.jl which uses WhaleWorld types
include("whale_world.jl")

# Include Whale Curriculum (Omniglot-style hierarchical refinement)
include("whale_curriculum.jl")

# Include SPC REPL (Symbolic · Possible · Compositional)
include("spc_repl.jl")

# Include xy-pic LaTeX diagram generation
include("xypic.jl")

# Include SDF-style Propagator system with chromatic identity
include("propagator.jl")
include("propagator_lisp.jl")
export Propagator, PropagatorLisp

# Include Traced Monoidal Category Structure (after Propagator)
include("traced_tensor.jl")
using .TracedTensor
export TracedMorphism, tensor_product, monoidal_unit, categorical_trace
export feedback_loop, TensorNetwork, add_node!, add_edge!, run_network!
export verify_traced_laws, demo_traced_tensor, network_fingerprint

# Include Thread Findings (Two Monad Structure)
include("thread_findings.jl")
using .ThreadFindings
export Finding, FindingsSet, ThreadContext, VerificationMonad
export bind_finding, return_finding, run_verification
export count_threads, fingerprint_threads, lazy_place!
export demo_thread_findings, LazyThreadStream, next_thread!, LAYER_NAMES
export run_all_verifications

# Include Verification Report
include("verification_report.jl")
using .VerificationReport
export generate_report, FullReport, ReportSection
export verify_coherence, attestation_fingerprint
export export_report_markdown, demo_report

# Include Amp Thread Connection
include("amp_threads.jl")
using .AmpThreads
export AmpThread, thread_seed, thread_color, thread_fingerprint
export ThreadGenealogy, add_thread!, genealogy_fingerprint
export verify_thread_chain, demo_amp_threads

# Include Cognitive Superposition (DisCoCat-style categorical semantics)
include("cognitive_superposition.jl")
using .CognitiveSuperposition
export CognitiveState, CognitiveMorphism, CognitiveCategory
export superpose, collapse, entails, induces, abduces
export BraidedSuperposition, HypergraphSuperposition
export cognitive_tensor, cognitive_trace, cognitive_spider
export verify_cognitive_laws, demo_cognitive_superposition

# Include SPI CLI
include("spi_cli.jl")
using .SPICLI
export spi_main, spi_verify

# Include Tuning Parameters
include("tuning.jl")
using .Tuning
export SPIConfig, default_config, with_config, preset, tune!, current_config, set_config!

# Include Kripke Semantics & Possible Worlds (Layers 6-8)
include("kripke_worlds.jl")
using .KripkeWorlds
export KripkeFrame, World, accessible, truth_at, necessity, possibility
export ModalProposition, box, diamond, verify_modal_laws
export SheafSemantics, local_truth, global_sections, stalk_at
export world_kripke, run_kripke_tests

# Include Random Topos (Layers 9-11: Simpson's Three Toposes)
include("random_topos.jl")
using .RandomTopos
export SampleSpace, RandomElement, RandomVariable, ProbabilitySheaf
export GrowingRandomTopos, grow_random_topos!, world_random_topos

# Include Strategic Differentiation (Semantic Blastoderm → Tower mapping)
include("strategic_differentiation.jl")
using .StrategicDifferentiation
export StrategicChoice, DifferentiationBasin, SemanticFate
export tower_basin, world_strategic_differentiation
export differentiate!, fate_fingerprint, basin_color, TOWER_BASINS, fate_at_layer

# Include Compositional World Bridge (Topos Institute research program)
include("compositional_world.jl")
using .CompositionalWorld
export SystemProperty, DynamicalDoctrine, CompositionalBridge
export property_layer, doctrine_fingerprint, world_compositional_world
export SYSTEM_PROPERTIES, compose_systems, behavioral_intersection

# Include Unified Tower (all 12 layers)
include("tower.jl")
using .Tower
export TowerState, world_tower, tower_fingerprint, run_tower_tests
export LAYER_INFO, layer_name, layer_category

# Include SplitMix64-CFT Verification (first principles SPI proof)
include("splitmix_cft_verify.jl")
using .SplitMixCFTVerify
export run_verification_suite

# Include Multiverse Geometric Morphisms (Hamkins + Dave White)
include("multiverse_geometric.jl")
using .MultiverseGeometric
export Verse, MultiverseFrame, GeometricMorphism
export create_verse, partition, pushdown!, pullup!, resolve!
export verse_fingerprint, verse_color, verify_multiverse_laws
export HolographicColorGame, game_state, make_move!, check_win
export world_multiverse, demo_holographic_game

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
    gay_at(index; seed=nothing)
    gay_at(indices...; seed=nothing)

Get color(s) at specific invocation index/indices.
Uses global seed from `gay_seed!()` unless `seed=` is provided.

# Universal seed support
```julia
gay_at(42)                    # Use global seed
gay_at(42; seed=123)          # Integer seed
gay_at(42; seed="experiment") # String seed
gay_at(42; seed=:test)        # Symbol seed
gay_at(42; seed=[1,2,3])      # Array seed
```
"""
function gay_at(idx::Integer; seed=nothing)
    s = seed === nothing ? gay_rng().seed : _to_seed(seed)
    color_at(idx, current_colorspace(); seed=s)
end
function gay_at(idx::Integer, cs::Symbol; seed=nothing)
    s = seed === nothing ? gay_rng().seed : _to_seed(seed)
    color_at(idx, sym_to_colorspace(cs); seed=s)
end

# Helper: convert anything to seed
_to_seed(x::Integer) = UInt64(x)
_to_seed(x::UInt64) = x
function _to_seed(x)
    # FNV-1a hash for strings, symbols, arrays, etc.
    h = UInt64(0xcbf29ce484222325)
    for byte in reinterpret(UInt8, [hash(x)])
        h ⊻= byte
        h *= 0x100000001b3
    end
    # SplitMix64 finalization
    h = (h ⊻ (h >> 30)) * 0xbf58476d1ce4e5b9
    h = (h ⊻ (h >> 27)) * 0x94d049bb133111eb
    h ⊻ (h >> 31)
end

"""
    gay_palette(n; seed=nothing)

Generate n visually distinct deterministic colors.
"""
function gay_palette(n::Int; seed=nothing)
    s = seed === nothing ? gay_rng().seed : _to_seed(seed)
    [color_at(i, current_colorspace(); seed=s) for i in 1:n]
end
function gay_palette(n::Int, cs::Symbol; seed=nothing)
    s = seed === nothing ? gay_rng().seed : _to_seed(seed)
    [color_at(i, sym_to_colorspace(cs); seed=s) for i in 1:n]
end

"""
    gay_seed(x)

Set the global RNG seed. Accepts any type:
- Integer: `gay_seed(42)`
- String: `gay_seed("my experiment")`
- Symbol: `gay_seed(:test_run)`
- Array: `gay_seed([1, 2, 3])`
"""
gay_seed(n::Integer) = gay_seed!(n)
gay_seed(x) = gay_seed!(_to_seed(x))

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
            # Initialize SPC REPL (press SPACE to enter)
            init_spc_repl()
        end
    else
        @info "Gay.jl loaded 🏳️‍🌈 - Wide-gamut colors + splittable determinism"
        @info "In REPL: init_spc_repl() for SPC mode (press SPACE to enter)"
    end
end

end # module Gay
