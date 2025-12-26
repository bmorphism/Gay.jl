# SPI Command Line Interface
# ==========================
#
# Provides a command-line interface for running SPI verification.
#
# Usage:
#   julia --project=. -e 'using Gay; spi_main()' -- [options]
#
# Options:
#   --verify           Run full verification (default)
#   --report           Generate verification report
#   --thread ID        Set thread ID for context
#   --size N           Set concept tensor size (default: 23)
#   --benchmark        Run performance benchmarks
#   --export FILE      Export report to markdown file
#   --quiet            Minimal output

module SPICLI

using Dates

export spi_main, run_cli, parse_args

# Import from parent
using ..Gay: GAY_SEED, splitmix64
using ..Gay: run_regression_suite
using ..VerificationReport: generate_report, attestation_fingerprint, verify_coherence
using ..VerificationReport: export_report_markdown
using ..AmpThreads: AmpThread, ThreadGenealogy, add_thread!, genealogy_fingerprint
using ..AmpThreads: thread_seed, verify_thread_chain
using ..ConceptTensor: ConceptLattice, lattice_fingerprint, world_concept_tensor
using ..ConceptTensor: world_exponential, world_higher_structure
using ..TracedTensor: world_traced_tensor
using ..ThreadFindings: world_thread_findings

# ═══════════════════════════════════════════════════════════════════════════════
# Argument Parsing
# ═══════════════════════════════════════════════════════════════════════════════

struct CLIArgs
    command::Symbol
    thread_id::Union{Nothing, String}
    tensor_size::Int
    n_threads::Int
    export_file::Union{Nothing, String}
    quiet::Bool
    benchmark::Bool
end

function parse_args(args::Vector{String})
    command = :verify
    thread_id = nothing
    tensor_size = 23
    n_threads = 69
    export_file = nothing
    quiet = false
    benchmark = false
    
    i = 1
    while i <= length(args)
        arg = args[i]
        
        if arg == "--verify"
            command = :verify
        elseif arg == "--report"
            command = :report
        elseif arg == "--demo"
            command = :demo
        elseif arg == "--thread" && i < length(args)
            i += 1
            thread_id = args[i]
        elseif arg == "--size" && i < length(args)
            i += 1
            tensor_size = parse(Int, args[i])
        elseif arg == "--threads" && i < length(args)
            i += 1
            n_threads = parse(Int, args[i])
        elseif arg == "--export" && i < length(args)
            i += 1
            export_file = args[i]
        elseif arg == "--quiet"
            quiet = true
        elseif arg == "--benchmark"
            benchmark = true
        elseif arg == "--help"
            command = :help
        end
        
        i += 1
    end
    
    CLIArgs(command, thread_id, tensor_size, n_threads, export_file, quiet, benchmark)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════════════

function cmd_verify(args::CLIArgs)
    !args.quiet && println("═" ^ 60)
    !args.quiet && println("SPI VERIFICATION")
    !args.quiet && println("═" ^ 60)
    !args.quiet && println()
    
    # Determine seed
    seed = if args.thread_id !== nothing
        thread_seed(args.thread_id)
    else
        GAY_SEED
    end
    
    !args.quiet && println("Configuration:")
    !args.quiet && println("  Seed: 0x$(string(seed, base=16, pad=16))")
    !args.quiet && println("  Tensor size: $(args.tensor_size)³")
    !args.quiet && println("  Threads: $(args.n_threads)")
    !args.quiet && println()
    
    # Run regression tests
    !args.quiet && println("Running regression tests...")
    result = run_regression_suite(; verbose=!args.quiet)
    
    # Generate report
    !args.quiet && println()
    !args.quiet && println("Generating verification report...")
    report = generate_report(; seed=seed, tensor_size=args.tensor_size, n_threads=args.n_threads)
    
    !args.quiet && println()
    !args.quiet && println("Results:")
    !args.quiet && println("  Attestation: 0x$(string(attestation_fingerprint(report), base=16, pad=8))")
    !args.quiet && println("  Coherent: $(verify_coherence(report) ? "◆" : "◇")")
    
    if args.quiet
        println(result && verify_coherence(report) ? "PASS" : "FAIL")
    end
    
    # Export if requested
    if args.export_file !== nothing
        md = export_report_markdown(report)
        open(args.export_file, "w") do f
            write(f, md)
        end
        !args.quiet && println("  Exported to: $(args.export_file)")
    end
    
    !args.quiet && println()
    !args.quiet && println("═" ^ 60)
    
    result && verify_coherence(report)
end

function cmd_report(args::CLIArgs)
    seed = if args.thread_id !== nothing
        thread_seed(args.thread_id)
    else
        GAY_SEED
    end
    
    report = generate_report(; seed=seed, tensor_size=args.tensor_size, n_threads=args.n_threads)
    
    if args.export_file !== nothing
        md = export_report_markdown(report)
        open(args.export_file, "w") do f
            write(f, md)
        end
        println("Report exported to: $(args.export_file)")
    else
        println(export_report_markdown(report))
    end
    
    true
end

function cmd_demo(args::CLIArgs)
    println("═" ^ 60)
    println("SPI DEMONSTRATION")
    println("═" ^ 60)
    println()
    
    println("1. Concept Tensor Demo")
    println("-" ^ 40)
    world_concept_tensor(; size=min(args.tensor_size, 23), n_steps=3)
    println()
    
    println("2. Exponential X^X Demo")
    println("-" ^ 40)
    world_exponential(; size=min(args.tensor_size, 11))
    println()
    
    println("3. Higher Structure Demo")
    println("-" ^ 40)
    world_higher_structure(; size=min(args.tensor_size, 11))
    println()
    
    println("4. Traced Tensor Demo")
    println("-" ^ 40)
    world_traced_tensor(; size=min(args.tensor_size, 11))
    println()
    
    println("5. Thread Findings Demo")
    println("-" ^ 40)
    world_thread_findings(; n_threads=min(args.n_threads, 20))
    println()
    
    println("═" ^ 60)
    println("DEMONSTRATION COMPLETE")
    println("═" ^ 60)
    
    true
end

function cmd_help()
    println("""
SPI Color Verification System

Usage:
  julia --project=. -e 'using Gay; spi_main()' -- [options]

Commands:
  --verify       Run full verification (default)
  --report       Generate verification report
  --demo         Run interactive demos

Options:
  --thread ID    Set thread ID for context
  --size N       Set concept tensor size (default: 23)
  --threads N    Set number of verification threads (default: 69)
  --export FILE  Export report to markdown file
  --quiet        Minimal output (just PASS/FAIL)
  --benchmark    Include performance benchmarks
  --help         Show this help

Examples:
  # Run verification with default settings
  julia --project=. -e 'using Gay; spi_main()'

  # Verify with specific thread context
  julia --project=. -e 'using Gay; spi_main()' -- --thread T-abc123

  # Generate and export report
  julia --project=. -e 'using Gay; spi_main()' -- --report --export report.md

  # Run with 69³ tensor
  julia --project=. -e 'using Gay; spi_main()' -- --size 69
""")
    true
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

function run_cli(args::CLIArgs)
    if args.command == :verify
        cmd_verify(args)
    elseif args.command == :report
        cmd_report(args)
    elseif args.command == :demo
        cmd_demo(args)
    elseif args.command == :help
        cmd_help()
    else
        cmd_verify(args)
    end
end

"""
    spi_main()

Main entry point for the SPI CLI.
"""
function spi_main()
    args = parse_args(ARGS)
    success = run_cli(args)
    exit(success ? 0 : 1)
end

"""
    spi_verify(; thread_id=nothing, size=23, n_threads=69)

Programmatic interface for verification.
"""
function spi_verify(; thread_id::Union{Nothing, String}=nothing, 
                     size::Int=23, n_threads::Int=69)
    args = CLIArgs(:verify, thread_id, size, n_threads, nothing, false, false)
    run_cli(args)
end

export spi_verify

end # module SPICLI
