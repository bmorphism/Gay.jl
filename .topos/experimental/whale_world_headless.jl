# Whale World Headless Test Harness
# Runs SPC REPL commands without interactive terminal, suitable for:
# - CI/CD pipelines
# - Unison Terminus integration
# - Cross-language verification (Julia ↔ Unison ↔ Rust)

using Test

# Include Gay.jl from parent
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Gay

# ═══════════════════════════════════════════════════════════════════════════
# Headless Command Evaluator
# ═══════════════════════════════════════════════════════════════════════════

"""
Capture output from SPC command execution.
Returns (output::String, success::Bool).
"""
function eval_spc_capture(cmd::String)
    io = IOBuffer()
    redirect_stdout(io) do
        try
            Gay.spc_eval(cmd)
            return true
        catch e
            println("ERROR: $e")
            return false
        end
    end
    output = String(take!(io))
    (output, !contains(output, "ERROR"))
end

"""
Run SPC command and return output string.
"""
function spc(cmd::String)::String
    Gay.init_world(Gay.GAY_SEED)
    io = IOBuffer()
    redirect_stdout(io) do
        Gay.spc_eval(cmd)
    end
    String(take!(io))
end

# ═══════════════════════════════════════════════════════════════════════════
# Test Protocol: Commands that Unison Terminus can drive
# ═══════════════════════════════════════════════════════════════════════════

"""
Protocol commands for Unison Terminus integration.
Each returns a structured result that Unison can parse.
"""
module WhaleWorldProtocol

using ..Gay

# Initialize world with N whales, return fingerprint
function init_world(n::Int, seed::UInt64)
    world = Gay.demo_whale_world(n_whales=n, seed=seed)
    Gay.WHALE_WORLD[] = world
    (
        n_whales = length(world.whales),
        fingerprint = Gay.world_state_hash(world),
        base_seed = world.base_seed
    )
end

# Compute synergies, return gadget distribution
function compute_synergies()
    world = Gay.WHALE_WORLD[]
    Gay.compute_all_synergies!(world)
    
    # Count gadget types
    gadgets = Dict{Symbol, Int}()
    for (_, syn) in world.synergies
        gadgets[syn.gadget_class] = get(gadgets, syn.gadget_class, 0) + 1
    end
    
    (
        n_triads = length(world.synergies),
        gadgets = gadgets,
        fingerprint = Gay.world_state_hash(world)
    )
end

# Get optimal triads
function optimal_triads(k::Int=5)
    world = Gay.WHALE_WORLD[]
    optimal = Gay.find_optimal_triads(world; k=k)
    
    [(
        whale_ids = key,
        gadget = string(syn.gadget_class),
        coupling = syn.coupling_score,
        xor_residue = syn.xor_residue,
        fingerprint = syn.color_fingerprint
    ) for (key, syn) in optimal]
end

# First-contact verification
function verify_fingerprint(expected::UInt64)
    world = Gay.WHALE_WORLD[]
    local_fp = Gay.world_state_hash(world)
    (
        verified = local_fp == expected,
        local_fingerprint = local_fp,
        expected_fingerprint = expected
    )
end

# Get whale color chain (for cross-language verification)
function whale_colors(whale_id::String)
    world = Gay.WHALE_WORLD[]
    if !haskey(world.whales, whale_id)
        return nothing
    end
    
    whale = world.whales[whale_id]
    rgb_values = [(
        r = round(Int, clamp(c.r, 0, 1) * 255),
        g = round(Int, clamp(c.g, 0, 1) * 255),
        b = round(Int, clamp(c.b, 0, 1) * 255)
    ) for c in whale.chain]
    
    (
        whale_id = whale_id,
        seed = whale.seed,
        colors = rgb_values,
        notes = whale.notes,
        intervals = whale.intervals
    )
end

# Run SPI demonstration and return verification result
function spi_demo()
    world = Gay.WHALE_WORLD[]
    result = Gay.spi_parallel_demo(world; verbose=false)
    result
end

end # module WhaleWorldProtocol

# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

@testset "Whale World Headless" begin
    
    @testset "World Initialization" begin
        result = WhaleWorldProtocol.init_world(6, Gay.GAY_SEED)
        @test result.n_whales == 6
        @test result.fingerprint isa UInt64
        @test result.base_seed == Gay.GAY_SEED
    end
    
    @testset "Synergy Computation" begin
        WhaleWorldProtocol.init_world(6, Gay.GAY_SEED)
        result = WhaleWorldProtocol.compute_synergies()
        
        # 6 choose 3 = 20 triads
        @test result.n_triads == 20
        @test haskey(result.gadgets, :XOR) || haskey(result.gadgets, :MAJ) || 
              haskey(result.gadgets, :PARITY) || haskey(result.gadgets, :CLAUSE)
    end
    
    @testset "Optimal Triads" begin
        WhaleWorldProtocol.init_world(6, Gay.GAY_SEED)
        triads = WhaleWorldProtocol.optimal_triads(5)
        
        @test length(triads) == 5
        @test all(t -> t.coupling >= 0.0 && t.coupling <= 1.0, triads)
        
        # Triads should be sorted by coupling (descending)
        couplings = [t.coupling for t in triads]
        @test couplings == sort(couplings, rev=true)
    end
    
    @testset "Fingerprint Verification" begin
        result1 = WhaleWorldProtocol.init_world(6, Gay.GAY_SEED)
        
        # Same seed should produce same fingerprint
        result2 = WhaleWorldProtocol.init_world(6, Gay.GAY_SEED)
        @test result1.fingerprint == result2.fingerprint
        
        # Verification should pass with correct fingerprint
        verify = WhaleWorldProtocol.verify_fingerprint(result2.fingerprint)
        @test verify.verified == true
        
        # Verification should fail with wrong fingerprint
        bad_verify = WhaleWorldProtocol.verify_fingerprint(UInt64(0xDEADBEEF))
        @test bad_verify.verified == false
    end
    
    @testset "Whale Colors Determinism" begin
        WhaleWorldProtocol.init_world(6, Gay.GAY_SEED)
        
        colors1 = WhaleWorldProtocol.whale_colors("W001")
        colors2 = WhaleWorldProtocol.whale_colors("W001")
        
        @test colors1 !== nothing
        @test colors2 !== nothing
        @test colors1.colors == colors2.colors
        @test colors1.seed == colors2.seed
        @test length(colors1.colors) == 12
    end
    
    @testset "SPI Verification" begin
        WhaleWorldProtocol.init_world(6, Gay.GAY_SEED)
        result = WhaleWorldProtocol.spi_demo()
        
        @test result.spi_verified == true
        @test result.n_triads == 20
        @test result.fingerprint isa UInt64
    end
    
    @testset "Cross-Seed Independence" begin
        # Different seeds should produce different fingerprints
        result1 = WhaleWorldProtocol.init_world(6, UInt64(0x1234))
        result2 = WhaleWorldProtocol.init_world(6, UInt64(0x5678))
        
        @test result1.fingerprint != result2.fingerprint
    end
    
end

# ═══════════════════════════════════════════════════════════════════════════
# Unison Terminus Integration: JSON Protocol
# ═══════════════════════════════════════════════════════════════════════════

"""
JSON-based protocol for Unison Terminus communication.
Commands come as JSON, responses go back as JSON.
"""
module TerminusProtocol

using JSON3
using ..WhaleWorldProtocol

const COMMANDS = Dict{String, Function}(
    "init" => (args) -> WhaleWorldProtocol.init_world(
        get(args, "n_whales", 6),
        UInt64(get(args, "seed", 0x6761795f636f6c6f))
    ),
    "synergy" => (_) -> WhaleWorldProtocol.compute_synergies(),
    "optimal" => (args) -> WhaleWorldProtocol.optimal_triads(get(args, "k", 5)),
    "verify" => (args) -> WhaleWorldProtocol.verify_fingerprint(UInt64(args["fingerprint"])),
    "colors" => (args) -> WhaleWorldProtocol.whale_colors(args["whale_id"]),
    "spi" => (_) -> WhaleWorldProtocol.spi_demo(),
)

"""
Process a JSON command from Terminus.
Returns JSON response.
"""
function process(json_input::String)::String
    try
        request = JSON3.read(json_input)
        cmd = request.command
        args = get(request, :args, Dict())
        
        if haskey(COMMANDS, cmd)
            result = COMMANDS[cmd](args)
            return JSON3.write((status = "ok", result = result))
        else
            return JSON3.write((status = "error", message = "Unknown command: $cmd"))
        end
    catch e
        return JSON3.write((status = "error", message = string(e)))
    end
end

"""
REPL loop for Terminus integration.
Reads JSON from stdin, writes JSON to stdout.
"""
function repl_loop()
    while !eof(stdin)
        line = readline(stdin)
        if isempty(line)
            continue
        end
        response = process(line)
        println(response)
        flush(stdout)
    end
end

end # module TerminusProtocol

# ═══════════════════════════════════════════════════════════════════════════
# Cross-Language Verification Data
# ═══════════════════════════════════════════════════════════════════════════

"""
Generate verification data for Unison/Rust cross-language tests.
"""
function generate_verification_data(seed::UInt64=Gay.GAY_SEED)
    # Initialize
    WhaleWorldProtocol.init_world(6, seed)
    
    # Get whale W001's colors (canonical reference)
    w001 = WhaleWorldProtocol.whale_colors("W001")
    
    # Compute synergies
    syn_result = WhaleWorldProtocol.compute_synergies()
    
    # Get optimal triads
    optimal = WhaleWorldProtocol.optimal_triads(3)
    
    Dict(
        "seed" => seed,
        "whale_w001" => Dict(
            "seed" => w001.seed,
            "colors" => w001.colors,
            "notes" => w001.notes,
            "intervals" => w001.intervals
        ),
        "fingerprint" => syn_result.fingerprint,
        "n_triads" => syn_result.n_triads,
        "gadget_distribution" => syn_result.gadgets,
        "top_3_triads" => [Dict(
            "whale_ids" => t.whale_ids,
            "gadget" => t.gadget,
            "coupling" => t.coupling,
            "xor_residue" => t.xor_residue
        ) for t in optimal]
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Main: Run tests or start Terminus protocol
# ═══════════════════════════════════════════════════════════════════════════

function main(args=ARGS)
    if "--terminus" in args
        # Terminus protocol mode
        TerminusProtocol.repl_loop()
    elseif "--verify" in args
        # Generate verification data
        data = generate_verification_data()
        using JSON3
        println(JSON3.write(data, allow_inf=true))
    else
        # Run tests
        include("runtests.jl")
    end
end

# Run if invoked directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
