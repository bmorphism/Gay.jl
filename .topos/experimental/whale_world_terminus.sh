#!/bin/bash
# Whale World Terminus Test
# Runs headless Julia tests with Unison verification
#
# Usage:
#   ./whale_world_terminus.sh          # Run Julia tests
#   ./whale_world_terminus.sh --verify # Generate verification data
#   ./whale_world_terminus.sh --ucm    # Run Unison tests (requires UCM)
#   ./whale_world_terminus.sh --all    # Run both Julia and Unison tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAY_JL_DIR="$(dirname "$SCRIPT_DIR")"
UNISON_DIR="$GAY_JL_DIR/../rio/unison-terminus"

# ANSI colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

print_header() {
    echo ""
    echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${CYAN}║${RESET}  ${BOLD}🐋 Whale World: Terminus Integration Test${RESET}                    ${BOLD}${CYAN}║${RESET}"
    echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════════════╝${RESET}"
    echo ""
}

run_julia_tests() {
    echo -e "${BOLD}Running Julia headless tests...${RESET}"
    cd "$GAY_JL_DIR"
    
    # Run tests with output capture
    julia --project=. -e '
        using Pkg
        Pkg.instantiate()
        include("test/whale_world_headless.jl")
    ' 2>&1 | while IFS= read -r line; do
        if [[ "$line" == *"Test Passed"* ]] || [[ "$line" == *"✓"* ]]; then
            echo -e "${GREEN}$line${RESET}"
        elif [[ "$line" == *"Test Failed"* ]] || [[ "$line" == *"✗"* ]]; then
            echo -e "${RED}$line${RESET}"
        elif [[ "$line" == *"@testset"* ]] || [[ "$line" == *"Test Summary"* ]]; then
            echo -e "${BOLD}$line${RESET}"
        else
            echo "$line"
        fi
    done
    
    echo ""
}

generate_verification() {
    echo -e "${BOLD}Generating verification data...${RESET}"
    cd "$GAY_JL_DIR"
    
    julia --project=. -e '
        include("test/whale_world_headless.jl")
        data = generate_verification_data()
        
        println("# Julia Whale World Verification Data")
        println("# GAY_SEED = 0x6761795f636f6c6f")
        println()
        
        println("seed: ", data["seed"])
        println("fingerprint: ", data["fingerprint"])
        println("n_triads: ", data["n_triads"])
        println()
        
        println("# Whale W001")
        w001 = data["whale_w001"]
        println("w001_seed: ", w001["seed"])
        println("w001_notes: ", join(w001["notes"], ","))
        println("w001_intervals: ", join(w001["intervals"], ","))
        println()
        
        println("# Colors (first 3)")
        for (i, c) in enumerate(w001["colors"][1:3])
            println("w001_color_", i, ": RGB(", c.r, ",", c.g, ",", c.b, ")")
        end
        println()
        
        println("# Gadget distribution")
        for (k, v) in data["gadget_distribution"]
            println("gadget_", k, ": ", v)
        end
        println()
        
        println("# Top 3 triads")
        for (i, t) in enumerate(data["top_3_triads"])
            println("triad_", i, ": ", join(t["whale_ids"], "+"), " ", t["gadget"], " coupling=", round(t["coupling"], digits=3))
        end
    '
}

run_unison_tests() {
    echo -e "${BOLD}Running Unison Terminus tests...${RESET}"
    
    if ! command -v ucm &> /dev/null; then
        echo -e "${YELLOW}UCM not found. Install Unison: https://www.unison-lang.org/install${RESET}"
        echo -e "${DIM}Skipping Unison tests${RESET}"
        return 1
    fi
    
    cd "$UNISON_DIR"
    
    # Run Unison tests
    echo -e "${DIM}Loading WhaleWorld.u...${RESET}"
    ucm transcript.fork <<EOF
.topos/main> load WhaleWorld.u
.topos/main> test.whaleWorld
EOF
}

run_terminus_protocol() {
    echo -e "${BOLD}Running Terminus JSON protocol test...${RESET}"
    cd "$GAY_JL_DIR"
    
    # Start Julia in Terminus protocol mode
    julia --project=. -e '
        include("test/whale_world_headless.jl")
        
        # Simulate Terminus commands
        commands = [
            """{"command": "init", "args": {"n_whales": 6, "seed": 7523094288207667311}}""",
            """{"command": "synergy", "args": {}}""",
            """{"command": "optimal", "args": {"k": 3}}""",
            """{"command": "spi", "args": {}}"""
        ]
        
        println("# Terminus Protocol Test")
        println()
        
        for cmd in commands
            println(">>> ", cmd)
            response = TerminusProtocol.process(cmd)
            println("<<< ", response)
            println()
        end
    '
}

spi_demo() {
    echo -e "${BOLD}Running SPI demonstration...${RESET}"
    cd "$GAY_JL_DIR"
    
    julia --project=. -e '
        include("test/whale_world_headless.jl")
        
        # Initialize and run SPI demo
        WhaleWorldProtocol.init_world(6, UInt64(0x6761795f636f6c6f))
        result = WhaleWorldProtocol.spi_demo()
        
        println()
        println("Strong Parallelism Invariance Verified: ", result.spi_verified)
        println("Number of triads: ", result.n_triads)
        println("World fingerprint: 0x", string(result.fingerprint, base=16))
        println()
        println("This proves: Sequential == Parallel == Reversed == Random Order")
    '
}

# Parse arguments
case "${1:-test}" in
    --verify)
        print_header
        generate_verification
        ;;
    --ucm)
        print_header
        run_unison_tests
        ;;
    --protocol)
        print_header
        run_terminus_protocol
        ;;
    --spi)
        print_header
        spi_demo
        ;;
    --all)
        print_header
        run_julia_tests
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo ""
        generate_verification
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo ""
        run_unison_tests
        ;;
    test|*)
        print_header
        run_julia_tests
        ;;
esac

echo ""
echo -e "${DIM}Done.${RESET}"
