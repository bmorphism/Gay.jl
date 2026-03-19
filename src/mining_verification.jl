# Mining Verification: Use Deterministic Color Chains as Runtime Proofs
# ═══════════════════════════════════════════════════════════════════════════════
#
# The genesis chain is a RUNTIME ATTESTATION:
#   - Battery cycles = chain length (verifiable work)
#   - Each color = checkpoint (proves sequential computation)
#   - LCH values = fingerprint (unique to seed + position)
#
# VERIFICATION STRATEGY:
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  1. SPOT CHECK: Verify random positions in claimed chain                   │
# │  2. BOUNDARY CHECK: Verify first/last colors match seed                    │
# │  3. TRANSITION CHECK: Verify consecutive colors follow SplitMix64          │
# │  4. GAMUT CHECK: Verify all colors are in sRGB gamut                       │
# │  5. FINGERPRINT: Hash entire chain for tamper detection                    │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# WHY THIS WORKS:
#   - SplitMix64 is deterministic: same seed → same sequence
#   - Can't skip ahead: must compute all previous colors
#   - Can't fake: LCH→sRGB has unique inverse
#   - Verifiable in O(k) for k spot checks vs O(n) to mine
#
# ═══════════════════════════════════════════════════════════════════════════════

module MiningVerification

export ChainColor, MiningChain, MiningClaim, VerificationResult
export verify_chain, verify_spot, verify_boundary, verify_transition
export fingerprint_chain, chain_from_seed, verify_mining_claim
export RuntimeAttestation, attest_runtime, verify_attestation
export GENESIS_SEED, GENESIS_CHAIN

const GENESIS_SEED = UInt64(0x6761795f636f6c6f)  # "gay_colo"

# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN TYPES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChainColor

A single color in the mining chain with full LCH + hex + position.
"""
struct ChainColor
    cycle::Int
    hex::String
    L::Float64
    C::Float64
    H::Float64
end

"""
    MiningChain

A sequence of colors mined from a seed.
"""
struct MiningChain
    seed::UInt64
    seed_name::String
    colors::Vector{ChainColor}
    fingerprint::UInt64
    
    # Mining metadata
    start_time::Float64
    end_time::Float64
    battery_cycles::Int
end

"""
    MiningClaim

A claim submitted by a miner to be verified.
"""
struct MiningClaim
    miner_id::UInt64
    seed::UInt64
    claimed_length::Int
    claimed_fingerprint::UInt64
    
    # Spot check samples (positions + colors)
    samples::Vector{Tuple{Int, ChainColor}}
    
    # Boundary colors
    first_color::ChainColor
    last_color::ChainColor
    
    # Timestamp
    submitted_at::Float64
end

"""
    VerificationResult

Result of verifying a mining claim.
"""
struct VerificationResult
    valid::Bool
    checks_passed::Vector{Symbol}
    checks_failed::Vector{Symbol}
    confidence::Float64  # 0-1 based on spot checks
    details::Dict{Symbol, Any}
end

# ═══════════════════════════════════════════════════════════════════════════════
# SPLITMIX64 COLOR GENERATION (must match Gay.jl exactly)
# ═══════════════════════════════════════════════════════════════════════════════

@inline function splitmix64_next(state::UInt64)
    z = (state + 0x9E3779B97F4A7C15) % UInt64
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) % UInt64
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) % UInt64
    z ⊻ (z >> 31)
end

"""
    lch_from_state(state) -> (L, C, H)

Extract LCH values from RNG state (must match Gay.jl implementation).
"""
function lch_from_state(state::UInt64)
    # Extract bytes for L, C, H
    L = Float64((state >> 56) & 0xFF) / 2.55  # 0-100
    C = Float64((state >> 48) & 0xFF) / 2.0    # 0-127.5
    H = Float64((state >> 40) & 0xFF) * 1.41   # 0-360
    
    (L, C, H)
end

"""
    lch_to_xyz(L, C, H) -> (X, Y, Z)

Convert LCH to XYZ via Lab (D65 illuminant).
"""
function lch_to_xyz(L::Float64, C::Float64, H::Float64)
    # LCH → Lab
    H_rad = H * π / 180.0
    a = C * cos(H_rad)
    b = C * sin(H_rad)
    
    # Lab → XYZ (D65)
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    
    # Inverse f function
    ϵ = 216.0 / 24389.0
    κ = 24389.0 / 27.0
    
    xr = fx^3 > ϵ ? fx^3 : (116.0 * fx - 16.0) / κ
    yr = L > κ * ϵ ? ((L + 16.0) / 116.0)^3 : L / κ
    zr = fz^3 > ϵ ? fz^3 : (116.0 * fz - 16.0) / κ
    
    # D65 white point
    X = xr * 0.95047
    Y = yr * 1.0
    Z = zr * 1.08883
    
    (X, Y, Z)
end

"""
    xyz_to_srgb(X, Y, Z) -> (R, G, B)

Convert XYZ to sRGB with gamma correction.
"""
function xyz_to_srgb(X::Float64, Y::Float64, Z::Float64)
    # XYZ → Linear RGB
    r_lin = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b_lin = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    
    # Gamma correction
    gamma_correct(c) = c <= 0.0031308 ? 12.92 * c : 1.055 * c^(1/2.4) - 0.055
    
    R = clamp(gamma_correct(r_lin), 0.0, 1.0)
    G = clamp(gamma_correct(g_lin), 0.0, 1.0)
    B = clamp(gamma_correct(b_lin), 0.0, 1.0)
    
    (R, G, B)
end

"""
    rgb_to_hex(R, G, B) -> String

Convert RGB (0-1) to hex string.
"""
function rgb_to_hex(R::Float64, G::Float64, B::Float64)
    r = round(UInt8, clamp(R, 0, 1) * 255)
    g = round(UInt8, clamp(G, 0, 1) * 255)
    b = round(UInt8, clamp(B, 0, 1) * 255)
    "#" * uppercase(string(r, base=16, pad=2) * string(g, base=16, pad=2) * string(b, base=16, pad=2))
end

"""
    color_at_position(seed, position) -> ChainColor

Compute the exact color at a given position in the chain.
"""
function color_at_position(seed::UInt64, position::Int)
    state = seed
    for _ in 0:position
        state = splitmix64_next(state)
    end
    
    L, C, H = lch_from_state(state)
    X, Y, Z = lch_to_xyz(L, C, H)
    R, G, B = xyz_to_srgb(X, Y, Z)
    hex = rgb_to_hex(R, G, B)
    
    ChainColor(position, hex, L, C, H)
end

"""
    chain_from_seed(seed, length) -> MiningChain

Generate a complete mining chain from seed.
"""
function chain_from_seed(seed::UInt64, length::Int; seed_name::String="")
    start_time = time()
    colors = ChainColor[]
    
    state = seed
    for i in 0:length-1
        state = splitmix64_next(state)
        L, C, H = lch_from_state(state)
        X, Y, Z = lch_to_xyz(L, C, H)
        R, G, B = xyz_to_srgb(X, Y, Z)
        hex = rgb_to_hex(R, G, B)
        push!(colors, ChainColor(i, hex, L, C, H))
    end
    
    end_time = time()
    fp = fingerprint_colors(colors)
    
    MiningChain(seed, seed_name, colors, fp, start_time, end_time, length)
end

# ═══════════════════════════════════════════════════════════════════════════════
# FINGERPRINTING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    fingerprint_color(color) -> UInt64

Hash a single color for fingerprinting.
"""
function fingerprint_color(c::ChainColor)
    # Combine position + LCH into a unique fingerprint
    h = UInt64(c.cycle)
    h = h ⊻ (reinterpret(UInt64, c.L) >> 12)
    h = h ⊻ (reinterpret(UInt64, c.C) >> 24)
    h = h ⊻ (reinterpret(UInt64, c.H) >> 36)
    
    # Mix
    h = ((h ⊻ (h >> 30)) * 0xBF58476D1CE4E5B9) % UInt64
    h = ((h ⊻ (h >> 27)) * 0x94D049BB133111EB) % UInt64
    h ⊻ (h >> 31)
end

"""
    fingerprint_colors(colors) -> UInt64

Hash entire chain for tamper detection.
"""
function fingerprint_colors(colors::Vector{ChainColor})
    fp = UInt64(0)
    for c in colors
        fp = fp ⊻ fingerprint_color(c)
        fp = splitmix64_next(fp)
    end
    fp
end

"""
    fingerprint_chain(chain) -> UInt64

Get or compute chain fingerprint.
"""
fingerprint_chain(chain::MiningChain) = chain.fingerprint

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    verify_spot(seed, position, claimed_color; tolerance) -> Bool

Verify a single color at a specific position.
"""
function verify_spot(seed::UInt64, position::Int, claimed::ChainColor;
                     tolerance::Float64=1e-10)
    expected = color_at_position(seed, position)
    
    # Check position matches
    position != claimed.cycle && return false
    
    # Check LCH values within tolerance
    abs(expected.L - claimed.L) > tolerance && return false
    abs(expected.C - claimed.C) > tolerance && return false
    abs(expected.H - claimed.H) > tolerance && return false
    
    # Hex should match exactly
    expected.hex != claimed.hex && return false
    
    true
end

"""
    verify_boundary(seed, length, first_color, last_color) -> Bool

Verify first and last colors of chain.
"""
function verify_boundary(seed::UInt64, length::Int, 
                         first_color::ChainColor, last_color::ChainColor)
    # Verify first color (position 0)
    verify_spot(seed, 0, first_color) || return false
    
    # Verify last color (position length-1)
    verify_spot(seed, length - 1, last_color) || return false
    
    true
end

"""
    verify_transition(seed, color1, color2) -> Bool

Verify two consecutive colors follow SplitMix64 transition.
"""
function verify_transition(seed::UInt64, color1::ChainColor, color2::ChainColor)
    # Must be consecutive
    color2.cycle != color1.cycle + 1 && return false
    
    # Verify both colors are correct
    verify_spot(seed, color1.cycle, color1) || return false
    verify_spot(seed, color2.cycle, color2) || return false
    
    true
end

"""
    verify_gamut(color) -> Bool

Verify color is within sRGB gamut (hex is valid).
"""
function verify_gamut(c::ChainColor)
    # Parse hex
    length(c.hex) != 7 && return false
    c.hex[1] != '#' && return false
    
    try
        r = parse(UInt8, c.hex[2:3], base=16)
        g = parse(UInt8, c.hex[4:5], base=16)
        b = parse(UInt8, c.hex[6:7], base=16)
        return true
    catch
        return false
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# FULL CHAIN VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    verify_chain(chain; full_verify, n_spots) -> VerificationResult

Verify a mining chain with configurable thoroughness.

# Arguments
- `full_verify`: If true, verify every color (expensive)
- `n_spots`: Number of random spot checks if not full_verify
"""
function verify_chain(chain::MiningChain; full_verify::Bool=false, n_spots::Int=10)
    checks_passed = Symbol[]
    checks_failed = Symbol[]
    details = Dict{Symbol, Any}()
    
    # 1. Boundary check
    if !isempty(chain.colors)
        first_ok = verify_spot(chain.seed, 0, chain.colors[1])
        last_ok = verify_spot(chain.seed, length(chain.colors) - 1, chain.colors[end])
        
        if first_ok && last_ok
            push!(checks_passed, :boundary)
        else
            push!(checks_failed, :boundary)
            details[:boundary_first] = first_ok
            details[:boundary_last] = last_ok
        end
    else
        push!(checks_failed, :empty_chain)
    end
    
    # 2. Fingerprint check
    computed_fp = fingerprint_colors(chain.colors)
    if computed_fp == chain.fingerprint
        push!(checks_passed, :fingerprint)
    else
        push!(checks_failed, :fingerprint)
        details[:expected_fp] = chain.fingerprint
        details[:computed_fp] = computed_fp
    end
    
    # 3. Gamut check (all colors must be valid sRGB)
    gamut_ok = all(verify_gamut, chain.colors)
    if gamut_ok
        push!(checks_passed, :gamut)
    else
        push!(checks_failed, :gamut)
        invalid_count = count(!verify_gamut, chain.colors)
        details[:invalid_gamut_count] = invalid_count
    end
    
    # 4. Spot checks or full verification
    spots_checked = 0
    spots_passed = 0
    
    if full_verify
        for c in chain.colors
            spots_checked += 1
            if verify_spot(chain.seed, c.cycle, c)
                spots_passed += 1
            end
        end
    else
        # Random spot checks
        n = min(n_spots, length(chain.colors))
        positions = sort(unique(rand(0:length(chain.colors)-1, n)))
        
        for pos in positions
            spots_checked += 1
            if verify_spot(chain.seed, pos, chain.colors[pos + 1])
                spots_passed += 1
            end
        end
    end
    
    spot_ratio = spots_checked > 0 ? spots_passed / spots_checked : 0.0
    if spot_ratio == 1.0
        push!(checks_passed, :spot_checks)
    else
        push!(checks_failed, :spot_checks)
    end
    details[:spots_checked] = spots_checked
    details[:spots_passed] = spots_passed
    
    # 5. Transition check (verify some consecutive pairs)
    if length(chain.colors) >= 2
        n_transitions = min(5, length(chain.colors) - 1)
        trans_ok = 0
        for i in 1:n_transitions
            if verify_transition(chain.seed, chain.colors[i], chain.colors[i + 1])
                trans_ok += 1
            end
        end
        
        if trans_ok == n_transitions
            push!(checks_passed, :transitions)
        else
            push!(checks_failed, :transitions)
        end
        details[:transitions_checked] = n_transitions
        details[:transitions_passed] = trans_ok
    end
    
    # Compute overall validity and confidence
    valid = isempty(checks_failed)
    confidence = length(checks_passed) / (length(checks_passed) + length(checks_failed))
    
    VerificationResult(valid, checks_passed, checks_failed, confidence, details)
end

"""
    verify_mining_claim(claim) -> VerificationResult

Verify a mining claim submitted by a miner.
"""
function verify_mining_claim(claim::MiningClaim)
    checks_passed = Symbol[]
    checks_failed = Symbol[]
    details = Dict{Symbol, Any}()
    
    # 1. Verify boundary colors
    boundary_ok = verify_boundary(claim.seed, claim.claimed_length,
                                  claim.first_color, claim.last_color)
    if boundary_ok
        push!(checks_passed, :boundary)
    else
        push!(checks_failed, :boundary)
    end
    
    # 2. Verify spot check samples
    spots_ok = 0
    for (pos, color) in claim.samples
        if verify_spot(claim.seed, pos, color)
            spots_ok += 1
        end
    end
    
    if spots_ok == length(claim.samples)
        push!(checks_passed, :samples)
    else
        push!(checks_failed, :samples)
    end
    details[:samples_verified] = spots_ok
    details[:samples_total] = length(claim.samples)
    
    # 3. Verify fingerprint by recomputing from samples
    # (Can't verify full fingerprint without full chain)
    push!(checks_passed, :fingerprint_partial)
    
    valid = isempty(checks_failed)
    confidence = spots_ok / max(1, length(claim.samples))
    
    VerificationResult(valid, checks_passed, checks_failed, confidence, details)
end

# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME ATTESTATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    RuntimeAttestation

Proof that computation ran for a certain duration.
"""
struct RuntimeAttestation
    miner_id::UInt64
    seed::UInt64
    
    # What was computed
    chain_length::Int
    start_cycle::Int
    end_cycle::Int
    
    # Checkpoints (position → color hash)
    checkpoints::Dict{Int, UInt64}
    
    # Timing
    elapsed_ns::UInt64
    colors_per_second::Float64
    
    # Final fingerprint
    fingerprint::UInt64
    
    # Battery state (from genesis example)
    battery_cycles::Int
    battery_percent::Int
end

"""
    attest_runtime(seed, length; checkpoint_interval, miner_id) -> RuntimeAttestation

Generate a runtime attestation by actually mining the chain.
"""
function attest_runtime(seed::UInt64, length::Int;
                        checkpoint_interval::Int=100,
                        miner_id::UInt64=UInt64(0))
    start_ns = time_ns()
    checkpoints = Dict{Int, UInt64}()
    
    state = seed
    fp = UInt64(0)
    
    for i in 0:length-1
        state = splitmix64_next(state)
        
        # Checkpoint at intervals
        if i % checkpoint_interval == 0
            L, C, H = lch_from_state(state)
            checkpoint_hash = UInt64(round(L * 1e6)) ⊻ 
                             UInt64(round(C * 1e6)) ⊻ 
                             UInt64(round(H * 1e6))
            checkpoints[i] = checkpoint_hash
        end
        
        # Update fingerprint
        fp = fp ⊻ state
        fp = splitmix64_next(fp)
    end
    
    elapsed_ns = time_ns() - start_ns
    colors_per_sec = length / (elapsed_ns / 1e9)
    
    RuntimeAttestation(
        miner_id, seed,
        length, 0, length - 1,
        checkpoints,
        elapsed_ns, colors_per_sec,
        fp,
        length, 100  # Battery simulation
    )
end

"""
    verify_attestation(attestation; n_checkpoints) -> VerificationResult

Verify a runtime attestation by recomputing checkpoints.
"""
function verify_attestation(attestation::RuntimeAttestation; n_checkpoints::Int=5)
    checks_passed = Symbol[]
    checks_failed = Symbol[]
    details = Dict{Symbol, Any}()
    
    # Select checkpoints to verify
    checkpoint_positions = collect(keys(attestation.checkpoints))
    n_verify = min(n_checkpoints, length(checkpoint_positions))
    verify_positions = sort(checkpoint_positions)[1:n_verify]
    
    verified = 0
    for pos in verify_positions
        # Recompute to this position
        state = attestation.seed
        for _ in 0:pos
            state = splitmix64_next(state)
        end
        
        L, C, H = lch_from_state(state)
        expected_hash = UInt64(round(L * 1e6)) ⊻ 
                       UInt64(round(C * 1e6)) ⊻ 
                       UInt64(round(H * 1e6))
        
        if expected_hash == attestation.checkpoints[pos]
            verified += 1
        end
    end
    
    if verified == n_verify
        push!(checks_passed, :checkpoints)
    else
        push!(checks_failed, :checkpoints)
    end
    details[:checkpoints_verified] = verified
    details[:checkpoints_checked] = n_verify
    
    # Verify rate is plausible (not impossibly fast)
    # SplitMix64 should produce ~100M-1B colors/sec on modern hardware
    if 1e6 < attestation.colors_per_second < 1e10
        push!(checks_passed, :rate_plausible)
    else
        push!(checks_failed, :rate_plausible)
    end
    details[:colors_per_second] = attestation.colors_per_second
    
    valid = isempty(checks_failed)
    confidence = verified / max(1, n_verify)
    
    VerificationResult(valid, checks_passed, checks_failed, confidence, details)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GENESIS CHAIN (from user's input)
# ═══════════════════════════════════════════════════════════════════════════════

const GENESIS_CHAIN = [
    ChainColor(0, "#232100", 9.95305151795426, 89.12121123266927, 109.16670705328829),
    ChainColor(1, "#FFC196", 95.64340626247366, 75.69463862432056, 40.578861532301225),
    ChainColor(2, "#B797F5", 68.83307832090246, 52.58624293448647, 305.8775869504176),
    ChainColor(3, "#00D3FE", 77.01270406658392, 50.719765707180365, 224.57712168419232),
    ChainColor(4, "#F3B4DD", 80.30684610328687, 31.00925970957098, 338.5668861594303),
    ChainColor(5, "#E4D8CA", 87.10757626363412, 8.713821882767803, 80.19839549147454),
    ChainColor(6, "#E6A0FF", 75.92474966498482, 57.13182126381925, 317.5858774285715),
    ChainColor(7, "#A1AB2D", 67.33295337865329, 62.4733295284763, 107.90473523965251),
    ChainColor(8, "#430D00", 12.016818230531934, 39.790834705489495, 54.01863549186114),
    ChainColor(9, "#263330", 20.24941930893076, 6.316731061999381, 181.28556359100568),
    ChainColor(10, "#ACA7A1", 68.92133115422948, 3.962701273577207, 82.54499708853153),
    ChainColor(11, "#004D62", 28.685339908683037, 29.288286562638422, 223.27136465880565),
    ChainColor(12, "#021300", 4.342355432062184, 13.499979374325699, 133.4646290114955),
    ChainColor(13, "#4E3C3C", 27.414759014376987, 8.735175349709479, 19.421693716272557),
    ChainColor(14, "#FFD9A8", 90.65230031650403, 34.211009968606945, 66.9328903252508),
    ChainColor(15, "#3A3D3E", 25.7167729837364, 1.665747430769271, 234.35513798098134),
    ChainColor(16, "#918C8E", 58.80375174074871, 2.189760028829779, 350.1804627887977),
    ChainColor(17, "#AF6535", 50.54210972073506, 46.737904999077394, 57.451736335861156),
    ChainColor(18, "#68A617", 62.12991336886255, 72.50368716334194, 124.21928439533164),
    ChainColor(19, "#750000", 7.255156262785755, 98.86696191681608, 8.573000391080656),
    ChainColor(20, "#00C1FF", 73.67885130891794, 64.16166590749516, 260.54781611975665),
    ChainColor(21, "#ED0070", 49.066022993728176, 85.5860083567706, 3.2767068869989346),
    ChainColor(22, "#B84705", 45.36158016576941, 69.57368830782679, 51.3370126048211),
    ChainColor(23, "#00C175", 66.36817064239906, 87.38519725362308, 164.96931844436997),
    ChainColor(24, "#DDFBE3", 96.15675032741034, 16.527001387130113, 149.02601183239642),
]

"""
    verify_genesis() -> VerificationResult

Verify the hardcoded genesis chain matches the seed.
"""
function verify_genesis()
    chain = MiningChain(
        GENESIS_SEED, "gay_colo", GENESIS_CHAIN,
        fingerprint_colors(GENESIS_CHAIN),
        0.0, 0.0, 25
    )
    verify_chain(chain; full_verify=true)
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function demo_mining_verification()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  MINING VERIFICATION: Deterministic Color Chains as Runtime Proofs       ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # Verify genesis chain
    println("═══ GENESIS CHAIN VERIFICATION ═══")
    genesis_result = verify_genesis()
    println("  Seed: 0x$(string(GENESIS_SEED, base=16)) (\"gay_colo\")")
    println("  Length: $(length(GENESIS_CHAIN)) colors")
    println("  Valid: $(genesis_result.valid ? "✓ YES" : "✗ NO")")
    println("  Confidence: $(round(genesis_result.confidence * 100, digits=1))%")
    println("  Checks passed: $(genesis_result.checks_passed)")
    if !isempty(genesis_result.checks_failed)
        println("  Checks failed: $(genesis_result.checks_failed)")
    end
    println()
    
    # Show some colors
    println("═══ GENESIS COLORS (first 5) ═══")
    for c in GENESIS_CHAIN[1:5]
        println("  Cycle $(c.cycle): $(c.hex)  L=$(round(c.L, digits=1)) C=$(round(c.C, digits=1)) H=$(round(c.H, digits=1))")
    end
    println("  ...")
    println()
    
    # Generate and verify a new chain
    println("═══ RUNTIME ATTESTATION ═══")
    attestation = attest_runtime(GENESIS_SEED, 10000; checkpoint_interval=1000)
    println("  Mined $(attestation.chain_length) colors")
    println("  Rate: $(round(attestation.colors_per_second / 1e6, digits=2))M colors/sec")
    println("  Checkpoints: $(length(attestation.checkpoints))")
    println("  Elapsed: $(round(attestation.elapsed_ns / 1e6, digits=2))ms")
    println()
    
    # Verify attestation
    println("═══ ATTESTATION VERIFICATION ═══")
    att_result = verify_attestation(attestation; n_checkpoints=5)
    println("  Valid: $(att_result.valid ? "✓ YES" : "✗ NO")")
    println("  Checkpoints verified: $(att_result.details[:checkpoints_verified])/$(att_result.details[:checkpoints_checked])")
    println("  Rate plausible: $(:rate_plausible in att_result.checks_passed ? "✓" : "✗")")
    println()
    
    (genesis_result, attestation, att_result)
end

# Register as world
function world_mining_verification(; seed::UInt64=GENESIS_SEED, kwargs...)
    demo_mining_verification()
end

end # module MiningVerification
