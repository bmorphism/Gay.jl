# tropical_semirings.jl - Multiple semiring variations with Gay.jl verification
#
# Semirings for different computational needs:
# - (min, +)  : Shortest path, scheduling, tropical geometry
# - (max, +)  : Longest path, critical path analysis  
# - (min, max): Fuzzy logic, capacity planning
# - (max, min): Bottleneck problems, widest path
# - (∨, ∧)    : Boolean reachability
# - (gcd, lcm): Number theory
#
# Each semiring gets a deterministic color signature for verification.

export AbstractSemiring, TropicalMinPlus, TropicalMaxPlus, TropicalMinMax
export TropicalMaxMin, BooleanSemiring, GcdLcmSemiring
export semiring_add, semiring_mul, semiring_zero, semiring_one
export semiring_color, verify_semiring_laws, SemiringMatcher
export match_semiring, run_tropical_regression

using Colors

# ═══════════════════════════════════════════════════════════════════════════
# Abstract Semiring Interface
# ═══════════════════════════════════════════════════════════════════════════

abstract type AbstractSemiring end

# Interface methods (to be implemented by each semiring)
semiring_add(::Type{S}, a, b) where S <: AbstractSemiring = error("Not implemented")
semiring_mul(::Type{S}, a, b) where S <: AbstractSemiring = error("Not implemented")
semiring_zero(::Type{S}) where S <: AbstractSemiring = error("Not implemented")
semiring_one(::Type{S}) where S <: AbstractSemiring = error("Not implemented")
semiring_name(::Type{S}) where S <: AbstractSemiring = string(S)

# ═══════════════════════════════════════════════════════════════════════════
# Tropical (min, +) Semiring - Shortest Path
# ═══════════════════════════════════════════════════════════════════════════

struct TropicalMinPlus <: AbstractSemiring end

semiring_add(::Type{TropicalMinPlus}, a::T, b::T) where T<:Real = min(a, b)
semiring_mul(::Type{TropicalMinPlus}, a::T, b::T) where T<:Real = a + b
semiring_zero(::Type{TropicalMinPlus}) = Inf
semiring_one(::Type{TropicalMinPlus}) = 0.0
semiring_name(::Type{TropicalMinPlus}) = "(min, +)"

# ═══════════════════════════════════════════════════════════════════════════
# Tropical (max, +) Semiring - Longest Path / Critical Path
# ═══════════════════════════════════════════════════════════════════════════

struct TropicalMaxPlus <: AbstractSemiring end

semiring_add(::Type{TropicalMaxPlus}, a::T, b::T) where T<:Real = max(a, b)
semiring_mul(::Type{TropicalMaxPlus}, a::T, b::T) where T<:Real = a + b
semiring_zero(::Type{TropicalMaxPlus}) = -Inf
semiring_one(::Type{TropicalMaxPlus}) = 0.0
semiring_name(::Type{TropicalMaxPlus}) = "(max, +)"

# ═══════════════════════════════════════════════════════════════════════════
# Fuzzy (min, max) Semiring - Capacity / Fuzzy Logic
# ═══════════════════════════════════════════════════════════════════════════

struct TropicalMinMax <: AbstractSemiring end

semiring_add(::Type{TropicalMinMax}, a::T, b::T) where T<:Real = min(a, b)
semiring_mul(::Type{TropicalMinMax}, a::T, b::T) where T<:Real = max(a, b)
semiring_zero(::Type{TropicalMinMax}) = Inf
semiring_one(::Type{TropicalMinMax}) = -Inf
semiring_name(::Type{TropicalMinMax}) = "(min, max)"

# ═══════════════════════════════════════════════════════════════════════════
# Bottleneck (max, min) Semiring - Widest Path
# ═══════════════════════════════════════════════════════════════════════════

struct TropicalMaxMin <: AbstractSemiring end

semiring_add(::Type{TropicalMaxMin}, a::T, b::T) where T<:Real = max(a, b)
semiring_mul(::Type{TropicalMaxMin}, a::T, b::T) where T<:Real = min(a, b)
semiring_zero(::Type{TropicalMaxMin}) = -Inf
semiring_one(::Type{TropicalMaxMin}) = Inf
semiring_name(::Type{TropicalMaxMin}) = "(max, min)"

# ═══════════════════════════════════════════════════════════════════════════
# Boolean (∨, ∧) Semiring - Reachability
# ═══════════════════════════════════════════════════════════════════════════

struct BooleanSemiring <: AbstractSemiring end

semiring_add(::Type{BooleanSemiring}, a::Bool, b::Bool) = a || b
semiring_mul(::Type{BooleanSemiring}, a::Bool, b::Bool) = a && b
semiring_zero(::Type{BooleanSemiring}) = false
semiring_one(::Type{BooleanSemiring}) = true
semiring_name(::Type{BooleanSemiring}) = "(∨, ∧)"

# ═══════════════════════════════════════════════════════════════════════════
# GCD/LCM Semiring - Number Theory
# ═══════════════════════════════════════════════════════════════════════════

struct GcdLcmSemiring <: AbstractSemiring end

semiring_add(::Type{GcdLcmSemiring}, a::Integer, b::Integer) = gcd(a, b)
semiring_mul(::Type{GcdLcmSemiring}, a::Integer, b::Integer) = lcm(a, b)
semiring_zero(::Type{GcdLcmSemiring}) = 0
semiring_one(::Type{GcdLcmSemiring}) = 1
semiring_name(::Type{GcdLcmSemiring}) = "(gcd, lcm)"

# ═══════════════════════════════════════════════════════════════════════════
# All Semiring Types
# ═══════════════════════════════════════════════════════════════════════════

const ALL_SEMIRINGS = [
    TropicalMinPlus,
    TropicalMaxPlus, 
    TropicalMinMax,
    TropicalMaxMin,
    BooleanSemiring,
    GcdLcmSemiring,
]

# ═══════════════════════════════════════════════════════════════════════════
# Gay.jl Color Signatures for Verification
# ═══════════════════════════════════════════════════════════════════════════

# Each semiring gets a deterministic color from its name hash
function semiring_seed(::Type{S}) where S <: AbstractSemiring
    name = semiring_name(S)
    h = UInt64(14695981039346656037)  # FNV-1a
    for c in name
        h = (h ⊻ UInt64(c)) * UInt64(1099511628211)
    end
    h
end

function semiring_color(::Type{S}) where S <: AbstractSemiring
    color_at(1; seed=semiring_seed(S))
end

# Color for a specific operation result
function operation_color(::Type{S}, op::Symbol, result, index::Int) where S <: AbstractSemiring
    base_seed = semiring_seed(S)
    op_offset = op == :add ? 1000 : op == :mul ? 2000 : 3000
    color_at(index + op_offset; seed=base_seed)
end

# ═══════════════════════════════════════════════════════════════════════════
# Semiring Law Verification with Colors
# ═══════════════════════════════════════════════════════════════════════════

struct SemiringLawResult
    law::Symbol
    passed::Bool
    color::RGB
    details::String
end

"""
Verify semiring laws for a given type with test values.
Returns colored results for each law.
"""
function verify_semiring_laws(::Type{S}, test_values::Vector{T}) where {S <: AbstractSemiring, T}
    results = SemiringLawResult[]
    seed = semiring_seed(S)
    
    zero = semiring_zero(S)
    one = semiring_one(S)
    
    a, b, c = if length(test_values) >= 3
        test_values[1], test_values[2], test_values[3]
    else
        error("Need at least 3 test values")
    end
    
    # Law 1: Additive associativity - (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
    lhs1 = semiring_add(S, semiring_add(S, a, b), c)
    rhs1 = semiring_add(S, a, semiring_add(S, b, c))
    pass1 = lhs1 == rhs1 || (isnan(lhs1) && isnan(rhs1)) || (isinf(lhs1) && isinf(rhs1) && sign(lhs1) == sign(rhs1))
    push!(results, SemiringLawResult(:add_assoc, pass1, color_at(1; seed=seed), 
        "(a⊕b)⊕c = a⊕(b⊕c): $lhs1 = $rhs1"))
    
    # Law 2: Additive commutativity - a ⊕ b = b ⊕ a
    lhs2 = semiring_add(S, a, b)
    rhs2 = semiring_add(S, b, a)
    pass2 = lhs2 == rhs2 || (isnan(lhs2) && isnan(rhs2))
    push!(results, SemiringLawResult(:add_comm, pass2, color_at(2; seed=seed),
        "a⊕b = b⊕a: $lhs2 = $rhs2"))
    
    # Law 3: Additive identity - a ⊕ 0 = a
    lhs3 = semiring_add(S, a, zero)
    pass3 = lhs3 == a || (isinf(a) && isinf(lhs3))
    push!(results, SemiringLawResult(:add_identity, pass3, color_at(3; seed=seed),
        "a⊕0 = a: $lhs3 = $a"))
    
    # Law 4: Multiplicative associativity - (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)
    lhs4 = semiring_mul(S, semiring_mul(S, a, b), c)
    rhs4 = semiring_mul(S, a, semiring_mul(S, b, c))
    pass4 = lhs4 == rhs4 || (isnan(lhs4) && isnan(rhs4)) || (isinf(lhs4) && isinf(rhs4))
    push!(results, SemiringLawResult(:mul_assoc, pass4, color_at(4; seed=seed),
        "(a⊗b)⊗c = a⊗(b⊗c): $lhs4 = $rhs4"))
    
    # Law 5: Multiplicative identity - a ⊗ 1 = a
    lhs5 = semiring_mul(S, a, one)
    pass5 = lhs5 == a || (isinf(a) && isinf(lhs5))
    push!(results, SemiringLawResult(:mul_identity, pass5, color_at(5; seed=seed),
        "a⊗1 = a: $lhs5 = $a"))
    
    # Law 6: Left distributivity - a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)
    lhs6 = semiring_mul(S, a, semiring_add(S, b, c))
    rhs6 = semiring_add(S, semiring_mul(S, a, b), semiring_mul(S, a, c))
    pass6 = lhs6 == rhs6 || (isnan(lhs6) && isnan(rhs6)) || (isinf(lhs6) && isinf(rhs6))
    push!(results, SemiringLawResult(:left_distrib, pass6, color_at(6; seed=seed),
        "a⊗(b⊕c) = (a⊗b)⊕(a⊗c): $lhs6 = $rhs6"))
    
    # Law 7: Right distributivity - (a ⊕ b) ⊗ c = (a ⊗ c) ⊕ (b ⊗ c)
    lhs7 = semiring_mul(S, semiring_add(S, a, b), c)
    rhs7 = semiring_add(S, semiring_mul(S, a, c), semiring_mul(S, b, c))
    pass7 = lhs7 == rhs7 || (isnan(lhs7) && isnan(rhs7)) || (isinf(lhs7) && isinf(rhs7))
    push!(results, SemiringLawResult(:right_distrib, pass7, color_at(7; seed=seed),
        "(a⊕b)⊗c = (a⊗c)⊕(b⊗c): $lhs7 = $rhs7"))
    
    # Law 8: Zero annihilation - a ⊗ 0 = 0 (for complete semirings)
    lhs8 = semiring_mul(S, a, zero)
    pass8 = lhs8 == zero || isinf(lhs8)  # Tropical semirings have Inf as zero
    push!(results, SemiringLawResult(:zero_annihil, pass8, color_at(8; seed=seed),
        "a⊗0 = 0: $lhs8 = $zero"))
    
    results
end

# ═══════════════════════════════════════════════════════════════════════════
# Semiring Matcher - Choose Right Semiring for Computation
# ═══════════════════════════════════════════════════════════════════════════

@enum ComputationType begin
    SHORTEST_PATH
    LONGEST_PATH
    CRITICAL_PATH
    CAPACITY_PLANNING
    BOTTLENECK
    REACHABILITY
    NUMBER_THEORY
    SCHEDULING
    FUZZY_LOGIC
end

const COMPUTATION_SEMIRING_MAP = Dict(
    SHORTEST_PATH => TropicalMinPlus,
    LONGEST_PATH => TropicalMaxPlus,
    CRITICAL_PATH => TropicalMaxPlus,
    CAPACITY_PLANNING => TropicalMinMax,
    BOTTLENECK => TropicalMaxMin,
    REACHABILITY => BooleanSemiring,
    NUMBER_THEORY => GcdLcmSemiring,
    SCHEDULING => TropicalMinPlus,
    FUZZY_LOGIC => TropicalMinMax,
)

"""
Match computation type to appropriate semiring.
"""
function match_semiring(comp::ComputationType)
    COMPUTATION_SEMIRING_MAP[comp]
end

struct SemiringMatcher
    computation::ComputationType
    semiring::Type{<:AbstractSemiring}
    color::RGB
    rationale::String
end

function SemiringMatcher(comp::ComputationType)
    S = match_semiring(comp)
    c = semiring_color(S)
    rationale = semiring_rationale(comp, S)
    SemiringMatcher(comp, S, c, rationale)
end

function semiring_rationale(comp::ComputationType, ::Type{S}) where S <: AbstractSemiring
    rationales = Dict(
        SHORTEST_PATH => "min finds shortest, + accumulates distances",
        LONGEST_PATH => "max finds longest, + accumulates durations", 
        CRITICAL_PATH => "max finds critical (longest) path in project network",
        CAPACITY_PLANNING => "min for bottleneck capacity, max for parallel capacity",
        BOTTLENECK => "max finds best option, min for weakest link in path",
        REACHABILITY => "∨ for any path exists, ∧ for both endpoints reachable",
        NUMBER_THEORY => "gcd for common factors, lcm for synchronization",
        SCHEDULING => "min for earliest finish time optimization",
        FUZZY_LOGIC => "min for conjunction, max for disjunction in fuzzy sets",
    )
    get(rationales, comp, "Standard semiring operations")
end

# ═══════════════════════════════════════════════════════════════════════════
# Tropical Regression Test Suite
# ═══════════════════════════════════════════════════════════════════════════

struct TropicalRegressionResult
    semiring::Type{<:AbstractSemiring}
    color::RGB
    law_results::Vector{SemiringLawResult}
    all_passed::Bool
    fingerprint::UInt64
end

"""
Run tropical regression tests across all semirings.
Each result is colored for visual verification.
"""
function run_tropical_regression(; verbose::Bool=true)
    results = TropicalRegressionResult[]
    
    # Test values for each semiring type
    real_vals = [1.0, 2.0, 3.0]
    bool_vals = [true, false, true]
    int_vals = [6, 10, 15]
    
    for S in ALL_SEMIRINGS
        test_vals = if S == BooleanSemiring
            bool_vals
        elseif S == GcdLcmSemiring
            int_vals
        else
            real_vals
        end
        
        law_results = verify_semiring_laws(S, test_vals)
        all_passed = all(r -> r.passed, law_results)
        base_color = semiring_color(S)
        
        # Fingerprint from color sequence
        fingerprint = UInt64(0)
        for (i, r) in enumerate(law_results)
            c = r.color
            rgb_val = UInt64(round(red(c)*255)) << 16 | 
                      UInt64(round(green(c)*255)) << 8 | 
                      UInt64(round(blue(c)*255))
            fingerprint = fingerprint ⊻ (rgb_val << (i * 8))
        end
        
        push!(results, TropicalRegressionResult(S, base_color, law_results, all_passed, fingerprint))
    end
    
    if verbose
        print_tropical_regression(results)
    end
    
    results
end

_hex(c::RGB) = begin
    r = round(Int, red(c) * 255)
    g = round(Int, green(c) * 255)
    b = round(Int, blue(c) * 255)
    uppercase("#$(lpad(string(r, base=16), 2, '0'))$(lpad(string(g, base=16), 2, '0'))$(lpad(string(b, base=16), 2, '0'))")
end

_ansi_bg(c::RGB) = begin
    r = round(Int, red(c) * 255)
    g = round(Int, green(c) * 255)
    b = round(Int, blue(c) * 255)
    "\e[48;2;$(r);$(g);$(b)m"
end

_ansi_fg(c::RGB) = begin
    r = round(Int, red(c) * 255)
    g = round(Int, green(c) * 255)
    b = round(Int, blue(c) * 255)
    "\e[38;2;$(r);$(g);$(b)m"
end

const _R = "\e[0m"
const _B = "\e[1m"
const _D = "\e[2m"

function print_tropical_regression(results::Vector{TropicalRegressionResult})
    println()
    println("$(_B)🌴 Tropical Semiring Regression Tests$(_R)")
    println("$(_D)Verifying semiring laws with Gay.jl color signatures$(_R)")
    println()
    
    total_passed = 0
    total_tests = 0
    
    for result in results
        name = semiring_name(result.semiring)
        status = result.all_passed ? "✓" : "✗"
        h = _hex(result.color)
        
        println("$(_ansi_bg(result.color))  $(_R) $(_B)$(name)$(_R) $h")
        
        for law in result.law_results
            total_tests += 1
            if law.passed
                total_passed += 1
                println("    $(_ansi_fg(law.color))✓$(_R) $(law.law)")
            else
                println("    $(_ansi_fg(law.color))✗$(_R) $(law.law): $(law.details)")
            end
        end
        
        println("    $(_D)Fingerprint: 0x$(string(result.fingerprint, base=16))$(_R)")
        println()
    end
    
    # Summary
    all_pass = total_passed == total_tests
    status_color = all_pass ? RGB(0.0, 1.0, 0.0) : RGB(1.0, 0.0, 0.0)
    
    println("$(_B)Summary:$(_R)")
    println("  $(_ansi_fg(status_color))$(total_passed)/$(total_tests) tests passed$(_R)")
    
    if all_pass
        println("  $(_ansi_fg(status_color))✓ All semiring laws verified$(_R)")
    else
        println("  $(_ansi_fg(status_color))✗ Some tests failed$(_R)")
    end
    
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Reference Values for Cross-Session Verification
# ═══════════════════════════════════════════════════════════════════════════

const TROPICAL_REFERENCE_FINGERPRINTS = Dict(
    TropicalMinPlus => 0x0000000000000000,  # To be filled after first run
    TropicalMaxPlus => 0x0000000000000000,
    TropicalMinMax => 0x0000000000000000,
    TropicalMaxMin => 0x0000000000000000,
    BooleanSemiring => 0x0000000000000000,
    GcdLcmSemiring => 0x0000000000000000,
)

"""
Verify fingerprints match reference values.
"""
function verify_tropical_fingerprints(results::Vector{TropicalRegressionResult})
    all_match = true
    for r in results
        ref = get(TROPICAL_REFERENCE_FINGERPRINTS, r.semiring, nothing)
        if ref !== nothing && ref != 0 && r.fingerprint != ref
            println("⚠ Fingerprint mismatch for $(semiring_name(r.semiring))")
            println("  Expected: 0x$(string(ref, base=16))")
            println("  Got:      0x$(string(r.fingerprint, base=16))")
            all_match = false
        end
    end
    all_match
end
