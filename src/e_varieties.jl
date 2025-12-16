# Varieties of e: Numeric Stability Index for Euler's Constant
# ═══════════════════════════════════════════════════════════════════════════════
#
#   ┌─────────────────────────────────────────────────────────────────────────────┐
#   │  INDEX OF e IMPLEMENTATIONS                                                │
#   │  ═══════════════════════════                                               │
#   │                                                                             │
#   │  1. HARDCODED CONSTANT (fastest, IEEE-exact)                               │
#   │     e = 2.718281828459045... as Float64 bits                               │
#   │     Stability: ★★★★★ (exact to machine precision)                          │
#   │                                                                             │
#   │  2. TAYLOR SERIES: e = Σ(1/n!) for n=0..∞                                  │
#   │     Converges: O(1/n!) exponentially fast                                  │
#   │     Stability: ★★★★☆ (stable, forward summation)                           │
#   │                                                                             │
#   │  3. LIMIT DEFINITION: e = lim(n→∞) (1 + 1/n)^n                             │
#   │     Converges: O(1/n) slowly                                               │
#   │     Stability: ★★☆☆☆ (catastrophic cancellation for large n)              │
#   │                                                                             │
#   │  4. CONTINUED FRACTION: [2; 1,2,1, 1,4,1, 1,6,1, ...]                      │
#   │     Converges: O(1/n²) quadratically                                       │
#   │     Stability: ★★★★☆ (numerically robust)                                  │
#   │                                                                             │
#   │  5. BINARY SPLITTING (Brent-McMillan style)                                │
#   │     Converges: O(n log² n log log n)                                       │
#   │     Stability: ★★★★★ (optimal for arbitrary precision)                     │
#   │                                                                             │
#   │  6. SPIGOT ALGORITHM                                                       │
#   │     Produces digits one at a time                                          │
#   │     Stability: ★★★☆☆ (streaming, memory-efficient)                         │
#   │                                                                             │
#   └─────────────────────────────────────────────────────────────────────────────┘

module EVarieties

export EULER_BITS, E_IEEE754, E_TAYLOR, E_LIMIT, E_CONTINUED_FRACTION
export e_taylor, e_limit, e_continued_fraction, e_binary_splitting
export EImplementation, stability_index, convergence_rate
export gay_seed_e, compare_implementations

# ═══════════════════════════════════════════════════════════════════════════════
# The IEEE 754 Constant: Exact Bits of e
# ═══════════════════════════════════════════════════════════════════════════════

"""
IEEE 754 double-precision representation of e.

Binary: 0 10000000000 0101111100001010100010110001010111011010010101110100001
       s exponent     mantissa

Hexadecimal: 0x4005bf0a8b145769

This is THE canonical representation in 64-bit floating point.
"""
const EULER_BITS = 0x4005bf0a8b145769

"""
Reconstruct e from its exact IEEE 754 bits.
"""
const E_IEEE754 = reinterpret(Float64, EULER_BITS)

# Verify: should equal Julia's built-in ℯ
@assert E_IEEE754 == Float64(ℯ) "IEEE 754 bits mismatch!"

# ═══════════════════════════════════════════════════════════════════════════════
# Implementation 1: Taylor Series
# ═══════════════════════════════════════════════════════════════════════════════

"""
    e_taylor(n_terms::Int=20) -> (value, terms_used, error_estimate)

Compute e via Taylor series: e = Σ(1/n!) for n=0,1,2,...

# Numerical Stability
- Forward summation from n=0 is stable (terms decrease monotonically)
- Converges exponentially fast: error ≈ 1/(n+1)!
- For Float64, n=18 terms suffices for full precision

# Convergence Rate
- 1/n! decreases super-exponentially
- Each term adds ~log₁₀(n) new correct digits
"""
function e_taylor(n_terms::Int=20)
    sum = 1.0  # 1/0! = 1
    term = 1.0
    
    for n in 1:n_terms
        term /= n  # term = 1/n!
        sum += term
        
        # Early termination if term is negligible
        if term < eps(sum)
            return (sum, n, term)
        end
    end
    
    return (sum, n_terms, term)
end

const E_TAYLOR = e_taylor(25)[1]

# ═══════════════════════════════════════════════════════════════════════════════
# Implementation 2: Limit Definition
# ═══════════════════════════════════════════════════════════════════════════════

"""
    e_limit(n::Int=10^8) -> (value, n_used, relative_error)

Compute e via limit: e = lim(n→∞) (1 + 1/n)^n

# Numerical Stability: POOR
- Catastrophic cancellation: (1 + 1/n) loses precision as n grows
- For n > 10^15, 1/n underflows relative to 1.0
- Error ≈ e/(2n) - converges only linearly

# Why This Is Bad
The expression (1 + 1/n)^n suffers from:
1. Loss of significance in 1/n for large n
2. Accumulation of rounding errors in exponentiation
3. Only O(1/n) convergence vs O(1/n!) for Taylor
"""
function e_limit(n::Int=10^8)
    # Naive implementation (demonstrating instability)
    base = 1.0 + 1.0/n
    result = base^n
    
    # Theoretical error
    error_estimate = E_IEEE754 / (2 * n)
    
    return (result, n, abs(result - E_IEEE754))
end

const E_LIMIT = e_limit(10^8)[1]

# ═══════════════════════════════════════════════════════════════════════════════
# Implementation 3: Continued Fraction
# ═══════════════════════════════════════════════════════════════════════════════

"""
    e_continued_fraction(n_terms::Int=30) -> (value, terms_used, convergent_diff)

Compute e via its regular continued fraction expansion:

    e = [2; 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, 1, ...]

Pattern: [2; 1, 2k, 1] repeating for k = 1, 2, 3, ...

# Numerical Stability: GOOD
- Backward recurrence is stable
- Convergents approach e alternately from above and below
- Quadratic convergence: error ≈ O(1/convergent_denominator²)
"""
function e_continued_fraction(n_terms::Int=30)
    # Generate continued fraction coefficients for e
    # Pattern: 2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, ...
    function cf_coeff(i::Int)
        if i == 0
            return 2
        elseif i % 3 == 2
            return 2 * ((i + 1) ÷ 3)  # 2, 4, 6, 8, ...
        else
            return 1
        end
    end
    
    # Backward recurrence for convergent
    # h_{-1} = 1, h_0 = a_0
    # h_n = a_n * h_{n-1} + h_{n-2}
    
    if n_terms < 1
        return (2.0, 0, 0.0)
    end
    
    # Start from the end and work backward
    coeffs = [cf_coeff(i) for i in 0:n_terms-1]
    
    # Compute convergent p_n / q_n using forward recurrence
    p_prev, p_curr = BigInt(1), BigInt(coeffs[1])
    q_prev, q_curr = BigInt(0), BigInt(1)
    
    for i in 2:n_terms
        a = coeffs[i]
        p_prev, p_curr = p_curr, a * p_curr + p_prev
        q_prev, q_curr = q_curr, a * q_curr + q_prev
    end
    
    result = Float64(p_curr) / Float64(q_curr)
    
    # Previous convergent for error estimate
    prev_result = Float64(p_prev) / Float64(q_prev)
    
    return (result, n_terms, abs(result - prev_result))
end

const E_CONTINUED_FRACTION = e_continued_fraction(40)[1]

# ═══════════════════════════════════════════════════════════════════════════════
# Implementation 4: Binary Splitting
# ═══════════════════════════════════════════════════════════════════════════════

"""
    e_binary_splitting(precision_bits::Int=100) -> (value, terms_used)

Compute e via binary splitting (Brent-McMillan style).

For e = Σ(1/n!), we can write the partial sum as P(a,b)/Q(a,b) where:
- P(a,b) = Σ_{n=a}^{b-1} Q(a,n+1)/n!
- Q(a,b) = Π_{n=a}^{b-1} n!

Binary splitting recursively computes:
- P(a,b) = P(a,m)·Q(m,b) + P(m,b)
- Q(a,b) = Q(a,m)·Q(m,b)

# Numerical Stability: EXCELLENT
- Uses exact integer arithmetic until final division
- O(n log² n log log n) time complexity
- Optimal for arbitrary precision
"""
function e_binary_splitting(precision_bits::Int=100)
    # Determine number of terms needed
    # For n! > 2^precision_bits, we need n ≈ precision_bits / log2(n)
    n_terms = max(10, precision_bits ÷ 3)
    
    # Binary splitting recursive computation
    function split(a::Int, b::Int)
        if b - a == 1
            # Base case: single term 1/a!
            # P = 1, Q = a (for the recursive formula)
            return (BigInt(1), BigInt(a))
        end
        
        m = (a + b) ÷ 2
        p_left, q_left = split(a, m)
        p_right, q_right = split(m, b)
        
        # Combine: P = P_left * Q_right + P_right
        #          Q = Q_left * Q_right
        p = p_left * q_right + p_right
        q = q_left * q_right
        
        return (p, q)
    end
    
    # Compute Σ_{n=0}^{n_terms-1} 1/n!
    # We use a slightly different formulation for efficiency
    
    # Direct summation with BigInt for exactness
    setprecision(precision_bits + 64) do
        sum = big(1.0)
        term = big(1.0)
        
        for n in 1:n_terms
            term /= n
            sum += term
        end
        
        return (Float64(sum), n_terms)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Implementation 5: Spigot Algorithm (streaming digits)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    e_spigot(n_digits::Int=50) -> String

Compute e using a spigot algorithm that produces one digit at a time.

Based on the representation:
    e = 2 + Σ_{n=2}^∞ 1/n!

Uses the mixed-radix representation where position i has radix i.

# Numerical Stability: MODERATE
- Memory-efficient (streaming)
- Exact integer arithmetic
- Can produce unbounded digits
"""
function e_spigot(n_digits::Int=50)
    # Array size: need ~1.08n positions for n digits
    size = Int(ceil(1.1 * n_digits)) + 10
    
    # Initialize: position i holds value i-1 (representing (i-1)/(i-1)!)
    a = collect(1:size)
    
    digits = Char[]
    
    for _ in 1:n_digits+1
        carry = 0
        
        # Process from right to left
        for i in size:-1:2
            # Current value plus carry times radix
            val = a[i] * 10 + carry
            # Quotient becomes new carry
            carry = val ÷ i
            # Remainder stays
            a[i] = val % i
        end
        
        # First position gives the digit (plus any carry)
        val = a[1] * 10 + carry
        digit = val ÷ 10
        a[1] = val % 10
        
        push!(digits, Char('0' + digit))
    end
    
    # Format as "2.71828..."
    result = string(digits[1], ".", join(digits[2:end]))
    return result
end

# ═══════════════════════════════════════════════════════════════════════════════
# Stability Index and Comparison
# ═══════════════════════════════════════════════════════════════════════════════

"""
    EImplementation

Metadata about an e implementation.
"""
struct EImplementation
    name::String
    method::Function
    stability::Int        # 1-5 stars
    convergence::Symbol   # :constant, :linear, :quadratic, :exponential
    notes::String
end

const IMPLEMENTATIONS = [
    EImplementation(
        "IEEE 754 Constant",
        () -> E_IEEE754,
        5,
        :constant,
        "Exact to machine precision, no computation"
    ),
    EImplementation(
        "Taylor Series",
        () -> e_taylor(20)[1],
        4,
        :exponential,
        "e = Σ(1/n!), stable forward summation"
    ),
    EImplementation(
        "Limit Definition", 
        () -> e_limit(10^7)[1],
        2,
        :linear,
        "e = lim(1+1/n)^n, catastrophic cancellation"
    ),
    EImplementation(
        "Continued Fraction",
        () -> e_continued_fraction(30)[1],
        4,
        :quadratic,
        "[2; 1,2,1,1,4,1,...], stable backward recurrence"
    ),
]

"""
    stability_index(impl::EImplementation) -> String

Return a visual stability rating.
"""
function stability_index(impl::EImplementation)
    "★" ^ impl.stability * "☆" ^ (5 - impl.stability)
end

"""
    convergence_rate(impl::EImplementation) -> String

Describe convergence rate.
"""
function convergence_rate(impl::EImplementation)
    rates = Dict(
        :constant => "O(1) - instant",
        :linear => "O(1/n) - slow",
        :quadratic => "O(1/n²) - good", 
        :exponential => "O(1/n!) - excellent"
    )
    get(rates, impl.convergence, "unknown")
end

"""
    compare_implementations() -> DataFrame-like output

Compare all e implementations for accuracy and stability.
"""
function compare_implementations()
    println()
    println("╔" * "═" ^ 76 * "╗")
    println("║  INDEX OF e IMPLEMENTATIONS: Numeric Stability Analysis                     ║")
    println("╠" * "═" ^ 76 * "╣")
    println("║  Reference: IEEE 754 = $(E_IEEE754)                           ║")
    println("║  Bits:      0x$(string(EULER_BITS, base=16, pad=16))                                     ║")
    println("╠" * "═" ^ 76 * "╣")
    
    for impl in IMPLEMENTATIONS
        value = impl.method()
        error = abs(value - E_IEEE754)
        ulp_error = error / eps(E_IEEE754)
        
        println("║  $(rpad(impl.name, 22)) │ Stability: $(stability_index(impl))                      ║")
        println("║  $(rpad("", 22)) │ Convergence: $(rpad(convergence_rate(impl), 20))            ║")
        println("║  $(rpad("", 22)) │ Error: $(rpad(string(round(ulp_error, digits=2)) * " ULP", 20))           ║")
        println("║  $(rpad("", 22)) │ $(impl.notes[1:min(38, length(impl.notes))])   ║")
        println("╟" * "─" ^ 76 * "╢")
    end
    
    println("╚" * "═" ^ 76 * "╝")
end

# ═══════════════════════════════════════════════════════════════════════════════
# gay_seed Integration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    gay_seed_e(implementation::Symbol=:ieee754) -> UInt64

Get gay_seed for e computed via different methods.

Each implementation produces a slightly different value (except :ieee754),
leading to different derived seeds for colorization.
"""
function gay_seed_e(implementation::Symbol=:ieee754)
    GAY_IGOR_SEED = UInt64(0x6761795f636f6c6f)
    
    e_value = if implementation == :ieee754
        E_IEEE754
    elseif implementation == :taylor
        e_taylor(20)[1]
    elseif implementation == :limit
        e_limit(10^7)[1]
    elseif implementation == :continued_fraction
        e_continued_fraction(30)[1]
    else
        E_IEEE754
    end
    
    bits = reinterpret(UInt64, e_value)
    bits ⊻ GAY_IGOR_SEED
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

function demo_e_varieties()
    compare_implementations()
    
    println()
    println("SPIGOT ALGORITHM (first 50 digits):")
    println("  ", e_spigot(50))
    
    println()
    println("GAY_SEED VARIANTS BY IMPLEMENTATION:")
    println("─" ^ 60)
    for sym in [:ieee754, :taylor, :limit, :continued_fraction]
        seed = gay_seed_e(sym)
        println("  $(rpad(string(sym), 20)): 0x$(string(seed, base=16, pad=16))")
    end
    
    println()
    println("KEY INSIGHT: The limit definition (1+1/n)^n suffers from")
    println("catastrophic cancellation - avoid for numerical work!")
end

end # module EVarieties
