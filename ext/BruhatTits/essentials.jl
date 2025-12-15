# Gay Essentials - The 3 Irreducible Primitives
#
# Everything in Gay.jl derives from these 3 operations.
# SEED is conventional (gauge choice), not essential.

module GayEssentials

export sm64, gf3, distinct3

# ============================================================================
# ESSENTIAL 1: sm64 (Mixing)
# ============================================================================

"""
    sm64(z::UInt64) -> UInt64

SplitMix64 mixing function. The fundamental dynamics of Gay.jl.
All color generation, fingerprinting, and random access derive from this.
"""
@inline function sm64(z::UInt64)::UInt64
    z += 0x9E3779B97F4A7C15  # Golden ratio
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    z ⊻ (z >> 31)
end

# ============================================================================
# ESSENTIAL 2: ⊻ (XOR Composition)
# ============================================================================

# XOR is built into Julia as ⊻
# It provides:
#   - Associativity: (a ⊻ b) ⊻ c = a ⊻ (b ⊻ c)
#   - Self-inverse: a ⊻ a = 0
#   - Identity: a ⊻ 0 = a
#   - Commutativity: a ⊻ b = b ⊻ a

# ============================================================================
# ESSENTIAL 3: GF(3) (3-Coloring Domain)
# ============================================================================

"""
    gf3(n) -> Int

Project to GF(3) = {0, 1, 2}.
The minimal NP-complete coloring domain.
"""
gf3(n::Integer) = mod(n, 3)

"""
    distinct3(a, b, c) -> Bool

Check if three values are pairwise distinct in GF(3).
This is the 3-coloring constraint.
"""
distinct3(a, b, c) = a ≠ b && b ≠ c && a ≠ c

# ============================================================================
# DERIVED (from essentials)
# ============================================================================

# Everything below is composite/derived, not essential.

"""
    fingerprint(seeds) -> UInt64

XOR composition of all seeds. Derived from ⊻.
"""
fingerprint(seeds) = reduce(⊻, seeds; init=UInt64(0))

"""
    color_at(n, seed) -> UInt64

Random access to n-th color. Derived from n × sm64.
"""
function color_at(n::Integer, seed::UInt64)
    state = seed
    for _ in 1:n
        state = sm64(state)
    end
    state
end

"""
    is_derangement(perm) -> Bool

Check if permutation has no fixed points. Derived from ≠.
"""
is_derangement(perm) = all(i -> perm[i] ≠ i, eachindex(perm))

end # module
