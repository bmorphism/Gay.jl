"""
    ZMod3 — GF(3) as a proper algebraic type

The discovery: Gay.jl's trit system IS ZMod 3. Instead of ad-hoc
`+1, 0, -1` with hand-rolled addition tables, use modular arithmetic.

This gives us for free:
- Commutativity, associativity, identity, inverses
- `char_three`: t + t + t ≡ 0 (mod 3)
- `balanced_iff`: a + b + c = 0 ↔ c = -(a + b)
- Möbius inversion = negation
- Padovan period conservation

Verified in Lean 4: see lean4/gf3_elegant.lean
"""
module ZMod3Module

export ZMod3, PLUS, ERGODIC, MINUS
export balanced, moebius, char_three
export padovan_mod3_period, padovan_residues

"""
    ZMod3

The field with three elements. Wraps a UInt8 ∈ {0, 1, 2}.
Maps: PLUS=1, ERGODIC=0, MINUS=2 (since -1 ≡ 2 mod 3).
"""
struct ZMod3
    val::UInt8
    ZMod3(n::Integer) = new(UInt8(mod(n, 3)))
end

const PLUS    = ZMod3(1)
const ERGODIC = ZMod3(0)
const MINUS   = ZMod3(2)  # -1 mod 3 = 2

# Ring operations
Base.:+(a::ZMod3, b::ZMod3) = ZMod3(a.val + b.val)
Base.:-(a::ZMod3) = ZMod3(3 - a.val)  # negation
Base.:-(a::ZMod3, b::ZMod3) = a + (-b)
Base.:*(a::ZMod3, b::ZMod3) = ZMod3(a.val * b.val)
Base.zero(::Type{ZMod3}) = ERGODIC
Base.one(::Type{ZMod3}) = PLUS
Base.:(==)(a::ZMod3, b::ZMod3) = a.val == b.val
Base.hash(a::ZMod3, h::UInt) = hash(a.val, h)

# Display
function Base.show(io::IO, t::ZMod3)
    names = Dict(UInt8(0) => "○", UInt8(1) => "+", UInt8(2) => "−")
    print(io, get(names, t.val, "?"))
end

"""
    balanced(a, b, c) -> Bool

Check if three trits form a balanced triad: a + b + c ≡ 0 (mod 3).
This is THE conservation law of Gay.jl.
"""
balanced(a::ZMod3, b::ZMod3, c::ZMod3) = (a + b + c) == ERGODIC

"""
    moebius(t) -> ZMod3

Möbius inversion on GF(3). Since μ(3) = -1 (3 is prime),
Möbius inversion IS negation. This is the "flavor of color" map.
"""
moebius(t::ZMod3) = -t

"""
    char_three(t) -> ZMod3

Characteristic 3: t + t + t = 0 for all t ∈ GF(3).
This is why three copies of any trit balance.
"""
char_three(t::ZMod3) = t + t + t  # always returns ERGODIC

# ═══ PADOVAN ═══

"""
    padovan_mod3_period() -> Vector{ZMod3}

The Padovan sequence mod 3 has period 13.
P(n) = P(n-2) + P(n-3), residues cycle through:
[1, 1, 1, 2, 2, 0, 1, 2, 1, 0, 0, 1, 0]

Properties (verified in Lean 4):
- Sum over period ≡ 0 (mod 3) — Noether conservation
- Net flow ≡ -1 (mod 3) — compression bias
- e (=1) appears 6/13 times — dominates (non-Boolean middle)
- ⊥ (=0) appears 4/13, ⊤ (=2) appears 3/13 — asymmetric
- 7/13 < 2/3 — strict Bumpus-Kocsis
"""
function padovan_mod3_period()
    ZMod3.([1, 1, 1, 2, 2, 0, 1, 2, 1, 0, 0, 1, 0])
end

"""
    padovan_residues(n) -> Vector{ZMod3}

First n Padovan numbers mod 3.
"""
function padovan_residues(n::Int)
    p = [1, 1, 1]
    for i in 4:n
        push!(p, p[end-1] + p[end-2])
    end
    ZMod3.(p[1:n])
end

# ═══ CONVERSION FROM LEGACY TRIT ═══

"""
    ZMod3(trit::Int) — convert legacy +1/0/-1 trit to ZMod3
"""
function from_legacy_trit(t::Int)
    @assert t ∈ (-1, 0, 1) "Legacy trit must be -1, 0, or +1"
    ZMod3(t)
end

"""
    to_legacy_trit(t::ZMod3) -> Int — convert back to +1/0/-1
"""
function to_legacy_trit(t::ZMod3)
    t == PLUS && return 1
    t == ERGODIC && return 0
    return -1  # MINUS
end

end # module ZMod3Module
