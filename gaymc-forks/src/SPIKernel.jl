# SPIKernel.jl - MINUS CONTRACTION: Minimal SPI Core
#
# CONSOLIDATION: Extract the 5x duplicated splitmix64 into ONE canonical source.
# All 4 forks (Plurigrid, TeglonLabs, Tritwies, bmorphism) import from here.
#
# 87 bytes of kernel. Everything else derives from this.

module SPIKernel

export sm64, skip, fnv1a, GAY_SEED, GOLDEN, fingerprint
export SPIState, next!, split!, color_at

# ═══════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS (cannot be compressed further)
# ═══════════════════════════════════════════════════════════════════════════

const GAY_SEED = 0x285508656870f24a  # fnv1a(69 underscores)
const GOLDEN   = 0x9e3779b97f4a7c15  # φ⁻¹ × 2⁶⁴
const MIX1     = 0xbf58476d1ce4e5b9
const MIX2     = 0x94d049bb133111eb
const FNV_OFFSET = 0xcbf29ce484222325
const FNV_PRIME  = 0x100000001b3

# GF(3) twists for triadic interleaving
const TWIST = (0x2d2d2d2d2d2d2d2d,  # MINUS
               0x5f5f5f5f5f5f5f5f,  # ERGODIC  
               0x2b2b2b2b2b2b2b2b)  # PLUS

# ═══════════════════════════════════════════════════════════════════════════
# MINIMAL KERNEL (3 functions, 9 LOC total)
# ═══════════════════════════════════════════════════════════════════════════

@inline function sm64(z::UInt64)::UInt64
    z = (z ⊻ (z >> 30)) * MIX1
    z = (z ⊻ (z >> 27)) * MIX2
    z ⊻ (z >> 31)
end

@inline step(s::UInt64)::UInt64 = s + GOLDEN
@inline skip(s::UInt64, n::Int)::UInt64 = s + n * GOLDEN

function fnv1a(s::AbstractString)::UInt64
    reduce((h, b) -> (h ⊻ b) * FNV_PRIME, codeunits(s); init=FNV_OFFSET)
end

# ═══════════════════════════════════════════════════════════════════════════
# COMPRESSED STATE (single mutable cell)
# ═══════════════════════════════════════════════════════════════════════════

mutable struct SPIState
    s::UInt64
end

SPIState() = SPIState(GAY_SEED)
SPIState(seed::Integer) = SPIState(UInt64(seed))

@inline function next!(rng::SPIState)::UInt64
    rng.s = step(rng.s)
    sm64(rng.s)
end

@inline function split!(rng::SPIState)::SPIState
    v1 = next!(rng)
    v2 = next!(rng)
    SPIState(v1 ⊻ v2)
end

# ═══════════════════════════════════════════════════════════════════════════
# MINIMAL FINGERPRINT (XOR-based, order-independent)
# ═══════════════════════════════════════════════════════════════════════════

@inline fingerprint(seed::UInt64, h::UInt64)::UInt64 = sm64(seed ⊻ h)
@inline fingerprint(contents::AbstractString...)::UInt64 = 
    reduce((fp, c) -> fp ⊻ sm64(fnv1a(c)), contents; init=GAY_SEED)

# ═══════════════════════════════════════════════════════════════════════════
# MINIMAL COLOR (3 values from state)
# ═══════════════════════════════════════════════════════════════════════════

function color_at(seed::UInt64, idx::Int)::NTuple{3,Float32}
    s = skip(seed, 3 * idx)
    h = Float32((sm64(step(s)) % 1000000) / 1000000.0 * 360.0 + 137.508) % 360
    s = step(s); sat = Float32(0.5 + (sm64(step(s)) % 1000000) / 1000000.0 * 0.4)
    s = step(s); lit = Float32(0.4 + (sm64(step(s)) % 1000000) / 1000000.0 * 0.3)
    (h, sat, lit)
end

# ═══════════════════════════════════════════════════════════════════════════
# TRIADIC INTERLEAVE (GF(3) polarity)
# ═══════════════════════════════════════════════════════════════════════════

polarity(seed::UInt64)::Int = Int(sm64(seed) % 3)
twist(seed::UInt64, pol::Int)::UInt64 = seed ⊻ TWIST[pol + 1]

end # module
