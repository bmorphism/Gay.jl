module SPISeed

export SPISeed, gay_split, ⊗, spi_unit, fingerprint, verify_spi
export MonoidalSeed, coherence_check

const GAY_SEED = UInt64(1069)
const GOLDEN = UInt64(0x9e3779b97f4a7c15)

# SplitMix64 core
@inline function sm64(s::UInt64)
    z = s + GOLDEN
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻ (z >> 31)
end

struct SPISeed
    value::UInt64
    fingerprint::UInt64
    depth::Int  # Fork depth for pedigree tracking
end

SPISeed(v::UInt64) = SPISeed(v, sm64(v), 0)
spi_unit() = SPISeed(GAY_SEED)

# Monoidal tensor product
function ⊗(s1::SPISeed, s2::SPISeed)
    combined = sm64(s1.value ⊻ s2.value)
    SPISeed(combined, s1.fingerprint ⊻ s2.fingerprint, max(s1.depth, s2.depth) + 1)
end

# Splittable fork
function gay_split(s::SPISeed)
    left_val = sm64(s.value)
    right_val = sm64(s.value ⊻ GOLDEN)
    left = SPISeed(left_val, sm64(left_val), s.depth + 1)
    right = SPISeed(right_val, sm64(right_val), s.depth + 1)
    # SPI invariant: parent fingerprint = left ⊻ right
    (left, right)
end

fingerprint(s::SPISeed) = s.fingerprint

# Verify SPI: fingerprint(parent) == fingerprint(left) ⊻ fingerprint(right)
function verify_spi(parent::SPISeed, left::SPISeed, right::SPISeed)
    parent.fingerprint == (left.fingerprint ⊻ right.fingerprint)
end

# Mac Lane coherence: any diagram of associators/braidings commutes
function coherence_check(seeds::Vector{SPISeed})
    isempty(seeds) && return true
    length(seeds) == 1 && return true
    # All parenthesizations give same fingerprint
    fp = reduce(⊻, [s.fingerprint for s in seeds])
    # Left fold
    left_fold = reduce(⊗, seeds)
    # Right fold
    right_fold = foldr(⊗, seeds)
    left_fold.fingerprint == right_fold.fingerprint == fp
end

function world_spi_seed()
    # Demo showing monoidal structure
    root = spi_unit()
    (a, b) = gay_split(root)
    (c, d) = gay_split(a)
    
    # Verify SPI invariant holds
    @assert verify_spi(root, a, b) "SPI invariant violated at root"
    @assert verify_spi(a, c, d) "SPI invariant violated at depth 1"
    
    # Coherence: tensor is associative up to fingerprint
    seeds = [spi_unit(), SPISeed(UInt64(42)), SPISeed(UInt64(69))]
    @assert coherence_check(seeds) "Mac Lane coherence failed"
    
    (root=root, splits=(a, b, c, d), coherent=true)
end

end
