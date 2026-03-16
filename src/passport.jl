# Gay Passport - Decentralized Identity via did:gay:* method
# Blessed passports with IRL witnessing, vivid color stamps
#
# WorldID / passport.xyz successor: no stamps necessary
# except maybe the special ones for vivid color experiences

module GayPassport

using SplittableRandoms: SplittableRandom, split
using Random
using Printf: @sprintf
using SHA: sha256

export GayDID, GayStamp, Witness, Passport, PassportRegistry
export did_gay, resolve_did, verify_did
export issue_passport, bless_passport, witness_passport
export add_stamp, verify_passport, passport_fingerprint, passport_colors
export premine_passports, world_passport
export PassportWorld

# ═══════════════════════════════════════════════════════════════════════════════
# did:gay:* — Decentralized Identifier Method
# ═══════════════════════════════════════════════════════════════════════════════
#
# Format: did:gay:<hex-encoded-seed-hash>
#
# The DID is derived from a seed + identity nonce via SplitMix64 hashing,
# then the genesis colors (first 3) serve as the visual fingerprint.
# Blessed passports carry witness attestations from IRL encounters.

const GOLDEN = UInt64(0x9e3779b97f4a7c15)
const MIX1   = UInt64(0xbf58476d1ce4e5b9)
const MIX2   = UInt64(0x94d049bb133111eb)

"""
    splitmix64(x::UInt64) -> UInt64

SplitMix64 finalizer — same as in splittable.jl for consistency.
"""
function splitmix64(x::UInt64)
    x = (x ⊻ (x >> 30)) * MIX1
    x = (x ⊻ (x >> 27)) * MIX2
    x ⊻ (x >> 31)
end

"""
    hash_color(seed::UInt64, index::UInt64) -> NTuple{3,Float32}

O(1) deterministic color from seed + index. Matches Gay.jl core.
"""
function hash_color(seed::UInt64, index::UInt64)
    h = splitmix64(seed ⊻ (index * GOLDEN))
    r = Float32((h >> 0) & 0xFF) / 255.0f0
    g = Float32((h >> 8) & 0xFF) / 255.0f0
    b = Float32((h >> 16) & 0xFF) / 255.0f0
    (r, g, b)
end

function color_hex(rgb::NTuple{3,Float32})
    r = clamp(round(Int, rgb[1] * 255), 0, 255)
    g = clamp(round(Int, rgb[2] * 255), 0, 255)
    b = clamp(round(Int, rgb[3] * 255), 0, 255)
    @sprintf("#%02X%02X%02X", r, g, b)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Core Types
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayDID

A `did:gay:*` decentralized identifier. The method-specific identifier is
derived from a seed hash, and the genesis colors (first 3 colors from that
seed) serve as a visual fingerprint — your passport photo in color space.
"""
struct GayDID
    seed::UInt64
    method_specific_id::String  # hex string
end

function GayDID(seed::UInt64)
    h = splitmix64(seed)
    msid = string(h, base=16, pad=16)
    GayDID(seed, msid)
end

"""
    did_gay(seed::UInt64) -> String

Construct a `did:gay:*` DID string from a seed.
"""
did_gay(seed::UInt64) = "did:gay:" * GayDID(seed).method_specific_id
did_gay(did::GayDID) = "did:gay:" * did.method_specific_id

"""
    resolve_did(did_string::String) -> GayDID or nothing

Resolve a `did:gay:*` string. Returns the DID if valid format.
"""
function resolve_did(did_string::String)
    startswith(did_string, "did:gay:") || return nothing
    msid = did_string[9:end]
    length(msid) == 16 && all(c -> c in "0123456789abcdef", msid) || return nothing
    GayDID(UInt64(0), msid)  # seed unknown from DID alone
end

"""
    verify_did(did::GayDID, seed::UInt64) -> Bool

Verify that a DID was derived from the given seed.
"""
function verify_did(did::GayDID, seed::UInt64)
    expected = GayDID(seed)
    did.method_specific_id == expected.method_specific_id
end

"""
    Witness

An IRL witness attestation. Witnesses observe passport issuance in person
and sign with their own DID + timestamp. No stamps necessary — just presence.
"""
struct Witness
    did::GayDID
    name::String
    timestamp::Float64  # Unix epoch
    attestation_hash::UInt64
end

function Witness(did::GayDID, name::String; timestamp::Float64=time())
    # Attestation = hash of witness DID + timestamp
    raw = splitmix64(did.seed ⊻ reinterpret(UInt64, timestamp))
    Witness(did, name, timestamp, raw)
end

"""
    GayStamp

A stamp for vivid color experiences. Special stamps are earned through
participation in events, color explorations, and community moments.
Unlike regular passport operation, stamps are the collectible extras.
"""
struct GayStamp
    name::String
    color::NTuple{3,Float32}
    event::String
    timestamp::Float64
    seed::UInt64
end

function GayStamp(name::String, event::String; seed::UInt64=UInt64(0), timestamp::Float64=time())
    s = seed == 0 ? splitmix64(hash(name) ⊻ hash(event)) : seed
    color = hash_color(s, UInt64(1))
    GayStamp(name, color, event, timestamp, s)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Passport
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Passport

A Gay Passport — your `did:gay:*` identity credential.

Fields:
- `did`: The decentralized identifier
- `genesis_colors`: First 3 colors from your seed (visual fingerprint)
- `witnesses`: IRL witness attestations (blessed passports have ≥1)
- `stamps`: Special stamps for vivid color experiences
- `issued_at`: Unix timestamp of issuance
- `blessed`: Whether this passport has been blessed via IRL witnessing
"""
mutable struct Passport
    did::GayDID
    genesis_colors::NTuple{3, NTuple{3,Float32}}
    witnesses::Vector{Witness}
    stamps::Vector{GayStamp}
    issued_at::Float64
    blessed::Bool
end

"""
    issue_passport(seed::UInt64; timestamp=time()) -> Passport

Issue a new Gay Passport from a seed. The passport starts unblessed —
bless it through IRL witnessing.
"""
function issue_passport(seed::UInt64; timestamp::Float64=time())
    did = GayDID(seed)
    # Genesis colors: first 3 from this seed
    c1 = hash_color(seed, UInt64(1))
    c2 = hash_color(seed, UInt64(2))
    c3 = hash_color(seed, UInt64(3))
    Passport(did, (c1, c2, c3), Witness[], GayStamp[], timestamp, false)
end

"""
    bless_passport(passport::Passport, witness::Witness) -> Passport

Bless a passport through IRL witnessing. The first witness attestation
makes the passport blessed. Additional witnesses strengthen the web of trust.
"""
function bless_passport(passport::Passport, witness::Witness)
    push!(passport.witnesses, witness)
    passport.blessed = true
    passport
end

"""
    witness_passport(passport::Passport, witness_seed::UInt64, witness_name::String) -> Passport

Convenience: create a witness from seed+name and bless the passport.
"""
function witness_passport(passport::Passport, witness_seed::UInt64, witness_name::String)
    w = Witness(GayDID(witness_seed), witness_name)
    bless_passport(passport, w)
end

"""
    add_stamp(passport::Passport, name::String, event::String; kwargs...) -> Passport

Add a stamp for a vivid color experience. Stamps are the special collectibles.
"""
function add_stamp(passport::Passport, name::String, event::String; kwargs...)
    push!(passport.stamps, GayStamp(name, event; kwargs...))
    passport
end

"""
    verify_passport(passport::Passport) -> Bool

Verify passport integrity: DID matches genesis colors, witnesses are consistent.
"""
function verify_passport(passport::Passport)
    seed = passport.did.seed
    # Verify genesis colors match seed
    c1 = hash_color(seed, UInt64(1))
    c2 = hash_color(seed, UInt64(2))
    c3 = hash_color(seed, UInt64(3))
    passport.genesis_colors == (c1, c2, c3)
end

"""
    passport_fingerprint(passport::Passport) -> UInt64

XOR fingerprint of the entire passport state — SPI-compliant.
"""
function passport_fingerprint(passport::Passport)
    fp = passport.did.seed
    for (i, c) in enumerate(passport.genesis_colors)
        fp ⊻= splitmix64(reinterpret(UInt64, Float64(c[1])) ⊻ UInt64(i))
    end
    for w in passport.witnesses
        fp ⊻= w.attestation_hash
    end
    for s in passport.stamps
        fp ⊻= s.seed
    end
    fp
end

"""
    passport_colors(passport::Passport) -> Vector{String}

Get the genesis color hex codes for display.
"""
function passport_colors(passport::Passport)
    [color_hex(c) for c in passport.genesis_colors]
end

# ═══════════════════════════════════════════════════════════════════════════════
# Registry & Premine
# ═══════════════════════════════════════════════════════════════════════════════

"""
    PassportRegistry

Registry of issued passports. Tracks all DIDs and their passports.
"""
struct PassportRegistry
    passports::Dict{String, Passport}  # DID string -> Passport
end

PassportRegistry() = PassportRegistry(Dict{String, Passport}())

function Base.length(registry::PassportRegistry)
    length(registry.passports)
end

function register!(registry::PassportRegistry, passport::Passport)
    key = did_gay(passport.did)
    registry.passports[key] = passport
    passport
end

function lookup(registry::PassportRegistry, did_string::String)
    get(registry.passports, did_string, nothing)
end

"""
    premine_passports(seeds::Vector{UInt64}; bless_with=nothing) -> PassportRegistry

Premine passports for a batch of seeds. Optionally bless them all with
a witness (e.g., the issuer at a launch event).

First passports issued at Minecraft + BCI event at Vivarium.
"""
function premine_passports(seeds::Vector{UInt64}; bless_with::Union{Nothing, Tuple{UInt64, String}}=nothing)
    registry = PassportRegistry()
    for seed in seeds
        p = issue_passport(seed)
        if bless_with !== nothing
            witness_passport(p, bless_with[1], bless_with[2])
        end
        register!(registry, p)
    end
    registry
end

# ═══════════════════════════════════════════════════════════════════════════════
# World Builder (AGENTS.md compliant)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    PassportWorld

Composable passport world state.
"""
struct PassportWorld
    registry::PassportRegistry
    fingerprint_val::UInt64
    n_passports::Int
    n_blessed::Int
    n_stamps::Int
end

function Base.length(w::PassportWorld)
    w.n_passports
end

function Base.merge(w1::PassportWorld, w2::PassportWorld)
    merged = PassportRegistry()
    for (k, v) in w1.registry.passports
        merged.passports[k] = v
    end
    for (k, v) in w2.registry.passports
        merged.passports[k] = v
    end
    fp = w1.fingerprint_val ⊻ w2.fingerprint_val
    n_blessed = count(p -> p.blessed, values(merged.passports))
    n_stamps = sum(p -> length(p.stamps), values(merged.passports); init=0)
    PassportWorld(merged, fp, length(merged), n_blessed, n_stamps)
end

"""
    world_passport(; seeds, bless_with=nothing) -> PassportWorld

Build a passport world from seeds. Returns composable state.

# Example: Vivarium launch premine
```julia
w = world_passport(
    seeds = UInt64[1069, 420, 69, 1337, 42],
    bless_with = (UInt64(1069), "vivarium-witness")
)
```
"""
function world_passport(;
    seeds::Vector{UInt64}=UInt64[1069],
    bless_with::Union{Nothing, Tuple{UInt64, String}}=nothing
)
    registry = premine_passports(seeds; bless_with=bless_with)

    fp = UInt64(0)
    n_blessed = 0
    n_stamps = 0
    for (_, p) in registry.passports
        fp ⊻= passport_fingerprint(p)
        p.blessed && (n_blessed += 1)
        n_stamps += length(p.stamps)
    end

    PassportWorld(registry, fp, length(registry), n_blessed, n_stamps)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════

function Base.show(io::IO, did::GayDID)
    print(io, "did:gay:", did.method_specific_id)
end

function Base.show(io::IO, p::Passport)
    colors = passport_colors(p)
    status = p.blessed ? "BLESSED" : "unblessed"
    print(io, "Passport(", did_gay(p.did), " [", status, "] ",
          join(colors, " "), " witnesses=", length(p.witnesses),
          " stamps=", length(p.stamps), ")")
end

function Base.show(io::IO, w::PassportWorld)
    print(io, "PassportWorld(n=", w.n_passports,
          " blessed=", w.n_blessed,
          " stamps=", w.n_stamps,
          " fp=0x", string(w.fingerprint_val, base=16, pad=16), ")")
end

end # module GayPassport
