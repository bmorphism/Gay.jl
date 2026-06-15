module GayIdentifiers

# GayIdentifiers.jl — the Gay-flavoured member of tecosaur's *Identifiers.jl family.
# Parse/normalise an identifier (academic: DOI/ORCID/arXiv … + our schemes:
# world:// vm:// morphism:// handle), then attach the Gay layer: a deterministic
# colour + GF(3) trit from Gay.jl's mix64 kernel. A set of identifiers carries a
# GF(3) identity audit (the world://groupoid Σ).

using Gay: hash_color_hex, trit, stable_seed

export GayID, idkind, shortcode, purl, gaycolor, gaytrit, gay_audit

abstract type AbstractGayIdentifier end

struct GayID <: AbstractGayIdentifier
    raw::String
    kind::Symbol
    shortcode::String
end

# (kind, regex with one capture, canonicaliser, purl-prefix) — tecosaur-idiom
const RECOGNIZERS = Tuple{Symbol,Regex,Function,String}[
    (:doi,      r"(?:doi:|https?://doi\.org/)?(10\.\d{4,9}/\S+)"i, lowercase, "https://doi.org/"),
    (:orcid,    r"(?:https?://orcid\.org/)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])"i, uppercase, "https://orcid.org/"),
    (:arxiv,    r"(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)"i, identity, "https://arxiv.org/abs/"),
    (:world,    r"^world://(\S+)$", identity, "world://"),
    (:vm,       r"^vm://(\S+)$",    identity, "vm://"),
    (:morphism, r"^morphism://(\S+)$", identity, "morphism://"),
    (:handle,   r"^(@?[A-Za-z][\w.\-]+|\+\d{6,})$", identity, ""),
]

function Base.parse(::Type{GayID}, s::AbstractString)
    for (kind, re, canon, _) in RECOGNIZERS
        m = match(re, s)
        if m !== nothing
            cap = (length(m.captures) >= 1 && m.captures[1] !== nothing) ? m.captures[1] : m.match
            return GayID(String(s), kind, String(canon(cap)))
        end
    end
    GayID(String(s), :unknown, String(s))
end
Base.tryparse(::Type{GayID}, s::AbstractString) = parse(GayID, s)

idkind(g::GayID) = g.kind
shortcode(g::GayID) = g.shortcode
Base.string(g::GayID) = g.shortcode
function purl(g::GayID)
    for (kind, _, _, pre) in RECOGNIZERS
        kind == g.kind && return string(pre, g.shortcode)
    end
    g.shortcode
end

# --- the Gay layer: deterministic colour + GF(3) trit (Gay.jl mix64 kernel) -----
_gayseed(g::GayID) = stable_seed(g.shortcode)
gaycolor(g::GayID) = hash_color_hex(_gayseed(g), 0)
gaytrit(g::GayID)  = trit(0; seed = _gayseed(g))

# --- GF(3) identity audit over a set of identifiers (world://groupoid Σ) ---------
function gay_audit(ids)
    gs = [parse(GayID, x) for x in ids]
    s = mod(sum(Int(gaytrit(g)) for g in gs), 3)
    (; n = length(gs), sigma = s, balanced = (s == 0),
       colors = Dict(shortcode(g) => gaycolor(g) for g in gs))
end

end # module GayIdentifiers
