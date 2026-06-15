# GayIdentifiers.jl

The Gay-flavoured member of tecosaur's `*Identifiers.jl` family
(`AcademicIdentifiers.jl` / `BioIdentifiers.jl` / `FastIdentifiers.jl`). It parses
and normalises an identifier, then attaches the **Gay layer**: a deterministic
colour + GF(3) trit drawn from `Gay.jl`'s `mix64` kernel.

## API (tecosaur idiom + Gay layer)
```julia
using GayIdentifiers
d = parse(GayID, "https://doi.org/10.1371/journal.pone.0068810")
idkind(d)     # :doi
shortcode(d)  # "10.1371/journal.pone.0068810"   (canonical)
purl(d)       # "https://doi.org/10.1371/journal.pone.0068810"
gaycolor(d)   # "#RRGGBB"   ← Gay.jl hash_color_hex(stable_seed(shortcode), 0)
gaytrit(d)    # Int8 ∈ {-1,0,1}
```
Equivalent inputs canonicalise to the same shortcode ⇒ the **same colour**
(`doi:…` ≡ `https://doi.org/…`); distinct identifiers ⇒ distinct colours.

## Recognised kinds
Academic (mirroring AcademicIdentifiers.jl): **doi, orcid, arxiv** — plus this
project's own schemes: **world:// , vm:// , morphism://** and bare **handle**
(Beeper/Signal-style). Unknown input → `:unknown` (still coloured).

## GF(3) identity audit (the world://groupoid Σ)
```julia
gay_audit(["world://securities","world://groupoid","world://morphism"])
# (; n=3, sigma, balanced, colors=Dict(shortcode => "#RRGGBB"))
```
`sigma = Σ gaytrit mod 3` over a set of identifiers — the scalar of the cross-
identifier identity groupoid (`world://groupoid`). Σ≡0 = balanced identity set.

## Dependency
Depends on **`Gay.jl`** by path (`[sources] Gay = {path = "../Gay.jl"}`), so the
colouring kernel is byte-identical to `world://securities` / `world://morphism`
(`hash_color_hex(1069,0) == #B35D38`). Lineage: tecosaur's identifier family ×
our Gay colour kernel × the GF(3) groupoid audit.

Test: `julia --project=. -e 'using Pkg; Pkg.test()'` — 17 assertions GREEN.
