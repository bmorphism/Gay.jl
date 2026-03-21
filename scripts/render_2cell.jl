#!/usr/bin/env julia
# Render the Tree Functor 2-cell diagram in terminal with ANSI true color
# Uses SplitMix64 hash_color for deterministic coloring from wavelength seeds
# Standalone — no Gay.jl dependency needed (avoids HTTP precompile issue)

# SplitMix64 finalizer (same as Gay.jl's splitmix64)
function splitmix64(x::UInt64)
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    x ⊻ (x >> 31)
end

function hash_color(seed::UInt64, index::UInt64)
    h = splitmix64(seed ⊻ splitmix64(index))
    r = Float32((h & 0xFFFF) / 65535.0)
    g = Float32(((h >> 16) & 0xFFFF) / 65535.0)
    b = Float32(((h >> 32) & 0xFFFF) / 65535.0)
    (r, g, b)
end

# ANSI helpers
ansi(r,g,b) = "\e[38;2;$(r);$(g);$(b)m"
bg(r,g,b) = "\e[48;2;$(r);$(g);$(b)m"
const R = "\e[0m"
const B = "\e[1m"
const D = "\e[2m"

# Colors seeded from wavelengths
to256(x) = round(Int, clamp(x, 0, 1) * 255)
function ac(seed::UInt64, idx::UInt64)
    r, g, b = hash_color(seed, idx)
    ansi(to256(r), to256(g), to256(b))
end

# Cherenkov/bioluminescence palette
cb = ac(UInt64(470), UInt64(1))   # bio
cm = ac(UInt64(470), UInt64(2))   # math
cg = ac(UInt64(470), UInt64(3))   # GF(3)
cl = ac(UInt64(420), UInt64(1))   # LUCA (deep blue-violet)
ce = ac(UInt64(420), UInt64(2))   # Euler
cf = ac(UInt64(420), UInt64(3))   # Fourier
ct = ac(UInt64(470), UInt64(7))   # tau (2-cell)

# Fixed trit colors
tm = "\e[38;2;100;140;255m-1$(R)"
tz = "\e[38;2;180;180;180m 0$(R)"
tp = "\e[38;2;255;120;100m+1$(R)"

println()
println("$(B)  ╔══════════════════════════════════════════════════════════════╗$(R)")
println("$(B)  ║     Tree Functor T : Bio → Math  (2-cell over GF(3))       ║$(R)")
println("$(B)  ╚══════════════════════════════════════════════════════════════╝$(R)")
println()

# Main 2-cell square
println("  $(cb)Δᵒᵖ$(R) ─────$(cb)F_bio$(R)─────▶ $(cb)Bio$(R)")
println("   │                        │")
println("   │    $(ct)╔═══╗$(R)               │")
println("   │    $(ct)║ τ ║$(R)  $(ct)⇒  T$(R)         │ $(cb)Trait$(R)")
println("   │    $(ct)╚═╤═╝$(R)               │")
println("  $(cm)U$(R) │      $(ct)⇓$(R)                │")
println("   │                        │")
println("   ▼                        ▼")
println("  $(cm)Δᵒᵖ$(R) ────$(cm)F_math$(R)────▶ $(cm)Math$(R)")
println("                      \\")
println("                $(cm)trit_m$(R) \\  $(cb)trit_b$(R)")
println("                        ▼")
println("                     $(cg)GF(3)$(R)")
println()

# Horizontal pasting
println("$(B)  ── Horizontal Pasting (Cherenkov ~ Bioluminescence) ──$(R)")
println()
println("  $(cl)LUCA$(R) ═══$(ct)τ₁$(R)═══▶ $(ce)Euler$(R) ═══$(ct)τ₂$(R)═══▶ $(cf)Fourier$(R)")
println("   │  $(cl)FeS$(R)           │  $(ce)energy$(R)         │  $(cf)spectral$(R)")
println("   │  $(cl)cluster$(R)       │  $(ce)methods$(R)        │  $(cf)decomp.$(R)")
println("   ▼                 ▼                  ▼")
println("  $(cl)470nm$(R) ◀──peak──▶ $(ce)1/λ²$(R) ◀──spectrum──▶ $(cf)f\u0302(ν)$(R)")
println("  $(D)(biolum.)          (Cherenkov)         (transform)$(R)")
println()

# Trit conservation
println("$(B)  ── Trit Conservation at Each Node ──$(R)")
println()
println("  $(cl)LUCA$(R)           $(ce)Euler$(R)          $(cf)Fourier$(R)")
println("  ┌──┬──┬──┐    ┌──┬──┬──┐    ┌──┬──┬──┐")
println("  │$(tm)│$(tz)│$(tp)│    │$(tp)│$(tm)│$(tz)│    │$(tz)│$(tp)│$(tm)│")
println("  └──┴──┴──┘    └──┴──┴──┘    └──┴──┴──┘")
println("   Σ≡0 mod 3    Σ≡0 mod 3    Σ≡0 mod 3")
println()

# Parallel trees
println("$(B)  ── Parallel Trees ──$(R)")
println()
println("         $(cl)●$(R) LUCA                    $(ce)●$(R) Reinhold")
println("        ╱ ╲                         ╱ ╲")
println("       ╱   ╲                       ╱   ╲")
println("     $(cl)●$(R)     $(cl)●$(R)  Archaea           $(ce)●$(R)     $(ce)●$(R)  Rheticus")
println("     │     │  Bacteria            │     │  Clavius")
println("    ╱╲   ╱╲                     ╱╲   ╱╲")
println("   $(cl)●$(R)  $(cl)●$(R) $(cl)●$(R)  $(cl)●$(R)                $(ce)●$(R)  $(ce)●$(R) $(ce)●$(R)  $(ce)●$(R)")
println("   │       │                   │       │")
println("   ⋮       ⋮                   ⋮       ⋮")
println("   $(cl)●$(R)       $(cl)●$(R)  luciferase      $(ce)●$(R)       $(ce)●$(R)  Fourier")
println("  $(D) (Aequorea) (dinoflag.)      (Lagrange) (Euler)$(R)")
println("    \\       ╱                   \\       ╱")
println("     ▼     ▼                     ▼     ▼")
println("     $(cf)470nm$(R)                      $(cf)∫eⁱωᵗ$(R)")
println()

# The punchline
println("$(B)  T$(R)(emit light) = decompose light")
println("$(B)  T$(R)(FeS → flavin → luciferin) = (Euler → Lagrange → Fourier)")
println("$(B)  T$(R)(conservation of energy) = conservation of proof validity")
println()
println("  $(D)\"Both paths: undifferentiated energy → structured transfer → visible spectrum\"$(R)")
println()
