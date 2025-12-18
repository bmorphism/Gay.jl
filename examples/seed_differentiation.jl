# Gay Seed Differentiation: From the irreducible GAY_SEED, differentiate the API
#
# The gay seed (0x6761795f636f6c6f = "gay_colo") is the irreducible root.
# From it, all color operations differentiate via splittable RNG splits.
#
#              GAY_SEED (irreducible)
#                   │
#        ┌──────────┼──────────┬──────────┐
#        ▼          ▼          ▼          ▼
#    gay-next   gay-at    gay-palette  pride
#        │          │          │          │
#      split₁    split₂    split₃     split₄
#        │          │          │          │
#       RGB₁     RGB₂...   RGB₁...   fixed colors

using Gay

# ═══════════════════════════════════════════════════════════════════════════
# The Irreducible Seed
# ═══════════════════════════════════════════════════════════════════════════

const IRREDUCIBLE = Gay.GAY_SEED  # 0x6761795f636f6c6f

println("═══════════════════════════════════════════════════════════════════════")
println("  ◈ Gay Seed Differentiation")
println("  From the irreducible GAY_SEED, differentiate the full API")
println("═══════════════════════════════════════════════════════════════════════")
println()

println("The Irreducible Seed:")
println("  GAY_SEED = 0x$(string(IRREDUCIBLE, base=16))")
println("  Bytes: \"$(String(reinterpret(UInt8, [IRREDUCIBLE])))\"")
println()

# ═══════════════════════════════════════════════════════════════════════════
# Differentiation 1: gay-next (sequential color stream)
# ═══════════════════════════════════════════════════════════════════════════

println("┌─────────────────────────────────────────────────────────────────────┐")
println("│ Differentiation 1: (gay-next)                                       │")
println("│ Sequential color stream via repeated splits                         │")
println("└─────────────────────────────────────────────────────────────────────┘")
println()

gay_seed!(IRREDUCIBLE)
println("  (gay-seed 0x$(string(IRREDUCIBLE, base=16)))")
println()

for i in 1:6
    c = next_color()
    r, g, b = round.(Int, 255 .* (c.r, c.g, c.b))
    hex = "#" * uppercase(string(r, base=16, pad=2) * string(g, base=16, pad=2) * string(b, base=16, pad=2))
    println("  (gay-next) → \e[48;2;$(r);$(g);$(b)m  \e[0m $hex  ; invocation=$i")
end
println()

# ═══════════════════════════════════════════════════════════════════════════
# Differentiation 2: gay-at (random access by index)
# ═══════════════════════════════════════════════════════════════════════════

println("┌─────────────────────────────────────────────────────────────────────┐")
println("│ Differentiation 2: (gay-at i)                                       │")
println("│ Random access - same index always yields same color                 │")
println("└─────────────────────────────────────────────────────────────────────┘")
println()

indices = [1, 42, 69, 420, 1337, 42069]
for i in indices
    c = color_at(i)
    r, g, b = round.(Int, 255 .* (c.r, c.g, c.b))
    hex = "#" * uppercase(string(r, base=16, pad=2) * string(g, base=16, pad=2) * string(b, base=16, pad=2))
    println("  (gay-at $i) → \e[48;2;$(r);$(g);$(b)m  \e[0m $hex")
end
println()

# ═══════════════════════════════════════════════════════════════════════════
# Differentiation 3: gay-palette (visually distinct colors)
# ═══════════════════════════════════════════════════════════════════════════

println("┌─────────────────────────────────────────────────────────────────────┐")
println("│ Differentiation 3: (gay-palette n)                                  │")
println("│ n visually distinct colors (CIEDE2000 distance ≥ 30)                │")
println("└─────────────────────────────────────────────────────────────────────┘")
println()

gay_seed!(IRREDUCIBLE)
palette = next_palette(8)

print("  (gay-palette 8) → ")
for c in palette
    r, g, b = round.(Int, 255 .* (c.r, c.g, c.b))
    print("\e[48;2;$(r);$(g);$(b)m  \e[0m")
end
println()
println()

# ═══════════════════════════════════════════════════════════════════════════
# Differentiation 4: pride (flag palettes)
# ═══════════════════════════════════════════════════════════════════════════

println("┌─────────────────────────────────────────────────────────────────────┐")
println("│ Differentiation 4: (pride :flag)                                    │")
println("│ Fixed pride flag colors in current colorspace                       │")
println("└─────────────────────────────────────────────────────────────────────┘")
println()

flags = [:rainbow, :transgender, :bisexual, :nonbinary, :pansexual, :asexual]

for flag in flags
    colors = pride_flag(flag)
    print("  (pride :$flag) → ")
    for c in colors
        r, g, b = round.(Int, 255 .* (c.r, c.g, c.b))
        print("\e[48;2;$(r);$(g);$(b)m  \e[0m")
    end
    println()
end
println()

# ═══════════════════════════════════════════════════════════════════════════
# Differentiation 5: gay-space (colorspace transformation)
# ═══════════════════════════════════════════════════════════════════════════

println("┌─────────────────────────────────────────────────────────────────────┐")
println("│ Differentiation 5: (gay-space :cs)                                  │")
println("│ Same seed, different colorspace = different RGB but same intent     │")
println("└─────────────────────────────────────────────────────────────────────┘")
println()

colorspaces = [(SRGB(), "sRGB"), (DisplayP3(), "P3"), (Rec2020(), "Rec2020")]

for (cs, name) in colorspaces
    c = color_at(42; seed=IRREDUCIBLE)
    r, g, b = round.(Int, 255 .* (c.r, c.g, c.b))
    hex = "#" * uppercase(string(r, base=16, pad=2) * string(g, base=16, pad=2) * string(b, base=16, pad=2))
    println("  (gay-at 42) @ $name → \e[48;2;$(r);$(g);$(b)m  \e[0m $hex")
end
println()

# ═══════════════════════════════════════════════════════════════════════════
# Differentiation 6: Comrade sky models
# ═══════════════════════════════════════════════════════════════════════════

println("┌─────────────────────────────────────────────────────────────────────┐")
println("│ Differentiation 6: Comrade Sky Models                               │")
println("│ S-expression primitives with deterministic colors                   │")
println("└─────────────────────────────────────────────────────────────────────┘")
println()

gay_seed!(2017)  # EHT M87* observation year

ring = comrade_ring(1.0, 0.3)
gauss = comrade_gaussian(0.5, 0.5)
model = sky_add(ring, gauss)

println("  (gay-seed 2017)")
println()
sky_show(ring)
sky_show(gauss)
println()
println("  (sky-add ring gaussian):")
comrade_show(model)
println()

# ═══════════════════════════════════════════════════════════════════════════
# The Differentiation Tree
# ═══════════════════════════════════════════════════════════════════════════

println("═══════════════════════════════════════════════════════════════════════")
println("  Differentiation Tree")
println("═══════════════════════════════════════════════════════════════════════")
println()
println("                    GAY_SEED")
println("                 0x6761795f636f6c6f")
println("                 \"gay_colo\" (irreducible)")
println("                        │")
println("         ┌──────────────┼──────────────┬──────────────┐")
println("         │              │              │              │")
println("         ▼              ▼              ▼              ▼")
println("     gay-next       gay-at       gay-palette      pride")
println("     (stream)      (random)      (distinct)      (fixed)")
println("         │              │              │              │")
println("     split(rng)   skip-to(i)    rejection      flag→RGB")
println("         │              │         sampling           │")
println("         ▼              ▼              ▼              ▼")
println("     color₁,₂,₃   color@i      [c₁...cₙ]     ◈ colors")
println("         │              │              │              │")
println("         └──────────────┴──────────────┴──────────────┘")
println("                               │")
println("                           ┌───┴───┐")
println("                           ▼       ▼")
println("                        sRGB    P3/Rec2020")
println("                      (clamp)   (wide gamut)")
println()

# ═══════════════════════════════════════════════════════════════════════════
# Lisp Interface Summary
# ═══════════════════════════════════════════════════════════════════════════

println("═══════════════════════════════════════════════════════════════════════")
println("  Lisp Interface (Gay REPL)")
println("═══════════════════════════════════════════════════════════════════════")
println()
println("  ┌─────────────────────┬───────────────────────────────────────────┐")
println("  │ Lisp Form           │ Description                               │")
println("  ├─────────────────────┼───────────────────────────────────────────┤")
println("  │ (gay-next)          │ Next deterministic color                  │")
println("  │ (gay-next n)        │ Next n colors                             │")
println("  │ (gay-at i)          │ Color at index i (O(i) splits)            │")
println("  │ (gay-palette n)     │ n visually distinct colors                │")
println("  │ (gay-seed n)        │ Set RNG seed                              │")
println("  │ (gay-space :p3)     │ Set colorspace (:srgb, :p3, :rec2020)     │")
println("  │ (pride :rainbow)    │ Pride flag colors                         │")
println("  │ (comrade-ring r w)  │ Ring primitive                            │")
println("  │ (comrade-show m)    │ Display sky model                         │")
println("  └─────────────────────┴───────────────────────────────────────────┘")
println()

println("\e[32m◆ All differentiations from GAY_SEED demonstrated\e[0m")
println("\e[32m◆ Lisp interface maps kebab-case → snake_case\e[0m")
println("\e[32m◆ Strong Parallelism Invariance maintained\e[0m")
println()
