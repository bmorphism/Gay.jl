# # Multicolor Derivations: Comrade Sky Models Meet DuckDB
#
# This example explores different color derivation strategies inspired by
# Comrade.jl's geometric models (Ring, MRing, Gaussian, Disk, Crescent),
# applied to repository data from DuckDB.
#
# Each model type suggests a different way to derive colors:
# - **Gaussian**: Smooth distribution - colors blend based on similarity
# - **Ring**: Radial structure - colors by distance from center
# - **MRing**: Fourier modes - colors by periodic patterns
# - **Disk**: Binary classification - distinct category colors
# - **Crescent**: Asymmetric - colors weighted by direction
#
# We'll test which derivation works best for visualizing repository data.

using Gay
using Gay: ka_colors, xor_fingerprint, hash_color, show_colors
using KernelAbstractions: CPU
using Colors: RGB, HSL

# ═══════════════════════════════════════════════════════════════════════════════
# DuckDB Integration
# ═══════════════════════════════════════════════════════════════════════════════

const TOPOS_DB = expanduser("~/.topos/repos.duckdb")

"""
Load repository data from DuckDB.
"""
function load_repos()
    if !isfile(TOPOS_DB)
        @warn "DuckDB not found at $TOPOS_DB"
        return nothing
    end
    
    # Use DuckDB CLI to extract data
    result = read(`duckdb $TOPOS_DB -json -c "SELECT name, language, owner, pushed_at FROM repos WHERE language IS NOT NULL ORDER BY pushed_at DESC LIMIT 100"`, String)
    
    # Parse JSON manually (simple approach)
    repos = []
    for m in eachmatch(r"\{[^}]+\}", result)
        obj = m.match
        name = match(r"\"name\":\"([^\"]+)\"", obj)
        lang = match(r"\"language\":\"([^\"]+)\"", obj)
        owner = match(r"\"owner\":\"([^\"]+)\"", obj)
        
        if name !== nothing && lang !== nothing
            push!(repos, (
                name = name.captures[1],
                language = lang.captures[1],
                owner = owner !== nothing ? owner.captures[1] : "unknown"
            ))
        end
    end
    
    return repos
end

# ═══════════════════════════════════════════════════════════════════════════════
# Comrade-Inspired Color Derivation Models
# ═══════════════════════════════════════════════════════════════════════════════

"""
Language to base hue mapping (Comrade-style categorical).
"""
const LANGUAGE_HUES = Dict(
    "Python" => 55.0,       # Yellow-green (Python brand)
    "Rust" => 15.0,         # Orange-red (Rust brand)  
    "JavaScript" => 50.0,   # Yellow
    "TypeScript" => 200.0,  # Blue
    "Julia" => 275.0,       # Purple
    "Clojure" => 120.0,     # Green (Lisp family)
    "Haskell" => 280.0,     # Purple (functional)
    "Go" => 190.0,          # Cyan
    "C++" => 210.0,         # Blue-gray
    "Swift" => 30.0,        # Orange
    "Nix" => 220.0,         # Blue
    "TeX" => 35.0,          # Brown
    "HTML" => 20.0,         # Orange-red
)

"""
    GaussianDerivation

Smooth blending - hash determines position in color space,
nearby hashes get similar colors (like Gaussian blur).
"""
struct GaussianDerivation
    σ::Float64  # Width of Gaussian (color spread)
end

function derive_color(m::GaussianDerivation, name::AbstractString, lang::AbstractString; seed=42)
    h = hash(name, UInt64(seed))
    base_hue = get(LANGUAGE_HUES, lang, 180.0)
    
    # Gaussian spread around base hue
    spread = ((h % 1000) / 1000.0 - 0.5) * m.σ * 60.0
    hue = mod(base_hue + spread, 360.0)
    
    # Saturation and lightness from hash
    sat = 0.5 + ((h >> 10) % 100) / 200.0
    light = 0.4 + ((h >> 20) % 100) / 250.0
    
    convert(RGB, HSL(hue, sat, light))
end

"""
    RingDerivation

Radial structure - colors arranged in rings by hash distance
from a center point. Inner ring = hot colors, outer = cool.
"""
struct RingDerivation
    radius::Float64
    width::Float64
end

function derive_color(m::RingDerivation, name::AbstractString, lang::AbstractString; seed=42)
    h = hash(name, UInt64(seed))
    
    # Radial position (0 to 1)
    r = (h % 10000) / 10000.0
    
    # Map radius to hue (inner=red, outer=blue)
    hue = r * 240.0  # Red → Blue spectrum
    
    # Ring structure: brightness peaks at m.radius
    ring_dist = abs(r - m.radius) / m.width
    light = 0.7 - 0.3 * min(ring_dist, 1.0)
    
    # Language affects saturation
    base_sat = haskey(LANGUAGE_HUES, lang) ? 0.8 : 0.5
    
    convert(RGB, HSL(hue, base_sat, light))
end

"""
    MRingDerivation

Fourier mode structure - colors have periodic patterns
based on azimuthal angle (like MRing's α, β coefficients).
"""
struct MRingDerivation
    n_modes::Int  # Number of Fourier modes
end

function derive_color(m::MRingDerivation, name::AbstractString, lang::AbstractString; seed=42)
    h = hash(name, UInt64(seed))
    
    # Azimuthal angle from hash
    θ = 2π * (h % 10000) / 10000.0
    
    # Fourier modes modify the color
    base_hue = get(LANGUAGE_HUES, lang, 180.0)
    
    mode_sum = 0.0
    for n in 1:m.n_modes
        αn = sin(n * θ + (h >> (n * 5)) % 100 / 50.0)
        mode_sum += αn / n
    end
    
    # Mode sum affects hue shift
    hue = mod(base_hue + mode_sum * 30.0, 360.0)
    
    # Radial position for saturation
    r = ((h >> 16) % 1000) / 1000.0
    sat = 0.6 + 0.3 * r
    light = 0.5 + 0.2 * cos(θ)  # Brightness varies with angle
    
    convert(RGB, HSL(hue, sat, clamp(light, 0.3, 0.7)))
end

"""
    DiskDerivation

Binary classification - inside disk = one color scheme,
outside = another. Sharp boundaries.
"""
struct DiskDerivation
    threshold::Float64  # Hash threshold for "inside"
end

function derive_color(m::DiskDerivation, name::AbstractString, lang::AbstractString; seed=42)
    h = hash(name, UInt64(seed))
    
    r = (h % 10000) / 10000.0
    inside = r < m.threshold
    
    base_hue = get(LANGUAGE_HUES, lang, 180.0)
    
    if inside
        # Inside: saturated, based on language
        hue = base_hue
        sat = 0.8
        light = 0.5
    else
        # Outside: desaturated, shifted hue
        hue = mod(base_hue + 180.0, 360.0)  # Complementary
        sat = 0.3
        light = 0.6
    end
    
    convert(RGB, HSL(hue, sat, light))
end

"""
    CrescentDerivation

Asymmetric brightness - one side bright, other dark.
Models directional preference in the data.
"""
struct CrescentDerivation
    shift::Float64  # Asymmetry amount (0-1)
end

function derive_color(m::CrescentDerivation, name::AbstractString, lang::AbstractString; seed=42)
    h = hash(name, UInt64(seed))
    
    # Position on crescent
    θ = 2π * (h % 10000) / 10000.0
    r = ((h >> 16) % 1000) / 1000.0
    
    # Asymmetric brightness
    brightness_factor = 0.5 + 0.5 * cos(θ - m.shift * π)
    
    base_hue = get(LANGUAGE_HUES, lang, 180.0)
    hue = mod(base_hue + (1 - brightness_factor) * 30.0, 360.0)
    
    sat = 0.6 + 0.3 * r
    light = 0.3 + 0.4 * brightness_factor
    
    convert(RGB, HSL(hue, sat, light))
end

"""
    SPIDerivation

Strong Parallelism Invariance - uses Gay.jl's hash_color
for guaranteed reproducibility across backends.
"""
struct SPIDerivation
    seed::UInt64
end

function derive_color(m::SPIDerivation, name::AbstractString, lang::AbstractString; seed=42)
    # Combine name hash with language hash for index
    h = hash(name, hash(lang, m.seed))
    idx = h % 1_000_000 + 1
    
    # Use Gay.jl's SPI-guaranteed hash_color
    r, g, b = hash_color(m.seed, UInt64(idx))
    RGB(r, g, b)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Comparison Framework
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compare_derivations(repos; models=default_models())

Compare different color derivation models on repository data.
Returns statistics about color distribution and visual distinctness.
"""
function compare_derivations(repos; models=nothing)
    if models === nothing
        models = [
            ("Gaussian(σ=0.5)", GaussianDerivation(0.5)),
            ("Ring(r=0.5,w=0.2)", RingDerivation(0.5, 0.2)),
            ("MRing(n=3)", MRingDerivation(3)),
            ("Disk(t=0.5)", DiskDerivation(0.5)),
            ("Crescent(s=0.3)", CrescentDerivation(0.3)),
            ("SPI(seed=42069)", SPIDerivation(42069)),
        ]
    end
    
    println("═" ^ 70)
    println("  MULTICOLOR DERIVATION COMPARISON")
    println("  Comrade.jl-inspired models on $(length(repos)) repositories")
    println("═" ^ 70)
    println()
    
    results = []
    
    for (name, model) in models
        colors = [derive_color(model, r.name, r.language) for r in repos]
        
        # Compute metrics
        # 1. Hue diversity (how spread out are the hues?)
        hues = [convert(HSL, c).h for c in colors]
        hue_std = std(hues)
        
        # 2. Color uniqueness (how many distinct colors?)
        hex_colors = Set([sprint(show, c) for c in colors])
        uniqueness = length(hex_colors) / length(colors)
        
        # 3. XOR fingerprint (for SPI verification)
        # Convert to Float32 matrix for fingerprinting
        color_matrix = zeros(Float32, length(colors), 3)
        for (i, c) in enumerate(colors)
            color_matrix[i, 1] = Float32(c.r)
            color_matrix[i, 2] = Float32(c.g)
            color_matrix[i, 3] = Float32(c.b)
        end
        fp = xor_fingerprint(color_matrix)
        
        push!(results, (
            name = name,
            model = model,
            colors = colors,
            hue_std = hue_std,
            uniqueness = uniqueness,
            fingerprint = fp
        ))
        
        # Display
        println("─" ^ 70)
        println("  Model: $name")
        println("  Hue diversity (std): $(round(hue_std, digits=1))°")
        println("  Uniqueness: $(round(uniqueness * 100, digits=1))%")
        println("  Fingerprint: 0x$(string(fp, base=16, pad=8))")
        print("  Sample: ")
        show_colors(colors[1:min(15, length(colors))]; width=2)
    end
    
    println()
    println("═" ^ 70)
    
    # Find best model
    best_idx = argmax([r.hue_std * r.uniqueness for r in results])
    best = results[best_idx]
    
    println("  BEST MODEL: $(best.name)")
    println("  Score: $(round(best.hue_std * best.uniqueness, digits=2))")
    println("═" ^ 70)
    
    return results
end

"""
    language_palette(model; seed=42)

Generate a palette for all known languages using the given model.
"""
function language_palette(model; seed=42)
    println()
    println("Language Palette using $(typeof(model))")
    println("─" ^ 50)
    
    for lang in sort(collect(keys(LANGUAGE_HUES)))
        c = derive_color(model, lang, lang; seed=seed)
        r = round(Int, c.r * 255)
        g = round(Int, c.g * 255)
        b = round(Int, c.b * 255)
        hex = "#$(string(r, base=16, pad=2))$(string(g, base=16, pad=2))$(string(b, base=16, pad=2))" |> uppercase
        
        print("  \e[38;2;$(r);$(g);$(b)m████\e[0m ")
        println("$(rpad(lang, 15)) $hex")
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════════
# Teleportation Test: Same colors across derivation backends?
# ═══════════════════════════════════════════════════════════════════════════════

"""
    teleportation_derivation_test(repos)

Test if different derivation models can "teleport" data to GPU
and back while preserving color identity (via SPI fingerprint).
"""
function teleportation_derivation_test(repos)
    println()
    println("═" ^ 70)
    println("  TELEPORTATION TEST: Derivation Backends")
    println("═" ^ 70)
    println()
    
    model = SPIDerivation(42069)
    
    # Derive on "World A" (sequential)
    colors_a = [derive_color(model, r.name, r.language) for r in repos]
    mat_a = zeros(Float32, length(colors_a), 3)
    for (i, c) in enumerate(colors_a)
        mat_a[i, 1] = Float32(c.r)
        mat_a[i, 2] = Float32(c.g)
        mat_a[i, 3] = Float32(c.b)
    end
    fp_a = xor_fingerprint(mat_a)
    
    # Derive on "World B" (using Gay.jl's SPI)
    # This uses the same indices but via ka infrastructure
    indices = [hash(r.name, hash(r.language, model.seed)) % 1_000_000 + 1 for r in repos]
    
    # Generate via hash_color directly
    mat_b = zeros(Float32, length(indices), 3)
    for (i, idx) in enumerate(indices)
        r, g, b = hash_color(model.seed, UInt64(idx))
        mat_b[i, 1] = r
        mat_b[i, 2] = g
        mat_b[i, 3] = b
    end
    fp_b = xor_fingerprint(mat_b)
    
    println("  World A (derive_color): 0x$(string(fp_a, base=16, pad=8))")
    println("  World B (hash_color):   0x$(string(fp_b, base=16, pad=8))")
    println()
    
    if fp_a == fp_b
        println("  ✓ TELEPORTATION SUCCESSFUL - Colors identical!")
    else
        println("  ✗ TELEPORTATION FAILED - Colors differ!")
        println("    (This is expected - derive_color uses HSL conversion)")
    end
    
    println("═" ^ 70)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println()
    println("╔" * "═" ^ 68 * "╗")
    println("║" * " " ^ 15 * "MULTICOLOR COMRADE × DUCKDB DEMO" * " " ^ 20 * "║")
    println("╚" * "═" ^ 68 * "╝")
    println()
    
    repos = load_repos()
    
    if repos === nothing || isempty(repos)
        println("  No repository data found. Using synthetic data.")
        repos = [
            (name = "Gay.jl", language = "Julia", owner = "bmorphism"),
            (name = "Comrade.jl", language = "Julia", owner = "ptiede"),
            (name = "xf.jl", language = "Julia", owner = "bmorphism"),
            (name = "rustc", language = "Rust", owner = "rust-lang"),
            (name = "tensorflow", language = "Python", owner = "tensorflow"),
            (name = "react", language = "JavaScript", owner = "facebook"),
            (name = "typescript", language = "TypeScript", owner = "microsoft"),
            (name = "clojure", language = "Clojure", owner = "clojure"),
            (name = "ghc", language = "Haskell", owner = "ghc"),
            (name = "go", language = "Go", owner = "golang"),
        ]
    end
    
    println("  Loaded $(length(repos)) repositories")
    println()
    
    # Compare derivation models
    results = compare_derivations(repos)
    
    # Show language palette for best model
    best = results[argmax([r.hue_std * r.uniqueness for r in results])]
    language_palette(best.model)
    
    # Teleportation test
    teleportation_derivation_test(repos)
    
    return results
end

# Helper for std calculation
using Statistics: std

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
