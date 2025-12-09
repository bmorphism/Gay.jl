#!/usr/bin/env julia
"""
Gheg Albanian Dialect Gradient as Markov Blanket

The dialect boundary is a linguistic Markov blanket:
- NE Gheg (s): Kosovo, Preshevo Valley — "environment" 
- Transition zone (b): Prizren, Has — "blanket" (screens off)
- NW Gheg (z): Shkodër, Malësia — "internal"

Noise (η) in linguistic variation creates conditional independence:
  p(NE, NW | transition) = p(NE | transition) × p(NW | transition)

This is a 2-transducer in Loregian's sense:
- Input A: phonemic inventory (NE features)
- Output B: phonemic inventory (NW features)  
- State category Q: transition zone dialects (with morphisms!)
- Profunctor t: feature transmission probability

See: Beck & Ramstead (2025) "Dynamic Markov Blanket Detection"
     Loregian (2025) "Two-dimensional transducers" arXiv:2509.06769
"""

using Gay

# ═══════════════════════════════════════════════════════════════════════════
# Gheg Dialect Regions with Approximate Coordinates
# ═══════════════════════════════════════════════════════════════════════════

const GHEG_REGIONS = Dict(
    # Northeastern Gheg (deeper pronunciation, Serbian calques)
    :northeastern => [
        ("Prishtinë", 42.67, 21.17),
        ("Mitrovicë", 42.88, 20.87),
        ("Gjilan", 42.47, 21.47),
        ("Preshevë", 42.31, 21.63),
        ("Bujanovac", 42.46, 21.77),
        ("Tropojë", 42.40, 20.16),
        ("Kukës", 42.08, 20.42),
        ("Pukë", 42.04, 19.90),
        ("Has", 42.20, 20.33),
    ],
    
    # Northwestern Gheg (softer, clearer tone)
    :northwestern => [
        ("Shkodër", 42.07, 19.51),
        ("Prizren", 42.22, 20.74),  # "Prizren old dialect"
        ("Pejë", 42.66, 20.29),
        ("Gjakovë", 42.38, 20.43),
        ("Lezhë", 41.78, 19.64),
        ("Ulcinj", 41.93, 19.21),
        ("Tuzi", 42.37, 19.33),
        ("Malësia", 42.45, 19.45),  # Highlander tribes
        ("Vermosh", 42.57, 19.79),
    ],
)

# ═══════════════════════════════════════════════════════════════════════════
# Directional Gradient Functions
# ═══════════════════════════════════════════════════════════════════════════

"""
Compute directional gradient value for a point.
- North (↑) = positive latitude contribution
- West (←) = negative longitude contribution
"""
function directional_gradient(lat::Float64, lon::Float64; 
                              center_lat=42.4, center_lon=20.5,
                              north_weight=0.6, west_weight=0.4)
    # Normalize to [-1, 1] range
    north_component = (lat - center_lat) / 1.0  # ~1 degree range
    west_component = (center_lon - lon) / 1.5   # ~1.5 degree range
    
    # Combine: higher value = more NW, lower = more NE
    gradient_value = north_weight * north_component + west_weight * west_component
    return clamp(gradient_value, -1.0, 1.0)
end

"""
Map gradient value to color using Gay.jl deterministic sampling.
NE Gheg (deep/dark tones) → NW Gheg (soft/light tones)
"""
function gradient_to_color(value::Float64; seed=42)
    # Map [-1, 1] to color index [1, 100]
    idx = round(Int, 50 + value * 49)
    idx = clamp(idx, 1, 100)
    return color_at(idx; seed=seed)
end

# ═══════════════════════════════════════════════════════════════════════════
# ASCII Map Rendering
# ═══════════════════════════════════════════════════════════════════════════

"""
Render Gheg dialect map with directional color gradient.
"""
function render_gheg_map(; seed=1069, width=60, height=20)
    println()
    println("  ╔════════════════════════════════════════════════════════════════╗")
    println("  ║     Gheg Albanian Dialect Gradient: NE ←→ NW                   ║")
    println("  ║     Seed: $seed (deterministic colors)                          ║")
    println("  ╚════════════════════════════════════════════════════════════════╝")
    println()
    
    # Bounding box: Albania/Kosovo region
    lat_min, lat_max = 41.7, 43.0
    lon_min, lon_max = 19.0, 22.0
    
    # Build character map
    map_chars = fill(' ', height, width)
    map_colors = fill((128, 128, 128), height, width)
    
    # Place region markers
    for (dialect, regions) in GHEG_REGIONS
        marker = dialect == :northeastern ? '●' : '○'
        for (name, lat, lon) in regions
            # Convert to map coordinates
            x = round(Int, (lon - lon_min) / (lon_max - lon_min) * (width - 1)) + 1
            y = round(Int, (lat_max - lat) / (lat_max - lat_min) * (height - 1)) + 1
            
            if 1 <= x <= width && 1 <= y <= height
                map_chars[y, x] = marker
                
                # Compute directional gradient color
                grad_val = directional_gradient(lat, lon)
                c = gradient_to_color(grad_val; seed=seed)
                rgb = (round(Int, c.r * 255), round(Int, c.g * 255), round(Int, c.b * 255))
                map_colors[y, x] = rgb
            end
        end
    end
    
    # Fill gradient background
    for y in 1:height
        for x in 1:width
            if map_chars[y, x] == ' '
                # Convert back to lat/lon
                lat = lat_max - (y - 1) / (height - 1) * (lat_max - lat_min)
                lon = lon_min + (x - 1) / (width - 1) * (lon_max - lon_min)
                
                grad_val = directional_gradient(lat, lon)
                c = gradient_to_color(grad_val; seed=seed)
                rgb = (round(Int, c.r * 255), round(Int, c.g * 255), round(Int, c.b * 255))
                map_colors[y, x] = rgb
                map_chars[y, x] = '░'
            end
        end
    end
    
    # Render with ANSI colors
    println("     West ←────────────────────────────────────────→ East")
    println("     19°E                                           22°E")
    println("  ┌" * "─"^width * "┐ 43°N")
    
    for y in 1:height
        print("  │")
        for x in 1:width
            r, g, b = map_colors[y, x]
            char = map_chars[y, x]
            print("\e[38;2;$(r);$(g);$(b)m$(char)\e[0m")
        end
        lat = lat_max - (y - 1) / (height - 1) * (lat_max - lat_min)
        println("│ $(round(lat, digits=1))°")
    end
    
    println("  └" * "─"^width * "┘ 41.7°N")
    println()
    
    # Legend
    println("  Legend:")
    println("    ● = Northeastern Gheg (deeper pronunciation)")
    println("    ○ = Northwestern Gheg (softer, clearer tone)")
    println()
    
    # Color gradient bar
    print("  Gradient: NE ")
    for i in 1:20
        val = -1.0 + (i - 1) * 2.0 / 19
        c = gradient_to_color(val; seed=seed)
        r, g, b = round(Int, c.r * 255), round(Int, c.g * 255), round(Int, c.b * 255)
        print("\e[38;2;$(r);$(g);$(b)m█\e[0m")
    end
    println(" NW")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Dialect Comparison Table
# ═══════════════════════════════════════════════════════════════════════════

function compare_dialects(; seed=1069)
    println("  ╔═══════════════════════════════════════════════════════════════════╗")
    println("  ║     Gheg Dialect Comparison: Northeastern vs Northwestern         ║")
    println("  ╚═══════════════════════════════════════════════════════════════════╝")
    println()
    
    # Sample colors for each region
    ne_colors = [gradient_to_color(directional_gradient(r[2], r[3]); seed=seed) 
                 for r in GHEG_REGIONS[:northeastern]]
    nw_colors = [gradient_to_color(directional_gradient(r[2], r[3]); seed=seed) 
                 for r in GHEG_REGIONS[:northwestern]]
    
    println("  Northeastern Gheg regions:")
    for (i, (name, lat, lon)) in enumerate(GHEG_REGIONS[:northeastern])
        c = ne_colors[i]
        r, g, b = round(Int, c.r * 255), round(Int, c.g * 255), round(Int, c.b * 255)
        grad = directional_gradient(lat, lon)
        print("    \e[38;2;$(r);$(g);$(b)m████\e[0m $(rpad(name, 12)) ")
        println("($(round(lat, digits=2))°N, $(round(lon, digits=2))°E) gradient=$(round(grad, digits=2))")
    end
    println()
    
    println("  Northwestern Gheg regions:")
    for (i, (name, lat, lon)) in enumerate(GHEG_REGIONS[:northwestern])
        c = nw_colors[i]
        r, g, b = round(Int, c.r * 255), round(Int, c.g * 255), round(Int, c.b * 255)
        grad = directional_gradient(lat, lon)
        print("    \e[38;2;$(r);$(g);$(b)m████\e[0m $(rpad(name, 12)) ")
        println("($(round(lat, digits=2))°N, $(round(lon, digits=2))°E) gradient=$(round(grad, digits=2))")
    end
    println()
    
    # Linguistic features
    println("  Linguistic Contrasts:")
    println("  ┌─────────────────────┬────────────────────┬────────────────────┐")
    println("  │ Feature             │ Northeastern       │ Northwestern       │")
    println("  ├─────────────────────┼────────────────────┼────────────────────┤")
    println("  │ Tone               │ Deeper, prolonged  │ Softer, clearer    │")
    println("  │ 'been' (past part.)│ kon                │ kjen / ken         │")
    println("  │ 'how?' (adverb)    │ qysh               │ si                 │")
    println("  │ /y/ → /i/ shift    │ ylberi → ilberi    │ (standard)         │")
    println("  │ Palatal stops      │ [t͡ʃ], [d͡ʒ]         │ [c], [ɟ]           │")
    println("  │ Serbian calques    │ Present (syntax)   │ Less common        │")
    println("  └─────────────────────┴────────────────────┴────────────────────┘")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Dialect as 2-Transducer
# ═══════════════════════════════════════════════════════════════════════════

"""
Model the dialect gradient as a 2-transducer:
  (Q, t) : NE_phonemes → NW_phonemes
  
where Q = transition zone dialects (Prizren, Has, Tropojë)
and t encodes feature transmission with noise η.
"""
struct DialectTransducer
    name::String
    input_features::Vector{Symbol}   # NE Gheg phonemic features
    output_features::Vector{Symbol}  # NW Gheg phonemic features
    transition_zones::Vector{String} # Blanket regions
    zone_colors::Vector{Any}         # Colored by gradient position
    noise_level::Float64             # η - linguistic variation
end

function DialectTransducer(; seed::Integer=1069, noise::Float64=0.15)
    gay_seed!(seed)
    
    # Phonemic features that distinguish NE from NW Gheg
    ne_features = [:deep_vowels, :serbian_calques, :qysh, :kon, :palatalized_tʃ]
    nw_features = [:clear_vowels, :italian_loans, :si, :kjen, :palatal_c]
    
    # Transition zones form the "blanket" - they have mixed features
    zones = ["Prizren", "Has", "Tropojë", "Pejë"]
    zone_colors = [next_color(SRGB()) for _ in zones]
    
    DialectTransducer("Gheg", ne_features, nw_features, zones, zone_colors, noise)
end

"""
Apply the dialect transducer: feature mapping with noise.
Returns probability of NW feature given NE feature and transition zone.
"""
function transduce(dt::DialectTransducer, ne_feature::Symbol, zone_idx::Int)
    # Base transmission probability (higher in western zones)
    base_p = zone_idx / length(dt.transition_zones)
    
    # Add noise η (linguistic variation)
    p = clamp(base_p + dt.noise_level * (rand() - 0.5), 0.0, 1.0)
    
    # Map to corresponding NW feature
    feature_idx = findfirst(==(ne_feature), dt.input_features)
    if isnothing(feature_idx) || feature_idx > length(dt.output_features)
        return (nothing, p)
    end
    
    return (dt.output_features[feature_idx], p)
end

"""
Visualize the dialect transducer as a Markov blanket.
"""
function render_dialect_transducer(dt::DialectTransducer; seed::Integer=1069)
    gay_seed!(seed)
    R = "\e[0m"
    
    function ansi(c)
        r = round(Int, c.r * 255)
        g = round(Int, c.g * 255)
        b = round(Int, c.b * 255)
        "\e[38;2;$(r);$(g);$(b)m"
    end
    
    # Colors for each partition
    c_ne = next_color(SRGB())  # Environment (s)
    c_nw = next_color(SRGB())  # Internal (z)
    
    ne = ansi(c_ne)
    nw = ansi(c_nw)
    
    println()
    println("  ╔════════════════════════════════════════════════════════════════════╗")
    println("  ║     Gheg Dialect as Markov Blanket / 2-Transducer                  ║")
    println("  ╚════════════════════════════════════════════════════════════════════╝")
    println()
    
    # The three-partition structure
    println("     $(ne)┌───────────────┐$(R)        Noise η_s")
    println("     $(ne)│   NE Gheg     │$(R)        (Serbian contact,")
    println("     $(ne)│ Environment s │$(R)         Ottoman legacy)")
    println("     $(ne)│ Kosovo, Preshevë│$(R)")
    println("     $(ne)└───────┬───────┘$(R)")
    println("             $(ne)│$(R)")
    println("             $(ne)▼$(R)         ┌─ Noise η_b (code-switching,")
    
    # Blanket zones with their colors
    print("     ")
    for (i, (zone, c)) in enumerate(zip(dt.transition_zones, dt.zone_colors))
        print("$(ansi(c))═$(R)")
    end
    println("        │   mixed marriages,")
    
    print("     ")
    for (i, (zone, c)) in enumerate(zip(dt.transition_zones, dt.zone_colors))
        print("$(ansi(c))║$(R)")
    end
    println("        │   trade routes)")
    
    println("     ╠═══════════════╣        └─ BLANKET b screens off")
    
    # Zone names
    print("     ")
    for (zone, c) in zip(dt.transition_zones, dt.zone_colors)
        print("$(ansi(c))$(first(zone))$(R)")
    end
    println("   Transition zones Q")
    println("             │")
    println("             ▼         ┌─ Noise η_z (Italian contact,")
    println("     $(nw)┌───────────────┐$(R)│   Adriatic trade)")
    println("     $(nw)│   NW Gheg     │$(R)│")
    println("     $(nw)│  Internal z   │$(R)└─")
    println("     $(nw)│ Shkodër, Ulcinj│$(R)")
    println("     $(nw)└───────────────┘$(R)")
    println()
    
    # The conditional independence
    println("  Markov property: p(NE, NW | transition) = p(NE | transition) × p(NW | transition)")
    println()
    
    # Feature transmission table
    println("  2-Transducer (Q, t): NE_features → NW_features")
    println("  ┌──────────────────┬────────────────────┬─────────────────┐")
    println("  │ NE Feature (A)   │ Transition Zone Q  │ NW Feature (B)  │")
    println("  ├──────────────────┼────────────────────┼─────────────────┤")
    
    for (ne_feat, nw_feat) in zip(dt.input_features, dt.output_features)
        zone_idx = rand(1:length(dt.transition_zones))
        zone = dt.transition_zones[zone_idx]
        c = dt.zone_colors[zone_idx]
        _, p = transduce(dt, ne_feat, zone_idx)
        
        ne_str = rpad(string(ne_feat), 16)
        zone_str = "$(ansi(c))$(rpad(zone, 18))$(R)"
        nw_str = rpad(string(nw_feat), 15)
        println("  │ $(ne_str) │ $(zone_str) │ $(nw_str) │")
    end
    println("  └──────────────────┴────────────────────┴─────────────────┘")
    println()
    
    println("  η = $(dt.noise_level) (linguistic variation / noise level)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main(; seed=1069)
    render_gheg_map(; seed=seed)
    compare_dialects(; seed=seed)
    
    # New: dialect as 2-transducer / Markov blanket
    dt = DialectTransducer(; seed=seed)
    render_dialect_transducer(dt; seed=seed)
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
