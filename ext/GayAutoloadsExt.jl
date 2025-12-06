# BasicAutoloads + Chairmarks extension for Gay.jl
# Enables "type this, run that" REPL convenience

module GayAutoloadsExt

using Gay
using BasicAutoloads: register_autoloads
using Chairmarks

function __init__()
    register_autoloads([
        # Chairmarks.jl for fast benchmarking (100x faster than BenchmarkTools)
        "@b" => :(using Chairmarks),
        "@be" => :(using Chairmarks),
        "@btime" => :(using Chairmarks),
        
        # Gay.jl color generation - auto-import when typed
        "gay_next" => :(using Gay: gay_next, gay_at, gay_palette, gay_seed!, GAY_SEED),
        "gay_at" => :(using Gay: gay_next, gay_at, gay_palette, gay_seed!),
        "gay_palette" => :(using Gay: gay_next, gay_at, gay_palette, next_palette),
        
        # O(1) hash color generation
        "hash_color" => :(using Gay: hash_color, splitmix64, xor_fingerprint, GAY_SEED),
        "splitmix64" => :(using Gay: hash_color, splitmix64),
        
        # KernelAbstractions GPU/parallel
        "ka_colors!" => :(using Gay: ka_colors!, ka_color_sums, ka_parallel_hash, set_backend!, get_backend),
        "ka_color_sums" => :(using Gay: ka_colors!, ka_color_sums, ka_parallel_hash),
        
        # Pride flags and display
        "pride_flag" => :(using Gay: pride_flag, rainbow, show_colors, show_palette),
        "rainbow" => :(using Gay: rainbow, pride_flag, show_colors),
        "show_colors" => :(using Gay: show_colors, show_palette),
        
        # Energy measurement
        "measure_energy" => :(using Gay: measure_energy, EnergyMeasurement, with_energy_measurement),
        "EnergyMeasurement" => :(using Gay: EnergyMeasurement, measure_energy),
        
        # Mortal/Immortal lifetimes
        "MortalComputation" => :(using Gay: MortalComputation, ImmortalComputation, mortal_step!, immortal_epoch!, harvest, ascend),
        "ImmortalComputation" => :(using Gay: MortalComputation, ImmortalComputation, mortal_step!, immortal_epoch!),
        
        # Interleaved streams for checkerboard/XOR decomposition
        "GayInterleaver" => :(using Gay: GayInterleaver, gay_interleave, gay_sublattice, gay_xor_color),
        "gay_interleave" => :(using Gay: GayInterleaver, gay_interleave, gay_sublattice),
        
        # S-expression coloring
        "GaySexpr" => :(using Gay: GaySexpr, gay_magnetized_sexpr, gay_render_sexpr, gay_sexpr_colors),
        "gay_sexpr_colors" => :(using Gay: gay_sexpr_colors, gay_magnetized_sexpr, gay_render_sexpr),
        
        # Color spaces
        "SRGB" => :(using Gay: SRGB, DisplayP3, Rec2020, ColorSpace),
        "DisplayP3" => :(using Gay: SRGB, DisplayP3, Rec2020),
        "Rec2020" => :(using Gay: SRGB, DisplayP3, Rec2020),
        
        # LispSyntax for S-expressions
        "lisp" => :(using LispSyntax),
        "@lisp_str" => :(using LispSyntax),
        
        # Enzyme autodiff (if integrated)
        "gay_autodiff" => :(using Gay: gay_autodiff, gay_gradient_color, EnzymeBinding),
    ])
    
    @info "Gay.jl autoloads active! Type any Gay.jl function to auto-import."
    @info "  Examples: hash_color, @b, gay_next, ka_colors!, measure_energy"
end

end # module
