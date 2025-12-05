using Documenter
using Literate
using Gay

# Directory paths
LITERATE_DIR = joinpath(@__DIR__, "src", "literate")
OUTPUT_DIR = joinpath(@__DIR__, "src", "examples")

# Ensure output directory exists
mkpath(OUTPUT_DIR)

# Process all literate files
literate_files = [
    "splittable_determinism.jl",
    "wide_gamut_colors.jl", 
    "comrade_sky_models.jl",
    "pride_palettes.jl",
    "parallel_spi.jl",
    "gay_invaders.jl",
    "parallel_color_determinism.jl"
]

for file in literate_files
    input = joinpath(LITERATE_DIR, file)
    if isfile(input)
        @info "Processing $file..."
        
        # Generate markdown
        Literate.markdown(input, OUTPUT_DIR; 
            documenter=true,
            credit=false
        )
        
        # Generate notebook
        Literate.notebook(input, OUTPUT_DIR;
            execute=false
        )
    else
        @warn "File not found: $input"
    end
end

# Build documentation
makedocs(
    sitename = "Gay.jl",
    authors = "bmorphism",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://bmorphism.github.io/Gay.jl",
        assets = String[],
        collapselevel = 1
    ),
    modules = [Gay],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Examples" => [
            "Splittable Determinism" => "examples/splittable_determinism.md",
            "Wide-Gamut Colors" => "examples/wide_gamut_colors.md",
            "Comrade Sky Models" => "examples/comrade_sky_models.md",
            "Pride Palettes" => "examples/pride_palettes.md",
            "Parallel SPI" => "examples/parallel_spi.md",
            "Gay Invaders" => "examples/gay_invaders.md",
            "Parallel Color Determinism" => "examples/parallel_color_determinism.md"
        ],
        "API Reference" => "api.md"
    ],
    doctest = false,
    warnonly = true
)

# Deploy docs (handled by GitHub Actions deploy-pages)
