# Semantic Coloring for XF.jl
# Colors derived from file meaning, not just index
#
# "If nature is unjust, change nature!" — Laboria Cuboniks

using Colors

export SemanticColorScheme, semantic_color, semantic_palette
export FileSemantics, analyze_semantics, semantic_hash

# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Categories
# ═══════════════════════════════════════════════════════════════════════════════

"""
Semantic categories for file classification.
Each category has a base hue for consistent visual language.
"""
@enum SemanticCategory begin
    # Code categories (cool colors: blue/purple)
    CODE_RUST        # 🦀 Orange-red (Rust brand)
    CODE_JULIA       # 💜 Purple (Julia brand)
    CODE_PYTHON      # 🐍 Yellow-green (Python brand)
    CODE_HASKELL     # λ Purple (functional)
    CODE_LISP        # λ Green (Lisp family)
    CODE_ML          # 🐫 Orange (OCaml/SML)
    CODE_JS          # 💛 Yellow (JavaScript)
    CODE_GO          # 🐹 Cyan (Go brand)
    CODE_C           # ⚙️ Gray-blue (systems)
    CODE_SHELL       # 🐚 Green (terminal)
    CODE_OTHER       # Generic code
    
    # Data categories (warm colors: orange/yellow)
    DATA_JSON        # 📦 Orange
    DATA_YAML        # 📋 Yellow
    DATA_TOML        # ⚙️ Brown
    DATA_CSV         # 📊 Green
    DATA_SQL         # 🗄️ Blue
    DATA_BINARY      # 🔢 Gray
    
    # Documentation (green spectrum)
    DOC_MARKDOWN     # 📝 Green
    DOC_ORG          # 🦄 Purple
    DOC_RST          # 📖 Teal
    DOC_TEX          # 📜 Brown
    DOC_HTML         # 🌐 Orange
    
    # Assets (varied)
    ASSET_IMAGE      # 🖼️ Magenta
    ASSET_VIDEO      # 🎬 Red
    ASSET_AUDIO      # 🎵 Cyan
    ASSET_FONT       # 🔤 Gray
    
    # Build/Config (muted)
    BUILD_ARTIFACT   # ⚙️ Gray
    BUILD_CONFIG     # 🔧 Brown
    BUILD_LOCK       # 🔒 Dark gray
    
    # Version control
    VCS_GIT          # 🔀 Orange-red
    
    # Unknown
    UNKNOWN          # ❓ Gray
end

# Base hues for each category (HSL hue: 0-360)
const CATEGORY_HUES = Dict{SemanticCategory, Float64}(
    CODE_RUST => 15.0,       # Orange-red
    CODE_JULIA => 275.0,     # Purple
    CODE_PYTHON => 55.0,     # Yellow-green
    CODE_HASKELL => 280.0,   # Purple
    CODE_LISP => 120.0,      # Green
    CODE_ML => 30.0,         # Orange
    CODE_JS => 50.0,         # Yellow
    CODE_GO => 190.0,        # Cyan
    CODE_C => 210.0,         # Blue-gray
    CODE_SHELL => 140.0,     # Green
    CODE_OTHER => 220.0,     # Blue
    
    DATA_JSON => 35.0,       # Orange
    DATA_YAML => 45.0,       # Yellow
    DATA_TOML => 25.0,       # Brown-orange
    DATA_CSV => 100.0,       # Yellow-green
    DATA_SQL => 200.0,       # Blue
    DATA_BINARY => 0.0,      # Gray (saturation=0)
    
    DOC_MARKDOWN => 150.0,   # Green
    DOC_ORG => 290.0,        # Purple
    DOC_RST => 170.0,        # Teal
    DOC_TEX => 35.0,         # Brown
    DOC_HTML => 20.0,        # Orange
    
    ASSET_IMAGE => 320.0,    # Magenta
    ASSET_VIDEO => 0.0,      # Red
    ASSET_AUDIO => 180.0,    # Cyan
    ASSET_FONT => 0.0,       # Gray
    
    BUILD_ARTIFACT => 0.0,   # Gray
    BUILD_CONFIG => 30.0,    # Brown
    BUILD_LOCK => 0.0,       # Dark gray
    
    VCS_GIT => 10.0,         # Orange-red
    
    UNKNOWN => 0.0,          # Gray
)

# Saturation levels (0-1)
const CATEGORY_SATURATIONS = Dict{SemanticCategory, Float64}(
    BUILD_ARTIFACT => 0.1,
    BUILD_LOCK => 0.05,
    DATA_BINARY => 0.1,
    ASSET_FONT => 0.1,
    UNKNOWN => 0.15,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Extension → Category mapping
# ═══════════════════════════════════════════════════════════════════════════════

const EXT_TO_CATEGORY = Dict{String, SemanticCategory}(
    # Rust
    ".rs" => CODE_RUST,
    ".rlib" => CODE_RUST,
    ".rmeta" => BUILD_ARTIFACT,
    
    # Julia
    ".jl" => CODE_JULIA,
    
    # Python
    ".py" => CODE_PYTHON,
    ".pyx" => CODE_PYTHON,
    ".pyi" => CODE_PYTHON,
    ".ipynb" => CODE_PYTHON,
    
    # Haskell
    ".hs" => CODE_HASKELL,
    ".lhs" => CODE_HASKELL,
    ".cabal" => BUILD_CONFIG,
    
    # Lisp family
    ".lisp" => CODE_LISP,
    ".cl" => CODE_LISP,
    ".el" => CODE_LISP,
    ".scm" => CODE_LISP,
    ".rkt" => CODE_LISP,
    ".clj" => CODE_LISP,
    ".cljs" => CODE_LISP,
    ".hy" => CODE_LISP,
    ".fnl" => CODE_LISP,
    
    # ML family
    ".ml" => CODE_ML,
    ".mli" => CODE_ML,
    ".mll" => CODE_ML,
    ".mly" => CODE_ML,
    ".sml" => CODE_ML,
    ".sig" => CODE_ML,
    ".fun" => CODE_ML,
    
    # JavaScript/TypeScript
    ".js" => CODE_JS,
    ".jsx" => CODE_JS,
    ".ts" => CODE_JS,
    ".tsx" => CODE_JS,
    ".mjs" => CODE_JS,
    ".cjs" => CODE_JS,
    ".vue" => CODE_JS,
    ".svelte" => CODE_JS,
    
    # Go
    ".go" => CODE_GO,
    ".mod" => BUILD_CONFIG,
    ".sum" => BUILD_LOCK,
    
    # C/C++
    ".c" => CODE_C,
    ".h" => CODE_C,
    ".cpp" => CODE_C,
    ".hpp" => CODE_C,
    ".cc" => CODE_C,
    ".hh" => CODE_C,
    ".cxx" => CODE_C,
    ".hxx" => CODE_C,
    
    # Shell
    ".sh" => CODE_SHELL,
    ".bash" => CODE_SHELL,
    ".zsh" => CODE_SHELL,
    ".fish" => CODE_SHELL,
    
    # Data formats
    ".json" => DATA_JSON,
    ".jsonl" => DATA_JSON,
    ".yaml" => DATA_YAML,
    ".yml" => DATA_YAML,
    ".toml" => DATA_TOML,
    ".csv" => DATA_CSV,
    ".tsv" => DATA_CSV,
    ".sql" => DATA_SQL,
    ".db" => DATA_BINARY,
    ".sqlite" => DATA_BINARY,
    ".duckdb" => DATA_BINARY,
    
    # Documentation
    ".md" => DOC_MARKDOWN,
    ".markdown" => DOC_MARKDOWN,
    ".org" => DOC_ORG,
    ".rst" => DOC_RST,
    ".tex" => DOC_TEX,
    ".latex" => DOC_TEX,
    ".html" => DOC_HTML,
    ".htm" => DOC_HTML,
    ".txt" => DOC_MARKDOWN,
    
    # Assets
    ".png" => ASSET_IMAGE,
    ".jpg" => ASSET_IMAGE,
    ".jpeg" => ASSET_IMAGE,
    ".gif" => ASSET_IMAGE,
    ".svg" => ASSET_IMAGE,
    ".webp" => ASSET_IMAGE,
    ".ico" => ASSET_IMAGE,
    ".mp4" => ASSET_VIDEO,
    ".webm" => ASSET_VIDEO,
    ".mov" => ASSET_VIDEO,
    ".avi" => ASSET_VIDEO,
    ".mp3" => ASSET_AUDIO,
    ".wav" => ASSET_AUDIO,
    ".flac" => ASSET_AUDIO,
    ".ogg" => ASSET_AUDIO,
    ".ttf" => ASSET_FONT,
    ".otf" => ASSET_FONT,
    ".woff" => ASSET_FONT,
    ".woff2" => ASSET_FONT,
    
    # Build artifacts
    ".o" => BUILD_ARTIFACT,
    ".a" => BUILD_ARTIFACT,
    ".so" => BUILD_ARTIFACT,
    ".dylib" => BUILD_ARTIFACT,
    ".dll" => BUILD_ARTIFACT,
    ".exe" => BUILD_ARTIFACT,
    ".d" => BUILD_ARTIFACT,
    ".timestamp" => BUILD_ARTIFACT,
    
    # Config
    ".lock" => BUILD_LOCK,
    ".nix" => BUILD_CONFIG,
    ".flake" => BUILD_CONFIG,
    
    # VCS
    ".gitignore" => VCS_GIT,
    ".gitmodules" => VCS_GIT,
    ".gitattributes" => VCS_GIT,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Analysis
# ═══════════════════════════════════════════════════════════════════════════════

"""
    FileSemantics

Semantic information extracted from a file path.
"""
struct FileSemantics
    path::String
    category::SemanticCategory
    depth::Int                    # Directory depth
    name_hash::UInt64            # Hash of filename for variation
end

"""
    analyze_semantics(path::String) -> FileSemantics

Extract semantic information from a file path.
"""
function analyze_semantics(path::String)
    # Get extension
    ext = lowercase(splitext(path)[2])
    
    # Categorize
    category = get(EXT_TO_CATEGORY, ext, UNKNOWN)
    
    # Special cases by filename
    filename = lowercase(basename(path))
    if filename in ["makefile", "gnumakefile"]
        category = BUILD_CONFIG
    elseif filename in ["dockerfile", "containerfile"]
        category = BUILD_CONFIG
    elseif filename == "cargo.toml"
        category = CODE_RUST
    elseif filename == "project.toml"
        category = CODE_JULIA
    elseif filename == "package.json"
        category = CODE_JS
    elseif filename == "pyproject.toml" || filename == "setup.py"
        category = CODE_PYTHON
    elseif startswith(filename, ".")
        category = BUILD_CONFIG
    end
    
    # Directory depth
    depth = count(==('/'), path) + count(==('\\'), path)
    
    # Hash filename for variation within category
    name_hash = hash(basename(path))
    
    FileSemantics(path, category, depth, name_hash)
end

"""
    semantic_hash(path::String) -> UInt64

Compute a semantic hash that groups similar files together.
Files with the same extension in the same directory get similar hashes.
"""
function semantic_hash(path::String)
    dir = dirname(path)
    ext = lowercase(splitext(path)[2])
    name = basename(path)
    
    # Combine directory, extension, and name
    h = hash(dir)
    h = hash(ext, h)
    h = hash(name, h)
    
    return UInt64(h & typemax(UInt64))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Color Generation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    semantic_color(sem::FileSemantics; seed::UInt64=XF_SEED) -> RGB

Generate a color based on file semantics.
- Hue determined by category
- Saturation/Lightness varied by filename hash
"""
function semantic_color(sem::FileSemantics; seed::UInt64=XF_SEED)
    # Base hue from category
    base_hue = get(CATEGORY_HUES, sem.category, 0.0)
    
    # Base saturation (some categories are muted)
    base_sat = get(CATEGORY_SATURATIONS, sem.category, 0.7)
    
    # Variation from filename hash
    h = xor(sem.name_hash, seed)
    
    # Hue variation: ±20°
    hue_var = (((h >> 0) & 0xFF) / 255.0 - 0.5) * 40.0
    hue = mod(base_hue + hue_var, 360.0)
    
    # Saturation variation: base ± 0.15
    sat_var = (((h >> 8) & 0xFF) / 255.0 - 0.5) * 0.3
    sat = clamp(base_sat + sat_var, 0.1, 0.9)
    
    # Lightness: 0.4-0.7 based on depth (deeper = darker)
    depth_factor = min(sem.depth / 10.0, 1.0)
    base_light = 0.65 - depth_factor * 0.2
    light_var = (((h >> 16) & 0xFF) / 255.0 - 0.5) * 0.2
    light = clamp(base_light + light_var, 0.35, 0.75)
    
    # Convert HSL to RGB
    hsl = HSL(hue, sat, light)
    return convert(RGB, hsl)
end

"""
    semantic_color(path::String; seed::UInt64=XF_SEED) -> RGB

Generate a semantic color for a file path.
"""
function semantic_color(path::String; seed::UInt64=XF_SEED)
    sem = analyze_semantics(path)
    semantic_color(sem; seed=seed)
end

"""
    semantic_palette(paths::Vector{String}; seed::UInt64=XF_SEED) -> Vector{RGB}

Generate semantic colors for multiple files.
"""
function semantic_palette(paths::Vector{String}; seed::UInt64=XF_SEED)
    return [semantic_color(p; seed=seed) for p in paths]
end

# ═══════════════════════════════════════════════════════════════════════════════
# Category Statistics
# ═══════════════════════════════════════════════════════════════════════════════

"""
    category_stats(paths::Vector{String}) -> Dict{SemanticCategory, Int}

Count files by semantic category.
"""
function category_stats(paths::Vector{String})
    counts = Dict{SemanticCategory, Int}()
    for p in paths
        cat = analyze_semantics(p).category
        counts[cat] = get(counts, cat, 0) + 1
    end
    return counts
end

"""
    show_category_colors()

Display all semantic category colors.
"""
function show_category_colors()
    println("Semantic Category Colors:")
    println("-" ^ 50)
    
    for cat in instances(SemanticCategory)
        hue = get(CATEGORY_HUES, cat, 0.0)
        sat = get(CATEGORY_SATURATIONS, cat, 0.7)
        
        hsl = HSL(hue, sat, 0.5)
        rgb = convert(RGB, hsl)
        
        r = round(Int, rgb.r * 255)
        g = round(Int, rgb.g * 255)
        b = round(Int, rgb.b * 255)
        
        hex = "#$(string(r, base=16, pad=2))$(string(g, base=16, pad=2))$(string(b, base=16, pad=2))" |> uppercase
        
        print("  \e[38;2;$(r);$(g);$(b)m████\e[0m ")
        println("$(rpad(string(cat), 20)) $hex  H=$(round(Int, hue))° S=$(round(sat, digits=2))")
    end
end
