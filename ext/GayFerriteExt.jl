# Ferrite.jl extension for Gay.jl
# SPI-compliant coloring for FE grids and implicit/explicit attribute integration
# Maximum embarrassing parallelism enabled by Gay

module GayFerriteExt

using Gay
using Gay: hash_color_rgb, GAY_SEED, splitmix64
using Ferrite
using OhMyThreads
using Colors: RGB, HSL, convert

export world_ferrite_grid, world_ferrite_solution
export parallel_assemble_spi, verify_ferrite_spi
export FerriteWorld

# ═══════════════════════════════════════════════════════════════════════════
# Ferrite World Structure (Persistent State)
# ═══════════════════════════════════════════════════════════════════════════

struct FerriteWorld{G, C}
    grid::G
    cell_colors::Vector{C}
    seed::UInt64
end

function Base.length(w::FerriteWorld)
    length(w.cell_colors)
end

function Base.merge(w1::FerriteWorld, w2::FerriteWorld)
    # Monoidal composition: combine colors and XOR seeds
    FerriteWorld(w1.grid, [w1.cell_colors; w2.cell_colors], w1.seed ⊻ w2.seed)
end

function Gay.fingerprint(w::FerriteWorld)
    fp = UInt64(0)
    for c in w.cell_colors
        # Assuming c is RGB{Float32} or similar
        r = Float32(c.r)
        g = Float32(c.g)
        b = Float32(c.b)
        r_bits = reinterpret(UInt32, r)
        g_bits = reinterpret(UInt32, g)
        b_bits = reinterpret(UInt32, b)
        fp ⊻= UInt64(r_bits) | (UInt64(g_bits) << 24) | (UInt64(b_bits) << 48)
    end
    fp
end

# ═══════════════════════════════════════════════════════════════════════════
# Explicit Attribute Coloring (Grid Cells)
# ═══════════════════════════════════════════════════════════════════════════

"""
    world_ferrite_grid(grid::Grid; seed=GAY_SEED) -> FerriteWorld

Color grid cells deterministically based on their ID (Explicit Attribute).
Uses maximum embarrassing parallelism (OhMyThreads).
"""
function world_ferrite_grid(grid::Grid; seed::UInt64=GAY_SEED)
    n_cells = getncells(grid)

    # SPI-compliant parallel coloring
    colors = tmap(1:n_cells) do i
        hash_color_rgb(UInt64(i), seed)
    end

    FerriteWorld(grid, colors, seed)
end

# ═══════════════════════════════════════════════════════════════════════════
# Implicit Attribute Coloring (Solution Field)
# ═══════════════════════════════════════════════════════════════════════════

"""
    world_ferrite_solution(grid::Grid, dh::DofHandler, u::Vector; seed=GAY_SEED) -> FerriteWorld

Color grid cells based on the solution field `u` (Implicit Attribute).
Uses average value in cell to determine lightness/saturation.
"""
function world_ferrite_solution(grid::Grid, dh::DofHandler, u::Vector; seed::UInt64=GAY_SEED)
    n_cells = getncells(grid)

    # SPI-compliant parallel coloring
    colors = tmap(1:n_cells) do cell_idx
        # Note: In a real parallel scenario, we need to be careful with DofHandler access if it's not thread-safe.
        # Ferrite DofHandler is generally read-only after creation, so it should be fine.

        # We can't allocate inside the loop easily without care, but celldofs! writes to a vector.
        # We allocate a small vector per thread/task.
        n = ndofs_per_cell(dh, cell_idx)
        dofs = Vector{Int}(undef, n)
        celldofs!(dofs, dh, cell_idx)

        val = sum(abs, u[d] for d in dofs) / n

        # Base color from cell ID
        base_color = hash_color_rgb(UInt64(cell_idx), seed)
        base_hsl = convert(HSL, base_color)

        # Modulate with value (Implicit Attribute Integration)
        # Normalize roughly (assuming u is somewhat bounded or we clamp)
        norm_val = clamp(val, 0.0, 1.0)

        new_hsl = HSL(base_hsl.h, base_hsl.s, Float32(0.2 + 0.6*norm_val))
        convert(RGB{Float32}, new_hsl)
    end

    FerriteWorld(grid, colors, seed)
end

# ═══════════════════════════════════════════════════════════════════════════
# Maximum Embarrassing Parallelism
# ═══════════════════════════════════════════════════════════════════════════

"""
    parallel_assemble_spi(f, grid::Grid)

Execute function `f(cell_idx)` for every cell in parallel.
Uses Gay's embarrassing parallelism (OhMyThreads).
"""
function parallel_assemble_spi(f, grid::Grid)
    tforeach(f, 1:getncells(grid))
end

"""
    verify_ferrite_spi(grid::Grid; seed=GAY_SEED)

Verify that parallel iteration over grid cells preserves SPI properties.
"""
function verify_ferrite_spi(grid::Grid; seed::UInt64=GAY_SEED)
    n = getncells(grid)

    # Helper to calculate hash of a color
    function color_hash(c)
        r = Float32(c.r)
        g = Float32(c.g)
        b = Float32(c.b)
        r_bits = reinterpret(UInt32, r)
        g_bits = reinterpret(UInt32, g)
        b_bits = reinterpret(UInt32, b)
        UInt64(r_bits) | (UInt64(g_bits) << 24) | (UInt64(b_bits) << 48)
    end

    # Sequential fingerprint
    seq_fp = UInt64(0)
    for i in 1:n
        c = hash_color_rgb(UInt64(i), seed)
        seq_fp ⊻= color_hash(c)
    end

    # Parallel fingerprint (using mapreduce for parallel reduction)
    par_hashes = tmap(1:n) do i
        c = hash_color_rgb(UInt64(i), seed)
        color_hash(c)
    end
    par_fp = reduce(⊻, par_hashes)

    (verified = seq_fp == par_fp, seq=seq_fp, par=par_fp)
end

function __init__()
    @info "Gay.jl Ferrite extension loaded - SPI coloring and parallelism available"
end

end # module
