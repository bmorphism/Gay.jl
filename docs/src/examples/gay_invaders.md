```@meta
EditURL = "../literate/gay_invaders.jl"
```

# Gay Invaders: Terminal Game with Deterministic Colors

This example demonstrates how Gay.jl's splittable random colors can be used
to add vibrant, reproducible color palettes to terminal games.

## Inspiration

This example was inspired by [Lilith Hafner's JuliaCon talk on SpaceInvaders.jl](https://www.youtube.com/watch?v=PgqrHm-wL1w),
which demonstrated building a terminal game in pure Julia. We extend it with
Gay.jl's deterministic color system to add:

- **Reproducible palettes**: Same seed = same colors every time
- **Pride flag colors**: Ship and bullets use trans pride colors
- **Rainbow effects**: Explosions cycle through rainbow colors
- **Per-row enemy colors**: Each row gets a unique deterministic color

## Game Screenshot

Here's what Gay Invaders looks like in action (colors render in terminal with ANSI true-color support):

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║         🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯                            ║
║         🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯                            ║
║         🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯                            ║
║         🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯                            ║
║          🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯                             ║
║          🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯 🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯                             ║
║           🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯 🙯🙯🙯🙯 🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯                              ║
║             🙯🙯🙯🙯 🙯🙯🙯  🙯🙯   🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯🙯                               ║
║             🙯🙯🙯🙯 🙯🙯 🙯 🙯  🙯 🙯  🙯🙯🙯🙯🙯🙯🙯 🙯                                ║
║              🙯 🙯    🙯    🙯  🙯 🙯🙯🙯🙯🙯🙯🙯                                  ║
║                               🢙    🙯🙯                                    ║
║                        🙯 🙯  ✦         🙯                                    ║
║                                                                              ║
║                               🢙                                              ║
║                                                                              ║
║                               🙭                                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**Game Elements:**
- `🙯` - Enemies (each row colored by `color_at(row; seed=seed)`)
- `🙭` - Ship (trans pride light blue)
- `🢙` - Bullets (trans pride pink)
- `✦` - Explosions (cycling rainbow colors)
- Rainbow border (cycles through pride colors)

## Quick Start

```julia
using Gay
include(joinpath(pkgdir(Gay), "examples", "spaceinvaders_colors.jl"))
GayInvaders.main()
```

## Controls

| Key | Action |
|-----|--------|
| ← / A | Move left |
| → / D | Move right |
| ↑ / W / Space | Fire |
| Q | Quit |

## How It Works

First, we set up color helpers that convert Gay.jl RGB colors to ANSI escape codes:

````@example gay_invaders
using Gay
using Colors: RGB

function rgb_to_ansi(c::RGB)
    r = round(Int, clamp(c.r, 0, 1) * 255)
    g = round(Int, clamp(c.g, 0, 1) * 255)
    b = round(Int, clamp(c.b, 0, 1) * 255)
    "\e[38;2;$(r);$(g);$(b)m"
end
````

### Deterministic Enemy Colors

Each enemy row gets a color from `color_at(row; seed=seed)`. This ensures
the same seed always produces the same color palette:

````@example gay_invaders
seed = 42
gay_seed!(seed)

println("Enemy row colors (seed=$seed):")
for row in 1:6
    c = color_at(row; seed=seed)
    ansi = rgb_to_ansi(c)
    println("  Row $row: $(ansi)████████\e[0m")
end
````

### Pride Flag Integration

The ship and bullets use colors from the transgender pride flag,
making the game a celebration of identity:

````@example gay_invaders
trans = transgender()
println("\nTrans pride colors for ship/bullets:")
for (i, c) in enumerate(trans)
    ansi = rgb_to_ansi(c)
    println("  $(ansi)████\e[0m")
end
````

Ship uses light blue (trans[3]), bullets use pink (trans[1]):

````@example gay_invaders
println("\nShip: $(rgb_to_ansi(trans[3]))🙭\e[0m  Bullet: $(rgb_to_ansi(trans[1]))🢙\e[0m")
````

### Rainbow Explosions

When enemies are hit, explosions cycle through rainbow colors:

````@example gay_invaders
println("\nExplosion colors (rainbow):")
for (i, c) in enumerate(rainbow())
    ansi = rgb_to_ansi(c)
    print("$(ansi)✦\e[0m ")
end
println()
````

## Reproducibility Demo

The key feature is reproducibility - same seed always gives same colors:

````@example gay_invaders
println("\n--- Reproducibility Test ---")
for s in [42, 1337, 2024]
    print("Seed $s: ")
    for i in 1:6
        c = color_at(i; seed=s)
        print("$(rgb_to_ansi(c))●\e[0m")
    end
    println()
end
````

## Running the Full Game

The complete interactive game is in `examples/spaceinvaders_colors.jl`:

```julia
using Gay
include(joinpath(pkgdir(Gay), "examples", "spaceinvaders_colors.jl"))

# Play with default colors
GayInvaders.main()

# Play with different color palette
GayInvaders.main(seed=1337)

# Skip intro animation
GayInvaders.main(splash=false)
```

## Credits

- **SpaceInvaders.jl** by [Lilith Hafner](https://github.com/LilithHafner/SpaceInvaders.jl)
  for the original terminal game implementation
- **JuliaCon Talk**: [Building a Terminal Game in Julia](https://www.youtube.com/watch?v=PgqrHm-wL1w)
- **Gay.jl** for splittable deterministic colors and pride palettes

The combination of terminal games with reproducible color palettes opens up
creative possibilities for accessible, visually distinctive game experiences.

