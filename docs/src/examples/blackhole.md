# Black Hole Imaging

Gay.jl includes a black hole visualization demo inspired by [Comrade.jl](https://github.com/ptiede/Comrade.jl), the Julia package used by the Event Horizon Telescope collaboration.

## Running the Demo

```julia
include("examples/blackhole.jl")

# Render a black hole with photon rings
render_blackhole(seed=1337)

# EHT-style multi-ring structure
eht_rings(seed=2017)
```

## Physics Background

### Photon Rings

Black holes create a series of **photon rings** - light that has orbited the black hole before escaping:

- **n=1 ring**: Direct emission (no orbits)
- **n=2 ring**: Light that orbited once
- **n=3 ring**: Light that orbited twice
- etc.

Each successive ring is thinner and closer to the photon sphere.

### Doppler Boosting

Material orbiting the black hole experiences **relativistic Doppler boosting**:
- Approaching side appears brighter
- Receding side appears dimmer

This creates the characteristic **crescent** shape seen in EHT images.

## Model Components

```julia
gay_seed!(1337)

# n=1 ring (direct emission)
ring1 = comrade_ring(0.8, 0.15)

# n=2 ring (one orbit)
ring2 = comrade_ring(0.65, 0.08)

# n=3 ring (two orbits)
ring3 = comrade_ring(0.55, 0.04)

# Doppler-boosted crescent
doppler = comrade_crescent(0.9, 0.4, 0.2)

# Combined model
model = ring1 + ring2 + ring3 + doppler
comrade_show(model)
```

## Output

```
═══════════════════════════════════════════════════════════
   BLACK HOLE VISUALIZATION (seed=1337)
═══════════════════════════════════════════════════════════

Colored S-Expression (parentheses colored by component):
(ring 0.8 0.15) + (ring 0.65 0.08) + (ring 0.55 0.04) + (crescent 0.9 0.4 0.2)

Intensity Map:
            ████████████████
          ████████████████████
        ████████████████████████
      ██████████████████████████████
     ██████████    ████    ██████████
    ████████          ██        ████████
   ████████            ██        ████████
   ███████              ██        ███████
   ███████              ██        ███████
   ████████            ██        ████████
    ████████          ██        ████████
     ██████████    ████    ██████████
      ██████████████████████████████
        ████████████████████████
          ████████████████████
            ████████████████
```

## Scientific Context

The Event Horizon Telescope captured the first image of a black hole (M87*) in 2019, and Sagittarius A* in 2022. These images show:

1. **Shadow**: The dark central region where light cannot escape
2. **Photon ring**: Bright ring of light bent around the black hole
3. **Asymmetry**: Doppler boosting makes one side brighter

Gay.jl's sky models provide a simplified but visually accurate representation of these features with deterministic, reproducible colors.
