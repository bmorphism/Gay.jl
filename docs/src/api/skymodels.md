# Sky Models (Comrade.jl-style)

Gay.jl includes a Comrade.jl-inspired sky model system for creating colored S-expressions that represent VLBI (Very Long Baseline Interferometry) sky models.

This is inspired by [Comrade.jl](https://github.com/ptiede/Comrade.jl), the Julia package used by the Event Horizon Telescope collaboration for black hole imaging.

## Types

```@docs
SkyModel
```

## Primitives

```@docs
comrade_ring
comrade_gaussian
comrade_crescent
comrade_disk
```

## Composition

```@docs
sky_add
sky_mul
```

## Display

```@docs
comrade_show
render_intensity
```

## Gallery

```@docs
generate_model
aesthetic_score
```

## Model Types

### Ring
A ring at radius `r` with Gaussian width `w`:

```
I(x,y) = exp(-((r - R) / w)^2 / 2)
```

where `r = sqrt(x^2 + y^2)`.

### Gaussian
A 2D Gaussian with widths σx, σy:

```
I(x,y) = exp(-(x^2/(2σx^2) + y^2/(2σy^2)))
```

### Crescent
An asymmetric crescent (shifted inner disk subtracted from outer):

```
I(x,y) = outer_disk - shifted_inner_disk
```

Used for modeling Doppler-boosted emission.

### Disk
A uniform disk of radius `R`:

```
I(x,y) = 1 if sqrt(x^2 + y^2) < R, else 0
```

## Examples

### M87* Style Model

```julia
using Gay

gay_seed!(2017)

# Ring + central gaussian (like Event Horizon Telescope M87* image)
ring = comrade_ring(0.8, 0.15)
core = comrade_gaussian(0.2)
m87 = ring + core

comrade_show(m87)
```

### Sgr A* Style Model

```julia
gay_seed!(2022)

# Asymmetric crescent (like Sagittarius A*)
crescent = comrade_crescent(0.8, 0.4, 0.2)
hotspot = comrade_disk(0.1)
sgra = crescent + hotspot

comrade_show(sgra)
```

### Multi-Ring Structure

```julia
gay_seed!(1069)

# Multiple concentric rings (photon ring orbits)
model = comrade_ring(0.5, 0.1)
for i in 2:4
    model = model + comrade_ring(0.5 + i*0.2, 0.08)
end

comrade_show(model)
```

## S-Expression Output

Models are displayed as colored S-expressions:

```
(ring 0.8 0.15) + (gaussian 0.2 0.2)
```

Where each component's parentheses are colored by its deterministic color assignment.

## ASCII Intensity Map

`render_intensity` produces ASCII art visualization:

```
        ████████████
      ████████████████
    ██████    ████    ██████
   █████        ██        █████
  ████          ██          ████
  ████          ██          ████
   █████        ██        █████
    ██████    ████    ██████
      ████████████████
        ████████████
```
