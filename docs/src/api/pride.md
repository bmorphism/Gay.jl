# Pride Flags

Gay.jl includes authentic color palettes for various pride flags.

## Functions

```@docs
pride_flag
rainbow
transgender
bisexual
nonbinary
pansexual
asexual
lesbian
intersex
progress
```

## Available Flags

| Flag | Function | Colors |
|------|----------|--------|
| Rainbow | `rainbow()` | 6 |
| Transgender | `transgender()` | 5 |
| Bisexual | `bisexual()` | 5 |
| Non-binary | `nonbinary()` | 4 |
| Pansexual | `pansexual()` | 3 |
| Asexual | `asexual()` | 4 |
| Lesbian | `lesbian()` | 7 |
| Intersex | `intersex()` | 2 |
| Progress | `progress()` | 11 |
| Genderqueer | `pride_flag(:genderqueer)` | 3 |
| Agender | `pride_flag(:agender)` | 7 |

## Examples

### Display Pride Flags

```julia
using Gay

# Rainbow flag
show_colors(rainbow())
# Output: ██████████████████████████████████████████████

# Transgender flag
show_colors(transgender())

# Progress Pride flag (includes trans + POC chevrons)
show_colors(progress())
```

### Pride Flags in Wide Gamut

```julia
# Standard sRGB
rb_srgb = rainbow()

# Display P3 (richer reds/greens)
rb_p3 = rainbow(DisplayP3())

# Rec.2020 (maximum color volume)
rb_2020 = rainbow(Rec2020())
```

### Access by Symbol

```julia
# All flags accessible via pride_flag()
pride_flag(:rainbow)
pride_flag(:transgender)
pride_flag(:bisexual)
pride_flag(:nonbinary)
pride_flag(:pansexual)
pride_flag(:asexual)
pride_flag(:lesbian)
pride_flag(:intersex)
pride_flag(:progress)
pride_flag(:genderqueer)
pride_flag(:agender)
```

## Color Values

### Rainbow Flag
```
#E40303  Red
#FF8C00  Orange
#FFED00  Yellow
#008026  Green
#24408E  Blue
#732982  Violet
```

### Transgender Flag
```
#5BCEFA  Light Blue
#F5A9B8  Light Pink
#FFFFFF  White
#F5A9B8  Light Pink
#5BCEFA  Light Blue
```

### Progress Pride Flag
```
Chevron:
  #FFFFFF  White
  #F5A9B8  Pink
  #5BCEFA  Blue
  #613915  Brown
  #000000  Black

Rainbow background:
  (same as rainbow flag)
```
