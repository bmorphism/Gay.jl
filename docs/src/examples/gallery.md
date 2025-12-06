# Gallery Generation

Generate a gallery of 1069 sky models using parallel-safe splittable random streams.

## Running the Gallery Script

```bash
julia --threads=auto scripts/generate_gallery.jl
```

## How It Works

```julia
using Gay

const MASTER_SEED = 42069
const N_MODELS = 1069

gay_seed!(MASTER_SEED)

for i in 1:N_MODELS
    style = [:m87, :sgra, :rings, :custom][mod1(i, 4)]
    model_seed = round(Int, rand() * 100000)

    model = generate_model(style, model_seed)
    score = aesthetic_score(model)

    # Each model is fully reproducible from its seed
end
```

## Model Styles

| Style | Description | Components |
|-------|-------------|------------|
| `:m87` | M87* style | Ring + Gaussian core |
| `:sgra` | Sgr A* style | Crescent + Disk |
| `:rings` | Multi-ring | 2-5 concentric rings |
| `:custom` | Random mix | 2-4 random primitives |

## Aesthetic Scoring

Models are ranked by an aesthetic score based on:

1. **Coverage** (30%): Fraction of image area with intensity > 0.1
2. **Contrast** (20%): Standard deviation of intensities
3. **Symmetry** (30%): Radial symmetry measure
4. **Golden Ratio** (20%): Bonus for ring counts near φ

```julia
score = coverage * 0.3 + contrast * 0.2 + symmetry * 0.3 + golden_bonus * 0.2
```

## Top Models

The gallery consistently produces beautiful multi-ring structures:

```
#1 [rings] seed=51749 (4 rings)
   (ring 0.63 0.23) + (ring 0.91 0.18) + (ring 1.22 0.11) + (ring 1.52 0.29)

#2 [rings] seed=73597 (4 rings)
   (ring 0.73 0.23) + (ring 0.99 0.14) + (ring 1.25 0.23) + (ring 1.56 0.12)

#3 [rings] seed=57547 (4 rings)
   (ring 0.76 0.25) + (ring 1.08 0.18) + (ring 1.38 0.13) + (ring 1.61 0.21)
```

## Output Format

The gallery generates:

1. **Console output**: Top 5 models by aesthetic score
2. **`gallery/catalog.jsonl`**: All 1069 models in JSON Lines format

```json
{"index":1,"style":"m87","seed":51749,"score":0.847,"sexpr":"(ring 0.63 0.23) + ..."}
{"index":2,"style":"sgra","seed":73597,"score":0.823,"sexpr":"(crescent 0.8 0.4 0.2) + ..."}
```

## Reproducibility

The entire gallery is **fully reproducible**:

```julia
# Run 1
gay_seed!(42069)
models_run1 = [generate_model(style, seed) for ...]

# Run 2 (any machine, any thread count)
gay_seed!(42069)
models_run2 = [generate_model(style, seed) for ...]

# Identical results guaranteed by SPI
@assert models_run1 == models_run2
```

This is the **Strong Parallelism Invariance** pattern from Pigeons.jl.
