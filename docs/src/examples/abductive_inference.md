# Abductive Inference

Abduction (ἀπαγωγή) is Peirce's "inference to the best explanation" - reasoning
from observed effects back to their causes.

In Gay.jl, colors are **signs** produced deterministically by the SPI system.
Abduction lets us recover the hidden parameters (seed, index) that produced them.

## The Three Modes of Inference

| Mode | Formula | Gay.jl Example |
|------|---------|----------------|
| **Deduction** | rule + case → result | `color_at(42, seed=0xDEADBEEF)` → color |
| **Induction** | cases + results → rule | Many (idx, color) pairs → "golden angle hue spacing" |
| **Abduction** | rule + result → case | color + rule → (seed=0xDEADBEEF, idx=42) |

## GayAbducer: The Inference Engine

`GayAbducer` accumulates observations and infers the underlying seed.

### Basic Usage

```julia
using Gay

# Someone generated these colors - what seed did they use?
mystery_colors = [
    RGB(0.87, 0.31, 0.38),
    RGB(0.92, 0.35, 0.39),
    RGB(0.21, 0.06, 0.81),
]

# Create abducer and register observations
abducer = GayAbducer()
for (i, c) in enumerate(mystery_colors)
    register_observation!(abducer, c; index=i)
end

# Infer the seed
seed = infer_seed(abducer)
println("Inferred seed: 0x$(string(seed, base=16))")
println("Confidence: $(abducer.confidence * 100)%")
```

### When You Know Candidate Seeds

If you have a list of likely seeds, provide them for faster inference:

```julia
abducer = GayAbducer()
# ... register observations ...

seed = infer_seed(abducer; seed_candidates=[
    0xDEADBEEF,
    0xCAFEBABE,
    0x12345678,
])
```

### Verifying SPI Consistency

Use abduction to verify that two systems produce identical colors:

```julia
# Machine A generates colors
seed_a = 0xDEADBEEF
colors_a = [color_at(i; seed=seed_a) for i in 1:10]

# Machine B receives colors over network
colors_b = receive_colors()  # Should match colors_a

# Verify consistency
abducer = GayAbducer()
for (i, c) in enumerate(colors_b)
    register_observation!(abducer, c; index=i)
end

inferred = infer_seed(abducer; seed_candidates=[seed_a])
if inferred == seed_a && abducer.confidence == 1.0
    println("✓ SPI verified: machines agree!")
else
    println("✗ SPI violation: colors differ")
end
```

## Direct Abduction Functions

For single-value inference without accumulating observations:

### Find Index (Given Seed)

```julia
# You know the seed, want to find which index produced a color
c = color_at(42; seed=0xDEADBEEF)

idx, dist, exact = abduce_index(c, 0xDEADBEEF; max_index=1000)
# idx = 42, exact = true
```

### Find Seed (Given Index)

```julia
# You know the index, want to find which seed produced a color
c = color_at(42; seed=0xDEADBEEF)

seed, dist, exact = abduce_seed(c, 42)
# seed = 0xDEADBEEF, exact = true
```

### Wrap in Abducible

```julia
# Track provenance alongside the value
ab = Abducible(some_color)
abduce(ab; seed=0xDEADBEEF, max_index=1000)

println("Index: $(ab.origin_index)")
println("Confidence: $(ab.confidence)")
println("Provenance: $(ab.provenance)")  # :inferred or :hypothesized
```

## Inverse Operations

Abduction also provides inverse operations for permutations:

### Inverse Permutation

```julia
d = Derangeable(6; seed=0xDEADBEEF)
perm = derange_indices(d, 1)  # [3, 5, 1, 6, 2, 4]

inv = abduce_inverse(perm)    # The inverse permutation

# Verify: applying both gives identity
for i in 1:6
    @assert inv[perm[i]] == i
end
```

### Recover Derangement from Before/After

```julia
original = [1, 2, 3, 4, 5, 6]
deranged = [3, 5, 1, 6, 2, 4]

perm = abduce_derangement(deranged, original)
# perm[i] tells you: deranged[i] = original[perm[i]]
```

### Cycle Decomposition

```julia
perm = [3, 5, 1, 6, 2, 4]
cycles = abduce_cycle(perm)
# [[1, 3], [2, 5], [4, 6]] - three 2-cycles
```

### Parity Analysis

```julia
perm = [3, 5, 1, 6, 2, 4]
even, odd = abduce_parity(perm)
# Counts elements where (i ⊻ σ(i)) & 1 == 0 vs 1
```

## Structure Inference

Beyond seed recovery, analyze structural patterns:

```julia
abducer = GayAbducer()
for i in 1:20
    register_observation!(abducer, color_at(i; seed=0xDEADBEEF); index=i)
end

structure = infer_structure(abducer)

println("Pattern: $(structure.pattern)")        # :clustered, :polarized, :dispersed
println("Mean hue: $(structure.mean_hue)°")
println("Hue variance: $(structure.hue_variance)")
println("Magnetization: $(structure.magnetization)")  # Spin bias
```

## Color Utilities

### Perceptual Distance

```julia
# CIELAB ΔE*ab (1.0 ≈ just noticeable difference)
dist = color_distance(color1, color2)
```

### Fast Fingerprinting

```julia
# 24-bit hash for fast lookup
fp = color_fingerprint(color)
```

### Nearest Neighbor

```julia
palette = [color_at(i) for i in 1:100]
matches = find_nearest_color(target_color, palette; top_k=3)
# Returns [(index, color, distance), ...]
```

## The Semiotic Triangle

```
        Object (seed, index)
           /           \
          /   Abduction \
         /               \
    Sign ←───────────── Interpretant
  (color)    Deduction    (rule: SPI)
```

- **Sign**: The observed color
- **Object**: The hidden (seed, index) pair that caused it  
- **Interpretant**: The SPI rule that connects them

Abduction completes the triangle by inferring the Object from Sign + Interpretant.

## Demo

Run the interactive demo:

```julia
using Gay
semiosis(:abduce)
```
