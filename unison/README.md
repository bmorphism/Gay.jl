# Gay.u - Unison Port of Gay.jl

Deterministic color generation for Unison using the builtin `splitmix` handler for the `Random` ability.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Unison Random Ability                     │
├─────────────────────────────────────────────────────────────┤
│  ability Random where                                        │
│    natIn : Nat -> Nat -> {Random} Nat                       │
│    nat : {Random} Nat                                        │
│    boolean : {Random} Boolean                                │
│    ...                                                       │
├─────────────────────────────────────────────────────────────┤
│                    splitmix Handler                          │
├─────────────────────────────────────────────────────────────┤
│  splitmix : Nat -> '{Random} a -> a                         │
│                                                              │
│  - Uses SplitMix64 algorithm internally                      │
│  - Same seed → same sequence (SPI guarantee)                 │
│  - Deterministic regardless of execution order               │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Strong Parallelism Invariance (SPI)

The `splitmix` handler ensures that:
- `splitmix 42 do someRandomComputation` always produces the same result
- Parallel execution produces same colors as sequential
- Colors are reproducible across sessions, machines, and time

### GF(3) Trit Assignment

Each color index maps to a balanced ternary trit:
- **Minus (-1)**: Validation, verification, analysis
- **Ergodic (0)**: Coordination, balance, infrastructure
- **Plus (+1)**: Generation, creation, synthesis

Conservation law: `(-1) + (0) + (+1) ≡ 0 (mod 3)`

## Usage

```unison
-- Load in UCM
scratch/main> load unison/Gay.u

-- Get color at index 42 using default Gay seed
> colorAtDefault 42

-- Get color at index with custom seed
> colorAt 1069 42

-- Generate a palette of 10 colors
> palette gay_seed 10

-- Generate a balanced triad
> balancedTriad gay_seed 1

-- Run demo
scratch/main> run demo
```

## Types

```unison
structural type Trit = Minus | Ergodic | Plus

structural type RGB = RGB Nat Nat Nat

structural type GayColor = GayColor RGB Trit Nat
```

## Core Functions

| Function | Type | Description |
|----------|------|-------------|
| `colorAt` | `Nat -> Nat -> GayColor` | Color at seed + index |
| `colorAtDefault` | `Nat -> GayColor` | Color using default seed |
| `palette` | `Nat -> Nat -> [GayColor]` | Generate n colors |
| `balancedTriad` | `Nat -> Nat -> [GayColor]` | 3 colors with balanced trits |
| `indexToTrit` | `Nat -> Trit` | Index → GF(3) trit |
| `Trit.isBalanced` | `[Trit] -> Boolean` | Check conservation |

## Correspondence with Gay.jl

| Gay.jl (Julia) | Gay.u (Unison) |
|----------------|----------------|
| `color_at(42)` | `colorAtDefault 42` |
| `gay_seed!(1069)` | `colorAt 1069 idx` |
| `splitmix64(x)` | `splitmix seed do ...` |
| `Trit` struct | `structural type Trit` |
| `GayRNG` | `Random` ability + handler |

## Seeds

```unison
gay_seed = 0x6761795f636f6c6f  -- "gay_colo" as bytes (canonical)
zubuyul  = 1069               -- from unison-acset skill
```

## Pride Flags

Built-in color palettes for pride flags:
- `rainbow` - Gilbert Baker rainbow flag
- `trans` - Transgender pride flag
- `bi` - Bisexual pride flag
- `nonbinary` - Nonbinary pride flag

## Integration with Cat# Bicomodule

Unison shares the same bicomodule structure as Clojure in the Cat# equipment:

```
Trit: 0 (ERGODIC)
Home: Prof
Poly Op: ⊗
Kan Role: Adj
```

This means Unison and Clojure can exchange color streams while preserving GF(3) conservation.

## License

Same as Gay.jl - MIT License
