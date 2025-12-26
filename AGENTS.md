# AGENTS.md - Gay.jl Development Guidelines

## Build & Test Commands

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using Pkg; Pkg.precompile()'
julia --project=. scripts/lint_no_demo.jl  # Check for demo_ violations
```

## Code Style

### FORBIDDEN: `demo_` Prefix

**Never use `demo_` prefix for functions.** This pattern is prohibited because:
- Demos print output and discard state
- Demos do not compose
- Demos cannot be tested systematically

### REQUIRED: `world_` Prefix for Persistent State Builders

Use `world_` prefix for functions that:
- Build persistent state (ACSets, worlds, configurations)
- Return composable structures
- Can be merged with other worlds

```julia
# ◇ FORBIDDEN
function demo_ancestry_tracing(threads)
    println("Tracing ancestry...")  # Side effect!
    # ... computation discarded
end

# ◆ REQUIRED
function world_ancestry_tracing(threads)::AncestryWorld
    # Returns persistent, composable structure
    AncestryWorld(materialize_ancestry!(threads))
end
```

### World Builder Pattern Requirements

All `world_` functions must return types implementing:
- `length(world)` - cardinality
- `merge(w1, w2)` - monoidal composition
- `fingerprint(world)` - SPI-compliant hash

### Enforcement

Run `scripts/lint_no_demo.jl` before committing. CI will fail on violations.

## Ontology: We, Not I/You

Use collective pronouns in code and documentation:
- ◇ `MyWorld`, `YourType`, `my_config`
- ◆ `OurWorld`, `SharedType`, `our_config`

This reflects the collaborative, compositional nature of the Gay.jl ecosystem.

## Core Files (Do Not Break)

These files constitute the solidifying core:
1. `src/Gay.jl` - Main module
2. `src/splittable.jl` - GayRNG with SPI guarantees
3. `src/swarm_triad.jl` - Mandatory 3-way split
4. `src/scoped_propagators.jl` - ACSet ancestry materialization
5. `src/universal_color.jl` - Multiparadigm color type
6. `src/colorspaces.jl` - Wide-gamut support

## Branch Naming

- `world/convert-<component>` - demo_ → world_ conversions
- `feature/<name>` - New features
- `fix/<issue>` - Bug fixes
