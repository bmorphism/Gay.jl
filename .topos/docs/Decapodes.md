# Decapodes.jl

> Source: https://algebraicjulia.github.io/Decapodes.jl/stable/

Decapodes.jl is a framework for developing, composing, and simulating physical systems.

## Overview

Decapodes.jl synthesizes:
- **Applied Category Theory (ACT)** techniques for formalizing and composing physics equations
- **Discrete Exterior Calculus (DEC)** techniques for formalizing differential operators

### Dependencies
- **CombinatorialSpaces.jl** - discretizing space, DEC operators on simplicial complexes
- **DiagrammaticEquations.jl** - representing equations as formal ACT diagrams

This repository compiles diagrams down to runnable simulation code.

## Key Features

- **Hierarchically composable** simulations
- **Generalizes over any type of manifold**
- **Performant and accurate** with declarative DSL
- **Human-readable** domain specific language

## Example: Grigoriev Ice Cap Dynamics

Demonstrates ice dynamics simulation on manifolds.

## Gay.jl Extension Target: GayDecapodesExt

### Already Implemented
See `/Users/bob/ies/Gay.jl/ext/GayDecapodesExt.jl`

### Key Types to Color
- `EmbeddedDeltaSet2D` - mesh structures
- `DualSimplicialSet` - dual complexes
- DEC operators: `:d`, `:δ`, `:Δ`, `:⋆`, `:♭`, `:♯`, `:∂`

### SPI Opportunities
- Mesh coloring with `color_mesh`
- Operator coloring by type (hue mapping)
- Solution field coloring (Poisson, Euler flow)
- Diffusion equation visualization

### Parallel Tractability
- Parallel mesh vertex coloring
- Distributed DEC operator application
- Multi-resolution multigrid coloring
