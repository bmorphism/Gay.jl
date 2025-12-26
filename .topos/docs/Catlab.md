# Catlab.jl

> Source: https://algebraicjulia.github.io/Catlab.jl/stable/

Catlab.jl is a framework for applied and computational category theory, written in Julia. It provides a programming library and interactive interface for applications of category theory to scientific and engineering fields. It emphasizes monoidal categories due to their wide applicability but can support any categorical structure formalizable as a generalized algebraic theory.

## What is Catlab?

**Programming library**: Data structures, algorithms, and serialization for applied category theory. Macros offer convenient syntax for specifying categorical doctrines and type-safe symbolic manipulation. Wiring diagrams (string diagrams) supported with GraphML and JSON serialization.

**Interactive computing environment**: Jupyter notebook support. LaTeX display for symbolic expressions, visualization via Compose.jl, Graphviz, or TikZ.

**Computer algebra system**: CAS for categorical algebra with expressions typed using generalized algebraic theories (GATs).

## Key Concepts

### Generalized Algebraic Theories (GATs)
See GATlab documentation: https://algebraicjulia.github.io/GATlab.jl

### Abstract Field Convention
Assumes all subtypes of abstract types have same field names/organization. Enables:
- 12+ struct subtypes with shared interface
- Copy methods via field access
- Shorter debug message names

## Table of Contents
- Standard library of theories
- Wiring diagrams
- Graphics
- Programs

## Gay.jl Extension Target: GayCatLabExt

### Key Types to Color
- `ACSet` - Attributed C-sets (copresheaves)
- `WiringDiagram` - String diagrams
- `FinSet`, `FinFunction` - Finite sets/functions
- GAT expressions

### SPI Opportunities
- Morphism fingerprinting via composition XOR
- Pullback/pushout color conservation
- Wiring diagram edge coloring
- Categorical limit/colimit verification

### Parallel Tractability
- Parallel morphism enumeration
- Distributed ACSet operations
- String diagram parallel composition
