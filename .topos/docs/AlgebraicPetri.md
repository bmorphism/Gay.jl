# AlgebraicPetri.jl

> Source: https://algebraicjulia.github.io/AlgebraicPetri.jl/stable/

AlgebraicPetri.jl is a Julia library for building Petri net agent-based models compositionally. It bridges Catlab.jl and Petri.jl, defining the category of Open Petri Nets as described in [Baez 2018](https://arxiv.org/abs/1808.05415).

## Core Concepts

### Open Petri Nets
- Compositional Petri net modeling
- Category-theoretic semantics
- Integration with Catlab's ACSet framework

### Key Types
- `PetriNet` - Basic Petri net structure
- `OpenPetriNet` - Petri net with exposed interfaces
- `LabelledPetriNet` - Petri nets with labeled places/transitions
- `ReactionNet` - Chemical reaction networks

## Gay.jl Extension Target: GayAlgebraicPetriExt

### Key Types to Color
- Places (species/states)
- Transitions (reactions/events)
- Tokens (population counts)
- Open interfaces (composition boundaries)

### SPI Opportunities
- Place coloring: `hash_color(place_id, seed)`
- Transition coloring: `hash_color(reduce(⊻, input_places ∪ output_places), seed)`
- Token count → lightness mapping
- Stoichiometry matrix entry coloring

### Parallel Tractability
- Parallel transition firing simulation
- Distributed composition of open Petri nets
- XOR fingerprint over marking vectors
- Concurrent reachability analysis

### Integration Points
- Combine with AlgebraicDynamics for ODE generation
- Epidemiological model visualization (SIR, SEIR)
- Chemical reaction network coloring
