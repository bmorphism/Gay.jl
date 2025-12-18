# AlgebraicDynamics.jl

> Source: https://algebraicjulia.github.io/AlgebraicDynamics.jl/stable/

AlgebraicDynamics.jl is a library for compositional dynamical systems built on Catlab.jl. It provides a software interface for specifying and solving dynamical systems with compositional and hierarchical structure using operads and operad algebras.

## Composition Process

1. **Pick composition syntax**: Undirected wiring diagrams, directed wiring diagrams, or open CPGs
2. **Define composition pattern**: Specify how primitive systems connect
3. **Define primitive systems**: Resource sharers (undirected) or machines (directed)
4. **Compose with `oapply`**: Apply operad algebra to get composite system

## Key Types

### Resource Sharers (Undirected Composition)
- `ContinuousResourceSharer{T}` - Continuous time: u̇(t) = f(u(t), p, t)
- `DiscreteResourceSharer{T}` - Discrete time: u_{n+1} = f(u_n, p, t)
- Components: ports, states, dynamics function f, port map m

### Machines (Directed Composition)
- `ContinuousMachine{T}` - Continuous dynamics with inputs/outputs
- `DiscreteMachine{T}` - Discrete dynamics with inputs/outputs
- Components: inputs, states, outputs, dynamics f, readout r

## Solving

- Continuous: `ODEProblem`, `DDEProblem` (use OrdinaryDiffEq.jl)
- Discrete: `DiscreteProblem` or `trajectory`
- Recommended solvers: `Tsit5()` with `dtmax`, `FRK65(w=0)`

## References
- [Schultz et al. 2019](https://arxiv.org/abs/1609.08086) - CDS, DDS operad algebras
- [Vagner et al. 2015](https://arxiv.org/abs/1408.1598) - Algebra of open systems
- [Baez and Pollard 2017](https://arxiv.org/abs/1704.02051) - Dynam operad
- [Libkind 2020](https://arxiv.org/abs/2007.14442) - Overview of operad algebras

## Gay.jl Extension Target: GayAlgebraicDynamicsExt

### Key Types to Color
- Resource sharers and machines
- Wiring diagrams (composition patterns)
- State trajectories
- Port/wire connections

### SPI Opportunities
- State trajectory coloring by variable index
- Wiring diagram edge coloring (box XOR)
- Composition pattern fingerprinting
- Phase portrait coloring

### Parallel Tractability
- Parallel `oapply` for independent subsystems
- Distributed trajectory simulation
- XOR fingerprint over composed system states
- Parallel sensitivity analysis

### Integration Points
- Combine with DifferentialEquations.jl solvers
- AlgebraicPetri integration for reaction networks
- Decapodes for PDE systems
