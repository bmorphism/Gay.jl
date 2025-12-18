# DifferentialEquations.jl

> Source: https://docs.sciml.ai/DiffEqDocs/stable/

Suite for numerically solving differential equations in Julia, Python, and R.

## Supported Equations

- Discrete equations (function maps, Gillespie/Markov simulations)
- Ordinary differential equations (ODEs)
- Split and Partitioned ODEs (Symplectic integrators, IMEX Methods)
- Stochastic ODEs (SODEs/SDEs)
- Stochastic DAEs (SDAEs)
- Random differential equations (RODEs/RDEs)
- Differential algebraic equations (DAEs)
- Delay differential equations (DDEs)
- Neutral, retarded, algebraic DDEs
- Stochastic DDEs
- Mixed discrete/continuous (Hybrid, Jump Diffusions)
- (Stochastic) PDEs (finite difference + FEM)

## Integrations

- GPU acceleration: CUDA.jl, DiffEqGPU.jl
- Automated sparsity detection: Symbolics.jl
- Automatic Jacobian coloring: SparseDiffTools.jl
- Linear solvers: LinearSolve.jl
- Arbitrary precision: BigFloats, ArbNumerics.jl
- Unit checked: Unitful
- Parallel Ensemble Simulations

## Analysis Features

- Forward/Adjoint Sensitivity Analysis
- Parameter Estimation, Bayesian Analysis
- Neural ODEs: DiffEqFlux.jl
- Global Sensitivity Analysis
- Uncertainty Quantification

## Gay.jl Extension Targets

### GayDiffEqExt (#6)
- Solution trajectory coloring
- Phase space visualization
- Bifurcation diagram colors

### GayODEExt (#7)
- ODE solver step coloring
- Timestep visualization

### GaySDEExt (#8)
- Colored noise paths
- Stochastic trajectory families

### SPI Opportunities
- Parallel ensemble with SPI fingerprints
- Trajectory XOR for divergence detection
- Colored Lyapunov exponents
