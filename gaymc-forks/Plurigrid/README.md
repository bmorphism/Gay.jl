# Plurigrid/gaymc

> **Compositional energy grid algorithms with chromatic SPI verification**

## Description

This fork of gaymc focuses on **distributed energy systems** where graph algorithms must be:
- Verifiably deterministic across heterogeneous compute (edge devices, cloud, TEEs)
- Compositional across grid partitions (microgrids, distribution, transmission)
- Traceable for regulatory compliance and audit

## Key Features

### 🔋 Grid Decomposition with Chromatic Identity

```julia
using GayMC.Plurigrid

# Load power grid topology
grid = load_ieee_bus_system("ieee118")

# Compute compositional decomposition
# Each partition gets deterministic color for verification
decomp = gay_grid_decomposition(grid; 
    seed=0xPLURIGRID,
    method=:structured  # Bumpus-style structured decomposition
)

# Verify: same decomposition regardless of compute node
@assert decomp.fingerprint == expected_fingerprint
```

### ⚡ Parallel Power Flow with SPI

```julia
# Run power flow analysis on partitions
# SPI guarantees identical results across distributed execution
results = gay_parallel_power_flow(grid, decomp;
    workers=[:edge1, :edge2, :cloud],
    verify_spi=true
)

# Each bus voltage has chromatic identity
for (bus_id, voltage) in results.voltages
    println("Bus $bus_id: V=$(voltage.magnitude), color=$(voltage.color)")
end
```

### 🔐 TEE-Compatible Verification

```julia
# Generate proof of correct execution
proof = gay_execution_proof(results)

# Proof includes chromatic fingerprints at each step
# Verifiable inside Trusted Execution Environment (SGX, TDX, SEV)
verify_in_tee(proof)
```

## Compositional Semantics

Following Bumpus et al.'s "Additive Invariants of Open Petri Nets":

```julia
# Microgrids compose sequentially and in parallel
mg1 = Microgrid(...)
mg2 = Microgrid(...)

# Sequential: power flows from mg1 to mg2
sequential = mg1 ⊗ mg2

# Parallel: mg1 and mg2 share a bus
parallel = mg1 ⊕ mg2

# Invariants (power balance, frequency stability) are additive
@assert power_balance(sequential) == power_balance(mg1) + power_balance(mg2)
```

## Connection to Plurigrid Mission

1. **Decentralized Energy Markets**: Chromatic identity enables trustless verification of grid computations
2. **Resilient Infrastructure**: SPI guarantees reproducibility after failures
3. **Interoperability**: Compositional algorithms work across different grid standards

## Repository

```
Plurigrid/gaymc
├── src/
│   ├── grid_decomposition.jl    # Structured grid partitioning
│   ├── power_flow.jl            # Chromatic power flow
│   ├── market_clearing.jl       # Compositional market algorithms
│   └── tee_verification.jl      # TEE-compatible proofs
├── examples/
│   ├── ieee_systems.jl          # Standard test cases
│   └── microgrid_composition.jl # Compositional examples
└── README.md
```

## Citation

```bibtex
@software{plurigrid_gaymc,
  title={Plurigrid/gaymc: Compositional Energy Grid Algorithms},
  organization={Plurigrid},
  year={2024},
  url={https://github.com/Plurigrid/gaymc}
}
```
