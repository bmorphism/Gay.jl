# Gay.jl Formal Verification Roadmap

## Overview

Gay.jl includes formal verification at two levels:
1. **Dafny**: Automated theorem proving for algorithmic properties
2. **Narya**: Higher observational type theory for categorical/compositional properties

## Release Schedule

### v0.4.0 (Current) - Foundation Proofs

**Dafny proofs** (automated verification):
- `SplitMixTernary.dfy` - Core PRNG properties
  - Determinism: same seed → same output
  - Path invariance: step(n) ∘ step(m) = step(m+n)
  - GF(3) conservation: triadic balance
  - Bounded output: triadic_amount in [-limit, +limit]
- `spi_galois.dfy` - Galois connection
  - α(γ(c)) = c closure
  - XOR fingerprint monoid laws
  - p-adic color integration
- `GayMcpCriticalProofs.dfy` - SPI properties
  - Roundtrip recovery: abduce ∘ color_at recovers seed
  - SPI guarantee: parallel = sequential
  - Reafference loop closure: self-recognition
  - Indexless property: O(1) random access

**Narya proofs** (type-theoretic):
- `gf3.ny` - GF(3) field structure
- `bridge_sheaf.ny` - Sheaf conditions on ordered locales
- `spi_conservation.ny` - SPI as dependent types
- `worldhop_narya_bridge.ny` - World hopping via bridge types

### v0.4.1 (Planned) - Ratchet & Game Proofs

**New Dafny proofs**:
- `SparsePQRatchet.dfy` - Post-quantum ratchet
  - Forward secrecy: past keys unrecoverable
  - Epoch puncture: subtree erasure correctness
  - SHA3-256 domain separation
- `OpenGame.dfy` - Compositional game theory
  - Nash equilibrium characterization
  - Marginal vanishing condition
  - WEV (World Extractable Value) bounds

**New Narya proofs**:
- `ratchet_forward_secrecy.ny` - Forward secrecy as bridge type
- `play_coplay_duality.ny` - Bidirectional game morphisms
- `equilibrium_fixpoint.ny` - Equilibrium as type-theoretic fixed point

### v0.4.2 (Planned) - Sheaf & Cohomology Proofs

**New Dafny proofs**:
- `StructuredDecomposition.dfy` - Tree decomposition verification
  - Bag adhesion conditions
  - FPT algorithm correctness
- `CechCohomology.dfy` - H^1 = 0 verification
  - Eventual consistency proof

**New Narya proofs**:
- `sheaf_neural_network.ny` - Sheaf Laplacian coordination
- `cohomological_consistency.ny` - CRDT conflict resolution
- `pentagon_hexagon.ny` - Monoidal coherence conditions

## File Structure

```
verification/
├── dafny/
│   ├── SplitMixTernary.dfy        # v0.4.0 ✓
│   ├── spi_galois.dfy             # v0.4.0 ✓
│   ├── GayMcpCriticalProofs.dfy   # v0.4.0 ✓
│   ├── SparsePQRatchet.dfy        # v0.4.1 (planned)
│   └── OpenGame.dfy               # v0.4.1 (planned)
└── narya/
    ├── gf3.ny                     # v0.4.0 ✓
    ├── bridge_sheaf.ny            # v0.4.0 ✓
    ├── spi_conservation.ny        # v0.4.0 ✓
    ├── worldhop_narya_bridge.ny   # v0.4.0 ✓
    ├── ratchet_forward_secrecy.ny # v0.4.1 (planned)
    └── play_coplay_duality.ny     # v0.4.1 (planned)
```

## Property Matrix

| Property | Dafny Proof | Narya Proof | Status |
|----------|-------------|-------------|--------|
| SplitMix64 determinism | `SplitMixTernary.dfy:Determinism` | `spi_conservation.ny:determinism_proof` | ✅ v0.4.0 |
| Path invariance | `SplitMixTernary.dfy:PathInvariance` | — | ✅ v0.4.0 |
| GF(3) conservation | `SplitMixTernary.dfy:GF3AlwaysConserved` | `gf3.ny:GF3Conserved` | ✅ v0.4.0 |
| Galois closure | `spi_galois.dfy:GaloisClosure` | — | ✅ v0.4.0 |
| XOR monoid laws | `spi_galois.dfy:XorAssociativity` | — | ✅ v0.4.0 |
| Reafference closure | `GayMcpCriticalProofs.dfy:ReafferenceLoopCloses` | `spi_conservation.ny:reafference_proof` | ✅ v0.4.0 |
| Indexless property | `GayMcpCriticalProofs.dfy:ColorAtIndexless` | `spi_conservation.ny:indexless_proof` | ✅ v0.4.0 |
| SPI guarantee | `GayMcpCriticalProofs.dfy:SpiGuarantee` | `spi_conservation.ny:SPIGuarantee` | ✅ v0.4.0 |
| Bridge sheaf | — | `bridge_sheaf.ny:BridgeSheaf` | ✅ v0.4.0 |
| World hopping | — | `worldhop_narya_bridge.ny:hop` | ✅ v0.4.0 |
| Forward secrecy | `SparsePQRatchet.dfy` (planned) | `ratchet_forward_secrecy.ny` (planned) | 📋 v0.4.1 |
| Nash equilibrium | `OpenGame.dfy` (planned) | `equilibrium_fixpoint.ny` (planned) | 📋 v0.4.1 |
| Sheaf cohomology | `CechCohomology.dfy` (planned) | `cohomological_consistency.ny` (planned) | 📋 v0.4.2 |

## Running Verification

### Dafny

```bash
# Install Dafny
brew install dafny  # macOS
# or
dotnet tool install --global Dafny

# Verify all proofs
dafny verify verification/dafny/*.dfy

# Verify single file
dafny verify verification/dafny/SplitMixTernary.dfy
```

### Narya

```bash
# Install Narya (from source)
git clone https://github.com/mikeshulman/narya
cd narya && dune build

# Type-check proofs
narya verification/narya/spi_conservation.ny
narya verification/narya/gf3.ny
```

## Correspondence to Julia Implementation

| Verified Property | Julia Implementation |
|-------------------|---------------------|
| `splitmix64` | `src/splittable.jl:sm64` |
| `color_at` | `src/splittable.jl:color_at` |
| `GF3Conservation` | `src/sparse_pq_ratchet.jl:check_gf3_conservation` |
| `forward_ratchet!` | `src/sparse_pq_ratchet.jl:forward_ratchet!` |
| `Nash equilibrium` | `src/gay_open_game.jl:is_equilibrium` |
| `World hopping` | `src/world_coworld_bridge.jl:handoff!` |

## Verification Philosophy

1. **Dafny** for algorithmic correctness: loops, bounds, determinism
2. **Narya** for compositional structure: types, bridges, coherence
3. **Both** verify the same invariants from different angles
4. **SPI** is the unifying property: same seed → same color, always

## Contributing

To add a new verified property:
1. Identify the invariant in Julia code
2. Write Dafny proof for algorithmic aspect
3. Write Narya proof for type-theoretic aspect
4. Update this roadmap and CHANGELOG.md
5. Run verification before merge
