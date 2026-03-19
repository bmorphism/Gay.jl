# Hatchery: Bumpus-Gay-Geb Random Walks

## Refinement Triad

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BUMPUS-GAY-GEB REFINEMENT                                                  │
│                                                                             │
│  BUMPUS (Obstruction)     GAY (Chromatic)      GEB (Self-Reference)         │
│  ┌─────────────────┐      ┌─────────────────┐  ┌─────────────────┐          │
│  │ Tree-depth      │      │ Seed bundles    │  │ Categorical     │          │
│  │ Left-recursion  │  ⊕   │ SPI guarantees  │  ⊕  │ Gödel encoding  │          │
│  │ Ambiguity       │      │ Color consistency│  │ Self-rewriting  │          │
│  └────────┬────────┘      └────────┬────────┘  └────────┬────────┘          │
│           │                        │                     │                   │
│           ▼                        ▼                     ▼                   │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │              COMPOSITIONAL GUARDRAILS                           │        │
│  │  Elected obstructions + Persistent unobstructed + Chromatic SPI │        │
│  └─────────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Random Walk Parameters

- **Max steps**: 1069 (configurable)
- **Origin seeds**: Gay seed bundle per repo
- **Clustering**: StUMAP for spectral gap optimization
- **Guardrails**: FM-specific temperature bounds

## Worlds Covered

| World | Tripos | Repos | Spectral Focus |
|-------|--------|-------|----------------|
| Anoma | geb_juvix_arm | 12 | Categorical composition |
| Penumbra | zk_privacy | 9 | ZK curve operations |
| Aptos | core_infrastructure+ | 90 | Block-STM parallelism |

## PEG Strategy Morphisms

```julia
# Parse ↔ Generate duality
parse_to_game(input) → OpenGameArrow
generate_from_game(arrow) → output

# Runtime embeddings
embed_to_runtime(rules, ANOMA_TARGET) → Juvix/ARM
embed_to_runtime(rules, PENUMBRA_TARGET) → decaf377/poseidon377
embed_to_runtime(rules, FM_TARGET) → Transformers with guardrails

# Inversion
invert_from_runtime(embedding) → inverted rules
```

## Usage

```julia
using Gay: GayPEGStrategy

# Parallel retrieve all orgs
repos = parallel_retrieve_orgs()

# Cluster origins for spectral gap
cluster = cluster_origins([r.seed for r in repos], 7)

# Walk from each origin
walks = [hatchery_walk(r, 69) for r in repos]

# Lazy topos glob
for match in topos_glob_lazy("worlds/aptos")
    println(match.path)
end
```
