# Gay.jl Contributors

Thank you to everyone who has contributed to Gay.jl!

## Core Team

- **bmorphism** - Creator, maintainer

## Contributors

| Contributor | Contribution | PR |
|-------------|--------------|-----|
| **@zubyul** (alice) | ZigZagBoomerang.jl integration - PDMP visualization with SPI-compliant colors, security fix for seed handling | [#185](https://github.com/bmorphism/Gay.jl/pull/185) |

## How to Contribute

1. Fork the repository
2. Create a feature branch from `gay`
3. Make your changes with proper Gay.jl integration:
   - Use `color_at()` not `MersenneTwister`
   - Use `gay_split()` for parallel branches
   - Verify SPI with XOR fingerprint composition
4. Submit a PR

## Contribution Types

- **Examples** (`examples/`) - Demonstrate Gay.jl with other packages
- **Core modules** (`src/`) - Extend chromatic identity system
- **Tests** (`test/`) - Statistical and compositional verification
- **Documentation** (`docs/`) - Tutorials, API docs

---

▽▽▽ Wide-gamut deterministic colors for all ▽▽▽
