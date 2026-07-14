# LINEAGE — semilattice promotion, not PR merge

This repo hosts divergent package lineages (distinct Julia UUIDs) under one
name. Three-way merges between them are meaningless, so `gay` is promoted by
**join**, bypassing the PR process:

1. **Archive**: before any promotion, the current `gay` HEAD is pushed to a
   long-living `lineage/<name>` ref. Nothing is ever deleted or rebased away.
2. **Promote**: the new HEAD is pushed to `gay` (force; histories are
   unrelated by construction).
3. **Tags never move.** Version tags accumulate monotonically across
   lineages (`v0.1.0 … v0.3.0` from the monorepo lineage, `v0.5.0+` from the
   kernel lineage).

Semilattice reading: refs are elements, promotion is the join — `gay` is
always the current top, and every prior element stays comparable via its
`lineage/*` ref. History is append-only at the ref level even though `gay`
itself is not fast-forward.

## Lineage registry

| ref | HEAD | identity |
|---|---|---|
| `lineage/monorepo-v0.1.0` | `e3c403a` | uuid `f3dee6b2-1ce2-4cc9-bfb1-25e98f6f315b`, v0.1.0 — 754-file monorepo (GayMC, Metal/Enzyme exts, DuckDB, LispSyntax REPL, lean4, GayIdentifiers.jl, GayLearnableColor.jl) |
| `gay` | *(current)* | uuid `8b449cd3-8280-14dd-1069-000000000042`, v0.5.0 — dep-free kernel + spi-race-canonical `spi_*` surface, weakdep extensions |
| `gay-v0.4-spi-kernel` | *(tracks current)* | development lineage of the current `gay` HEAD (v0.4.0 → v0.5.0) |

To recover anything from a prior lineage: `git checkout lineage/<name> -- <path>`,
or port it forward as a weakdep extension on the current kernel.
