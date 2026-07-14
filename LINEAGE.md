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

## Family-tree census (measured 2026-07-14)

There are exactly **two** lineage roots in this repo, not many: the orphan
kernel root (`c58702c`, current `gay`) and the `bb1d1d8` family, of which the
archived monorepo HEAD and all ten legacy branches are divergent limbs.
Divergence points below are measured merge-bases against
`lineage/monorepo-v0.1.0`; these branches are family, so ordinary git
machinery (merge/rebase/cherry-pick) applies among them — the semilattice
join is only needed across the two roots.

| branch | commits | diverged | last | contents |
|---|---|---|---|---|
| `master` | 76 | `0f27610` 2025-12-15 | 2026-06-15 PR #229 | parallel main line through PR #229 |
| `xf-integration` | 98 | `9034077` 2026-03-25 | 2026-06-15 PR #231 | splitmixrgb-xf — direct ancestor of kernel `hash_color_*`; **port-to-kernel candidate** |
| `zmod3-elegant` | 89 | (bb1d1d8 family) | 2026-03-25 | 176 Lean 4 GF(3) theorems + trit type; **pairs with `spi_trit`, port candidate** |
| `claude/launch-gay-passport-LW6hS` | 87 | (family) | 2026-03-26 | Ghani–Hedges open games on did:gay holders |
| `feature/topos-staging-area` | 95 | (family) | 2026-01-16 | dafny `spi_galois.dfy`, seed-1069 alignment |
| `propagator` | 58 | `df0e481` 2025-12-11 | 2025-12-11 | LearnableColorSpace + Enzyme autodiff |
| `slave` | 71 | = its own HEAD `0f27610` | 2025-12-15 | frozen fork-point marker of the master/gay split (GaySplittableRNG) |
| `add-zigzagboomerang-rebased` | — | (family) | 2025-12-15 | seed-as-secret hygiene |
| `bruhat-tits-curriculum` | — | (family) | 2025-12-15 | `world()` runner, Split3 + Sentinel |
| `integrate-ferrite-…` | — | (family) | 2026-03-26 | CI extras/targets fix |

`compathelper/*` (192 refs) are lineage citizens of the family tree — the
bots' proposal record. They were deleted once in error and restored
SHA-identical from the Activity API; they are not sweepable ("we are we").
