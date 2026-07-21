# IACR paper-readiness rubric

This is an evidence ledger, not a submission claim. The current manuscript is
**not submission-ready**. In particular, finite-field identities and software
tests do not establish a cryptographic security theorem.

## Standards baseline

- The [Cryptology ePrint Archive](https://eprint.iacr.org/) performs only a
  scope and minimal-conditions screen; posting is not peer review.
- The [IACR Guidelines for Authors](https://www.iacr.org/docs/author.pdf)
  govern confidentiality, conflicts, and submission conduct. A target venue's
  current call for papers remains authoritative for anonymity, format, page
  limits, supplementary material, and simultaneous-submission rules.
- The [IACR Artifact Archive](https://artifacts.iacr.org/) describes artifacts
  as support for reproducibility and reuse. A software artifact needs source,
  dependencies, build and test instructions, the evaluation platform, and a
  path from raw output to each paper result. Artifact review does not prove the
  paper's scientific claims.
- CHES artifact evaluations distinguish availability, functionality, and
  reproduction. We must claim only the badge actually awarded.

Because venue rules change, we must freeze a dated copy or URL of the exact
call before submission. The repository currently names ePrint, CHES 2026, and
ASIACRYPT 2026 without selecting one authoritative submission contract.

## Rubric and current evidence

| Gate | Required evidence | Current status |
|---|---|---|
| Scope and novelty | Precise problem, closest cryptographic work, delta over prior art | **Missing**: related work is sparse and several entries need bibliographic verification |
| Syntax and semantics | Algorithms with typed inputs, outputs, state, failure behavior, and parameters | **Incomplete**: the color projection is specified; the entropy-source and composition semantics are not |
| Security model | Parties, trust assumptions, adversary class, corruption, setup, leakage, and security parameter | **Missing** |
| Security definition | Game or ideal functionality with quantified advantage and success event | **Missing** |
| Theorem | Assumptions and conclusion matching the definition | **Contradicted**: GF(3) neutrality alone is asserted to imply UC/random-oracle security |
| Proof | Simulator or reduction with explicit hybrids and bounds | **Missing** |
| Entropy reasoning | Correct source model, conditional min-entropy, independence assumptions, extractor theorem, output length and error | **Contradicted**: XOR is called amplification without an extractor analysis; the leftover hash lemma is invoked for concatenation |
| Algebraic claim | Well-typed R-matrix/braiding and a proof of the Yang--Baxter equation | **Missing**: the displayed scalar-valued map is not enough to establish the claimed braided category |
| Formal verification | Toolchain lock, source theorem names, clean build, and claim-to-theorem crosswalk | **Incomplete**: Lean proves some GF(3) identities, not the manuscript's security theorem |
| Implementation fidelity | Paper algorithm mapped to package symbols and tests | **Missing**: README points to one source file but provides no crosswalk |
| Evaluation | Research questions, baselines, datasets/sources, platform, repetitions, statistics, and limitations | **Missing** |
| Artifact functionality | Clean build, pinned dependencies, one-command tests, expected output, runtime, resource bounds | **Incomplete** |
| Artifact reproduction | Script regenerates every paper table/figure/result from raw inputs | **Missing** |
| Claim hygiene | Every numeric, empirical, novelty, formal-verification, and security claim has a source or executable witness | **Failing** |
| Identity boundary | Typed referents carry identity; colors/hashes are representations and evidence only | **In progress**; executable EDN gates exist |
| Ethics and submission | Authors approve; conflicts and overlap disclosed; exact venue anonymity and dual-submission rules checked | **Not yet attestable** |
| Archival package | Source-only clean tree, license, citation metadata, immutable artifact identifier, checksums | **Missing** |

## Blocking claim repairs

1. Remove the UC/random-oracle theorem and entropy-amplification corollary until
   a real construction, definition, and reduction exist. A visualization or
   policy label cannot supply entropy or security.
2. Recast GF(3) conservation as a bookkeeping invariant unless a cryptographic
   consequence is separately proved.
3. Specify whether the construction concatenates, XORs, or extracts. These are
   different operations with different entropy guarantees.
4. Replace the QCD analogy with a clearly non-security-bearing presentation
   layer, or prove the exact algebraic structure being claimed.
5. Generate a machine-readable claim ledger mapping each theorem and empirical
   statement to a proof, test, dataset, or explicit `unverified` state.

## Local gate

Run:

```bash
julia --project=. scripts/audit_iacr_paper.jl
julia --project=. scripts/audit_iacr_paper.jl --strict
```

The first command reports the ledger. `--strict` must remain failing until all
blocking items have pointable evidence; submission readiness requires it to
exit successfully in a clean checkout.
