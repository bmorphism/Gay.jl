# Gay.jl .topos/ Staging Area Manifest

**Purpose**: Refinement-in-progress directory for incomplete features, experimental code, and reference materials awaiting graduation to versioned repository.

**Status**: Production-ready for public view with core committed; .topos/ contains staging for next release.

---

## Directory Structure

```
.topos/
├── STAGING_MANIFEST.md          ← This file
├── GRADUATION_CHECKLIST.md      ← Review criteria for promotion
│
├── experimental/                ← Work-in-progress code
│   ├── src/                     (37 untracked source modules)
│   ├── test/                    (3 test files needing completion)
│   ├── scripts/                 (2 utility scripts)
│   └── GayCliqueTreesExt.jl    (Extension needing completion)
│
├── llm-context/                 ← Auto-generated documentation
│   ├── llms.txt                 (Summary for LLM context)
│   ├── llms-small.txt           (Compact version)
│   └── llms-full.txt            (Complete version)
│
├── editor-support/              ← IDE/Editor integrations
│   └── whale-world.el           (Emacs lisp integration)
│
├── reference/                   ← Design documentation
│   ├── AGENTS.md                (Naming conventions, patterns)
│   ├── SPI_VERIFICATION_REPORT.md (Correctness proofs)
│   ├── TOPOS_TALKS_REFERENCE.md (Research references)
│   ├── DECIDING_SHEAVES_FLOWCHART.mermaid (Visual guide)
│   ├── WORLD_PATTERN.md         (Design patterns)
│   └── assets/                  (Documentation images)
│
├── docs/                        ← Integration research
│   ├── integrations/            (12 Catlab ecosystem sketches)
│   ├── assets/                  (Collages, diagrams)
│   └── 3xor3match.edn          (Color model reference)
│
└── colors.css                   ← Style definitions
```

---

## File Classifications & Graduation Criteria

### experimental/src/ (37 FILES)

**Modules Needing Triage**:

| File | Purpose | Status | Decision |
|------|---------|--------|----------|
| `acset_tower.jl` | ACSet composition | TBD | Test coverage needed |
| `amp_threads.jl` | Async threading | TBD | Integration test needed |
| `bench_spi_regression.jl` | SPI benchmarks | RUNNABLE | Add to benchmark suite |
| `chaos_vibing.jl` | Chaos theory | TBD | Examples needed |
| `color_chain_cft_test.jl` | CFT testing | TBD | Verification needed |
| `color_logic_pullback.jl` | Category theory | PARTIAL | Documentation needed |
| `compositional_world.jl` | Composition patterns | TBD | Interface design |
| `concept_tensor.jl` | Lattice operations | CORE-CANDIDATE | Review against parallel.jl |
| `ergodic_bridge.jl` | Ergodic theory | RESEARCH | Reference material |
| `gamut_learnable.jl` | Learnable colors | PARTIAL | Examples needed |
| `genetic_search.jl` | GA algorithms | RUNNABLE | Add to examples |
| `hyperdoctrine.jl` | Logic framework | RESEARCH | Catlab integration |
| `kripke_worlds.jl` | Modal semantics | RESEARCH | Documentation needed |
| `modal_descent.jl` | Optimization | TBD | Tests needed |
| `nash_prop_catalog.jl` | Game theory | RESEARCH | Examples needed |
| `parallel_color_scheduler.jl` | **CRITICAL** | READY | **COMMIT NOW** |
| `parallel_seed_search.jl` | Seed mining | RUNNABLE | Benchmark needed |
| `polarized_split.jl` | Splitting logic | TBD | Verification needed |
| `proof_of_color.jl` | PoCP consensus | RESEARCH | Reference only |
| `push_pull_sequence.jl` | Categorical ops | CORE-CANDIDATE | Review vs traced_tensor.jl |
| `quic_interleave.jl` | Protocol coloring | TBD | Tests needed |
| `splitmix_cft_verify.jl` | Hash verification | RUNNABLE | Add to regression |
| `strategic_differentiation.jl` | AD strategy | TBD | Benchmark needed |
| `thread_findings.jl` | Threading analysis | REFERENCE | Keep as doc |
| `tower.jl` | Topological layers | CORE-CANDIDATE | Full test suite |
| `traced_tensor.jl` | Monoidal traces | CORE-CANDIDATE | Review vs push_pull |
| `trajectory_db.jl` | History tracking | TBD | Integration test |
| `tropical_semirings.jl` | Tropical algebra | RESEARCH | Catlab integration |
| `tuning.jl` | Parameter tuning | TBD | Benchmark harness |
| `universal_color.jl` | Color abstraction | EXTRACTABLE | Consider GayColors.jl |
| `verification_report.jl` | Verification | REFERENCE | Auto-generate |
| `whale_bridge.jl` | Whale integration | RUNNABLE | Add to examples |
| `whale_curriculum.jl` | Learning pipeline | RUNNABLE | Add to examples |
| `whale_data.jl` | Data loading | RUNNABLE | Add to examples |
| `whale_demo.jl` | Demo code | RUNNABLE | Add to examples |
| `whale_world.jl` | Whale ecosystem | **CRITICAL** | **EXTRACTABLE** |
| `world_rotators.jl` | World operations | TBD | Tests needed |

**Action Items**:
- [ ] **COMMIT IMMEDIATELY**: `parallel_color_scheduler.jl` (blocking parallelization work)
- [ ] **EXTRACTABLE**: `whale_world.jl` + bridge/curriculum/data/demo → `GayWhaleWorld.jl` package
- [ ] **REVIEW CANDIDATES**: `concept_tensor.jl`, `tower.jl`, `traced_tensor.jl`, `push_pull_sequence.jl`
- [ ] **ADD TESTS**: 15 modules need test coverage before graduation
- [ ] **CONSOLIDATE**: Pair similar files (whale_* files, color_* files, etc.)

---

### experimental/test/ (3 FILES)

**Status**: Needed for complete test coverage

| File | Purpose | Needs |
|------|---------|-------|
| `regression_ternary.jl` | Ternary regression tests | Validation against CI |
| `whale_world_headless.jl` | Headless integration tests | Data mocking |
| `whale_world_terminus.sh` | Shell integration | CI workflow addition |

**Decision**: Add to `test/` after validation.

---

### experimental/scripts/ (2 FILES)

| File | Purpose | Action |
|------|---------|--------|
| `diagram_collage.jl` | Gallery generation | Move to `scripts/` + test |
| `lint_no_demo.jl` | Anti-pattern checker | Move to `.githooks/` |

---

### experimental/GayCliqueTreesExt.jl

**Status**: Untracked extension (incomplete?)

**Decision Needed**:
- [ ] Is CliqueTrees integration complete?
- [ ] Does it have tests?
- [ ] If complete: move to `ext/`
- [ ] If incomplete: document status here

---

### llm-context/ (3 FILES)

**Purpose**: Automatically-generated LLM context files

**Status**: Not versioned (regenerated on release)

**Action**:
- [ ] Add `llm-context/` to `.gitignore`
- [ ] Auto-generate these files on:
  - Release tagging
  - Documentation build
  - CI/CD workflow

**Current Files**:
- `llms.txt` - 53 lines, summary for LLM context
- `llms-small.txt` - 125 lines, compact version
- `llms-full.txt` - 433 lines, complete documentation

---

### editor-support/ (1 FILE)

**Purpose**: IDE/Editor integrations

| File | Target | Status |
|------|--------|--------|
| `whale-world.el` | Emacs lisp | Orphaned from main project |

**Action**: Document setup in README or archive to separate repo.

---

### reference/ (5 FILES + assets/)

**Purpose**: Design documentation and reference materials

| File | Purpose | Status |
|------|---------|--------|
| `AGENTS.md` | Naming conventions (world_* pattern) | POLICY |
| `SPI_VERIFICATION_REPORT.md` | Correctness proofs | REFERENCE |
| `TOPOS_TALKS_REFERENCE.md` | Research references | DOCUMENTATION |
| `DECIDING_SHEAVES_FLOWCHART.mermaid` | Decision guide diagram | DESIGN |
| `WORLD_PATTERN.md` | Design patterns | DESIGN |
| `assets/` | Documentation diagrams/collages | VISUAL |

**Action**: Promote to `docs/src/reference/` after review.

---

### docs/ (12 FILES + assets/)

**Purpose**: Research on Catlab.jl ecosystem integration

**Catlab Integration Sketches**:
1. AlgebraicDynamics.md - Compositional dynamical systems
2. AlgebraicPetri.md - Petri net semantics
3. AlgebraicRewriting.md - Graph rewriting
4. AnanasParallelism.md - Parallelism patterns
5. BlumeCapel23x3.md - 23×3 lattice models
6. Catlab.md - Categorical systems library
7. ColorableFlavorableLattice.md - Color lattice theory
8. CombinatorialSpaces.md - Discrete topology
9. Decapodes.md - Exterior calculus
10. DifferentialEquations.md - ODE/PDE systems
11. Enzyme.md - Automatic differentiation
12. Graphs.md - Graph algorithms

**Status**: Well-organized sketches ready to integrate.

**Action**:
- [ ] Review for completeness
- [ ] Add examples for each integration
- [ ] Promote to `docs/src/integrations/` after examples added

---

### docs/assets/ & docs/integrations/ (EXISTING)

**3xor3match.edn**:
- Color model with seeds, polarities, GF(3) structure
- Ready for: `examples/color_models/3xor3match.edn`

**colors.css**:
- Color styling definitions
- Ready for: `docs/src/assets/colors.css`

---

## Graduation Pipeline

```
.topos/experimental/
    ↓
[Add tests, verify, document]
    ↓
[Review & approve]
    ↓
Core source/ (if essential)
OR
docs/examples/ (if illustrative)
OR
Separate package repo (if extractable)
```

### Criteria for Graduation

**To `src/` (Core)**:
- ◆ Full test coverage (unit + integration)
- ◆ Passes SPI verification (if parallel)
- ◆ Documented in API docs
- ◆ No breaking changes to exports
- ◆ Performance profiled

**To `examples/` (Illustrations)**:
- ◆ Runnable without errors
- ◆ Demonstrates a concept
- ◆ Has docstring explaining purpose
- ◆ Appropriate for new users

**To Separate Package** (if high-extractability):
- ◆ No dependencies on internal src/ modules
- ◆ Complete API design
- ◆ Comprehensive tests
- ◆ Published as independent package
- ◆ Cross-linked in main README

---

## Critical Path Items (UNBLOCK GITHUB INTEGRATION)

**MUST COMMIT (from earlier session)**:
- [ ] `parallel_color_scheduler.jl` - Blocking GPU parallelization work

**DECISION NEEDED**:
- [ ] Keep or extract: `whale_world.jl` ecosystem (5 files)
- [ ] Keep or reject: `GayCliqueTreesExt.jl` (incomplete?)
- [ ] Test coverage: 15+ modules need test suites

---

## Summary Statistics

| Category | Count | Status | Action |
|----------|-------|--------|--------|
| Source modules | 37 | Limbo | Review & triage |
| Test files | 3 | Incomplete | Complete + integrate |
| Scripts | 2 | Partial | Integrate |
| Extensions | 1 | Unknown | Decision |
| Documentation | 8 | Reference | Promote to docs/src/ |
| Research sketches | 12 | Ready | Review + examples |
| Reference files | 1 | Policy | Document |
| Editor support | 1 | Orphaned | Archive or integrate |
| LLM context | 3 | Generated | Auto-gen on release |

**Total**: 68 files staged for triage/graduation.

---

## Next Steps

1. **Immediate** (Today)
   - [ ] Review CRITICAL files: `parallel_color_scheduler.jl`, `whale_world.jl`
   - [ ] Decide: Extract whale_world to separate package?
   - [ ] Commit parallelization blocker

2. **Short-term** (This week)
   - [ ] Add test coverage to top 10 modules
   - [ ] Promote completed items from .topos/reference/ to docs/src/
   - [ ] Set up auto-generation for llms*.txt files

3. **Medium-term** (This sprint)
   - [ ] Complete remaining modules (15 modules × 2-4 hours each)
   - [ ] Full review of Catlab integration sketches
   - [ ] Publish extracted packages (if applicable)

4. **Long-term** (Next release)
   - [ ] All .topos/experimental → versioned repos or reject
   - [ ] All .topos/docs → docs/src/
   - [ ] Zero untracked files in main Gay.jl

---

**Generated**: December 18, 2025
**Status**: Staging Area Operational
**Purpose**: Keep main repository clean while supporting active development

