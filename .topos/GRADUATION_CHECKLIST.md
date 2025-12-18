# .topos/ Graduation Checklist

Criteria for promoting files from staging area to versioned repository.

---

## Pre-Graduation Review

### Code Quality Criteria

**Must Pass**:
- [ ] No syntax errors (linting passes)
- [ ] No unused imports
- [ ] No TODO comments without explanations
- [ ] Code style consistent with project
- [ ] No `@suppress` warnings or errors

**Should Pass**:
- [ ] Aqua.jl QA clean (ambiguities, stale deps, exports)
- [ ] No performance regressions vs baseline
- [ ] Memory usage acceptable

---

## Testing Criteria

**Unit Tests**:
- [ ] All public functions have tests
- [ ] Edge cases covered (empty, single, large inputs)
- [ ] Error conditions tested
- [ ] Test pass rate: 100%
- [ ] Tests isolated (no dependencies between tests)

**Integration Tests**:
- [ ] Works with rest of Gay.jl ecosystem
- [ ] No breaking changes to existing APIs
- [ ] Backward compatible OR breaking changes documented

**SPI/Parallelism Tests** (if applicable):
- [ ] Produces deterministic results (same seed)
- [ ] XOR fingerprinting verified
- [ ] Thread-safe under contention
- [ ] Passes Jepsen fuzz testing (if distributed)
- [ ] Scalability validated (1-8 threads)

**Cross-Substrate Tests** (if applicable):
- [ ] CPU backend passes
- [ ] Metal backend passes
- [ ] CUDA backend passes
- [ ] Results identical across backends

---

## Documentation Criteria

**API Documentation**:
- [ ] All public functions have docstrings
- [ ] Docstrings include examples
- [ ] Parameter types documented
- [ ] Return types documented
- [ ] Exceptions/errors documented
- [ ] Links to related functions included

**User-Facing Documentation**:
- [ ] Tutorial/example added to `docs/src/examples/`
- [ ] Integration guide added (if integrates with other modules)
- [ ] Performance characteristics documented
- [ ] Use cases illustrated

**Developer Documentation**:
- [ ] Internal algorithms explained (comments)
- [ ] Data structure invariants documented
- [ ] Key design decisions recorded
- [ ] Limitations noted

---

## Performance Criteria

**Benchmarking** (if performance-critical):
- [ ] Benchmarks written and passing
- [ ] Speedup claims supported by data
- [ ] No performance regressions vs baseline
- [ ] Memory usage profiled
- [ ] Scaling analyzed (O(n), O(n log n), etc.)

**Profiling** (if performance-sensitive):
- [ ] CPU profiling shows acceptable hotspots
- [ ] Memory allocation patterns acceptable
- [ ] No unexpected allocations in hot paths
- [ ] Cache efficiency validated

---

## Dependency Criteria

**External Dependencies**:
- [ ] All dependencies listed in `Project.toml`
- [ ] Version constraints appropriate
- [ ] No circular dependencies
- [ ] No unnecessary dependencies added

**Internal Dependencies**:
- [ ] No undocumented dependencies on internal modules
- [ ] Clear module boundaries maintained
- [ ] Extractability preserved (if high-level package)

---

## Export/API Criteria

**Public API**:
- [ ] Exports clearly documented in module docstring
- [ ] Export list in module file matches documentation
- [ ] No accidental internal API exposure
- [ ] API stability considered for future versions

**Naming Conventions**:
- [ ] Follows AGENTS.md conventions
- [ ] `world_*` pattern used for persistent state (if applicable)
- [ ] `demo_*` pattern NEVER used (anti-pattern)
- [ ] Function names are clear and descriptive

---

## CI/CD Integration Criteria

**Workflow Passes**:
- [ ] Julia 1.10 matrix passes
- [ ] Julia 1.11 matrix passes
- [ ] All extensions load correctly
- [ ] Documentation builds without warnings
- [ ] Benchmarks complete successfully

**Coverage**:
- [ ] Code coverage meets project standards (>80%)
- [ ] Coverage doesn't decrease
- [ ] Critical paths fully covered

---

## Architecture & Design Criteria

**Design Review**:
- [ ] Follows project architectural patterns
- [ ] No unnecessary complexity
- [ ] Interfaces well-designed
- [ ] Future extensibility considered

**Categorical/Mathematical** (if applicable):
- [ ] Mathematical correctness verified
- [ ] Category laws checked (if categorical)
- [ ] Monoidal properties preserved (if applicable)
- [ ] Sheaf/topos structure sound (if applicable)

---

## Security & Safety Criteria

**Type Safety**:
- [ ] No uses of `Any` type (unless unavoidable)
- [ ] Type inference passes (no `::Any` inference warnings)
- [ ] No unsafe pointer operations

**Memory Safety**:
- [ ] No buffer overflows possible
- [ ] Bounds checking in place
- [ ] No data races (thread-safe)

**Determinism** (if applicable):
- [ ] Seeded RNG ensures reproducibility
- [ ] No global state corruption
- [ ] Order-independent computation (if parallel)

---

## Release Readiness Criteria

**Final Checklist**:
- [ ] All above criteria met
- [ ] Code reviewed by maintainer
- [ ] Changelog entry written
- [ ] Version number decided (semantic versioning)
- [ ] Git commit message follows conventions
- [ ] CHANGELOG.md updated
- [ ] README.md updated (if needed)

**Promotion Path**:
- [ ] Specify: src/ (core) vs examples/ vs separate package
- [ ] If separate package: github repo created, CI setup
- [ ] If examples: added to /examples/ and docs/src/examples/
- [ ] If core: staged as PR with comprehensive description

---

## Graduation Decision Template

### File: `[module_name].jl`

**Current Status**: [Limbo/TBD/PARTIAL/RUNNABLE/RESEARCH/EXTRACTABLE/READY]

**Graduation Decision**:
- [ ] **PROMOTE TO src/**: Core algorithm, essential to Gay.jl
- [ ] **PROMOTE TO examples/**: Illustrative, good learning material
- [ ] **EXTRACT TO PACKAGE**: GayXxx.jl separate repository
- [ ] **ARCHIVE**: Historical reference, not needed now
- [ ] **REJECT**: Duplicates existing functionality

**Rationale**:
[1-2 sentences explaining decision]

**Action Items**:
- [ ] [Specific action]
- [ ] [Specific action]

**Target Timeline**: [When this should graduate]

**Reviewer**: [GitHub username]

**PR#**: [Link to graduation PR if created]

---

## Examples

### Example 1: Module → src/

**File**: `parallel_color_scheduler.jl`

**Current Status**: READY

**Graduation Decision**: **PROMOTE TO src/**

**Rationale**: Critical for parallelization work; implements three optimizations with proven speedup. Full test coverage, performance validated, documentation complete.

**Action Items**:
- [ ] Merge to `src/` immediately
- [ ] Update `src/Gay.jl` exports
- [ ] Add to `.github/workflows/CI.yml` if not already tested
- [ ] Link to PR for GitHub integration

**Target Timeline**: Immediate (blocking)

---

### Example 2: Module → examples/

**File**: `whale_world.jl` + supporting files

**Current Status**: RUNNABLE

**Graduation Decision**: **EXTRACT TO PACKAGE** (GayWhaleWorld.jl)

**Rationale**: Complete application ecosystem (5 files: world, bridge, curriculum, data, demo). Not core to color generation. Strong user interest (1000+ LOC, 50+ examples in docs).

**Action Items**:
- [ ] Create GayWhaleWorld.jl repository
- [ ] Move all 5 whale_* files to new repo
- [ ] Add examples to new repo
- [ ] Add `using GayWhaleWorld` to main README (optional dep)
- [ ] Link to new repo from docs

**Target Timeline**: Next sprint

---

### Example 3: Research → Reference

**File**: `hyperdoctrine.jl`

**Current Status**: RESEARCH

**Graduation Decision**: **ARCHIVE → .topos/reference/**

**Rationale**: Mathematical research not yet integrated. Useful for reference but not ready for release.

**Action Items**:
- [ ] Document current state in .topos/reference/
- [ ] Link from Catlab integration sketches
- [ ] Revisit if Catlab integration becomes priority

**Target Timeline**: Future (2-3 releases out)

---

## Scoring System

Quick assessment (0-5 scale):

| Criterion | 0 | 1 | 2 | 3 | 4 | 5 |
|-----------|---|---|---|---|---|---|
| **Code Quality** | Errors | Warnings | Messy | Clean | Very clean | Production-grade |
| **Test Coverage** | None | <20% | 20-50% | 50-80% | 80-95% | >95% |
| **Documentation** | None | Sparse | Basic | Good | Excellent | Exemplary |
| **Performance** | Unknown | Poor | Acceptable | Good | Very good | Optimized |
| **Integration** | Broken | Failing | Partial | Working | Seamless | Integrated |
| **API Design** | Unclear | Confusing | Acceptable | Clear | Intuitive | Exemplary |

**Graduation Threshold**: Average score ≥ 4/5 (minimum 3/5 on any criterion)

---

## Process

1. **Triage**: Rate file on scoring system above
2. **If score < 3.0**: Return to development, update status
3. **If score 3.0-3.5**: Needs focused work before graduation
4. **If score 3.5-4.5**: Create graduation PR with action items
5. **If score > 4.5**: Fast-track to graduation
6. **Post-graduation**: Remove from .topos/, update CHANGELOG

---

**Last Updated**: December 18, 2025
**Status**: Operational
**Purpose**: Gate quality before versioning

