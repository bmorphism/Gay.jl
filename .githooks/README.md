# Gay.jl Git Hooks 🏳️‍🌈

Git hooks for Strong Parallelism Invariance (SPI) verification.

## Installation

```bash
# Use .githooks directory
git config core.hooksPath .githooks

# Make executable
chmod +x .githooks/*
```

Or symlink individually:
```bash
ln -sf ../../.githooks/pre-commit .git/hooks/pre-commit
ln -sf ../../.githooks/commit-msg .git/hooks/commit-msg
ln -sf ../../.githooks/pre-push .git/hooks/pre-push
```

## Hooks

### `pre-commit`
Runs before each commit. Verifies:
- ✓ Sequential and parallel generation produce identical colors
- ✓ Fingerprint matches expected value (if locked)
- ✓ Distribution is sane (RGB channels uniform)

### `commit-msg`
Runs after commit message is written. Appends:
```
Gay-SPI-Fingerprint: 0x3addddae
```

This creates an immutable record of the color algorithm state at each commit.

### `pre-push`
Runs before pushing. Comprehensive verification:
- ✓ Scale independence (same colors regardless of batch size)
- ✓ Seed sensitivity (different seeds → different colors)
- ✓ Reproducibility (same seed → same colors, always)
- ✓ Distribution sanity (uniform RGB)
- ✓ Performance (>100M colors/sec)

## Cherry-Picking

To add these hooks to another project:

```bash
# From your project root
git cherry-pick --no-commit <commit-with-hooks>
git checkout HEAD -- .githooks/
git config core.hooksPath .githooks
```

Or copy directly:
```bash
cp -r /path/to/Gay.jl/.githooks .
git config core.hooksPath .githooks
chmod +x .githooks/*
```

## Locking Fingerprints

To prevent accidental color algorithm changes, set `EXPECTED_FINGERPRINT` in `pre-commit`:

```julia
const EXPECTED_FINGERPRINT = 0x3addddae  # Lock to specific value
```

Any commit that changes the fingerprint will be blocked until you:
1. Update the expected value (intentional change)
2. Or fix the regression (unintentional change)

## CI Integration

Add to your CI workflow:
```yaml
- name: Verify SPI
  run: julia --project=. .githooks/pre-push
```

## Why This Matters

Gay.jl guarantees **Strong Parallelism Invariance**:
- Same seed → same colors
- Sequential == parallel
- Reproducible across machines

These hooks enforce this guarantee at the git level, preventing commits or pushes that break determinism.
