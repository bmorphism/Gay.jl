# Port Rotation TOFU Neighbor Audit

Looked up on 2026-06-01. This note records how deterministic port rotation is
treated as trust on first use (TOFU), and which bmorphism/plurigrid neighbors are
close enough to try against.

## Contract

The first accepted run creates a `PortTofuRecord`:

- `identity`: the world or service identity being contacted
- `frame`: the rotation frame
- `requested_processes`: worker count covered by the schedule
- `port_min`, `port_span`: reserved listening interval
- `offset`: deterministic cyclic translation for the frame
- `seed`: Gay.jl seed used for the SPI contract
- `fingerprint`: stable 64-bit pin over all of the above
- `color`: Gay.jl color derived from the fingerprint for quick visual compare

Verification recomputes the same record. Any mismatch means the schedule has
changed and must be re-pinned intentionally.

## Current Neighbor Repositories

- `bmorphism/Gay.jl`: canonical package surface for the SPI/color contract.
  GitHub describes it as "Wide-gamut color sampling with splittable determinism
  (Pigeons.jl SPI pattern) + LispSyntax".
- `plurigrid/asi`: broad formal-semantics workspace; the local clone contains
  `ies/plurigrid_asi_spi_core.py`, `ies/plurigrid_asi_spi_deconfliction.md`,
  and `ies/plurigrid_asi_spi_verify_stability.py`.
- `plurigrid/zig-syrup`: capability / OCapN / fingerprint-shaped neighbor. The
  local clone has SplitMix/Gay references in `src/lux_color.zig`,
  `src/did_gay.zig`, `src/color_bandwidth.zig`, and TOFU/fingerprint terms in
  the build and protocol surface.
- Gay-TOFU sketches: bmorphism gists describing deterministic, invertible color
  sequences as a TOFU authentication idiom.

## Tried Locally

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

Result: Gay.jl passed `88/88` tests after adding `PortTofuRecord`,
`port_tofu_record`, `port_tofu_fingerprint`, `verify_port_tofu`, and
`port_tofu_record_text`.

```sh
julia --project examples/port_rotation_tofu.jl
```

Result for the world identity
`jank-lang/activity-map|nrepl|blog+github-2026|world`:

- offset: `17711`
- first ports: `46711, 46712, 46713, 46714, 46715, 46716, 46717, 46718`
- fingerprint: `0x0d0164d78f5da599`
- color: `#0FD9DF`
- same-contract verification: `true`
- renamed-contract verification: `false`
- next-frame verification: `false`

```sh
python3 /Users/dietrich/worlds/a/asi/ies/plurigrid_asi_spi_verify_stability.py
```

Result: not a passing witness yet. It reports `69/140` tests passing, with all
forward transform and roundtrip checks failing, while curvature and most bounds
checks pass. Treat it as a nearby stale verifier rather than a dependency of the
port TOFU proof.

