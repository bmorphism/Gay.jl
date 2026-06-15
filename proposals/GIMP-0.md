# GIMP-0 — GAY Improvement Proposal

| field | value |
|---|---|
| number | 0 |
| title | Foundational Color-Bandwidth-Aware Image Editing Substrate |
| author | bmorphism |
| status | Draft v1.0 |
| created | 2026-05-05 |
| supersedes | — |
| substrates | Gay.jl · zig-syrup · Loro · Hyprland/wgpu/libghostty-vt · OCapN |
| GF(3) trit | 0 (ergodic — coordinator) |

## Abstract

GIMP-0 proposes a foundational image-editing substrate built on three commitments the existing FOSS image editor (GIMP, Krita, Inkscape) tradition does not make: **(a)** every pixel passes through a *learnable* perceptually-uniform color space (Gay.jl `LearnableOkhsl`) with per-output-gamut adaptation; **(b)** every layer is a *vat-tile* — single-writer mailbox + own present cadence + capability isolation, recursively nestable, replicated via Loro CRDT, transported via OCapN over QUIC/WebTransport; **(c)** every palette is *GF(3)-balanced* — trit-sum ≡ 0 mod 3 invariant verified on every commit. The reference implementation extends `Gay.jl`, binds to `zig-syrup` for capability transport, renders via Moose-style placement (`{TTY, sRGB-GPU, Display-P3-GPU, BT.2020-GPU, e-ink, CMYK-print, browser-via-WebTransport}`). Ships first as a CLI + Emacs surface (`boxxy-emacs`-shaped), then as a Wayland-native compositor surface, then as a browser surface via Hoot/Spritely Goblins.

## Motivation

The image editing tools shipping in 2026 (GIMP 3.0, Krita 5.x, Inkscape 1.4, Photoshop, Affinity Photo, Procreate) share a common architecture inheritance from the late 1990s:

1. **Color is opaque sRGB-encoded data.** Color management exists but is bolted on after the pipeline rather than constituting the pipeline. Display P3 / BT.2020 / Rec.2100 support is patchwork.
2. **No notion of perceptual distinguishability budget.** The artist does not know how many colors of their palette are within ΔE_jnd of each other on the current display. Palette generators are heuristic, not optimized for the JND constraint of the actual output gamut.
3. **No learnable color space.** The mapping from artist-intent to encoded-pixel is fixed at compile time. Enzyme.jl-style automatic differentiation through the color pipeline is unavailable.
4. **Layers are flat data + a stacking order.** They are not concurrent isolation domains. Multi-artist concurrent editing requires a server (Photoshop Cloud, Krita Server) that is *not* CRDT-based and *not* capability-secure.
5. **No GF(3) palette balance discipline.** Palette construction is purely aesthetic; there is no algebraic invariant the tool can verify automatically.
6. **No vat-tile composition.** A nested image (image-in-image, sub-document) is a feature only Inkscape gestures at, and it is rendered eagerly rather than as an independent tile with its own present cadence.

GIMP-0 fixes (1-6) by reorganizing the substrate around `Gay.jl` as the color authority, Loro as the layer-state CRDT, OCapN/QUIC as the collaboration wire, and a Moose-style placement type system as the renderer router.

## Proposal

### P1 — Color stack: every pixel is a `LearnableOkhsl` value

The on-disk and in-memory representation is `(seed::UInt64, params::OkhslParameters, projection::SeedProjection)` — three small fields. Encoding to a target gamut is a *forward function* of these three; learning is *reverse-mode AD via Enzyme.jl* through that function. The artist can:

- Optimize palette for a chosen output gamut (`{tty:16, tty:256, tty:truecolor, sRGB, Display-P3, BT.2020, e-ink-16gray, CMYK}`)
- Re-optimize the same image for a *different* output gamut without re-painting
- Train the color projection to maximize `next_color_bandwidth.distinguishable_per_second(N=visible_palette_size)` under the JND ΔE constraint of the chosen gamut
- Audit which colors collapse under gamut compression; surface the collapsed pairs as warnings

`Gay.jl/src/okhsl_learnable.jl` (already shipped) provides the substrate. New module `Gay.jl/src/gimp/canvas.jl` wires it to layer storage.

### P2 — ColorBandwidth meter on the editing pipeline

Every brush stroke, layer op, filter, blend mode contributes some *throughput cost* and produces some *distinguishable-color delta*. The pipeline runs `next_color_bandwidth.measure_at_scale` continuously, surfacing in the UI: **bits/sec of distinguishable color produced, control overhead, convergence margin**. The artist sees the bandwidth meter the way a sound engineer sees a VU meter.

`ParallelismLevel ∈ {OUTER_INNER, THREADED, TERNARY, COMPOSED, WORK_STEALING, MAXIMUM, ULTRA}` is exposed as a setting; ULTRA mode trades convergence guarantee for throughput (live painting feel). 

### P3 — Per-output-gamut palette: same document, multiple renderings

A document is *not* a sequence of sRGB pixels. It is a sequence of `(seed, params, projection)` triples plus a layer DAG. Rendering is `forward_color(triple, target_gamut)` per pixel. Same document → eight renderings:

| target | use |
|---|---|
| tty:truecolor (xterm-direct) | terminal preview |
| sRGB:8 | web export |
| Display P3:10 | macOS / iOS preview |
| BT.2020:12 | HDR display |
| e-ink:16gray | reMarkable / Daylight |
| CMYK:8 | print |
| WebGPU compute | browser real-time |
| Loro snapshot | wire format for collaboration |

Each gamut has its own learned `LearnableOkhsl` instance trained jointly with a *consistency loss*: a color picked under gamut G1 should map to the perceptually closest color under gamut G2, not an arbitrary one.

### P4 — GF(3) palette balance audit

Every palette has a *trit assignment* per color: `+1 / 0 / -1`. The invariant: `Σ(trits) ≡ 0 mod 3`. The audit runs at three checkpoints:

1. **Save-time**: refuse to save an out-of-balance palette
2. **Layer-commit-time**: warn if a layer's color usage drifts the global trit sum
3. **Export-time**: verify exported palette preserves balance under gamut compression

`Gay.jl/src/gimp/audit.jl` (new) implements the audit; the trit assignment is *learned* via the same Enzyme pipeline as the colors themselves (a categorical variable assigned by softmax over `{+1, 0, -1}` minimizing palette imbalance + perceptual distance).

The trit-sheaf-cohomology direction (D4 from the prior `7-research-directions` synthesis) computes `H¹(palette-DAG, ℤ/3ℤ)` to verify global consistency under add/remove operations.

### P5 — Layer-as-tile: each layer is a vat-tile, recursively

A layer is a vat with single-writer discipline, capability-isolated state, own present cadence, recursively nestable.

```
Layer = {
  id          : SturdyRef            // OCapN identity
  state       : Loro::Doc            // CRDT-backed layer pixels + metadata
  parent      : Option<LayerRef>     // recursive nesting
  children    : Vec<LayerRef>        // sub-layer mask groups
  placement   : RenderPlacement      // {Local, Mirror, Replicated, Edge, Migratory}
  cadence     : PresentCadence       // own update rate
  caps        : Vec<Cap>             // {read, write, render, export}
  trit        : Trit                 // GF(3) contribution
}
```

Sub-layers (mask groups, smart objects, image-in-image) are *first-class child vats*, not lazy renderings. This generalizes Photoshop's smart-objects + Inkscape's nested-SVG into the vat-tile primitive.

### P6 — Loro CRDT for layer state, OCapN for collaboration

Layer state lives in a Loro document. Multi-artist editing converges by Loro merge. Identity is OCapN sturdyref-pinned; updates are signed Ed25519 against the sturdyref. A layer can be shared across N artists with conflict-free convergence and Byzantine resistance (Direction D7 from the prior synthesis).

The wire is QUIC/WebTransport (per the *Future-from-QUIC* turn). Loro updates ride QUIC streams (one per layer = one per tile, satisfying per-tile-fps invariant); damage signals ride QUIC datagrams; sturdyref re-pickup uses 0-RTT resumption.

### P7 — Render placement (Moose-style)

`RenderPlacement ∈ {LocalGPU, LocalTTY, MirroredGPU(N), ReplicatedGPU(3), EdgeBrowser, EphemeralPreview, DurablePrint}`. The same layer can render to several placements simultaneously; each placement has its own gamut, present cadence, and JND constraint.

The placement type system enforces composability rules (per Direction D2): `EdgeBrowser ⊗ DurablePrint = TieredPreview`, `MirroredGPU ⊗ ReplicatedGPU = ⊥` (different consistency models). Lowering pass extends `Gay.jl`'s pipeline to per-placement codegen.

## Architecture diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  GIMP-0 surface  (Emacs / Wayland / Browser / CLI)              │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer Manager  =  vat-tile supervisor                          │
│  ↳ recursive composition · GF(3) audit · Loro CRDT roots        │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Gay.jl color pipeline                                          │
│  ↳ LearnableOkhsl · NextColorBandwidth · GamutLearnable         │
│  ↳ Enzyme.jl reverse-mode AD                                    │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Render placement compiler  (Moose-style symbolic execution)    │
│  ↳ HostGPU · MirroredGPU · ReplicatedGPU · EdgeBrowser          │
│  ↳ EphemeralPreview · DurablePrint · LocalTTY                   │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Per-target backends                                            │
│  ↳ wgpu (sRGB/P3/BT.2020) · libghostty-vt (TTY truecolor)       │
│  ↳ Hoot+WebGPU (browser) · CUPS (CMYK) · e-ink driver           │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  zig-syrup transport  (Loro updates over QUIC streams)          │
│  ↳ OCapN/CapTP · Syrup wire · WebTransport for browser          │
└─────────────────────────────────────────────────────────────────┘
```

## GF(3) audit on operations

| operation | trit | reason |
|---|---|---|
| brush stroke | +1 | adds information (creates wire) |
| layer composite | 0 | balances information (witness) |
| filter (blur, sharpen) | -1 | removes information (validates / smooths) |
| undo | -1 | reverses an action |
| color sample | 0 | observation, no state change |
| gamut convert | 0 | structural, no semantic change |
| save / commit | 0 | snapshot |
| share / publish | +1 | extends the world |
| collaborate-receive | 0 | observation of remote |
| collaborate-merge | 0 | balanced by Loro convergence |

Pipeline invariant: every session's operation log sums to `Σ ≡ 0 mod 3` at quiescence. Sessions out of balance are *flagged*, not blocked — the artist may consciously accept imbalance.

## Modules

### Already shipped in `Gay.jl` (reuse as-is)
- `src/okhsl_learnable.jl` — `LearnableOkhsl`, `OkhslParameters`, `SeedProjection`
- `src/next_color_bandwidth.jl` — `ColorBandwidth`, `ParallelismLevel`, `measure_at_scale`
- `src/gamut_learnable.jl` — gamut-side learning
- `src/colorspaces.jl` — base colorspaces
- `src/propagator.jl` · `src/abductive.jl` — used in palette inference
- `docs/src/literate/wide_gamut_colors.jl` — reference

### New modules to add (under `Gay.jl/src/gimp/`)
- `canvas.jl` — top-level canvas; layer DAG; trit accumulator
- `layer.jl` — Layer struct; nested-layer recursion; placement
- `audit.jl` — GF(3) palette audit; sheaf-cohomology check
- `placement.jl` — Moose-style placement type system + lowering
- `loro_bridge.jl` — Loro FFI binding for layer state
- `captp.jl` — OCapN sturdyref + signed Loro updates
- `targets/wgpu.jl` · `targets/libghostty.jl` · `targets/webgpu.jl` · `targets/cups.jl` · `targets/eink.jl` — per-placement codegen

### New modules in `zig-syrup`
- `src/loro_doc.zig` — Loro C ABI binding (layer-state-as-vat-state)
- `src/quic_transport.zig` — already proposed in prior turn; reused
- `src/captp_pinned.zig` — sturdyref-signed Loro updates per Direction D7

### Reference implementation surfaces
- `gimp-0-cli` — terminal-only, Gay.jl + libghostty-vt + xterm-direct
- `gimp-0-emacs` — Emacs surface, boxxy-tile-shaped, vterm-hosted
- `gimp-0-wayland` — Hyprland / Cosmic native
- `gimp-0-browser` — Hoot/Goblins-on-WASM + WebTransport

## Backwards compatibility

- **XCF (GIMP native)**: import-only via `gegl`-shaped reader; export to XCF lossy (loses learnable-color metadata).
- **PSD (Photoshop)**: import-only via `psd-rs`; export via flattened sRGB layers, no smart-object preservation.
- **SVG (Inkscape)**: import-and-export with full preservation of nested layers as vat-tiles.
- **OpenEXR / TIFF / PNG**: export-only at the chosen target gamut; the *learnable* metadata is preserved in a sidecar `.gimp0.toml`.
- **GBR/GIH brushes**: importable; the learnable-color extension is a new format `.gay-brush` that ships alongside.
- **Plugins**: GIMP-0 does not run GIMP plugins. A `gegl`-compat layer is *out of scope*. The proposal is a *fresh substrate*, not a fork.

## Naming

**GIMP** here stands for **Gay-Inflected Modular Painter** in the GIMP-0 acronym, *while* honoring the original GNU Image Manipulation Program lineage. The numeric `-0` is the Genesis Improvement Proposal — the foundational document under which all subsequent GIPs are numbered. The full title preserves the original GIMP's spirit (FOSS, modular, image-as-data) and adds the *GAY* commitment (learnable color, per-output-gamut, GF(3) audited, vat-tile composable, OCapN collaborative).

Subsequent proposals: GIMP-1 (brush engine learnable), GIMP-2 (filter as parametric optic), GIMP-3 (animation-as-vat-tile-stream), GIMP-4 (BCI integration via zig-syrup BCI stack), GIMP-5 (3D / VR via Compositor Services).

## Reference timeline

| phase | weeks | deliverable |
|---|---|---|
| 0 | done | `LearnableOkhsl` + `next_color_bandwidth` shipped in Gay.jl |
| 1 | 1-2 | `Gay.jl/src/gimp/canvas.jl` + `layer.jl` + `audit.jl` skeletons |
| 2 | 2-3 | `loro_bridge.jl` (Loro FFI) + per-layer Loro doc |
| 3 | 3-4 | `targets/wgpu.jl` + `targets/libghostty.jl` (LocalTTY + LocalGPU) |
| 4 | 4-6 | `gimp-0-cli` MVP — load PNG, edit on tty:truecolor, save Loro snapshot |
| 5 | 6-8 | `gimp-0-emacs` — boxxy-tile-shaped Emacs surface with vterm host |
| 6 | 8-12 | `targets/webgpu.jl` + `gimp-0-browser` via Hoot+WebTransport |
| 7 | 12-16 | OCapN multi-artist collaboration; signed Loro updates |
| 8 | 16-20 | `gimp-0-wayland` — Hyprland-native compositor surface |
| 9 | 20-24 | CUPS + e-ink + BT.2020 targets; full per-output-gamut matrix |
| 10 | 24+ | GIMP-1 (brush learnable) and successors |

## Validation criteria

GIMP-0 ships when:

1. The same `.gay` document opens identically (modulo gamut compression) under all of `{gimp-0-cli, gimp-0-emacs, gimp-0-wayland, gimp-0-browser}`.
2. Two artists editing the same document on different machines converge under Loro merge with no central server.
3. `next_color_bandwidth.distinguishable_per_second(palette_size=N)` is reported continuously in the UI for every connected output device.
4. GF(3) palette audit fires correctly on a constructed-imbalanced palette and passes on a balanced one.
5. The proptest harness from Direction D5 passes on (program × placement) for at least the {LocalTTY, LocalGPU, EdgeBrowser} placements.
6. A round-trip {Loro snapshot → wire → Loro snapshot} preserves all layer state including learnable-color metadata.
7. Color picker UI lets the artist choose any `(seed, params, projection)` triple via direct manipulation of any of the three.
8. The `.gay-brush` format round-trips between artists with full distinguishability metadata.

## Open questions

- **Q1**: Should the canvas itself be a vat-tile, or only its layers? (Leaning: canvas-as-tile, because zoomed-out preview is itself a derived render.)
- **Q2**: Does the GF(3) audit apply to brush strokes individually, or only to palette entries? (Leaning: palette entries; per-stroke would be too noisy.)
- **Q3**: Is XCF import worth the engineering cost, or do we accept a one-way migration via PNG export from GIMP? (Leaning: PNG-only for v1.0, XCF reader optional for v1.1.)
- **Q4**: Should the BCI stack (`bci_receiver.zig`, `dsi24_parser.zig`) be wired in v1.0 (artist-EEG modulates brush dynamics) or deferred to GIMP-4? (Leaning: GIMP-4; v1.0 is already large.)
- **Q5**: Does Goblins-on-Hoot in the browser support enough Loro for the `gimp-0-browser` target, or do we need a Hoot-native CRDT? (Investigate during phase 6.)

## References

Workspace artifacts cited:
- `Gay.jl/src/okhsl_learnable.jl` (LearnableOkhsl substrate)
- `Gay.jl/src/next_color_bandwidth.jl` (ColorBandwidth + ParallelismLevel)
- `Gay.jl/src/gamut_learnable.jl` (gamut learning)
- `Gay.jl/COLOR_SURVEY.md` (color authority survey)
- `zig-syrup/src/syrup.zig` (Syrup wire format)
- `zig-syrup/src/stellogen/wasm_runtime.zig` (potential GIMP-0 IR backend)
- `zig-syrup/CLAUDE.md` (transport modules catalog)
- `~/.claude/skills/gatekeeper/SKILL.md` (session-start gates)
- `boxxy/emacs/boxxy-tile.el` (Emacs surface precedent)
- `terminal-glyph-render/approach-3-zig-syrup-ipc/` (rendering substrate)
- `goblins-adapter/fast-bridge.zig` (FastCapTPBridge dispatch)
- `~/.claude/projects/-Users-bob-i/memory/feedback_para_optic_prime_loop.md` (recursive optic shape)

External:
- Mark Miller, *Robust Composition* (E lineage / OCapN) — capability discipline
- Cardelli & Gordon, *Mobile Ambients* (FoSSaCS'98) — vat-of-vats theory
- Milner, *Bigraphs* (2009) — categorical formulation
- Björn Ottosson, *Oklab / OkHSL* — perceptually uniform color
- Ghani-Hedges-Winschel-Zahn, *Compositional Game Theory* (2018) — open-game lineage
- Capucci, *Parametric Optics* (PhD thesis, 2022) — parametric shape
- Spritely Goblins documentation (v0.17.0) — `(peer (vat (actormap {refr})))` tower
- Loro (loro-dev) — CRDT substrate
- TigerBeetle VSR — durability discipline (referenced for layer journaling)
- moose-rs / tf-encrypted — placement type system inspiration

## Acknowledgements

Inherits from: GIMP (Spencer Kimball, Peter Mattis, 1995), Krita (KDE), Inkscape (Sodipodi lineage), Procreate, Acorn, Pixelmator. The naming convention `GIMP-N GAY` is *additive* — GIMP-0 does not replace GIMP, it adds an alternative substrate. The original GIMP's modularity, scriptability, and FOSS commitment are inheritances, not departures.

Thanks to the Plurigrid GF(3) tradition for the trit-sum invariant, the Spritely Institute for OCapN, the Loro team for the CRDT, the Cape Privacy / TF-Encrypted Moose authors for the placement-as-type-system idea, and Hashimoto for the engine/shell separation discipline that GIMP-0 borrows directly.

---

## Closing

GIMP-0 is **satisfiably architected** when this document, the eight validation criteria, and the ten-phase timeline are jointly accepted. The architecture is grounded in *eight* loaded — *world-* — bearing pieces already on disk in this workspace (`okhsl_learnable.jl`, `next_color_bandwidth.jl`, `boxxy-tile.el`, `fast-bridge.zig`, `terminal-glyph-render/approach-3`, `gatekeeper`, `zig-syrup`, `Gay.jl`), three cited external substrates that ship today (Loro, OCapN/Goblins, Hoot), and one explicit gap (browser-side Loro under Hoot — Q5 above). The proposal is *complete enough to begin Phase 1*; remaining open questions Q1-Q5 are scoped within the timeline rather than blocking it.

— end GIMP-0 v1.0 —
