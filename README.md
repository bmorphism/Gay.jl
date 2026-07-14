# Gay.jl

`SplittableRandom(seed) → split(index×) → Okhsl color`, plus a cross-runtime
O(1) kernel byte-identical to `spi-race`'s `libspi`.

A small, dep-free Julia package (Printf/Random/SHA/Unicode only) authored from
a single 64-bit hash — the amp-thread tag `0x8b449cd3828014dd` — to answer the
question *"this hash, into Gay.jl as a gay seed, how?"* Heavier color science
and topology load lazily as package extensions.

## The essential feature set

Triangulated across this package, `spi-race`, and `xf.jl`, five things are
load-bearing; everything else is an optional layer:

```
color(seed, index) = extract(mix64(seed + γ·index))    # pure, O(1), stateless
γ = 0x9e3779b97f4a7c15                                  # GOLDEN_GAMMA
```

1. **SplitMix64 kernel** — the constants every implementation shares.
2. **Deterministic seed×index→color** — same inputs ⇒ same color, forever.
3. **SPI** (Strong Parallelism Invariance) — sequential ≡ reversed ≡ shuffled
   ≡ parallel.
4. **XOR-fold fingerprint** — commutative + associative ⇒ any partition, any
   order, same answer. This is what makes SPI *checkable*.
5. **GF(3) trit** — `(r+g+b) mod 3` centered to `{-1,0,+1}`; Σ conserved.

## Cross-runtime SPI kernel (`spi_*`)

The canonical constant-time kernel, byte-identical to
`spi-race/libspi.zig`'s C ABI — so Julia, Zig, Swift/Metal, C/NEON, Python,
Ruby, and anything with FFI agree on every color:

```julia
using Gay

spi_color_hex(42, 0)                       # "#727622"  == spi_color_at(42,0)
spi_color_hex(42, 69)                      # "#A8E8BD"
spi_color_u32(42, 0)                       # 0x00727622 (packed 0x00RRGGBB)

spi_trit(42, 0)                            # +1   (centered, {-1,0,+1})
spi_trit_sum(42, 0, 100)                   # 2    (raw mod-3 residue, ABI-exact)

spi_xor_fingerprint(42, 0, 1_000_000)      # 0x0000000010de88
spi_xor_fingerprint_parallel(42, 1_000_000; chunks=4)
                                           # 0x0000000010de88 — SPI holds
```

Verified two independent ways:

- **Pinned vectors** in `test/runtests.jl` (colors, fingerprints at 1M/10M,
  trits) taken from the Zig reference binary.
- **Live ABI cross-validation**: `scripts/spi_ffi_crossvalidate.jl` `ccall`s
  into `libspi.dylib` and compares 2144 checks across all five entry points —
  including Julia `Threads.@threads` chunking vs Zig pthreads producing the
  same fingerprint. 0 mismatches.

```sh
SPI_LIB=/path/to/libspi.dylib julia --project scripts/spi_ffi_crossvalidate.jl
```

### Three index→color conventions, disambiguated

This package deliberately keeps three conventions with distinct jobs; do not
mix their outputs:

| function | recurrence | cost | at `(42,0)` |
|---|---|---|---|
| `color_at` | repeated `split()`, Okhsl | O(index) | `#6E33B8` |
| `hash_color_*` | `mix64(seed ⊻ γ·index)`, low-byte-first | O(1) | `#227672` |
| `spi_color_*` | `mix64(seed + γ·index)`, `0xRRGGBB` | O(1) | `#727622` |

`spi_*` is the cross-runtime canonical one. `color_at` is the original
split-lattice palette (perceptually nicer, order-dependent lattice).
`hash_color_*` is the GPU-portable Float32 port of splitmixrgb-xf.

## Born from a seed

```julia
Gay.HASH_SEED                       # 0x8b449cd3828014dd
color_at(0; seed=Gay.HASH_SEED)     # "#55DB2A"
color_at(1; seed=Gay.HASH_SEED)     # "#CF851D"
```

The seed slot takes any `UInt64`; the canonical Gay seed is
`GAY_SEED = 1069 = 0x42D` (Douglas Adams + Deterministic).

The gamma slot is the interesting one:

```julia
color_at(0; seed=Gay.HASH_SEED, gamma=Gay.HASH_SEED | 1)   # "#D06BE7"
```

Forcing the hash as the odd gamma instead of the seed walks a different
SplitMix lattice — the per-index palette stays in a narrow chromatic band
instead of spanning the hue circle. Useful for a *thematic* palette tied to a
tag rather than a uniform sampling.

## GF(3) trits from the same stream

```julia
trit(0; seed=Gay.HASH_SEED)  # -1
trit(1; seed=Gay.HASH_SEED)  # -1
trit(2; seed=Gay.HASH_SEED)  # +1
```

`-1 / 0 / +1` = Coplay / Witness / Play. The triad sum mod 3 is the **scalar**
Čech audit (necessary-not-sufficient: canceling pairs hide). For the
holonomy-vector audit, see `~/worlds/gay-lisp/colorholonomy.py`.

## Semantic fault atlas

Distributed-systems audit findings named as semantic keys map to stable
addresses:

```julia
uri = "jepsen://tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash/elle/pass"
seed = stable_seed(uri)             # 0xb701dde86a270bcc
color_at(0; seed=seed)              # "#D70E86"
trit(0; seed=seed)                  # 1, the +1 Play lane
hierarchical_colors(uri)[end]       # ("jepsen/tigerbeetle/.../elle/pass", "#82E4F6")
```

Gay.jl does not replace Jepsen, Elle, Knossos, or a model checker. It gives
the finding a deterministic visible chip, trit lane, and prefix trail so
passes, faults, regressions, and coverage gaps can be browsed and pinned.
See `docs/semantic_fault_atlas.md` and `examples/semantic_fault_atlas.jl`.

## Deterministic port rotation + TOFU

Assign listening ports to parallel workers with no scheduler state:

```julia
identity = "jank-lang/activity-map|nrepl|blog+github-2026|world"

port_for_worker(0, identity)   # 46711
port_for_worker(1, identity)   # 46712

report = assert_port_noncontention(20_000, identity)
report.unique_ports            # 20000
report.collisions              # 0
```

The default interval `29000:48999` holds exactly 20,000 ports below the macOS
ephemeral range; `20_000` workers form a full cyclic permutation, `20_001`
must collide by pigeonhole (`port_rotation_report(20_001, identity).collisions == 1`).
`port_proof_catalog` gives 13 independent witness families for the same fact;
`frames_in_flight_bound` composes planner throughput with socket-drain time.

Treat a schedule as a first-use contract — pin it like an SSH host key:

```julia
pin = port_tofu_record(identity; requested_processes=17)
verify_port_tofu(pin)                                  # true
verify_port_tofu(pin; identity=identity * "|renamed")  # false
verify_port_tofu(pin; frame=1)                         # false
```

Not a replacement for transport security; an opt-in drift detector for
deterministic local-world infrastructure. See
`docs/port_rotation_tofu_neighbors.md` and `examples/port_rotation*.jl`.

## Induced colors across Jank and Basilisp

A portable `.cljc` core keeps one root color while each host runtime receives
a stable accent and an induced carrier motif:

```julia
core = cljc_core_id(read("portable.cljc", String))
jank = cljc_runtime_color(core, :jank)
basilisp = cljc_runtime_color(core, :basilisp)

jank.core_color == basilisp.core_color       # same declared portable root
transition = cljc_runtime_transition(jank, basilisp)
verify_cljc_transition_structure(transition)
```

Authoritative identities are the full SHA-256 descriptor and runtime label;
seeds and RGB remain deterministic presentation only. See
`docs/cljc_runtime_color.md`.

## Private `iphone://` color identifiers

A paired Mac maps only coarse, consented interaction outcomes into a learnable
neighborhood and issues an opaque color reference:

```julia
observation = macos_iphone_observation()
probe = materialize_iphone_probe(observation)
key = generate_iphone_pair_key()
record = iphone_color_record(probe; pair_key=key, scope="external-mac",
    epoch="session-1", semantic_root="passport.gay", space=IPhoneColorSpace())

iphone_uri(record)      # iphone://g1-<model>-<color>/...
passport_uri(record)    # passport://gay/iphone/g1-<model>-<color>/...
```

The URI never contains the semantic alias, exact counts, hardware/account
identifiers, or raw timing; pair and color tokens are domain-separated
HMAC-SHA-256 tags under a fresh per-enrollment key. Color is presentation, not
authentication. Only a keyed-verified local registry resolves it back. See
`docs/iphone_color_uri.md`.

## Optional extensions (weakdeps)

The core stays dep-free; loading a companion package activates its extension:

| load | extension | gives |
|---|---|---|
| `using Colors` | `GayColorsExt` | `gay_colordiff`, CIEDE2000 on hex chips |
| `using Ripserer` | `GayRipsererExt` | `gay_ripserer`, persistent homology of color walks |
| `using FractalDimensions` | `GayFractalExt` | `gay_fractal_dimension` (Grassberger–Procaccia) |
| `using PersistenceDiagrams, Ripserer` | `GayPersistenceDiagramsExt` | `gay_bottleneck`, `gay_wasserstein`, `gay_matching` |

See `docs/color_topology_integration_memo.md` for the integration audit and
the MiniQhull/Nix repair script under `scripts/`.

## Honest gaps

- **Simplified Okhsl** matches the Python bridge / GayMCP.jl, not Björn
  Ottosson's perceptually-uniform Oklab→HSL. A future minor version can add
  real Okhsl without breaking the seed contract.
- **`SplittableRandom` is inline** rather than a dependency on
  `SplittableRandoms.jl`. The algorithm matches; interop via the
  `seed`/`gamma` fields.
- **`spi_trit_sum` returns the raw residue `{0,1,2}`**, matching `libspi.zig`
  exactly; the balanced representative is `r == 2 ? -1 : r`. Only the single
  `spi_trit` is centered. (This asymmetry is upstream's; we mirror it rather
  than silently diverge — the FFI cross-validation caught exactly this.)
- **The hash has no payload.** `0x8b449cd3828014dd` is an opaque 64-bit tag;
  splitmix erases provenance.

## Run

```sh
julia --project -e 'using Pkg; Pkg.test()'          # 385 tests
julia --project examples/semantic_fault_atlas.jl
julia --project examples/port_rotation.jl
julia --project scripts/spi_ffi_crossvalidate.jl    # needs spi-race libspi.dylib
```
