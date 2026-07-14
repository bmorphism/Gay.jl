# `iphone://` as a private learnable Gay.jl color reference

The contract has three disentangled roots:

```text
consented coarse observation ──→ learnable metric ──→ local Gay.jl RGB
             │                        │
             └──────── keyed tags + model digest ──→ iphone:// reference
```

The RGB makes a motif visible. The metric decides whether two motifs match.
The keyed tag keeps an enrollment distinct. None of those roles substitutes
for either of the others.

## Canonical forms

```text
iphone://g1-<model-id>-<color-token>/<scope-token>/<epoch-token>/<pair-tag>
passport://gay/iphone/g1-<model-id>-<color-token>/<scope-token>/<epoch-token>/<pair-tag>
```

The authority component after `iphone://` is the color identifier. It contains
a version, a digest of the frozen metric model, and a keyed color token. The
remaining path carries 128-bit keyed scope and epoch tokens plus a 128-bit
keyed pair tag. Parsers accept only lowercase canonical forms and reject queries,
fragments, userinfo, ports, extra path components, and unknown versions.

This package defines and validates the URI data contract. It does not register
an operating-system URL handler and does not claim that dispatching a custom
scheme authenticates its sender.

## Probe space

`IPhoneProbe` deliberately admits only four coarse coordinates:

| Coordinate | Values | Excluded source detail |
|---|---:|---|
| connection outcome | unavailable, interrupted, available, connected | device ID and exact error trace |
| Voice Memos sync | false or true | account identity and recording content |
| recording-count bin | 0, 1–4, 5–16, 17+ | exact count, titles, timestamps, duration, audio |
| interaction bin | 0–3 | precise latency and behavioral trace |

The embedding normalizes these coordinates to `[0,1]^4`. A strictly positive
diagonal metric defines distance. `learn_iphone_color_space` estimates its
weights from labeled matching and nonmatching probe pairs using regularized
between-class versus within-class dispersion. The weights are normalized to a
mean of one so distance scale remains comparable across learned models.

Gay.jl then deterministically projects the probe around a semantic root such as
`passport.gay`. A single RGB cannot isometrically preserve the four-dimensional
probe neighborhood, and this projection makes no rank-preservation guarantee.
The package's simplified Okhsl RGB is presentation only. Matching uses
`iphone_probe_distance`, never RGB distance.

## Read-only macOS observation

```sh
/usr/bin/swift scripts/macos_iphone_probe.swift --format json
```

```julia
observation = macos_iphone_observation()
if macos_iphone_observation_complete(observation)
    probe = materialize_iphone_probe(observation)
end
```

The Swift producer inspects only already-open Accessibility surfaces and a
CoreDevice fallback. It does not activate apps, click controls, create keys, or
register a URL handler. Its JSON contains nullable coarse fields and fixed
evidence codes; it never emits device labels/identifiers, recording metadata,
raw UI strings, exact counts, or CoreDevice JSON.

Connection labels are allowlisted. A paused/resumable session maps to
`available`; connection failure, local-auth lock, and remote-control gates map
to `interrupted`; an explicit not-available label maps to `unavailable`. A
generic window is not a positive connected witness. When AX has no known state,
CoreDevice is consulted through its structured JSON output: zero or multiple
known iPhones stay unknown, while exactly one row is projected from only pairing
and tunnel state. The producer routes that output through a bounded pipe instead
of a temporary file; `devicectl` documents only disk-file JSON as a stable
programmatic interface, so a failure or future output change remains unknown.
The local subprocess necessarily decodes the row transiently, but no name or
identifier field is selected, retained after projection, or emitted, and no row
is correlated with an AX window. CoreDevice availability remains a separate
witness from Continuity mirroring control.

Voice Memos is especially easy to mismeasure: `RecordingsList` is virtualized.
During validation the rendered child count was initially mistaken for the
total, but the selected All Recordings summary showed that the total belongs in
bin 3. The adapter therefore parses only that summary, bins it immediately
(`17+` → `3`), and retains neither number nor title. If the summary is not
exposed by the current UI state, `recording_count_bin` remains `nothing` and
`materialize_iphone_probe` refuses to guess. The iCloud sync flag is likewise
unknown unless the Voice Memos checkbox is already visible in System Settings.

`interaction_bin` currently derives from the same connection witness as
`state`; those axes are correlated and a uniform metric double-weights that
evidence. A learned model should down-weight one axis or replace interaction
with an independently observed motif before claiming four independent roots.

## Identity, model, and clock separation

```text
scope-token= HMAC(K_pair, scope-domain || scope || epoch)[0:128]
epoch-token= HMAC(K_pair, epoch-domain || scope || epoch)[0:128]
pair-tag   = HMAC(K_pair, pair-domain || scope || epoch)[0:128]
model-id   = SHA-256(version || canonical weight bits)[0:128]
color-token= HMAC(K_pair, color-domain || scope || epoch || model || motif)[0:128]
```

`K_pair` is a fresh 256-bit secret for exactly one enrollment. A different Mac
should receive a different key. The model does not enter `pair-tag`, so learning
can change the model and color without silently changing pair identity.

The caller supplies the epoch. That is the clock: advancing it rotates the
epoch, pair, and color tags without relying on wall-clock precision. Independent
tiles or observers may advance at independent rates; they agree only when a
coordinator explicitly supplies a shared epoch. This module does not hide an
autonomous timer or infer time from interaction latency.

`generate_iphone_pair_key` creates a key in memory with the operating system's
random source but does not persist it. A persistent deployment must store it in
an appropriate device-only secret store and implement explicit rotation and
revocation outside this library.

Input strings are normalized to stable Unicode NFC before hashing. The URI is
intentionally opaque: passive recipients cannot recover either RGB or the
probe embedding. An authorized Mac registers `IPhoneColorRecord`s in an
`IPhoneColorRegistry` with the enrollment key and resolves the URI there. The
key is verified and discarded; the registry retains the record and frozen
metric model. `iphone_record_distance` rejects unresolved or model-mismatched
URIs. Registry synchronization, consent, authenticated exchange, and persistent
secret storage remain deployment responsibilities rather than hidden behavior
in this package. Vat-local records can be revoked with
`unregister_iphone_color!` or `purge_iphone_epoch!`.

For a tiled or Goblins-style deployment, confine one registry, metric model,
and epoch cursor to each vat. The registry records its creating Julia `Task`
and rejects access from another task. Advance a tile by sending its owning vat a
next-motif message. A coordinator may exchange explicit epoch witnesses without
forcing all vats to run at the same rate.

## Measuring control optimality

A useful controller must satisfy all four objectives; no single score proves
optimality:

1. **Motif discrimination:** cross-validated AUC or triplet accuracy in
   `iphone_probe_distance`; compare against the uniform-weight baseline.
2. **Continuity:** bound color change for a one-bin probe change and measure
   rank correlation between metric distance and perceptual color distance.
3. **Stability:** pin model digests and colors for a frozen training set;
   measure drift only across an explicit model version change.
4. **Privacy and separation:** HMAC known-answer vectors, strict parse/format
   round trips, scope/epoch unlinkability, forced RGB-collision non-merge, key
   rotation, and tests that forbidden raw fields cannot enter the data model.

For a sequence controller, add regret against the best fixed model, transition
cost, probe budget, and per-tile clock skew. Report these separately rather than
compressing them into one reward that can be Goodharted.

## Security boundary

Never derive this URI from UDID, serial number, IMEI, ECID, MAC/BSSID, Apple ID,
DSID, APNs token, IP address, hostname, Voice Memo content, titles, timestamps,
or exact timing. `stable_seed` and a 24-bit RGB are public deterministic color
tools, not pseudonymization. Secure actions need a separate authenticated
transport with nonce, expiry, and authorization; a parsed `iphone://` string is
only a reference.
