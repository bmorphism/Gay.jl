# Semantic Fault Atlas: `jepsen://` Addresses

Date: 2026-07-06
Status: example use case
Trit: -1 (Coplay - validation atlas)

Gay.jl can make distributed-systems audit findings addressable without becoming
the checker. Jepsen-style histories, nemeses, Elle, Knossos, and model checkers
still carry the semantics. Gay.jl supplies the deterministic visual address:

```text
jepsen://system/version/workload/model/nemesis/checker/finding
```

That address gives three useful things:

- a stable 64-bit seed for replay and pinning
- a Gay color chip for human-visible navigation
- a GF(3) trit lane for routing work across play / witness / coplay queues

The URI is the semantic key. The color is not a proof and not a database primary
key; it is a reproducible handle for browsing, regression memory, and first-look
triage.

## REPL Session: One Finding

```julia-repl
julia> using Gay

julia> uri = "jepsen://tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash/elle/pass";

julia> seed = stable_seed(uri)
0xb701dde86a270bcc

julia> color_at(0; seed=seed)
"#D70E86"

julia> trit(0; seed=seed)
1

julia> hierarchical_colors(uri)[end]
("jepsen/tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash/elle/pass", "#82E4F6")
```

The raw URI color is the atlas chip. The hierarchical color trail is a prefix
browser: `jepsen`, `jepsen/tigerbeetle`, `jepsen/tigerbeetle/0.16.11`, and so
on. That lets a UI keep the leaf finding visible while also showing where it
sits in the audit tree.

## REPL Session: A Small Atlas

```julia-repl
julia> include("examples/semantic_fault_atlas.jl")
Semantic fault atlas (Gay.jl)
scheme: jepsen://system/version/workload/model/nemesis/checker/finding

trit   color    status  semantic key
+1     #D70E86  pass    jepsen://tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash/elle/pass
+1     #783CE0  fault   jepsen://paxos/origin/register/linearizable/dueling-leaders/knossos/lost-update
0      #298E9F  fault   jepsen://raft/origin/log/linearizable/leader-partition/knossos/divergent-log
0      #76DDDD  fault   jepsen://sql/lab/txn/snapshot-isolation/clock-skew+partition/elle/read-skew
+1     #DD1A4A  gap     jepsen://kv/lab/cas/linearizable/process-pause/knossos/indeterminate

prefix trail for jepsen://tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash/elle/pass
#C15AF2  jepsen
#BE7C1D  jepsen/tigerbeetle
#679DD0  jepsen/tigerbeetle/0.16.11
#74C1E7  jepsen/tigerbeetle/0.16.11/transfer
#F775A5  jepsen/tigerbeetle/0.16.11/transfer/strict-serializable
#D36682  jepsen/tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash
#DF4B97  jepsen/tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash/elle
#82E4F6  jepsen/tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash/elle/pass

read-skew family
#76DDDD witness/collect jepsen://sql/lab/txn/snapshot-isolation/clock-skew+partition/elle/read-skew

coverage-gap address
uri: jepsen://tigerbeetle/next/transfer/strict-serializable/clock-skew+partition/elle/unknown
seed: 0x10de85992ed0badf
color: #411FB6 trit: +1 lane: play/explore
nearest visible atlas chip: #783CE0 jepsen://paxos/origin/register/linearizable/dueling-leaders/knossos/lost-update
rgb distance: 75.03
```

The runnable example keeps a few local atlas fixtures:

- `tigerbeetle` for fast transaction-core regression pins
- `paxos/origin` for consensus-origin failure shapes
- `raft/origin` for replicated-log failure shapes
- `sql/lab` for transaction anomaly families such as read skew
- `kv/lab` for incomplete or indeterminate checker evidence

It also mints an address for a future gap:

```text
jepsen://tigerbeetle/next/transfer/strict-serializable/clock-skew+partition/elle/unknown
```

That is the useful move: a missing audit can be given a seed, color, trit, route,
and nearest visible atlas chip before a report exists. The atlas can therefore
track negative space, not just published findings.

## Non-Trivial Uses

1. **Fault atlas navigation.** Prefix colors let a browser collapse and expand
   by system, model, nemesis, checker, or finding while keeping deterministic
   visual continuity.
2. **Regression memory.** A finding URI pins the exact semantic path that should
   be rerun when a system, checker, or workload changes.
3. **Coverage gaps.** Future or unknown findings still get addresses, so the
   test plan can show which nemesis/model combinations are not yet witnessed.
4. **Runbook routing.** The trit lane can send work to `play/explore`,
   `witness/collect`, or `coplay/reproduce` queues without central scheduler
   state.
5. **Cross-audit comparison.** Gay.jl colors are stable visible chips; semantic
   comparison still comes from the URI fields and the checker histories. This
   separation keeps the atlas useful without pretending color distance proves
   behavioral similarity.

## Run

```sh
cd ~/worlds/g/Gay.jl
julia --project examples/semantic_fault_atlas.jl
```
