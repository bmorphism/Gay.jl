using Gay
using Printf

const AUDITS = [
    (
        uri = "jepsen://tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash/elle/pass",
        status = "pass",
        note = "fast transfer core regression pin",
    ),
    (
        uri = "jepsen://paxos/origin/register/linearizable/dueling-leaders/knossos/lost-update",
        status = "fault",
        note = "consensus origin failure shape",
    ),
    (
        uri = "jepsen://raft/origin/log/linearizable/leader-partition/knossos/divergent-log",
        status = "fault",
        note = "replicated-log origin failure shape",
    ),
    (
        uri = "jepsen://sql/lab/txn/snapshot-isolation/clock-skew+partition/elle/read-skew",
        status = "fault",
        note = "transaction anomaly family",
    ),
    (
        uri = "jepsen://kv/lab/cas/linearizable/process-pause/knossos/indeterminate",
        status = "gap",
        note = "needs sharper checker evidence",
    ),
]

atlas_seed(uri) = stable_seed(uri)
atlas_color(uri) = color_at(0; seed=atlas_seed(uri))
atlas_trit(uri) = trit(0; seed=atlas_seed(uri))

function atlas_record(a)
    seed = atlas_seed(a.uri)
    (; a..., seed=seed, color=color_at(0; seed=seed), trit=trit(0; seed=seed))
end

trit_label(t) = t == 1 ? "+1" : string(t)
lane(t) = t == -1 ? "coplay/reproduce" : t == 0 ? "witness/collect" : "play/explore"

function hex_rgb(hex)
    s = hex[2:end]
    (
        parse(Int, s[1:2]; base=16),
        parse(Int, s[3:4]; base=16),
        parse(Int, s[5:6]; base=16),
    )
end

function rgb_distance(a, b)
    ar, ag, ab = hex_rgb(a)
    br, bg, bb = hex_rgb(b)
    sqrt((ar - br)^2 + (ag - bg)^2 + (ab - bb)^2)
end

function nearest_color(record, records)
    best = nothing
    best_distance = Inf
    for candidate in records
        candidate.uri == record.uri && continue
        distance = rgb_distance(record.color, candidate.color)
        if distance < best_distance
            best = candidate
            best_distance = distance
        end
    end
    best === nothing && error("nearest_color needs at least one distinct record")
    best, best_distance
end

records = atlas_record.(AUDITS)

println("Semantic fault atlas (Gay.jl)")
println("scheme: jepsen://system/version/workload/model/nemesis/checker/finding")
println()
println(rpad("trit", 7), rpad("color", 9), rpad("status", 8), "semantic key")
for record in records
    println(rpad(trit_label(record.trit), 7),
            rpad(record.color, 9),
            rpad(record.status, 8),
            record.uri)
end

focus = first(records)
println()
println("prefix trail for ", focus.uri)
for (prefix, color) in hierarchical_colors(focus.uri)
    println(rpad(color, 9), prefix)
end

println()
println("read-skew family")
for record in filter(r -> occursin("read-skew", r.uri) || occursin("stale-read", r.uri), records)
    println(record.color, " ", lane(record.trit), " ", record.uri)
end

gap = atlas_record((
    uri = "jepsen://tigerbeetle/next/transfer/strict-serializable/clock-skew+partition/elle/unknown",
    status = "gap",
    note = "address a future nemesis before there is a report",
))
neighbor, distance = nearest_color(gap, records)

println()
println("coverage-gap address")
println("uri: ", gap.uri)
println("seed: ", repr(gap.seed))
println("color: ", gap.color, " trit: ", trit_label(gap.trit), " lane: ", lane(gap.trit))
println("nearest visible atlas chip: ", neighbor.color, " ", neighbor.uri)
println("rgb distance: ", @sprintf("%.2f", distance))
