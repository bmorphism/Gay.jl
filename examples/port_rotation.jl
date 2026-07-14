using Gay

identity = "jank-lang/activity-map|nrepl|blog+github-2026|world"
capacity = 20_000

report = assert_port_noncontention(capacity, identity)
over = port_rotation_report(capacity + 1, identity)
bound = frames_in_flight_bound(capacity;
    assignments_per_second=4.7e6,
    drain_seconds=0.25,
)

println("Gay.jl deterministic port rotation")
println("identity: ", identity)
println("interval: ", report.port_min, "..", report.port_min + report.port_span - 1)
println("capacity: ", report.upper_bound)
println("frame offset: ", report.offset)
println("first ports: ", join(report.ports[1:8], ", "))
println("capacity collisions: ", report.collisions)
println("overflow collisions: ", over.collisions,
        " (pigeonhole minimum ", over.pigeonhole_min_collisions, ")")
println("planner-limited Hz: ", round(bound.planner_limited_hz; digits=2))
println("safe rotation Hz: ", round(bound.max_rotation_hz; digits=2))
println()
println(port_proof_catalog_text(17, identity))
