using Gay

identity = "jank-lang/activity-map|nrepl|blog+github-2026|world"
workers = 17

pin = port_tofu_record(identity; requested_processes=workers)

println(port_tofu_record_text(pin))
println()
println("verify same contract: ", verify_port_tofu(pin))
println("verify renamed contract: ",
        verify_port_tofu(pin; identity=identity * "|renamed"))
println("verify next frame: ", verify_port_tofu(pin; frame=pin.frame + 1))

report = assert_port_noncontention(workers, identity)
println()
println("first assigned ports: ", join(report.ports[1:min(workers, 8)], ", "))
println("collisions: ", report.collisions)
