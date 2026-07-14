using Test
using Gay

@testset "privacy-coarsened macOS iPhone observation" begin
    complete_line = join([
        "gay-iphone-probe/v1",
        "available",
        "1",
        "3",
        "2",
        "ax-connection-paused",
        "ax-icloud-voice-memos-toggle",
        "ax-selected-all-recordings",
    ], '\t')
    observation = Gay._parse_macos_iphone_probe_tsv(complete_line * "\n")
    @test macos_iphone_observation_complete(observation)
    @test observation.state == :available
    @test observation.voice_memos_sync === true
    @test observation.recording_count_bin == 3
    @test observation.interaction_bin == 2

    probe = materialize_iphone_probe(observation)
    @test probe.state == :available
    @test probe.voice_memos_sync
    @test probe.recording_count_bin == 3
    @test probe.interaction_bin == 2

    partial_line = join([
        "gay-iphone-probe/v1", "-", "-", "-", "-",
        "ax-status-unknown", "ax-toggle-unavailable",
        "ax-all-recordings-unavailable",
    ], '\t')
    partial = Gay._parse_macos_iphone_probe_tsv(partial_line)
    @test !macos_iphone_observation_complete(partial)
    @test_throws ArgumentError materialize_iphone_probe(partial)

    @test_throws ArgumentError Gay._parse_macos_iphone_probe_tsv(
        replace(complete_line, "gay-iphone-probe/v1" => "unknown/v9"))
    @test_throws ArgumentError Gay._parse_macos_iphone_probe_tsv(
        replace(complete_line, "available" => "connected-ish"))
    @test_throws ArgumentError Gay._parse_macos_iphone_probe_tsv(
        replace(complete_line, "\t3\t2\t" => "\t4\t2\t"))
    @test_throws ArgumentError MacOSIPhoneObservation(
        :available, true, 3, 2, Symbol("raw evidence"), :safe, :safe)
    @test_throws ArgumentError MacOSIPhoneObservation(
        :connected, true, 3, 0, Symbol("ax-status-unknown"),
        Symbol("ax-icloud-voice-memos-toggle"), Symbol("ax-selected-all-recordings"))
    @test_throws ArgumentError Gay._parse_macos_iphone_probe_tsv(
        "\n" * complete_line * "\n\n")
    @test_throws ArgumentError Gay._parse_macos_iphone_probe_tsv("x"^1025)
    if Sys.isapple()
        script = normpath(joinpath(@__DIR__, "..", "scripts", "macos_iphone_probe.swift"))
        @test_throws ErrorException macos_iphone_observation(
            script=script, swift="/usr/bin/false")
        oversized = joinpath(@__DIR__, "fixtures", "oversized_probe.sh")
        @test_throws ArgumentError macos_iphone_observation(
            script=oversized, swift="/bin/sh")
        if Sys.which("swift") !== nothing
            @test success(run(`$(Sys.which("swift")) $script --self-test`))
        end
    else
        @test_throws ArgumentError macos_iphone_observation()
    end
end
