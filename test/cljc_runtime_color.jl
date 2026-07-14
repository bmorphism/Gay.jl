using Test
using Gay

@testset "induced .cljc runtime colors" begin
    portable = """(ns transition.portable-probe)
    (println (transduce (map inc) + 0 [1 2 3]))
    """
    core = cljc_core_id(portable)
    jank = cljc_runtime_color(core, :jank)
    basilisp = cljc_runtime_color(core, :basilisp)

    @test core == "1d7bcf1c802dd3db3b7429ceced6a9e0b92a2d9ef535b02ef0ffa3a954178b54"
    @test occursin(r"^[0-9a-f]{64}$", core)
    @test cljc_core_id(portable) == core
    @test cljc_core_id(Vector{UInt8}(codeunits(portable))) == core
    @test cljc_core_id(replace(portable, "\n" => "\r\n")) != core
    @test cljc_core_id("é") != cljc_core_id("e\u0301")

    # Both fibers are induced from one root; runtime accents are global and the
    # carrier motif depends on the product (root, runtime).
    @test jank.core_id == basilisp.core_id == core
    @test jank.core_seed == basilisp.core_seed
    @test jank.core_color == basilisp.core_color
    @test jank.reader_feature == :jank
    @test basilisp.reader_feature == :lpy
    @test jank.runtime_seed != basilisp.runtime_seed
    @test jank.runtime_color != basilisp.runtime_color
    @test jank.carrier_seed != basilisp.carrier_seed
    @test jank.carrier_color != basilisp.carrier_color
    @test (jank.core_color, jank.runtime_color, jank.carrier_color) ==
          ("#E794BB", "#7AEB76", "#C89E2F")
    @test (basilisp.core_color, basilisp.runtime_color, basilisp.carrier_color) ==
          ("#E794BB", "#91E358", "#30CC2E")

    @test cljc_runtime_color(core, "jank") == jank
    @test length(Set([jank, cljc_runtime_color(core, :jank)])) == 1
    @test length(Set([jank, basilisp])) == 2
    @test verify_cljc_runtime_color(jank)
    @test verify_cljc_runtime_color(basilisp)
    @test cljc_runtime_identity(jank) == ("clj1", core, :jank)
    @test cljc_runtime_identity(jank) != cljc_runtime_identity(basilisp)
    @test cljc_runtime_uri(jank) == "clojure://jank/cljc/clj1/gay-sha256/$core"
    @test cljc_runtime_uri(basilisp) == "clojure://basilisp/cljc/clj1/gay-sha256/$core"

    other = cljc_runtime_color(cljc_core_id(portable * "; changed\n"), :basilisp)
    @test other.core_id != core
    @test other.core_color != jank.core_color
    @test other.runtime_color == basilisp.runtime_color
    @test other.carrier_color != basilisp.carrier_color

    transition = cljc_runtime_transition(jank, basilisp)
    reverse_transition = cljc_runtime_transition(basilisp, jank)
    @test transition.source == jank
    @test transition.target == basilisp
    @test transition.transition_id != reverse_transition.transition_id
    @test transition.transition_color != reverse_transition.transition_color
    @test transition.transition_id ==
          "3e7b94f477b94ac97e50c1e630739db10138170ad455f7f148d7c8b6fb959d87"
    @test transition.transition_color == "#33A661"
    @test transition.required_roles == (Int8(0), Int8(1), Int8(-1))
    @test mod(sum(Int, transition.required_roles), 3) == 0
    @test verify_cljc_transition_structure(transition)
    @test cljc_runtime_transition(core, :jank, :basilisp) == transition

    @test_throws ArgumentError cljc_runtime_color(core, :Jank)
    @test_throws ArgumentError cljc_runtime_color(core, :clojure)
    @test_throws ArgumentError cljc_runtime_color(uppercase(core), :jank)
    @test_throws ArgumentError cljc_runtime_color(first(core, 63), :jank)
    @test_throws ArgumentError cljc_runtime_transition(jank, jank)
    @test_throws ArgumentError cljc_runtime_transition(jank, other)

    future_version = CljcRuntimeColor(
        "clj2",
        jank.core_id,
        jank.runtime,
        jank.reader_feature,
        jank.core_seed,
        jank.runtime_seed,
        jank.carrier_seed,
        jank.core_color,
        jank.runtime_color,
        jank.carrier_color,
    )
    @test cljc_runtime_identity(future_version) != cljc_runtime_identity(jank)
    @test !verify_cljc_runtime_color(future_version)
    @test_throws ArgumentError cljc_runtime_uri(future_version)

    # Negative controls: RGB equality never merges exact identities, and three
    # Play legs are not a valid structural requirement even though 1+1+1 = 0 mod 3.
    forced_collision = CljcRuntimeColor(
        basilisp.version,
        basilisp.core_id,
        basilisp.runtime,
        basilisp.reader_feature,
        basilisp.core_seed,
        basilisp.runtime_seed,
        basilisp.carrier_seed,
        jank.core_color,
        jank.runtime_color,
        jank.carrier_color,
    )
    @test cljc_runtime_identity(forced_collision) != cljc_runtime_identity(jank)
    @test !verify_cljc_runtime_color(forced_collision)

    false_green = CljcRuntimeTransition(
        transition.version,
        transition.source,
        transition.target,
        transition.transition_id,
        transition.transition_seed,
        transition.transition_color,
        (Int8(1), Int8(1), Int8(1)),
    )
    @test mod(sum(Int, false_green.required_roles), 3) == 0
    @test !verify_cljc_transition_structure(false_green)
end
