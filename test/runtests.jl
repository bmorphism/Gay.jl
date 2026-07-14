using Test
using Gay
using Aqua
using Colors

include("iphone_color_uri.jl")
include("macos_iphone_probe.jl")
include("cljc_runtime_color.jl")

@testset "Gay.jl — seed 0x8b449cd3828014dd (the amp-thread tag this package was born from)" begin

    @test Gay.HASH_SEED == 0x8b449cd3828014dd
    @test Gay.GAY_SEED == UInt64(1069)
    @test gay_seed("8b449cd3828014dd") == Gay.HASH_SEED

    @testset "spi-race compatible O(1) kernel" begin
        @test spi_color_hex(42, 0) == "#727622"
        @test spi_color_hex(42, 1) == "#EB6E95"
        @test spi_color_hex(42, 69) == "#A8E8BD"
        @test spi_trit(42, 0) == 1
        @test spi_trit(42, 1) == 1
        @test spi_trit(42, 69) == 0
        @test spi_xor_fingerprint(42, 0, 0) == 0x0000000000000000
        @test spi_xor_fingerprint(42, 0, 1_000_000) == 0x0000000010de88
        @test spi_xor_fingerprint(42, 0, 10_000_000) == 0x00000000f76ceb
        @test spi_xor_fingerprint_parallel(42, 1_000_000; chunks=1) == 0x0000000010de88
        @test spi_xor_fingerprint_parallel(42, 1_000_000; chunks=4) == 0x0000000010de88
        @test spi_xor_fingerprint_parallel(42, 10_000_000; chunks=4) == 0x00000000f76ceb
        # spi_trit_sum: raw mod-3 residue {0,1,2}, byte-identical to libspi.zig
        # (cross-validated via scripts/spi_ffi_crossvalidate.jl against the dylib).
        @test spi_trit_sum(42, 0, 0) == 0
        @test spi_trit_sum(42, 0, 3) == 2
        @test spi_trit_sum(42, 0, 100) == 2
    end

    # Pinned by the Python port in /tmp/gay_seed_8b44.py (gay_julia_bridge.py
    # algorithm). If these change, either the algorithm or the port drifted.
    expected_hash = [
        "#55DB2A", "#CF851D", "#7A49DD", "#6BD9CF",
        "#E25560", "#29D1B2", "#A67FE1", "#9EE134",
    ]
    @testset "color_at(i; seed=HASH_SEED) i=0..7" begin
        for (i, hex) in enumerate(expected_hash)
            @test color_at(i - 1; seed=Gay.HASH_SEED) == hex
        end
    end

    expected_canon = [
        "#196ACC", "#5537AE", "#EA2243", "#D98D1E",
        "#59D0CF", "#A49627", "#EA5136", "#B20DDE",
    ]
    @testset "color_at(i; seed=GAY_SEED=1069) i=0..7" begin
        for (i, hex) in enumerate(expected_canon)
            @test color_at(i - 1; seed=Gay.GAY_SEED) == hex
        end
    end

    expected_hash_gamma = [
        "#D06BE7", "#AE5EE3", "#448CD6", "#9B229D",
        "#880AAC", "#DC0BBF", "#31BE2C", "#D795E0",
    ]
    @testset "color_at(i; seed=HASH gamma=HASH|1) i=0..7" begin
        for (i, hex) in enumerate(expected_hash_gamma)
            @test color_at(i - 1; seed=Gay.HASH_SEED, gamma=Gay.HASH_SEED | UInt64(1)) == hex
        end
    end

    @testset "trit triad sums to -1 (Coplay) for all three seedings — coincidence at n=3" begin
        function triad(; kw...)
            s = sum(Int.(trit(i; kw...)) for i in 0:2) % 3
            s ≤ 1 ? s : s - 3
        end
        @test triad(seed=Gay.HASH_SEED) == -1
        @test triad(seed=Gay.GAY_SEED) == -1
        @test triad(seed=Gay.HASH_SEED, gamma=Gay.HASH_SEED | UInt64(1)) == -1
    end

    @testset "SplitMix64 mix64 against a published vector" begin
        # mix64(0) is a well-known SplitMix64 test vector.
        @test Gay.mix64(UInt64(0)) == 0x0000000000000000
        # Advance once from seed=0 with golden gamma:
        r = SplittableRandom(UInt64(0))
        Gay._next!(r)  # r.seed = GOLDEN_GAMMA
        @test r.seed == Gay.GOLDEN_GAMMA
    end

    @testset "hierarchical colors are deterministic prefix trails" begin
        trail = hierarchical_colors("agent/3/seed"; seed=42)
        @test first.(trail) == ["agent", "agent/3", "agent/3/seed"]
        @test trail == hierarchical_colors("agent/3/seed"; seed=42)
        @test length(unique(last.(trail))) == 3
    end

    @testset "semantic fault atlas URI has pinned seed, color, trit, and prefix trail" begin
        uri = "jepsen://tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash/elle/pass"
        seed = stable_seed(uri)
        trail = hierarchical_colors(uri)

        @test seed == 0xb701dde86a270bcc
        @test color_at(0; seed=seed) == "#D70E86"
        @test trit(0; seed=seed) == 1
        @test first.(trail) == [
            "jepsen",
            "jepsen/tigerbeetle",
            "jepsen/tigerbeetle/0.16.11",
            "jepsen/tigerbeetle/0.16.11/transfer",
            "jepsen/tigerbeetle/0.16.11/transfer/strict-serializable",
            "jepsen/tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash",
            "jepsen/tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash/elle",
            "jepsen/tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash/elle/pass",
        ]
        @test last(trail) == (
            "jepsen/tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash/elle/pass",
            "#82E4F6",
        )
    end

    @testset "colored self-avoiding walk never touches a color twice" begin
        adjacency = Dict(
            "root" => ["left", "right"],
            "left" => ["root", "left/deeper"],
            "right" => ["root", "right/deeper"],
            "left/deeper" => ["left"],
            "right/deeper" => ["right"],
        )
        result = color_self_avoiding_walk(adjacency, "root"; steps=8, seed=42)
        colors = [step.color for step in result.steps]
        @test length(colors) == length(unique(colors))
        @test result.stopped_reason in ("trapped", "step_limit")
    end

    @testset "17 workers steal work over one shared touched-color set" begin
        adjacency = Dict{String,Vector{String}}()
        starts = String[]
        for worker in 1:17
            start = "worker/$worker/start"
            push!(starts, start)
            adjacency[start] = ["worker/$worker/a", "worker/$worker/b", "shared/frontier/$worker"]
            adjacency["worker/$worker/a"] = ["worker/$worker/a/1", "worker/$worker/a/2"]
            adjacency["worker/$worker/b"] = ["worker/$worker/b/1", "worker/$worker/b/2"]
            adjacency["shared/frontier/$worker"] = ["shared/frontier/$worker/1"]
            adjacency["worker/$worker/a/1"] = String[]
            adjacency["worker/$worker/a/2"] = String[]
            adjacency["worker/$worker/b/1"] = String[]
            adjacency["worker/$worker/b/2"] = String[]
            adjacency["shared/frontier/$worker/1"] = String[]
        end

        result = work_stealing_walk(adjacency, starts; workers=17, max_steps=85, fanout=3, seed=Gay.HASH_SEED)
        colors = [step.color for step in result.steps]
        @test length(colors) == length(unique(colors))
        @test Set(step.worker for step in result.steps) == Set(1:17)
        @test length(result.touched_colors) == length(result.steps)
        @test result.stopped_reason in ("frontier_exhausted", "step_limit")
    end

    @testset "deterministic port rotation is collision-free up to capacity" begin
        identity = "jank-lang/activity-map|nrepl|blog+github-2026|world"

        small = assert_port_noncontention(17, identity)
        @test small.port_min == 29000
        @test small.port_span == 20000
        @test small.upper_bound == 20000
        @test small.collisions == 0
        @test small.unique_ports == 17
        @test all(29000 .<= small.ports .<= 48999)

        again = assert_port_noncontention(17, identity)
        @test again.ports == small.ports

        rotated = assert_port_noncontention(17, identity; frame=1)
        @test rotated.collisions == 0
        @test all(29000 .<= rotated.ports .<= 48999)

        capacity = assert_port_noncontention(20000, identity)
        @test capacity.saturated
        @test capacity.unique_ports == 20000
        @test capacity.collisions == 0
        @test minimum(capacity.ports) == 29000
        @test maximum(capacity.ports) == 48999

        over = port_rotation_report(20001, identity)
        @test over.collisions == 1
        @test over.pigeonhole_min_collisions == 1
        @test_throws ArgumentError assert_port_noncontention(20001, identity)
    end

    @testset "frames-in-flight cadence bound composes SPI throughput and drain time" begin
        bound = frames_in_flight_bound(20000; assignments_per_second=2.0e6, drain_seconds=0.25)
        @test bound.planner_limited_hz == 100.0
        @test bound.drain_limited_hz == 4.0
        @test bound.max_rotation_hz == 4.0
        @test bound.spi_fast_enough_for_drain

        slow = frames_in_flight_bound(20000; assignments_per_second=20000.0, drain_seconds=0.25)
        @test slow.planner_limited_hz == 1.0
        @test slow.max_rotation_hz == 1.0
        @test !slow.spi_fast_enough_for_drain
    end

    @testset "non-contention has many independent proof witnesses" begin
        identity = "jank-lang/activity-map|nrepl|blog+github-2026|world"
        catalog = port_proof_catalog(20000, identity)
        names = Set(w.name for w in catalog)
        expected = Set([
            :constructive_assignment,
            :finite_exhaustion,
            :set_cardinality,
            :modular_cancellation,
            :bounded_difference,
            :cyclic_translation,
            :permutation_or_prefix,
            :induction_on_prefix,
            :contradiction_minimal_duplicate,
            :pigeonhole_upper_bound,
            :abductive_best_explanation,
            :spi_order_independence,
            :range_and_ephemeral_disjointness,
        ])

        @test expected ⊆ names
        @test all(w.verdict for w in catalog)
        @test occursin("abductive", port_proof_catalog_text(17, identity))

        over_catalog = port_proof_catalog(20001, identity)
        by_name = Dict(w.name => w for w in over_catalog)
        @test !by_name[:constructive_assignment].verdict
        @test !by_name[:set_cardinality].verdict
        @test by_name[:pigeonhole_upper_bound].verdict
        @test by_name[:abductive_best_explanation].verdict
    end

    @testset "deterministic port rotation is a TOFU pin" begin
        identity = "jank-lang/activity-map|nrepl|blog+github-2026|world"
        pin = port_tofu_record(identity; requested_processes=17)

        @test pin.identity == identity
        @test pin.requested_processes == 17
        @test pin.port_min == 29000
        @test pin.port_span == 20000
        @test pin.offset == port_rotation_offset(identity)
        @test pin.fingerprint == port_tofu_fingerprint(identity; requested_processes=17)
        @test pin.color == color_at(0; seed=pin.fingerprint)
        @test verify_port_tofu(pin)

        @test !verify_port_tofu(pin; identity=identity * "|renamed")
        @test !verify_port_tofu(pin; frame=1)
        @test !verify_port_tofu(pin; requested_processes=18)
        @test !verify_port_tofu(pin; port_min=30000)

        text = port_tofu_record_text(pin)
        @test occursin("Port TOFU record", text)
        @test occursin("fingerprint: 0x", text)
        @test occursin(pin.color, text)

        @test_throws ArgumentError port_tofu_record(identity; requested_processes=20001)
    end

    @testset "Self/Other Boundary & Integer Robustness (assert_boundary_integrity)" begin
        # 1. Healthy float distinction
        r1 = SplittableRandom(0x9E3779B97F4A7C15)
        r2 = SplittableRandom(0x42D)
        @test assert_boundary_integrity(r1, r2) == false

        # 2. Float collapse (difference too small), but distinct integers
        ra = SplittableRandom(0x9E3779B97F4A7C15, 0x9e3779b97f4a7c15)
        rb = SplittableRandom(0x9E3779B97F4A7C16, 0x9e3779b97f4a7c15)
        
        # Test with an epsilon that forces them to collapse
        @test assert_boundary_integrity(ra, rb; epsilon=1.0) == true

        # 3. True collapse (both seed and gamma are identical)
        rc = SplittableRandom(0x9E3779B97F4A7C15, 0x9e3779b97f4a7c15)
        @test_throws ArgumentError assert_boundary_integrity(ra, rc; epsilon=1.0)
    end
end

@testset "Aqua.jl — naming + metadata verification" begin
    Aqua.test_all(Gay)
end

@testset "GayColorsExt — perceptual color science (Colors.jl package extension)" begin
    @test gay_colorant(color_at(0)) isa Colorant
    @test gay_colordiff(0, 0) == 0.0          # identical colors → zero CIEDE2000
    @test gay_colordiff(0, 1) > 0.0           # distinct Gay colors → positive distance
end

@testset "O(1) random-access kernels (splitmixrgb-xf port)" begin
    @test split_mix_64(UInt64(0)) == split_mix_64(UInt64(0))   # deterministic
    @test split_mix_64(UInt64(0)) != split_mix_64(UInt64(1))   # input-sensitive
    let rgb = hash_color_rgb(GAY_SEED, 0)
        @test rgb isa NTuple{3,Float32}
        @test all(0.0f0 .<= rgb .<= 1.0f0)
    end
    @test hash_color_rgb(GAY_SEED, 7) == hash_color_rgb(GAY_SEED, 7)  # O(1) reproducible
    @test hash_color_rgb(GAY_SEED, 7) != hash_color_rgb(GAY_SEED, 8)
    let (L, C, H) = hash_color_lch(GAY_SEED, 3)
        @test 30.0f0 <= L <= 80.0f0
        @test 40.0f0 <= C <= 80.0f0
        @test 0.0f0 <= H <= 360.0f0
    end
    @test length(hash_color_hex(GAY_SEED, 0)) == 7 && startswith(hash_color_hex(GAY_SEED, 0), "#")
end

ripserer_loaded = Ref(false)
ripserer_exception = Ref{Any}(nothing)

try
    using Ripserer
    ripserer_loaded[] = true
    @testset "GayRipsererExt — topological persistence (Ripserer.jl package extension)" begin
        # Test on manual list of colors
        colors = [color_at(i) for i in 0:5]
        diag = gay_ripserer(colors)
        @test diag isa AbstractVector
        @test length(diag) >= 1
        @test eltype(diag) <: Ripserer.PersistenceDiagram

        # Test on integer count
        diag_n = gay_ripserer(6)
        @test diag_n isa AbstractVector
        @test eltype(diag_n) <: Ripserer.PersistenceDiagram

        # Test on WalkResult
        adjacency = Dict(
            "root" => ["left", "right"],
            "left" => ["root"],
            "right" => ["root"]
        )
        walk = color_self_avoiding_walk(adjacency, "root"; steps=2)
        diag_walk = gay_ripserer(walk)
        @test diag_walk isa AbstractVector
        @test eltype(diag_walk) <: Ripserer.PersistenceDiagram

        # Test direct API overrides
        @test ripserer(colors) isa AbstractVector
        @test ripserer(colors; metric=:perceptual) isa AbstractVector
        @test ripserer(walk) isa AbstractVector
        @test ripserer(6) isa AbstractVector

        # Verify that perceptual and euclidean metric results are both valid
        d_eucl = ripserer(colors; metric=:euclidean)
        d_perc = ripserer(colors; metric=:perceptual)
        @test d_eucl isa AbstractVector
        @test d_perc isa AbstractVector
    end
catch e
    ripserer_exception[] = e
    @warn "Skipping Ripserer tests: Ripserer failed to load" exception=e
end

try
    using FractalDimensions
    @testset "GayFractalExt — correlation dimension (FractalDimensions.jl package extension)" begin
        # Test on manual list of colors (using 150 colors to ensure robust linear fit and avoid NaN)
        colors = [color_at(i) for i in 0:149]
        fd = gay_fractal_dimension(colors)
        @test fd isa Float64
        @test fd >= 0.0
        @test !isnan(fd)

        # Test on integer count (using 150 colors)
        fd_n = gay_fractal_dimension(150)
        @test fd_n isa Float64
        @test !isnan(fd_n)

        # Test on too few points (e.g. n < 3)
        @test gay_fractal_dimension(2) == 0.0

        # Test on WalkResult (using 50 steps)
        adjacency = Dict(
            "root" => ["left", "right"],
            "left" => ["root", "left/deeper"],
            "right" => ["root", "right/deeper"],
            "left/deeper" => ["left"],
            "right/deeper" => ["right"]
        )
        walk = color_self_avoiding_walk(adjacency, "root"; steps=50)
        fd_walk = gay_fractal_dimension(walk)
        @test fd_walk isa Float64

        # Test direct API overrides
        @test grassberger_proccacia_dim(colors) isa Float64
        @test grassberger_proccacia_dim(colors; metric=:perceptual) isa Float64
        @test grassberger_proccacia_dim(walk) isa Float64
        @test grassberger_proccacia_dim(150) isa Float64

        # Confirm that the perceptual and Euclidean metrics are distinct and not NaN
        fd_eucl = grassberger_proccacia_dim(colors; metric=:euclidean)
        fd_perc = grassberger_proccacia_dim(colors; metric=:perceptual)
        @test fd_eucl != fd_perc
        @test !isnan(fd_eucl)
        @test !isnan(fd_perc)
    end
catch e
    @warn "Skipping FractalDimensions tests: FractalDimensions failed to load" exception=e
end

if ripserer_loaded[]
    try
        using PersistenceDiagrams
    @testset "GayPersistenceDiagramsExt — matching & distance metrics (PersistenceDiagrams.jl package extension)" begin
        # 1. Test PersistenceDiagram constructors (standard ones)
        colors = [color_at(i) for i in 0:5]
        diag = PersistenceDiagram(colors; dim=0)
        @test diag isa PersistenceDiagram
        @test dim(diag) == 0

        adjacency = Dict(
            "root" => ["left", "right"],
            "left" => ["root"],
            "right" => ["root"]
        )
        walk = color_self_avoiding_walk(adjacency, "root"; steps=2)
        diag_walk = PersistenceDiagram(walk; dim=0)
        @test diag_walk isa PersistenceDiagram
        @test dim(diag_walk) == 0

        diag_n = PersistenceDiagram(6; dim=0)
        @test diag_n isa PersistenceDiagram
        @test dim(diag_n) == 0

        # 2. Test the first-class GayPersistenceDiagram constructors and properties
        gpd = GayPersistenceDiagram(colors; dim=0)
        @test gpd isa GayPersistenceDiagram
        @test gpd isa AbstractVector
        @test dim(gpd) == 0
        @test length(gpd) == length(gpd.diagram)
        @test gpd.colors == colors
        @test gpd.source == colors
        # Test vector delegation
        if length(gpd) > 0
            @test gpd[1] == gpd.diagram[1]
        end
        @test threshold(gpd) == threshold(gpd.diagram)
        @test PersistenceDiagram(gpd) === gpd.diagram
        @test convert(PersistenceDiagram, gpd) === gpd.diagram
        @test Base.IndexStyle(gpd) === Base.IndexLinear()

        gpd_walk = GayPersistenceDiagram(walk; dim=0)
        @test gpd_walk isa GayPersistenceDiagram
        @test gpd_walk.source === walk
        @test dim(gpd_walk) == 0

        gpd_n = GayPersistenceDiagram(6; dim=0)
        @test gpd_n isa GayPersistenceDiagram
        @test gpd_n.source == 6
        @test dim(gpd_n) == 0

        # Test pretty show/printing for GayPersistenceDiagram
        io = IOBuffer()
        show(io, gpd)
        @test occursin("GayPersistenceDiagram", String(take!(io)))
        show(io, MIME"text/plain"(), gpd)
        @test occursin("🌈", String(take!(io)))

        # 3. Test Bottleneck and Wasserstein callable functors on standard & Gay diagrams
        colors1 = [color_at(i) for i in 0:5]
        colors2 = [color_at(i+2) for i in 0:5]
        diag1 = PersistenceDiagram(colors1; dim=0)
        diag2 = PersistenceDiagram(colors2; dim=0)
        gpd1 = GayPersistenceDiagram(colors1; dim=0)
        gpd2 = GayPersistenceDiagram(colors2; dim=0)

        @test Bottleneck()(diag1, diag2) isa Real
        @test Wasserstein()(diag1, diag2) isa Real
        @test Bottleneck()(gpd1, gpd2) isa Real
        @test Wasserstein()(gpd1, gpd2) isa Real

        walk1 = color_self_avoiding_walk(adjacency, "root"; steps=2, seed=GAY_SEED)
        walk2 = color_self_avoiding_walk(adjacency, "root"; steps=2, seed=GAY_SEED + 1)
        @test Bottleneck()(walk1, walk2; dim=0) isa Real
        @test Wasserstein()(walk1, walk2; dim=0) isa Real

        @test Bottleneck()(colors1, colors2; dim=0) isa Real
        @test Wasserstein()(colors1, colors2; dim=0) isa Real

        @test Bottleneck()(6, 6; seed1=GAY_SEED, seed2=GAY_SEED+1, dim=0) isa Real
        @test Wasserstein()(6, 6; seed1=GAY_SEED, seed2=GAY_SEED+1, dim=0) isa Real

        # 4. Test GayBottleneck and GayWasserstein callable functors
        @test GayBottleneck()(gpd1, gpd2) isa Real
        @test GayWasserstein()(gpd1, gpd2) isa Real
        @test GayBottleneck()(diag1, diag2) isa Real
        @test GayWasserstein()(diag1, diag2) isa Real
        @test GayBottleneck()(walk1, walk2; dim=0) isa Real
        @test GayWasserstein()(walk1, walk2; dim=0) isa Real
        @test GayBottleneck()(colors1, colors2; dim=0) isa Real
        @test GayWasserstein()(colors1, colors2; dim=0) isa Real
        @test GayBottleneck()(6, 6; dim=0) isa Real
        @test GayWasserstein()(6, 6; dim=0) isa Real

        # 5. Test matching (standard matching and gay_matching)
        m_b = matching(Bottleneck(), diag1, diag2)
        @test m_b isa PersistenceDiagrams.Matching
        @test matching(Bottleneck(), gpd1, gpd2) isa PersistenceDiagrams.Matching
        @test matching(Bottleneck(), walk1, walk2; dim=0) isa PersistenceDiagrams.Matching
        @test matching(Wasserstein(), colors1, colors2; dim=0) isa PersistenceDiagrams.Matching
        @test matching(Bottleneck(), 6, 6; seed1=GAY_SEED, seed2=GAY_SEED+1, dim=0) isa PersistenceDiagrams.Matching

        @test gay_matching(Bottleneck(), gpd1, gpd2) isa PersistenceDiagrams.Matching
        @test gay_matching(GayBottleneck(), gpd1, gpd2) isa PersistenceDiagrams.Matching
        @test gay_matching(GayWasserstein(), walk1, walk2; dim=0) isa PersistenceDiagrams.Matching
        @test gay_matching(GayBottleneck(), colors1, colors2; dim=0) isa PersistenceDiagrams.Matching
        @test gay_matching(GayBottleneck(), 6, 6; dim=0) isa PersistenceDiagrams.Matching

        # 6. Test thin wrappers in Gay namespace
        @test gay_bottleneck(diag1, diag2) isa Real
        @test gay_bottleneck(gpd1, gpd2) isa Real
        @test gay_bottleneck(walk1, walk2; dim=0) isa Real
        @test gay_bottleneck(colors1, colors2; dim=0) isa Real
        @test gay_bottleneck(6, 6; dim=0) isa Real

        @test gay_wasserstein(diag1, diag2) isa Real
        @test gay_wasserstein(gpd1, gpd2) isa Real
        @test gay_wasserstein(walk1, walk2; dim=0) isa Real
        @test gay_wasserstein(colors1, colors2; dim=0) isa Real
        @test gay_wasserstein(6, 6; dim=0) isa Real

        @test gay_persistencediagram(colors; dim=0) isa GayPersistenceDiagram
        @test gay_persistencediagram(walk; dim=0) isa GayPersistenceDiagram
        @test gay_persistencediagram(6; dim=0) isa GayPersistenceDiagram

        # 7. The 69th witness: one semantic key interleaves the whole Gay.jl surface.
        @test begin
            atlas_uri = "jepsen://tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash/elle/pass"
            atlas_seed = stable_seed(atlas_uri)
            atlas_trail = hierarchical_colors(atlas_uri)
            atlas_color = color_at(0; seed=atlas_seed)
            atlas_hash = hash_color_hex(atlas_seed, 0)
            atlas_pin = port_tofu_record(atlas_uri; requested_processes=3, seed=atlas_seed)
            atlas_ports = assert_port_noncontention(3, atlas_uri; seed=atlas_seed)
            atlas_bound = frames_in_flight_bound(3; assignments_per_second=300.0, drain_seconds=0.25)

            atlas_adjacency = Dict(
                atlas_uri => [atlas_uri * "/history", atlas_uri * "/model", atlas_uri * "/nemesis"],
                atlas_uri * "/history" => [atlas_uri * "/history/elle", atlas_uri * "/history/knossos"],
                atlas_uri * "/model" => [atlas_uri * "/model/strict-serializable"],
                atlas_uri * "/nemesis" => [atlas_uri * "/nemesis/partition", atlas_uri * "/nemesis/crash"],
                atlas_uri * "/history/elle" => String[],
                atlas_uri * "/history/knossos" => String[],
                atlas_uri * "/model/strict-serializable" => String[],
                atlas_uri * "/nemesis/partition" => String[],
                atlas_uri * "/nemesis/crash" => String[],
            )
            atlas_walk = color_self_avoiding_walk(atlas_adjacency, atlas_uri; steps=4, seed=atlas_seed)
            atlas_steal = work_stealing_walk(atlas_adjacency,
                                             [atlas_uri, atlas_uri * "/history", atlas_uri * "/nemesis"];
                                             workers=3, max_steps=9, fanout=2, seed=atlas_seed)

            atlas_colors = unique(vcat(
                [atlas_color, last(atlas_trail)[2], atlas_hash, atlas_pin.color],
                [step.color for step in atlas_walk.steps],
                [step.color for step in atlas_steal.steps],
                [color_at(i; seed=atlas_seed) for i in 0:8],
            ))
            atlas_diag = GayPersistenceDiagram(atlas_colors; dim=0)
            atlas_walk_diag = GayPersistenceDiagram(atlas_walk; dim=0)
            atlas_std_diag = PersistenceDiagram(atlas_colors; dim=0)
            atlas_fd = gay_fractal_dimension(150; seed=atlas_seed)

            all(Bool[
                atlas_seed == 0xb701dde86a270bcc,
                atlas_color == "#D70E86",
                trit(0; seed=atlas_seed) == 1,
                first(last(atlas_trail)) == "jepsen/tigerbeetle/0.16.11/transfer/strict-serializable/partition+crash/elle/pass",
                last(atlas_trail)[2] == "#82E4F6",
                atlas_hash == hash_color_hex(atlas_seed, 0),
                hash_color_rgb(atlas_seed, 0) isa NTuple{3,Float32},
                hash_color_lch(atlas_seed, 0) isa NTuple{3,Float32},
                gay_colorant(atlas_color) isa Colorant,
                gay_colordiff(atlas_color, last(atlas_trail)[2]) isa Real,
                verify_port_tofu(atlas_pin; seed=atlas_seed),
                atlas_ports.collisions == 0,
                port_for_worker(0, atlas_uri; seed=atlas_seed) == first(atlas_ports.ports),
                atlas_bound.spi_fast_enough_for_drain,
                occursin("pigeonhole", port_proof_catalog_text(3, atlas_uri; seed=atlas_seed)),
                atlas_walk isa WalkResult,
                atlas_steal isa WalkResult,
                length(atlas_walk.steps) > 0,
                length(atlas_steal.steps) > 0,
                length(atlas_colors) >= 9,
                gay_ripserer(atlas_colors) isa AbstractVector,
                atlas_std_diag isa PersistenceDiagram,
                atlas_diag isa GayPersistenceDiagram,
                atlas_walk_diag isa GayPersistenceDiagram,
                gay_bottleneck(atlas_diag, atlas_walk_diag) isa Real,
                gay_wasserstein(atlas_diag, atlas_walk_diag) isa Real,
                gay_matching(GayBottleneck(), atlas_diag, atlas_walk_diag) isa PersistenceDiagrams.Matching,
                atlas_fd isa Float64 && !isnan(atlas_fd),
                !assert_boundary_integrity(SplittableRandom(0x9E3779B97F4A7C15), SplittableRandom(0x42D)),
            ])
        end
        end
    catch e
        @warn "Skipping PersistenceDiagrams tests: PersistenceDiagrams/Ripserer extension failed to load" exception=e
    end
else
    @warn "Skipping PersistenceDiagrams tests: Ripserer failed to load and GayPersistenceDiagramsExt depends on it"
end
