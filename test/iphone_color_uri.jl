using Test
using Gay
using SHA

@testset "iphone:// Gay.jl learnable color identifiers" begin
    # RFC 4231 test case 1 proves that the cryptographic primitive in this
    # environment is HMAC-SHA-256, not the public stable_seed color hash.
    @test bytes2hex(SHA.hmac_sha256(fill(UInt8(0x0b), 20), "Hi There")) ==
          "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"

    @test iphone_recording_count_bin(0) == 0
    @test iphone_recording_count_bin(4) == 1
    @test iphone_recording_count_bin(5) == 2
    @test iphone_recording_count_bin(16) == 2
    @test iphone_recording_count_bin(17) == 3
    @test_throws ArgumentError iphone_recording_count_bin(-1)

    interrupted = IPhoneProbe(:interrupted;
        voice_memos_sync=true,
        recording_count_bin=2,
        interaction_bin=1)
    connected = IPhoneProbe(:connected;
        voice_memos_sync=true,
        recording_count_bin=2,
        interaction_bin=3)
    unavailable = IPhoneProbe(:unavailable)

    @test iphone_probe_embedding(interrupted) == (1 / 3, 1.0, 2 / 3, 1 / 3)
    base_space = IPhoneColorSpace()
    @test iphone_probe_distance(base_space, interrupted, interrupted) == 0
    @test iphone_probe_distance(base_space, interrupted, connected) > 0
    @test iphone_probe_distance(base_space, interrupted, unavailable) > 0
    @test_throws ArgumentError IPhoneProbe(:timed_out)
    @test_throws ArgumentError IPhoneProbe(:connected; recording_count_bin=4)

    unavailable_synced = IPhoneProbe(:unavailable; voice_memos_sync=true)
    training = [
        (unavailable, unavailable_synced, true),
        (unavailable_synced, unavailable, true),
        (unavailable, connected, false),
        (unavailable_synced, connected, false),
    ]
    learned = learn_iphone_color_space(training)
    @test learned isa IPhoneColorSpace
    @test isapprox(sum(learned.weights), 4.0)
    @test all(>(0), learned.weights)
    @test learned.weights[1] > learned.weights[2]
    @test learn_iphone_color_space(reverse(training)).weights == learned.weights
    @test IPhoneColorSpace(weights=(1, 2, 3, 4)).weights ==
          IPhoneColorSpace(weights=(2, 4, 6, 8)).weights
    @test IPhoneColorSpace(weights=(floatmax(Float64), floatmax(Float64),
                                    floatmax(Float64), floatmax(Float64))).weights ==
          (1.0, 1.0, 1.0, 1.0)
    @test_throws ArgumentError IPhoneColorSpace(weights=(1, 0, 1, 1))
    @test_throws ArgumentError IPhoneColorSpace(
        weights=(nextfloat(0.0), 1.0, 1.0, floatmax(Float64)))
    @test_throws ArgumentError learn_iphone_color_space([
        (interrupted, interrupted, true),
        (unavailable, connected, false),
    ]; regularization=Inf)

    @test occursin(r"^#[0-9A-F]{6}$", iphone_root_color("passport.gay"))
    @test occursin(r"^#[0-9A-F]{6}$", iphone_probe_color(interrupted; space=learned))
    @test iphone_probe_color(interrupted; space=learned) ==
          iphone_probe_color(interrupted; space=learned)

    key_a = collect(UInt8(0x00):UInt8(0x1f))
    key_b = reverse(key_a)
    record = iphone_color_record(interrupted;
        pair_key=key_a,
        scope="external-mac",
        epoch="session-2026-07-12",
        semantic_root="passport.gay",
        space=learned)

    @test record.root_color == "#87EFBC"
    @test record.color == "#82F1AC"
    # Complete protocol vector: framing, domains, truncation, model digest,
    # canonical grammar, and all keyed tags are pinned together.
    @test iphone_uri(record) ==
          "iphone://g1-ba55723008865c5392b9baf653f7bac0-68d085099be2d23bb2d5b7555d5ec388/7d5c9c4ab6174b76f309072495c12627/259a0e1859ca57d3f780d59ccd6bf38b/f4ee76293f1701eadf07ac02b61e2ed6"
    reordered = iphone_color_record(interrupted;
        pair_key=key_a, scope="external-mac", epoch="session-2026-07-12",
        semantic_root="passport.gay", space=learn_iphone_color_space(reverse(training)))
    @test iphone_uri(reordered) == iphone_uri(record)
    @test startswith(iphone_uri(record), "iphone://$(iphone_color_identifier(record))/")
    @test startswith(passport_uri(record), "passport://gay/iphone/")
    @test iphone_uri(parse_iphone_uri(iphone_uri(record))) == iphone_uri(record)
    @test passport_uri(parse_passport_uri(passport_uri(record))) == passport_uri(record)
    @test verify_iphone_color_record(record, interrupted;
        pair_key=key_a, scope="external-mac", epoch="session-2026-07-12",
        semantic_root="passport.gay", space=learned)
    @test !verify_iphone_color_record(record, connected;
        pair_key=key_a, scope="external-mac", epoch="session-2026-07-12",
        semantic_root="passport.gay", space=learned)
    @test !verify_iphone_color_record(record, interrupted;
        pair_key=key_b, scope="external-mac", epoch="session-2026-07-12",
        semantic_root="passport.gay", space=learned)
    @test !verify_iphone_color_record(record, interrupted;
        pair_key=key_a, scope="other-mac", epoch="session-2026-07-12",
        semantic_root="passport.gay", space=learned)
    @test !verify_iphone_color_record(record, interrupted;
        pair_key=key_a, scope="external-mac", epoch="other-session",
        semantic_root="passport.gay", space=learned)

    registry = IPhoneColorRegistry()
    enrollment = (pair_key=key_a, scope="external-mac", epoch="session-2026-07-12",
                  semantic_root="passport.gay", space=learned)
    @test register_iphone_color!(registry, record, interrupted; enrollment...) === record
    @test register_iphone_color!(registry, record, interrupted; enrollment...) === record
    @test resolve_iphone_color(registry, iphone_uri(record)) === record
    @test resolve_iphone_color(registry, passport_uri(record)) === record
    crossed_vat = fetch(@async try
        resolve_iphone_color(registry, iphone_uri(record))
    catch error
        error
    end)
    @test crossed_vat isa ArgumentError
    @test_throws ArgumentError IPhoneColorRecord(
        record.ref, "red", record.color, record.embedding)
    @test_throws ArgumentError IPhoneColorRecord(
        record.ref, record.root_color, record.color, (NaN, 2.0, -1.0, Inf))
    conflicting = IPhoneColorRecord(record.ref, "#000000", record.color, record.embedding)
    @test_throws ArgumentError register_iphone_color!(
        registry, conflicting, interrupted; enrollment...)

    # Model learning changes the color/model token, never the enrollment tag.
    alternate_space = IPhoneColorSpace(version="alternate-v2", weights=(4, 1, 1, 1))
    alternate = iphone_color_record(interrupted;
        pair_key=key_a, scope="external-mac", epoch="session-2026-07-12",
        semantic_root="passport.gay", space=alternate_space)
    @test alternate.ref.pair_tag == record.ref.pair_tag
    @test alternate.ref.model_id != record.ref.model_id
    @test alternate.ref.color_token != record.ref.color_token
    alternate_enrollment = merge(enrollment, (space=alternate_space,))
    @test register_iphone_color!(
        registry, alternate, interrupted; alternate_enrollment...) === alternate
    @test_throws ArgumentError iphone_record_distance(
        registry, iphone_uri(record), iphone_uri(alternate))

    # Forced presentation-color collision across two enrollments cannot merge identity.
    other_pair = iphone_color_record(interrupted;
        pair_key=key_b, scope="external-mac", epoch="session-2026-07-12",
        semantic_root="passport.gay", space=learned)
    @test other_pair.color == record.color
    @test other_pair.ref.pair_tag != record.ref.pair_tag
    @test other_pair.ref.color_token != record.ref.color_token
    @test other_pair.ref.scope_token != record.ref.scope_token
    @test other_pair.ref.epoch_token != record.ref.epoch_token
    @test resolve_iphone_color(registry, iphone_uri(other_pair)) === nothing
    other_enrollment = merge(enrollment, (pair_key=key_b,))
    @test register_iphone_color!(
        registry, other_pair, interrupted; other_enrollment...) === other_pair
    @test iphone_record_distance(registry, iphone_uri(record), iphone_uri(other_pair)) == 0

    # Scope and epoch rotation make the same enrollment unlinkable at the URI layer.
    rotated = iphone_color_record(interrupted;
        pair_key=key_a, scope="external-mac", epoch="next-session",
        semantic_root="passport.gay", space=learned)
    @test rotated.ref.pair_tag != record.ref.pair_tag
    @test rotated.ref.epoch_token != record.ref.epoch_token
    @test rotated.ref.scope_token != record.ref.scope_token
    rotated_enrollment = merge(enrollment, (epoch="next-session",))
    @test register_iphone_color!(
        registry, rotated, interrupted; rotated_enrollment...) === rotated
    @test purge_iphone_epoch!(registry, rotated.ref.epoch_token) == 1
    @test resolve_iphone_color(registry, iphone_uri(rotated)) === nothing

    rescope = iphone_color_record(interrupted;
        pair_key=key_a, scope="other-mac", epoch="session-2026-07-12",
        semantic_root="passport.gay", space=learned)
    @test rescope.ref.scope_token != record.ref.scope_token
    @test rescope.ref.epoch_token != record.ref.epoch_token
    @test unregister_iphone_color!(registry, iphone_uri(other_pair)) === other_pair
    @test resolve_iphone_color(registry, iphone_uri(other_pair)) === nothing

    # Raw labels and forbidden device/recording values never appear in the URI.
    rendered = iphone_uri(record)
    for forbidden in ("passport.gay", "external-mac", "session-2026-07-12")
        @test !occursin(forbidden, rendered)
    end
    @test_throws ArgumentError parse_iphone_uri(uppercase(rendered))
    @test_throws ArgumentError parse_iphone_uri(rendered * "?raw=1")
    @test_throws ArgumentError parse_iphone_uri(rendered * "\n")
    @test_throws ArgumentError iphone_color_record(interrupted;
        pair_key=UInt8[1, 2, 3], scope="external-mac", epoch="session")

    # Canonically equivalent Unicode labels must agree across paired Macs.
    composed = iphone_color_record(interrupted;
        pair_key=key_a, scope="caf\u00e9", epoch="session",
        semantic_root="passport.gay", space=learned)
    decomposed = iphone_color_record(interrupted;
        pair_key=key_a, scope="cafe\u0301", epoch="session",
        semantic_root="passport.gay", space=learned)
    @test iphone_uri(composed) == iphone_uri(decomposed)

    fresh = generate_iphone_pair_key()
    @test fresh isa Vector{UInt8}
    @test length(fresh) == 32
end
