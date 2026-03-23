using Test

@testset "Gay Passport - did:gay:* DID System" begin

    @testset "DID construction and resolution" begin
        did = Gay.GayPassport.GayDID(UInt64(1069))
        did_str = Gay.did_gay(UInt64(1069))
        @test startswith(did_str, "did:gay:")
        @test length(did_str) == 8 + 16  # "did:gay:" + 16 hex chars

        # Round-trip resolution
        resolved = Gay.resolve_did(did_str)
        @test resolved !== nothing
        @test resolved.method_specific_id == did.method_specific_id

        # Invalid DIDs
        @test Gay.resolve_did("did:web:example.com") === nothing
        @test Gay.resolve_did("did:gay:tooshort") === nothing
        @test Gay.resolve_did("did:gay:ZZZZZZZZZZZZZZZZ") === nothing
    end

    @testset "DID verification" begin
        seed = UInt64(1069)
        did = Gay.GayPassport.GayDID(seed)
        @test Gay.verify_did(did, seed)
        @test !Gay.verify_did(did, UInt64(420))
    end

    @testset "Passport issuance" begin
        p = Gay.issue_passport(UInt64(1069))
        @test !p.blessed
        @test length(p.witnesses) == 0
        @test length(p.stamps) == 0

        # Genesis colors are deterministic
        colors = Gay.passport_colors(p)
        @test length(colors) == 3
        @test all(c -> startswith(c, "#") && length(c) == 7, colors)

        # Same seed = same passport colors (SPI)
        p2 = Gay.issue_passport(UInt64(1069))
        @test Gay.passport_colors(p) == Gay.passport_colors(p2)

        # Different seed = different colors
        p3 = Gay.issue_passport(UInt64(420))
        @test Gay.passport_colors(p) != Gay.passport_colors(p3)
    end

    @testset "Passport verification" begin
        p = Gay.issue_passport(UInt64(1069))
        @test Gay.verify_passport(p)
    end

    @testset "Blessing via IRL witnessing" begin
        p = Gay.issue_passport(UInt64(42))
        @test !p.blessed

        # Witness from another seed holder
        Gay.witness_passport(p, UInt64(1069), "vivarium-witness")
        @test p.blessed
        @test length(p.witnesses) == 1
        @test p.witnesses[1].name == "vivarium-witness"

        # Multiple witnesses strengthen web of trust
        Gay.witness_passport(p, UInt64(420), "minecraft-bci-witness")
        @test length(p.witnesses) == 2
    end

    @testset "Stamps for vivid color experiences" begin
        p = Gay.issue_passport(UInt64(69))
        Gay.add_stamp(p, "rainbow-flash", "Vivarium Launch Wed-Fri")
        @test length(p.stamps) == 1
        @test p.stamps[1].name == "rainbow-flash"
        @test p.stamps[1].event == "Vivarium Launch Wed-Fri"

        # Stamp has a color
        stamp = p.stamps[1]
        @test all(0.0f0 .<= stamp.color .<= 1.0f0)
    end

    @testset "Passport fingerprint (SPI)" begin
        p1 = Gay.issue_passport(UInt64(1069); timestamp=1000.0)
        p2 = Gay.issue_passport(UInt64(1069); timestamp=1000.0)
        @test Gay.passport_fingerprint(p1) == Gay.passport_fingerprint(p2)

        # Different seeds = different fingerprints
        p3 = Gay.issue_passport(UInt64(420); timestamp=1000.0)
        @test Gay.passport_fingerprint(p1) != Gay.passport_fingerprint(p3)
    end

    @testset "Premine batch issuance" begin
        seeds = UInt64[1069, 420, 69, 1337, 42]
        registry = Gay.premine_passports(seeds)
        @test length(registry) == 5

        # All unblessed
        for (_, p) in registry.passports
            @test !p.blessed
        end

        # Premine with blessing
        blessed_registry = Gay.premine_passports(seeds;
            bless_with=(UInt64(1069), "vivarium-issuer"))
        @test length(blessed_registry) == 5
        for (_, p) in blessed_registry.passports
            @test p.blessed
            @test length(p.witnesses) == 1
        end
    end

    @testset "World builder (AGENTS.md compliant)" begin
        w = Gay.world_passport(
            seeds=UInt64[1069, 420, 69, 1337, 42],
            bless_with=(UInt64(1069), "vivarium-witness")
        )
        @test length(w) == 5
        @test w.n_blessed == 5
        @test w.n_stamps == 0
        @test w.fingerprint_val != UInt64(0)

        # Merge two worlds
        w2 = Gay.world_passport(seeds=UInt64[7, 13])
        merged = merge(w, w2)
        @test length(merged) == 7
        @test merged.n_blessed == 5  # only first world was blessed
    end

    @testset "~10 random ape-in passports" begin
        # Simulate the ~10 random people who helped test the core flow
        rng = Random.MersenneTwister(1069)
        ape_seeds = [UInt64(rand(rng, 1:typemax(UInt32))) for _ in 1:10]
        registry = Gay.premine_passports(ape_seeds;
            bless_with=(UInt64(1069), "irl-ape-in"))

        @test length(registry) == 10
        # All verified
        for (_, p) in registry.passports
            @test Gay.verify_passport(p)
            @test p.blessed
        end
    end

    @testset "Display" begin
        p = Gay.issue_passport(UInt64(1069))
        s = sprint(show, p)
        @test contains(s, "did:gay:")
        @test contains(s, "unblessed")

        Gay.witness_passport(p, UInt64(42), "test")
        s2 = sprint(show, p)
        @test contains(s2, "BLESSED")
    end

    @testset "Verification levels (WorldID lineage)" begin
        # WorldID has Device / Orb levels
        # did:gay:* has UNVERIFIED / DEVICE_VERIFIED / WITNESS_VERIFIED / MULTI_WITNESS
        p = Gay.issue_passport(UInt64(1069))

        # Fresh passport with valid colors = DEVICE_VERIFIED
        @test Gay.verification_level(p) == Gay.DEVICE_VERIFIED

        # One witness = WITNESS_VERIFIED (like WorldID Orb, but no hardware)
        Gay.witness_passport(p, UInt64(42), "witness-1")
        @test Gay.verification_level(p) == Gay.WITNESS_VERIFIED

        # Two witnesses still WITNESS_VERIFIED
        Gay.witness_passport(p, UInt64(420), "witness-2")
        @test Gay.verification_level(p) == Gay.WITNESS_VERIFIED

        # Three witnesses = MULTI_WITNESS (strong web of trust)
        Gay.witness_passport(p, UInt64(69), "witness-3")
        @test Gay.verification_level(p) == Gay.MULTI_WITNESS
    end

    @testset "Trust score (passport.xyz lineage)" begin
        # passport.xyz uses stamp-weighted scoring (threshold ~20 points)
        # did:gay:* uses witness count + stamps → [0, 1] score
        p = Gay.issue_passport(UInt64(1069))

        # Base score: valid genesis colors = 0.2
        @test Gay.trust_score(p) ≈ 0.2

        # One witness: 0.2 + 0.2 = 0.4 (≈ passport.xyz threshold)
        Gay.witness_passport(p, UInt64(42), "witness-1")
        @test Gay.trust_score(p) ≈ 0.4

        # Three witnesses: 0.2 + 0.6 = 0.8
        Gay.witness_passport(p, UInt64(420), "witness-2")
        Gay.witness_passport(p, UInt64(69), "witness-3")
        @test Gay.trust_score(p) ≈ 0.8

        # Add stamps for bonus
        Gay.add_stamp(p, "vivid-1", "Vivarium")
        Gay.add_stamp(p, "vivid-2", "Minecraft BCI")
        @test Gay.trust_score(p) ≈ 0.84
    end

    @testset "is_human (the WorldID question)" begin
        # WorldID: iris scan → human
        # passport.xyz: score ≥ 20 → human
        # did:gay:*: blessed (≥1 IRL witness) → human
        p = Gay.issue_passport(UInt64(1069))
        @test !Gay.is_human(p)

        Gay.witness_passport(p, UInt64(42), "human-recognizer")
        @test Gay.is_human(p)
    end
end
