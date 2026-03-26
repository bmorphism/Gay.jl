using Test

@testset "Open Passport Game — Ghani-Hedges on did:gay:*" begin

    # Helper: create a blessed passport
    function blessed_passport(seed::UInt64, witness_seed::UInt64=UInt64(1069))
        p = Gay.issue_passport(seed)
        Gay.witness_passport(p, witness_seed, "vivarium-witness")
        p
    end

    @testset "Bounty creation" begin
        alice = blessed_passport(UInt64(1069))
        b = Gay.OpenPassportGame.Bounty("Build BCI integration", 100.0, alice;
            tags=[:bci, :minecraft], location="Vivarium")
        @test b.reward == 100.0
        @test b.description == "Build BCI integration"
        @test :bci in b.tags
        @test b.seed != UInt64(0)
    end

    @testset "Bounty board posting" begin
        alice = blessed_passport(UInt64(1069))
        board = Gay.BountyBoard()
        b = Gay.OpenPassportGame.Bounty("Test color pipeline", 50.0, alice; tags=[:testing])
        Gay.post_bounty!(board, b)
        @test length(board.bounties) == 1
        @test length(board.players) == 1
    end

    @testset "Play: commit to bounty" begin
        alice = blessed_passport(UInt64(1069))
        bob = blessed_passport(UInt64(420), UInt64(1069))
        board = Gay.BountyBoard()
        b = Gay.OpenPassportGame.Bounty("Build thing", 100.0, alice; tags=[:build])
        Gay.post_bounty!(board, b)

        # Bob commits (Play forward pass)
        play = Gay.commit!(board, bob, b.id)
        @test play.commitment.spin == Int8(1)
        @test play.commitment.stake > 0.0
        @test length(board.commitments) == 1
    end

    @testset "CoPlay: verify and pay" begin
        alice = blessed_passport(UInt64(1069))
        bob = blessed_passport(UInt64(420), UInt64(1069))
        board = Gay.BountyBoard()
        b = Gay.OpenPassportGame.Bounty("Build thing", 100.0, alice; tags=[:build])
        Gay.post_bounty!(board, b)

        play = Gay.commit!(board, bob, b.id)
        # Alice verifies Bob's work (CoPlay backward pass)
        outcome = Gay.verify_and_pay!(board, alice, play)
        @test outcome.coplay.verified
        @test outcome.coplay.reward_released == 100.0
        @test outcome.net_utility > 0.0
        @test outcome.fingerprint != UInt64(0)
        @test length(board.completed) == 1
    end

    @testset "Cannot verify own work" begin
        alice = blessed_passport(UInt64(1069))
        board = Gay.BountyBoard()
        b = Gay.OpenPassportGame.Bounty("Self-deal", 100.0, alice; tags=[:nope])
        Gay.post_bounty!(board, b)
        play = Gay.commit!(board, alice, b.id)
        @test_throws ErrorException Gay.verify_and_pay!(board, alice, play)
    end

    @testset "Verification level gates" begin
        # Unverified passport cannot commit to WITNESS_VERIFIED bounty
        alice = blessed_passport(UInt64(1069))
        noob = Gay.issue_passport(UInt64(999))  # unblessed
        board = Gay.BountyBoard()
        b = Gay.OpenPassportGame.Bounty("Gated task", 100.0, alice;
            tags=[:gated], required_level=Gay.WITNESS_VERIFIED)
        Gay.post_bounty!(board, b)
        @test_throws ErrorException Gay.commit!(board, noob, b.id)
    end

    @testset "Magnetization — Ising order parameter" begin
        alice = blessed_passport(UInt64(1069))
        bob = blessed_passport(UInt64(420), UInt64(1069))
        carol = blessed_passport(UInt64(69), UInt64(1069))

        board = Gay.BountyBoard()
        b = Gay.OpenPassportGame.Bounty("Task", 50.0, alice; tags=[:test])
        Gay.post_bounty!(board, b)

        # Empty board: magnetization = 0
        @test Gay.board_magnetization(board) == 0.0

        # Bob commits (+1)
        Gay.commit!(board, bob, b.id; spin=Int8(1))
        @test Gay.board_magnetization(board) == 1.0  # all supply

        # Carol requests (-1)
        Gay.commit!(board, carol, b.id; spin=Int8(-1))
        @test Gay.board_magnetization(board) ≈ 0.0  # equilibrium!
        @test Gay.OpenPassportGame.is_equilibrium(board)
    end

    @testset "Board fingerprint (SPI)" begin
        alice = blessed_passport(UInt64(1069))
        board = Gay.BountyBoard()
        b = Gay.OpenPassportGame.Bounty("Task", 50.0, alice; tags=[:test])
        Gay.post_bounty!(board, b)
        fp = Gay.board_fingerprint(board)
        @test fp != UInt64(0)
    end

    @testset "World builder" begin
        alice = blessed_passport(UInt64(1069))
        bob = blessed_passport(UInt64(420), UInt64(1069))

        w = Gay.world_open_passport_game(
            passports=[alice, bob],
            bounties=[
                ("Build BCI integration", 100.0, alice, [:bci, :minecraft]),
                ("Test color pipeline", 50.0, bob, [:testing, :colors]),
            ]
        )
        @test length(w) == 2
        @test w.n_players == 2
        @test w.fingerprint_val != UInt64(0)

        # Merge two worlds
        carol = blessed_passport(UInt64(69), UInt64(1069))
        w2 = Gay.world_open_passport_game(
            passports=[carol],
            bounties=[("New task", 75.0, carol, [:new])]
        )
        merged = merge(w, w2)
        @test length(merged) == 3
        @test merged.n_players == 3
    end

    @testset "is_human required for verification" begin
        alice = blessed_passport(UInt64(1069))
        bob = blessed_passport(UInt64(420), UInt64(1069))
        unblessed = Gay.issue_passport(UInt64(777))  # not blessed

        board = Gay.BountyBoard()
        b = Gay.OpenPassportGame.Bounty("Task", 50.0, alice;
            tags=[:test], required_level=Gay.DEVICE_VERIFIED)
        Gay.post_bounty!(board, b)
        play = Gay.commit!(board, bob, b.id)

        # Unblessed passport cannot verify (not is_human)
        @test_throws ErrorException Gay.verify_and_pay!(board, unblessed, play)
    end
end
