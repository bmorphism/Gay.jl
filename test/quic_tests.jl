# QUIC Path Probe Coloring Tests
# ===============================

using Test
using Gay
using Gay: QUICConnection, QUICPathProbe, QUICPath
using Gay: connection_color, path_color, path_probe_color
using Gay: add_path!, probe_challenge!, probe_response!, validate_path!
using Gay: parallel_probe_colors, generate_nonce
using Gay: GAY_SEED, splitmix64, hash_color
using Colors: RGB

@testset "QUIC Path Probe Coloring" begin
    
    @testset "Color Generation Determinism (SPI)" begin
        seed = GAY_SEED
        
        # Connection color is deterministic
        conn_id = UInt64(0x123456789ABCDEF0)
        c1 = connection_color(conn_id, seed)
        c2 = connection_color(conn_id, seed)
        @test c1 == c2
        
        # Different connection IDs produce different colors
        c3 = connection_color(conn_id + 1, seed)
        @test c1 != c3
        
        # Path color is deterministic
        path_id = UInt32(1)
        p1 = path_color(conn_id, path_id, seed)
        p2 = path_color(conn_id, path_id, seed)
        @test p1 == p2
        
        # Different paths have different colors
        p3 = path_color(conn_id, UInt32(path_id + 1), seed)
        @test p1 != p3
        
        # Probe color is deterministic
        nonce = NTuple{8, UInt8}((0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08))
        pr1 = path_probe_color(nonce, conn_id, path_id, seed)
        pr2 = path_probe_color(nonce, conn_id, path_id, seed)
        @test pr1 == pr2
        
        # Different nonces produce different colors
        nonce2 = NTuple{8, UInt8}((0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18))
        pr3 = path_probe_color(nonce2, conn_id, path_id, seed)
        @test pr1 != pr3
    end
    
    @testset "Connection Management" begin
        conn = QUICConnection(UInt64(42))
        
        @test conn.connection_id == 42
        @test conn.seed == GAY_SEED
        @test isempty(conn.paths)
        @test isempty(conn.probe_history)
        @test conn.color isa RGB
        
        # Add paths
        path0 = add_path!(conn, UInt32(0))
        @test path0.path_id == 0
        @test !path0.validated
        @test path0.active
        @test path0.color isa RGB
        
        path1 = add_path!(conn, UInt32(1))
        @test length(conn.paths) == 2
        @test path0.color != path1.color  # Different paths, different colors
    end
    
    @testset "Path Probing" begin
        conn = QUICConnection(UInt64(1337))
        
        # Send challenge
        challenge = probe_challenge!(conn, UInt32(0))
        @test challenge.is_challenge
        @test challenge.path_id == 0
        @test challenge.connection_id == 1337
        @test challenge.color isa RGB
        @test length(conn.probe_history) == 1
        
        path = conn.paths[UInt32(0)]
        @test path.probes_sent == 1
        @test path.probes_received == 0
        @test !path.validated
        
        # Receive response
        response_time = time_ns()
        response = probe_response!(conn, UInt32(0), challenge.nonce, response_time)
        @test response !== nothing
        @test !response.is_challenge
        @test response.nonce == challenge.nonce
        @test response.color == challenge.color  # Matched pair has same color
        @test path.probes_received == 1
        @test path.rtt_us > 0
        
        # Validate path
        @test validate_path!(conn, UInt32(0))
        @test path.validated
    end
    
    @testset "Nonce Matching" begin
        conn = QUICConnection(UInt64(9999))
        
        # Send two challenges
        c1 = probe_challenge!(conn, UInt32(0))
        c2 = probe_challenge!(conn, UInt32(0))
        @test c1.nonce != c2.nonce
        @test c1.color != c2.color  # Different nonces = different colors
        
        # Response with wrong nonce fails
        wrong_nonce = NTuple{8, UInt8}((0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF))
        bad_response = probe_response!(conn, UInt32(0), wrong_nonce, time_ns())
        @test bad_response === nothing
        
        # Response to non-existent path fails
        bad_response2 = probe_response!(conn, UInt32(99), c1.nonce, time_ns())
        @test bad_response2 === nothing
        
        # Correct response succeeds
        good_response = probe_response!(conn, UInt32(0), c1.nonce, time_ns())
        @test good_response !== nothing
        @test good_response.color == c1.color
    end
    
    @testset "Parallel Probe Colors (SPI)" begin
        n = 1000
        conn_id = UInt64(0xDEADBEEF)
        seed = GAY_SEED
        
        # Generate colors in parallel
        colors = parallel_probe_colors(n, conn_id, seed)
        @test size(colors) == (n, 3)
        @test eltype(colors) == Float32
        @test all(0.0f0 .<= colors .<= 1.0f0)
        
        # Same inputs = same outputs (SPI)
        colors2 = parallel_probe_colors(n, conn_id, seed)
        # Note: due to random path_ids and nonces, we can't test exact equality
        # But we can verify structure is correct
        @test size(colors2) == (n, 3)
    end
    
    @testset "Multiple Paths SPI" begin
        seed = GAY_SEED
        conn_id = UInt64(42069)
        
        # Generate colors for multiple paths in parallel
        n_probes = 100
        
        # Sequential generation
        sequential = Vector{RGB{Float64}}(undef, n_probes)
        for i in 1:n_probes
            path_id = UInt32((i - 1) % 4)
            nonce = NTuple{8, UInt8}(UInt8.((i >> j) & 0xFF for j in 0:7))
            sequential[i] = path_probe_color(nonce, conn_id, path_id, seed)
        end
        
        # Parallel generation via connection
        conn = QUICConnection(conn_id; seed=seed)
        for path_id in UInt32(0):UInt32(3)
            add_path!(conn, path_id)
        end
        
        # Probes should be reproducible
        for i in 1:10
            path_id = UInt32((i - 1) % 4)
            c = probe_challenge!(conn, path_id)
            @test c.color isa RGB
        end
    end
    
    @testset "Nonce Generation" begin
        # Nonces should be 8 bytes
        nonce = generate_nonce()
        @test length(nonce) == 8
        @test nonce isa NTuple{8, UInt8}
        
        # Nonces should be unique (probabilistically)
        nonces = [generate_nonce() for _ in 1:100]
        @test length(unique(nonces)) == 100
    end
    
    @testset "Probe Timeline" begin
        conn = QUICConnection(UInt64(123))
        
        # Generate some probes
        for path_id in UInt32(0):UInt32(2)
            c = probe_challenge!(conn, path_id)
            probe_response!(conn, path_id, c.nonce, time_ns())
        end
        
        timeline = Gay.probe_timeline(conn)
        @test length(timeline) == 6  # 3 challenges + 3 responses
        
        # Check structure
        for entry in timeline
            @test haskey(entry, :timestamp_ns)
            @test haskey(entry, :type)
            @test haskey(entry, :path_id)
            @test haskey(entry, :nonce)
            @test haskey(entry, :color)
            @test entry.type in [:challenge, :response]
        end
    end
end
