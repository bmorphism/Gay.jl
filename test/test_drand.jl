using Test
using Gay
using Gay.Drand
using SHA

@testset "Gay.jl Seeding Strategies" begin
    # 1. Known (Deterministic) - Congruence Check
    # Ensure strings use the stable hash/FNV logic, not SHA
    s1 = "test_phrase"
    gay_seed!(s1)
    c1 = next_color()
    
    gay_seed!(s1)
    c2 = next_color()
    @test c1 == c2
    
    # 2. Random, Known (Replay)
    # Mock a DrandRound
    mock_randomness = collect(UInt8(1):UInt8(32))
    mock_round = DrandRound(123, mock_randomness, UInt8[], nothing, mainnet())
    
    # gay_seed!(:record, round_id) - Network mock needed, skip for unit test
    # But we can test gay_seed!(round) directly
    gay_seed!(mock_round)
    c3 = next_color()
    
    gay_seed!(mock_round)
    c4 = next_color()
    @test c3 == c4
    
    # 3. Content-Derived (SHA-256)
    data = Vector{UInt8}("test_content")
    gay_seed!(data)
    c5 = next_color()
    
    # Verify it's different from the string "test_content" (Strategy 1)
    gay_seed!("test_content")
    c6 = next_color()
    @test c5 != c6  # SHA-256 vs FNV/Hash should differ
    
    # Verify stability of content seeding
    gay_seed!(Vector{UInt8}("test_content"))
    c7 = next_color()
    @test c5 == c7
end

@testset "Drand Integration" begin
    # Test utility functions
    t = round_at_time(1692803367, quicknet())
    @test t == 1
    
    # Verify SHA logic in to_seed
    bytes = zeros(UInt8, 32)
    bytes[1] = 0x01
    # 0x01000000... -> UInt64(1) if little endian? 
    # reinterpret depends on endianness, but consistency is what matters
    dr = DrandRound(1, bytes, UInt8[], nothing, mainnet())
    s = to_seed(dr)
    @test isa(s, UInt64)
end
