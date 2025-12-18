module SeedMining

using Printf

export SeedQuality, spectral_test, gf3_balance, mine_seeds, generate_move_registration, world_seed_mining

struct SeedQuality
    seed::UInt64
    spectral_ratio::Float64
    gf3_balance::Float64
    is_valid::Bool
end

# SM64 PRNG - same constants as Nintendo 64
function sm64_next(state::UInt64)::UInt64
    (state * 0x5D588B656C078965 + 0x0000000000269EC3) & 0xFFFFFFFFFFFFFFFF
end

function generate_hues(seed::UInt64, n::Int)::Vector{Float64}
    hues = Vector{Float64}(undef, n)
    state = seed
    for i in 1:n
        state = sm64_next(state)
        hues[i] = (state >> 48) / 65536.0  # normalized [0,1)
    end
    hues
end

# Simple DFT when FFTW unavailable
function simple_dft(x::Vector{Float64})::Vector{ComplexF64}
    N = length(x)
    X = Vector{ComplexF64}(undef, N)
    for k in 0:N-1
        s = 0.0 + 0.0im
        for n in 0:N-1
            s += x[n+1] * exp(-2π * im * k * n / N)
        end
        X[k+1] = s
    end
    X
end

# Try FFTW, fall back to simple DFT
function compute_fft(x::Vector{Float64})::Vector{ComplexF64}
    try
        @eval using FFTW
        return FFTW.fft(x)
    catch
        return simple_dft(x)
    end
end

function spectral_test(seed::UInt64; n::Int=1024)::Float64
    hues = generate_hues(seed, n)
    
    # Center the sequence
    centered = hues .- mean(hues)
    
    # Compute FFT magnitude spectrum
    spectrum = abs.(compute_fft(centered))
    
    # Skip DC component, analyze rest
    magnitudes = spectrum[2:div(n,2)]
    
    peak = maximum(magnitudes)
    μ = sum(magnitudes) / length(magnitudes)
    
    μ > 0 ? peak / μ : Inf
end

function mean(x::Vector{Float64})::Float64
    sum(x) / length(x)
end

function gf3_balance(seed::UInt64; samples::Int=1000)::Float64
    counts = zeros(Int, 3)
    state = seed
    for _ in 1:samples
        state = sm64_next(state)
        trit = mod(state >> 32, 3)
        counts[trit + 1] += 1
    end
    # Perfect balance = 1.0, deviation reduces score
    expected = samples / 3
    deviation = sum(abs.(counts .- expected)) / samples
    1.0 - deviation
end

function mine_seeds(n::Int; threshold::Float64=8.0)::Vector{SeedQuality}
    results = Vector{SeedQuality}()
    
    for seed in UInt64(1):UInt64(n)
        ratio = spectral_test(seed; n=512)  # smaller n for speed
        balance = gf3_balance(seed)
        is_valid = ratio < threshold && balance > 0.9
        
        if ratio < threshold
            push!(results, SeedQuality(seed, ratio, balance, is_valid))
        end
    end
    
    # Sort by spectral ratio (lower = stronger descent)
    sort!(results, by = sq -> sq.spectral_ratio)
    results
end

function generate_move_registration(seeds::Vector{SeedQuality})::String
    lines = String[]
    push!(lines, "module gay::seed_registry {")
    push!(lines, "    use std::vector;")
    push!(lines, "")
    push!(lines, "    public entry fun register_validated_seeds(account: &signer) {")
    
    for sq in seeds
        if sq.is_valid
            push!(lines, "        register_seed(account, $(sq.seed), $(round(sq.spectral_ratio, digits=3)), $(round(sq.gf3_balance, digits=3)));")
        end
    end
    
    push!(lines, "    }")
    push!(lines, "")
    push!(lines, "    fun register_seed(account: &signer, seed: u64, ratio: u64, balance: u64) {")
    push!(lines, "        // Validate descent: ratio < 8000 (scaled by 1000)")
    push!(lines, "        assert!(ratio < 8000, 0x1);")
    push!(lines, "        // Store in registry")
    push!(lines, "    }")
    push!(lines, "}")
    
    join(lines, "\n")
end

function world_seed_mining()
    println("═══════════════════════════════════════════════════════")
    println("  GAY WORLD SEED MINING - Spectral Descent Validation")
    println("═══════════════════════════════════════════════════════")
    println()
    
    # Mine seeds
    println("Mining 10000 seeds with spectral threshold < 8.0...")
    seeds = mine_seeds(10000; threshold=8.0)
    println("Found $(length(seeds)) valid descent seeds")
    println()
    
    # Show top 10
    println("TOP 10 STRONGEST DESCENT SEEDS:")
    println("─────────────────────────────────────────────────────")
    println(" Rank │    Seed │  Spectral │ GF3 Balance │ Status")
    println("─────────────────────────────────────────────────────")
    
    for (i, sq) in enumerate(seeds[1:min(10, length(seeds))])
        marker = sq.seed == 5980 ? " ★★★" : ""
        status = sq.is_valid ? "VALID" : "weak"
        println(@sprintf(" %4d │ %7d │    %5.3f │      %5.3f │ %s%s", 
                        i, sq.seed, sq.spectral_ratio, sq.gf3_balance, status, marker))
    end
    println("─────────────────────────────────────────────────────")
    
    # Check for seed 5980
    idx_5980 = findfirst(sq -> sq.seed == 5980, seeds)
    if idx_5980 !== nothing
        sq = seeds[idx_5980]
        println()
        println("★ SEED 5980 (KNOWN STRONGEST): rank #$idx_5980")
        println("  Spectral ratio: $(round(sq.spectral_ratio, digits=3)) (target: 2.761)")
        println("  GF3 balance: $(round(sq.gf3_balance, digits=3))")
    else
        # Test 5980 directly
        ratio = spectral_test(UInt64(5980))
        balance = gf3_balance(UInt64(5980))
        println()
        println("★ SEED 5980 (KNOWN STRONGEST):")
        println("  Spectral ratio: $(round(ratio, digits=3)) (expected ~2.761)")
        println("  GF3 balance: $(round(balance, digits=3))")
    end
    
    # Generate Move code
    println()
    println("═══════════════════════════════════════════════════════")
    println("  MOVE CONTRACT REGISTRATION CODE")
    println("═══════════════════════════════════════════════════════")
    println()
    move_code = generate_move_registration(seeds[1:min(10, length(seeds))])
    println(move_code)
    
    seeds
end

end # module
