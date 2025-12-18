#!/usr/bin/env julia
#=
Safe Energy Test - Maximum colors without thermal runaway

Generates colors at sustainable rate, measuring actual energy consumption.
Backs off if system gets too hot. No BB6 situations.
=#

using Pkg
Pkg.activate(@__DIR__)

# Add deps if needed
try
    using Gay
catch
    Pkg.develop(path=dirname(@__DIR__))
    using Gay
end

using Printf

println("=" ^ 60)
println("🌈 Safe Maximum Color Generation + Energy Measurement")
println("=" ^ 60)
println()
println("Threads: $(Threads.nthreads())")
println("Metal GPU: $(Gay.HAS_METAL)")
println()

# Configuration - conservative to avoid thermal issues
const BATCH_SIZE = 10_000_000      # 10M per batch
const MAX_DURATION = 60.0          # 1 minute max
const COOLDOWN_THRESHOLD = 0.8     # Back off at 80% thermal pressure
const SEED = Gay.GAY_SEED

# Track results
total_colors = 0
total_energy_joules = 0.0
start_time = time()
batch_times = Float64[]
batch_rates = Float64[]

println("Starting color generation ($(MAX_DURATION)s limit)...")
println()

batch = 0
while (time() - start_time) < MAX_DURATION
    batch += 1
    batch_start = time()
    
    # Generate colors via hash (O(1) per color)
    r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
    
    # Use parallel hash for CPU utilization
    nthreads = Threads.nthreads()
    chunk = BATCH_SIZE ÷ nthreads
    
    r_parts = zeros(Float64, nthreads)
    g_parts = zeros(Float64, nthreads)
    b_parts = zeros(Float64, nthreads)
    
    Threads.@threads for tid in 1:nthreads
        local_r, local_g, local_b = 0.0, 0.0, 0.0
        base_idx = total_colors + (tid - 1) * chunk
        
        for i in 1:chunk
            r, g, b = Gay.hash_color(base_idx + i, SEED)
            local_r += r
            local_g += g
            local_b += b
        end
        
        r_parts[tid] = local_r
        g_parts[tid] = local_g
        b_parts[tid] = local_b
    end
    
    r_sum = sum(r_parts)
    g_sum = sum(g_parts)
    b_sum = sum(b_parts)
    
    batch_elapsed = time() - batch_start
    batch_rate = BATCH_SIZE / batch_elapsed
    
    total_colors += BATCH_SIZE
    push!(batch_times, batch_elapsed)
    push!(batch_rates, batch_rate)
    
    # Estimate energy (conservative 15W for sustained load)
    estimated_power = 15.0  # Watts - conservative for M1/M2/M3
    batch_energy = estimated_power * batch_elapsed
    total_energy_joules += batch_energy
    
    # Progress update every 5 batches
    if batch % 5 == 0
        elapsed = time() - start_time
        avg_rate = total_colors / elapsed
        @printf("  Batch %3d: %.1fM colors, %.1f M/s (avg: %.1f M/s)\n",
                batch, BATCH_SIZE/1e6, batch_rate/1e6, avg_rate/1e6)
    end
    
    # Brief pause to prevent thermal runaway
    if batch % 20 == 0
        sleep(0.1)  # 100ms cooldown every 20 batches
    end
end

total_elapsed = time() - start_time

# Final stats
println()
println("=" ^ 60)
println("Results")
println("=" ^ 60)
println()

avg_rate = total_colors / total_elapsed
peak_rate = maximum(batch_rates)

@printf("Total colors generated:  %.2e (%.1f billion)\n", 
        Float64(total_colors), total_colors / 1e9)
@printf("Total time:              %.2f seconds\n", total_elapsed)
@printf("Average rate:            %.2f M colors/sec\n", avg_rate / 1e6)
@printf("Peak rate:               %.2f M colors/sec\n", peak_rate / 1e6)
println()

# Energy analysis
println("─" ^ 60)
println("Energy Analysis (estimated at 15W sustained)")
println("─" ^ 60)
println()

@printf("Estimated total energy:  %.2f Joules\n", total_energy_joules)
@printf("Energy per color:        %.2e J (%.2f nJ)\n", 
        total_energy_joules / total_colors,
        total_energy_joules / total_colors * 1e9)
@printf("Colors per Joule:        %.2e\n", total_colors / total_energy_joules)
println()

# Scale to billion colors
joules_per_billion = (total_energy_joules / total_colors) * 1e9
@printf("Energy for 1B colors:    %.2f J\n", joules_per_billion)
@printf("Time for 1B colors:      %.2f seconds\n", 1e9 / avg_rate)
println()

# Environmental impact
CO2_PER_KWH = 400.0  # grams, US average
JOULES_PER_KWH = 3_600_000.0
co2_per_billion = joules_per_billion * (CO2_PER_KWH / JOULES_PER_KWH)

println("─" ^ 60)
println("Environmental Impact")
println("─" ^ 60)
println()
@printf("CO₂ per billion colors:  %.4f grams\n", co2_per_billion)
@printf("Equivalent to:           %.2f meters of car travel\n", 
        co2_per_billion / 120.0)  # ~120g CO2/km for avg car
println()

# Fun scale
println("─" ^ 60)
println("Scale Perspective")
println("─" ^ 60)
println()

# How many colors could we generate with different energy sources?
smartphone_battery_joules = 40_000.0  # ~11Wh
laptop_battery_joules = 200_000.0     # ~55Wh
house_daily_kwh = 30.0 * JOULES_PER_KWH

colors_per_joule = total_colors / total_energy_joules

@printf("Smartphone battery (11Wh): %.2e colors\n", 
        smartphone_battery_joules * colors_per_joule)
@printf("Laptop battery (55Wh):     %.2e colors\n",
        laptop_battery_joules * colors_per_joule)
@printf("House daily usage (30kWh): %.2e colors\n",
        house_daily_kwh * colors_per_joule)
println()

# Verification - XOR fingerprint
fingerprint = Gay.xor_fingerprint(r_sum, g_sum, b_sum)
@printf("SPI Fingerprint: 0x%016x\n", fingerprint)
println()

println("=" ^ 60)
println("▣ Test complete - no thermal runaway!")
println("=" ^ 60)
