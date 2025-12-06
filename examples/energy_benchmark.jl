#!/usr/bin/env julia
#=
⚡ Gay.jl Energy Benchmark ⚡

Measures actual power consumption of color generation workloads
using macOS powermetrics on Apple Silicon.

Run with: sudo julia -t auto examples/energy_benchmark.jl

Note: sudo is required for powermetrics access to read power sensors.
Without sudo, estimates are used based on typical Apple Silicon power draw.
=#

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Gay
using Printf

println("=" ^ 70)
println("⚡ Gay.jl Energy Benchmark - Measure Your Colors' Carbon Footprint ⚡")
println("=" ^ 70)
println()

# Check platform
println("Platform Detection:")
println("  macOS: $(Sys.isapple())")
println("  Apple Silicon: $(Sys.isapple() && Sys.ARCH == :aarch64)")
println("  Metal.jl: $(Gay.HAS_METAL)")
println("  Julia threads: $(Threads.nthreads())")
println()

# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Baseline - Small Workload
# ═══════════════════════════════════════════════════════════════════════════

println("─" ^ 70)
println("Test 1: Baseline - 1M Colors (CPU)")
println("─" ^ 70)

n_small = 1_000_000
energy = Gay.measure_energy(n_small) do
    Gay.ka_parallel_hash(n_small, Gay.GAY_SEED)
end
println(energy)
println()

# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Medium Workload  
# ═══════════════════════════════════════════════════════════════════════════

println("─" ^ 70)
println("Test 2: Medium - 100M Colors (CPU)")
println("─" ^ 70)

n_medium = 100_000_000
energy = Gay.measure_energy(n_medium) do
    Gay.ka_parallel_hash(n_medium, Gay.GAY_SEED)
end
println(energy)
@printf("  Nanojoules per color: %.2f nJ\n", energy.joules_per_op * 1e9)
println()

# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Billion-Scale Workload
# ═══════════════════════════════════════════════════════════════════════════

println("─" ^ 70)
println("Test 3: Large - 1B Colors ($(Gay.HAS_METAL ? "GPU" : "CPU"))")
println("─" ^ 70)

n_large = 1_000_000_000

if Gay.HAS_METAL
    @eval using Metal
    backend = Metal.MetalBackend()
else
    backend = KernelAbstractions.CPU()
end

energy = Gay.measure_energy(n_large) do
    Gay.ka_color_sums(n_large, Gay.GAY_SEED; backend=backend)
end
println(energy)
@printf("  Nanojoules per color: %.2f nJ\n", energy.joules_per_op * 1e9)
@printf("  Colors per Joule: %.2e\n", energy.ops_per_joule)
println()

# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Sustained Load - 30 Seconds
# ═══════════════════════════════════════════════════════════════════════════

println("─" ^ 70)
println("Test 4: Sustained 30-Second Load")
println("─" ^ 70)

total_colors = 0
start_time = time()
target_duration = 30.0

energy = Gay.measure_energy(0) do
    local count = 0
    while (time() - start_time) < target_duration
        Gay.ka_parallel_hash(10_000_000, Gay.GAY_SEED)
        count += 10_000_000
    end
    global total_colors = count
    return count
end

# Recalculate with actual count
actual_energy = Gay.EnergyMeasurement(
    energy.cpu_power_watts,
    energy.gpu_power_watts,
    energy.ane_power_watts,
    energy.total_power_watts,
    energy.duration_seconds,
    energy.total_power_watts * energy.duration_seconds,
    total_colors,
    (energy.total_power_watts * energy.duration_seconds) / total_colors,
    total_colors / (energy.total_power_watts * energy.duration_seconds)
)

println(actual_energy)
@printf("  Total colors generated: %.2e\n", Float64(total_colors))
@printf("  Sustained power: %.2f W\n", actual_energy.total_power_watts)
println()

# ═══════════════════════════════════════════════════════════════════════════
# Test 5: GPU vs CPU Efficiency Comparison
# ═══════════════════════════════════════════════════════════════════════════

if Gay.HAS_METAL
    println("─" ^ 70)
    println("Test 5: GPU vs CPU Efficiency Comparison")
    println("─" ^ 70)
    
    n_compare = 100_000_000
    
    # CPU
    cpu_energy = Gay.measure_energy(n_compare) do
        Gay.ka_color_sums(n_compare, Gay.GAY_SEED; backend=KernelAbstractions.CPU())
    end
    
    # GPU
    gpu_energy = Gay.measure_energy(n_compare) do
        Gay.ka_color_sums(n_compare, Gay.GAY_SEED; backend=Metal.MetalBackend())
    end
    
    println("CPU:")
    @printf("  Energy: %.2f J, Efficiency: %.2e colors/J\n", 
            cpu_energy.energy_joules, cpu_energy.ops_per_joule)
    
    println("GPU:")
    @printf("  Energy: %.2f J, Efficiency: %.2e colors/J\n",
            gpu_energy.energy_joules, gpu_energy.ops_per_joule)
    
    efficiency_ratio = gpu_energy.ops_per_joule / cpu_energy.ops_per_joule
    @printf("\nGPU is %.1fx more energy efficient than CPU\n", efficiency_ratio)
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Mortal/Immortal Computation Energy
# ═══════════════════════════════════════════════════════════════════════════

println("─" ^ 70)
println("Test 6: Mortal Computation Energy")
println("─" ^ 70)

n_mortals = 1000
lifetime = 500
rounds = 50
total_ops = n_mortals * lifetime * rounds

energy = Gay.measure_energy(total_ops) do
    for r in 1:rounds
        mortals = [Gay.MortalComputation(i, r, lifetime) for i in 1:n_mortals]
        Threads.@threads for m in mortals
            while Gay.mortal_step!(m, 1.0)
                Gay.hash_color(m.id ⊻ UInt64(m.steps_remaining), Gay.GAY_SEED)
            end
        end
    end
end

println(energy)
@printf("  Steps per Joule: %.2e\n", energy.ops_per_joule)
println()

# ═══════════════════════════════════════════════════════════════════════════
# Summary: Environmental Impact
# ═══════════════════════════════════════════════════════════════════════════

println("=" ^ 70)
println("Environmental Impact Summary")
println("=" ^ 70)

# Calculate CO2 equivalent
# Average grid: ~400g CO2 per kWh
# 1 kWh = 3,600,000 J
CO2_PER_JOULE = 400.0 / 3_600_000  # grams CO2 per joule

billion_colors_joules = energy.joules_per_op * 1e9
billion_colors_co2 = billion_colors_joules * CO2_PER_JOULE

println()
@printf("Energy per billion colors: %.2f J\n", billion_colors_joules)
@printf("CO₂ per billion colors:    %.4f g\n", billion_colors_co2)
println()

# Fun comparisons
println("Fun comparisons (1 billion colors ≈):")
@printf("  %.2e LED blinks (1mJ each)\n", billion_colors_joules / 0.001)
@printf("  %.4f seconds of 60W lightbulb\n", billion_colors_joules / 60.0)
@printf("  %.6f smartphone charges (40kJ)\n", billion_colors_joules / 40000.0)
println()

# Throughput at sustainable power levels
sustainable_watts = 15.0  # Reasonable for laptop without fans
colors_per_second_at_sustainable = sustainable_watts / energy.joules_per_op
@printf("At %dW sustainable power: %.2e colors/sec\n", 
        Int(sustainable_watts), colors_per_second_at_sustainable)

println("=" ^ 70)
println("⚡ Energy benchmark complete!")
println("=" ^ 70)
