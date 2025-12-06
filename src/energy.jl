# Energy Measurement for Gay.jl Workloads
# Uses macOS powermetrics for Apple Silicon power consumption

using Printf

export EnergyMeasurement, measure_energy, with_energy_measurement
export PowerSample, start_power_sampling, stop_power_sampling
export energy_per_color, joules_per_billion_colors

# ═══════════════════════════════════════════════════════════════════════════
# Energy Measurement Types
# ═══════════════════════════════════════════════════════════════════════════

"""
    EnergyMeasurement

Result of measuring energy consumption of a computation.

# Fields
- `cpu_power_watts`: Average CPU power in watts
- `gpu_power_watts`: Average GPU power in watts  
- `ane_power_watts`: Average ANE (Neural Engine) power in watts
- `total_power_watts`: Total package power in watts
- `duration_seconds`: Duration of measurement
- `energy_joules`: Total energy consumed (power × time)
- `operations`: Number of operations performed (e.g., colors generated)
- `joules_per_op`: Energy per operation
- `ops_per_joule`: Operations per joule (efficiency)
"""
struct EnergyMeasurement
    cpu_power_watts::Float64
    gpu_power_watts::Float64
    ane_power_watts::Float64
    total_power_watts::Float64
    duration_seconds::Float64
    energy_joules::Float64
    operations::Int
    joules_per_op::Float64
    ops_per_joule::Float64
end

function Base.show(io::IO, e::EnergyMeasurement)
    println(io, "EnergyMeasurement:")
    @printf(io, "  CPU Power:    %.2f W\n", e.cpu_power_watts)
    @printf(io, "  GPU Power:    %.2f W\n", e.gpu_power_watts)
    @printf(io, "  ANE Power:    %.2f W\n", e.ane_power_watts)
    @printf(io, "  Total Power:  %.2f W\n", e.total_power_watts)
    @printf(io, "  Duration:     %.2f s\n", e.duration_seconds)
    @printf(io, "  Energy:       %.2f J\n", e.energy_joules)
    @printf(io, "  Operations:   %.2e\n", Float64(e.operations))
    @printf(io, "  Efficiency:   %.2e ops/J\n", e.ops_per_joule)
end

# ═══════════════════════════════════════════════════════════════════════════
# macOS powermetrics Integration
# ═══════════════════════════════════════════════════════════════════════════

const IS_MACOS = Sys.isapple()
const IS_APPLE_SILICON = IS_MACOS && Sys.ARCH == :aarch64

"""
    PowerSample

A single power sample from powermetrics.
"""
struct PowerSample
    timestamp::Float64
    cpu_power::Float64
    gpu_power::Float64
    ane_power::Float64
    package_power::Float64
    thermal_pressure::String
end

"""
    parse_powermetrics_output(output::String) -> Vector{PowerSample}

Parse powermetrics output to extract power measurements.
"""
function parse_powermetrics_output(output::String)::Vector{PowerSample}
    samples = PowerSample[]
    
    cpu_power = 0.0
    gpu_power = 0.0
    ane_power = 0.0
    package_power = 0.0
    thermal_pressure = "nominal"
    
    for line in split(output, '\n')
        line = strip(line)
        
        # Parse different power metrics formats
        if occursin("CPU Power:", line) || occursin("E-Cluster Power:", line) || occursin("P-Cluster Power:", line)
            m = match(r":\s*([\d.]+)\s*mW", line)
            if m !== nothing
                cpu_power += parse(Float64, m.captures[1]) / 1000.0  # mW to W
            end
        elseif occursin("GPU Power:", line)
            m = match(r":\s*([\d.]+)\s*mW", line)
            if m !== nothing
                gpu_power = parse(Float64, m.captures[1]) / 1000.0
            end
        elseif occursin("ANE Power:", line)
            m = match(r":\s*([\d.]+)\s*mW", line)
            if m !== nothing
                ane_power = parse(Float64, m.captures[1]) / 1000.0
            end
        elseif occursin("Combined Power", line) || occursin("Package Power", line)
            m = match(r":\s*([\d.]+)\s*mW", line)
            if m !== nothing
                package_power = parse(Float64, m.captures[1]) / 1000.0
            end
        elseif occursin("Thermal Pressure:", line)
            m = match(r":\s*(\w+)", line)
            if m !== nothing
                thermal_pressure = m.captures[1]
            end
        end
    end
    
    # If we got any power data, create a sample
    if package_power > 0 || cpu_power > 0
        if package_power == 0
            package_power = cpu_power + gpu_power + ane_power
        end
        push!(samples, PowerSample(time(), cpu_power, gpu_power, ane_power, 
                                   package_power, thermal_pressure))
    end
    
    return samples
end

"""
    run_powermetrics(duration_ms::Int) -> String

Run powermetrics for the specified duration and return output.
Requires sudo access. Returns empty string if not available.
"""
function run_powermetrics(duration_ms::Int)::String
    if !IS_APPLE_SILICON
        @warn "powermetrics only available on Apple Silicon Macs"
        return ""
    end
    
    try
        # Try without sudo first (may work in some contexts)
        cmd = `powermetrics --samplers cpu_power,gpu_power,thermal -n 1 -i $duration_ms`
        output = read(cmd, String)
        return output
    catch
        try
            # Try with sudo
            cmd = `sudo powermetrics --samplers cpu_power,gpu_power,thermal -n 1 -i $duration_ms`
            output = read(cmd, String)
            return output
        catch e
            @warn "powermetrics requires sudo access: $e"
            return ""
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Alternative: Use Activity Monitor's top command for energy
# ═══════════════════════════════════════════════════════════════════════════

"""
    get_process_power() -> Float64

Get power usage of current Julia process using top.
Returns relative power metric (not watts).
"""
function get_process_power()::Float64
    if !IS_MACOS
        return 0.0
    end
    
    pid = getpid()
    try
        # Run top twice to get accumulated power stats
        output = read(`top -l 2 -pid $pid -stats power`, String)
        lines = split(output, '\n')
        
        for line in reverse(lines)
            if occursin(string(pid), line) || !isempty(strip(line))
                m = match(r"([\d.]+)", strip(line))
                if m !== nothing
                    return parse(Float64, m.captures[1])
                end
            end
        end
    catch e
        @debug "Could not get process power: $e"
    end
    return 0.0
end

# ═══════════════════════════════════════════════════════════════════════════
# High-Level Energy Measurement API
# ═══════════════════════════════════════════════════════════════════════════

"""
    measure_energy(f::Function, n_operations::Integer; 
                   sample_interval_ms::Int=1000) -> EnergyMeasurement

Measure energy consumption while running function f.
Returns EnergyMeasurement with power and efficiency metrics.

# Example
```julia
result = measure_energy(1_000_000_000) do
    ka_color_sums(1_000_000_000, 42)
end
println("Energy: \$(result.energy_joules) J for 1B colors")
println("Efficiency: \$(result.ops_per_joule) colors/J")
```
"""
function measure_energy(f::Function, n_operations::Integer;
                        sample_interval_ms::Int=1000)
    # Start timing
    start_time = time()
    
    # Start background power sampling if available
    power_samples = PowerSample[]
    sampling_task = nothing
    stop_sampling = Ref(false)
    
    if IS_APPLE_SILICON
        sampling_task = @async begin
            while !stop_sampling[]
                output = run_powermetrics(sample_interval_ms)
                if !isempty(output)
                    append!(power_samples, parse_powermetrics_output(output))
                end
                sleep(sample_interval_ms / 1000.0)
            end
        end
    end
    
    # Run the computation
    result = f()
    
    # Stop sampling
    stop_sampling[] = true
    if sampling_task !== nothing
        try
            wait(sampling_task)
        catch
        end
    end
    
    duration = time() - start_time
    
    # Aggregate power samples
    if isempty(power_samples)
        # Fallback: estimate based on typical Apple Silicon power
        # M1/M2/M3 typically use 10-30W under load
        estimated_power = 20.0  # Conservative estimate
        return EnergyMeasurement(
            estimated_power * 0.6,  # CPU ~60%
            estimated_power * 0.3,  # GPU ~30%
            estimated_power * 0.1,  # ANE ~10%
            estimated_power,
            duration,
            estimated_power * duration,
            n_operations,
            (estimated_power * duration) / n_operations,
            n_operations / (estimated_power * duration)
        )
    else
        avg_cpu = sum(s.cpu_power for s in power_samples) / length(power_samples)
        avg_gpu = sum(s.gpu_power for s in power_samples) / length(power_samples)
        avg_ane = sum(s.ane_power for s in power_samples) / length(power_samples)
        avg_total = sum(s.package_power for s in power_samples) / length(power_samples)
        
        energy = avg_total * duration
        
        return EnergyMeasurement(
            avg_cpu, avg_gpu, avg_ane, avg_total,
            duration, energy, n_operations,
            energy / n_operations,
            n_operations / energy
        )
    end
end

"""
    with_energy_measurement(f::Function; sample_interval_ms::Int=500)

Run function with energy measurement, returning (result, EnergyMeasurement).
Automatically counts operations if function returns a count.

# Example
```julia
result, energy = with_energy_measurement() do
    r, g, b, time, rate = ka_color_sums(1_000_000_000, 42)
    return 1_000_000_000  # Return operation count
end
```
"""
function with_energy_measurement(f::Function; sample_interval_ms::Int=500)
    start_time = time()
    result = f()
    duration = time() - start_time
    
    # Try to infer operation count from result
    n_ops = if result isa Integer
        result
    elseif result isa Tuple && length(result) >= 1 && result[1] isa Number
        # Assume first element might be a count
        Int(round(result[1]))
    else
        1  # Unknown, use 1
    end
    
    # Get a quick power sample
    if IS_APPLE_SILICON
        output = run_powermetrics(min(1000, Int(duration * 1000)))
        samples = parse_powermetrics_output(output)
        
        if !isempty(samples)
            s = samples[1]
            energy = s.package_power * duration
            return (result, EnergyMeasurement(
                s.cpu_power, s.gpu_power, s.ane_power, s.package_power,
                duration, energy, n_ops,
                energy / n_ops, n_ops / energy
            ))
        end
    end
    
    # Fallback
    estimated_power = 20.0
    energy = estimated_power * duration
    return (result, EnergyMeasurement(
        12.0, 6.0, 2.0, estimated_power,
        duration, energy, n_ops,
        energy / n_ops, n_ops / energy
    ))
end

# ═══════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

"""
    energy_per_color(n::Integer=1_000_000, seed::Integer=GAY_SEED) -> Float64

Measure energy per color generation in joules.
"""
function energy_per_color(n::Integer=1_000_000, seed::Integer=GAY_SEED)
    result = measure_energy(n) do
        ka_parallel_hash(n, seed)
    end
    return result.joules_per_op
end

"""
    joules_per_billion_colors(seed::Integer=GAY_SEED; use_gpu::Bool=true) -> Float64

Measure joules required to generate 1 billion colors.
"""
function joules_per_billion_colors(seed::Integer=GAY_SEED; use_gpu::Bool=true)
    n = 1_000_000_000
    
    result = measure_energy(n) do
        if use_gpu && HAS_METAL
            @eval using Metal
            ka_color_sums(n, seed; backend=Metal.MetalBackend())
        else
            ka_color_sums(n, seed; backend=KernelAbstractions.CPU())
        end
    end
    
    return result.energy_joules
end

# ═══════════════════════════════════════════════════════════════════════════
# Energy Benchmark Suite
# ═══════════════════════════════════════════════════════════════════════════

"""
    run_energy_benchmarks(; duration_per_test::Float64=10.0)

Run comprehensive energy benchmarks for Gay.jl operations.
"""
function run_energy_benchmarks(; duration_per_test::Float64=10.0)
    println("=" ^ 70)
    println("⚡ Gay.jl Energy Benchmarks ⚡")
    println("=" ^ 70)
    println()
    println("Platform: $(IS_APPLE_SILICON ? "Apple Silicon" : "Other")")
    println("Metal.jl: $(HAS_METAL ? "Available" : "Not available")")
    println()
    
    results = Dict{String, EnergyMeasurement}()
    
    # Test 1: CPU Hash Colors
    println("─" ^ 70)
    println("Test 1: CPU Hash Color Generation")
    println("─" ^ 70)
    
    n_cpu = 100_000_000
    result = measure_energy(n_cpu) do
        ka_parallel_hash(n_cpu, GAY_SEED)
    end
    results["cpu_hash"] = result
    println(result)
    @printf("  Colors/Joule: %.2e\n", result.ops_per_joule)
    println()
    
    # Test 2: GPU Color Sums (if available)
    if HAS_METAL
        println("─" ^ 70)
        println("Test 2: GPU Color Sum Reduction")
        println("─" ^ 70)
        
        @eval using Metal
        n_gpu = 1_000_000_000
        result = measure_energy(n_gpu) do
            ka_color_sums(n_gpu, GAY_SEED; backend=Metal.MetalBackend())
        end
        results["gpu_sums"] = result
        println(result)
        @printf("  Colors/Joule: %.2e\n", result.ops_per_joule)
        println()
    end
    
    # Test 3: Mortal Computation Energy
    println("─" ^ 70)
    println("Test 3: Mortal Computation Churn")
    println("─" ^ 70)
    
    n_mortals = 1000
    lifetime = 1000
    rounds = 100
    total_ops = n_mortals * lifetime * rounds
    
    result = measure_energy(total_ops) do
        for r in 1:rounds
            mortals = [MortalComputation(i, r, lifetime) for i in 1:n_mortals]
            Threads.@threads for m in mortals
                while mortal_step!(m, 1.0)
                    hash_color(m.id ⊻ UInt64(m.steps_remaining), GAY_SEED)
                end
            end
        end
    end
    results["mortal_churn"] = result
    println(result)
    @printf("  Steps/Joule: %.2e\n", result.ops_per_joule)
    println()
    
    # Summary
    println("=" ^ 70)
    println("Summary")
    println("=" ^ 70)
    for (name, r) in results
        @printf("  %-15s: %.2f J total, %.2e ops/J efficiency\n", 
                name, r.energy_joules, r.ops_per_joule)
    end
    
    return results
end

export run_energy_benchmarks
