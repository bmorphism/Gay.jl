#!/usr/bin/env julia
# Spectral Test Investigation for Seed 42
# This script investigates the marginal spectral test failure

using Pkg
Pkg.activate(dirname(@__DIR__))

using Gay
using Statistics

function investigate_spectral()
    println("=== DETAILED SPECTRAL ANALYSIS FOR SEED 42 ===")
    println()

    # Detailed result for seed 42
    result = spectral_test(1024, 42)
    println("Test Parameters:")
    println("  n = 1024 samples")
    println("  seed = 42")
    println()
    println("Results:")
    println("  max_peak = ", result.max_peak)
    println("  mean_power = ", result.mean_power)
    println("  ratio = ", result.ratio)
    println("  peak_freq = ", result.peak_freq)
    println("  threshold = 12.0")
    println("  passed = ", result.passed)
    println("  margin = ", round((result.ratio - 12.0) / 12.0 * 100, digits=2), "% over threshold")
    println()

    # Reproducibility
    println("Reproducibility check (5 runs):")
    for i in 1:5
        r = spectral_test(1024, 42)
        println("  Run ", i, ": ratio=", round(r.ratio, digits=4), ", peak_freq=", round(r.peak_freq, digits=4))
    end
    println()

    # Statistical context
    println("Statistical context (seeds 1-100):")
    failures = 0
    ratios = Float64[]
    failing_seeds = Int[]

    for s in 1:100
        r = spectral_test(1024, s)
        push!(ratios, r.ratio)
        if !r.passed
            failures += 1
            push!(failing_seeds, s)
        end
    end

    println("  Failures: ", failures, "/100 (", failures, "%)")
    println("  Mean ratio: ", round(mean(ratios), digits=2))
    println("  Max ratio: ", round(maximum(ratios), digits=2))
    println("  Min ratio: ", round(minimum(ratios), digits=2))
    println("  Std dev: ", round(std(ratios), digits=2))
    println()

    if !isempty(failing_seeds)
        println("Seeds 1-100 that fail spectral test:")
        for s in failing_seeds
            r = spectral_test(1024, s)
            println("  Seed ", s, ": ratio=", round(r.ratio, digits=2), " (", round((r.ratio-12)/12*100, digits=1), "% over)")
        end
    else
        println("No seeds 1-100 fail the spectral test")
    end
    println()

    # Wider seed range
    println("Extended analysis (seeds 1-1000):")
    extended_failures = 0
    extended_ratios = Float64[]
    extended_failing = Int[]

    for s in 1:1000
        r = spectral_test(1024, s)
        push!(extended_ratios, r.ratio)
        if !r.passed
            extended_failures += 1
            push!(extended_failing, s)
        end
    end

    println("  Failures: ", extended_failures, "/1000 (", round(extended_failures/10, digits=1), "%)")
    println("  Mean ratio: ", round(mean(extended_ratios), digits=2))
    println("  95th percentile: ", round(sort(extended_ratios)[950], digits=2))
    println("  99th percentile: ", round(sort(extended_ratios)[990], digits=2))
    println()

    if length(extended_failing) <= 20
        println("All failing seeds (1-1000):")
        for s in extended_failing
            r = spectral_test(1024, s)
            println("  Seed ", s, ": ratio=", round(r.ratio, digits=2))
        end
    else
        println("First 20 failing seeds:")
        for s in extended_failing[1:20]
            r = spectral_test(1024, s)
            println("  Seed ", s, ": ratio=", round(r.ratio, digits=2))
        end
    end

    return (failures=extended_failures, failing_seeds=extended_failing, ratios=extended_ratios)
end

investigate_spectral()
