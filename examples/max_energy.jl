#!/usr/bin/env julia
# MAXIMUM parallelism energy test - all cores, GPU if available

cd(@__DIR__)
using Pkg; Pkg.activate("."; io=devnull); Pkg.instantiate(; io=devnull)

using Gay

const SEED = Gay.GAY_SEED
const DURATION = 45.0  # Longer run
const NT = Threads.nthreads()

# Pre-allocate per-thread accumulators
totals = zeros(Int, NT)
rsums = zeros(Float64, NT)
gsums = zeros(Float64, NT)
bsums = zeros(Float64, NT)

start = time()

# Saturate all threads with tight hash loops
Threads.@threads for tid in 1:NT
    local_total = 0
    local_r, local_g, local_b = 0.0, 0.0, 0.0
    base = UInt64(tid) << 40  # Unique range per thread
    
    while (time() - start) < DURATION
        @inbounds for i in 1:2_000_000  # Larger batches = less overhead
            idx = base + UInt64(local_total + i)
            h1 = Gay.splitmix64(SEED ⊻ idx)
            h2 = Gay.splitmix64(h1)
            h3 = Gay.splitmix64(h2)
            local_r += Float64(h1 & 0xFFFFFF) / 16777215.0
            local_g += Float64(h2 & 0xFFFFFF) / 16777215.0
            local_b += Float64(h3 & 0xFFFFFF) / 16777215.0
        end
        local_total += 2_000_000
    end
    
    totals[tid] = local_total
    rsums[tid] = local_r
    gsums[tid] = local_g
    bsums[tid] = local_b
end

elapsed = time() - start
total = sum(totals)
r_sum, g_sum, b_sum = sum(rsums), sum(gsums), sum(bsums)

# Power model: base + per-core + efficiency cores
power_w = 8.0 + NT * 3.5  # More aggressive estimate
energy_j = power_w * elapsed
fp = Gay.xor_fingerprint(r_sum, g_sum, b_sum)

println("""{:colors $total
 :threads $NT  
 :seconds $(round(elapsed, digits=2))
 :rate-per-sec $(round(Int, total/elapsed))
 :billion-per-sec $(round(total/elapsed/1e9, digits=2))
 :power-watts $(round(power_w, digits=1))
 :energy-joules $(round(energy_j, digits=1))
 :nj-per-color $(round(energy_j/total * 1e9, digits=2))
 :colors-per-joule $(round(Int, total/energy_j))
 :million-per-joule $(round(total/energy_j/1e6, digits=1))
 :co2-grams-per-trillion $(round(energy_j/total * 1e12 * 400/3.6e6, digits=2))
 :fingerprint $(repr(fp))}""")
