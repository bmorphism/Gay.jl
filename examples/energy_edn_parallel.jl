#!/usr/bin/env julia
# Parallel energy test - EDN output

cd(@__DIR__)
using Pkg; Pkg.activate("."; io=devnull); Pkg.instantiate(; io=devnull)

using Gay

const SEED = Gay.GAY_SEED
const DURATION = 30.0
const NT = Threads.nthreads()

totals = zeros(Int, NT)
rsums = zeros(Float64, NT)
gsums = zeros(Float64, NT)
bsums = zeros(Float64, NT)

start = time()

Threads.@threads for tid in 1:NT
    local_total = 0
    local_r, local_g, local_b = 0.0, 0.0, 0.0
    base = tid * 1_000_000_000
    
    while (time() - start) < DURATION
        for i in 1:1_000_000
            r, g, b = Gay.hash_color(base + local_total + i, SEED)
            local_r += r
            local_g += g
            local_b += b
        end
        local_total += 1_000_000
    end
    
    totals[tid] = local_total
    rsums[tid] = local_r
    gsums[tid] = local_g
    bsums[tid] = local_b
end

elapsed = time() - start
total = sum(totals)
r_sum, g_sum, b_sum = sum(rsums), sum(gsums), sum(bsums)
power_w = 15.0 + (NT - 1) * 2.0  # Scale power with threads
energy_j = power_w * elapsed
fp = Gay.xor_fingerprint(r_sum, g_sum, b_sum)

println("""{:colors $total
 :threads $NT
 :seconds $(round(elapsed, digits=2))
 :rate-per-sec $(round(Int, total/elapsed))
 :power-watts $power_w
 :energy-joules $(round(energy_j, digits=2))
 :nj-per-color $(round(energy_j/total * 1e9, digits=3))
 :colors-per-joule $(round(Int, total/energy_j))
 :rgb-sums [$(round(r_sum, digits=1)) $(round(g_sum, digits=1)) $(round(b_sum, digits=1))]
 :fingerprint $(repr(fp))}""")
