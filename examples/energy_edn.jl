#!/usr/bin/env julia
# Minimal energy test - EDN output only

cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using Gay

const SEED = Gay.GAY_SEED
const BATCH = 10_000_000
const DURATION = 30.0

total = 0
r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
start = time()

while (time() - start) < DURATION
    for i in 1:BATCH
        r, g, b = Gay.hash_color(total + i, SEED)
        global r_sum += r
        global g_sum += g
        global b_sum += b
    end
    global total += BATCH
end

elapsed = time() - start
power_w = 15.0  # estimated
energy_j = power_w * elapsed
fp = Gay.xor_fingerprint(r_sum, g_sum, b_sum)

# EDN output
println("""{:colors $total
 :seconds $(round(elapsed, digits=2))
 :rate-per-sec $(round(Int, total/elapsed))
 :power-watts $power_w
 :energy-joules $(round(energy_j, digits=2))
 :nj-per-color $(round(energy_j/total * 1e9, digits=3))
 :colors-per-joule $(round(Int, total/energy_j))
 :rgb-sums [$(round(r_sum, digits=1)) $(round(g_sum, digits=1)) $(round(b_sum, digits=1))]
 :fingerprint $(repr(fp))}""")
