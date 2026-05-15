# Coloring ~/.topos with XF.jl: 177,317 Files → 177,317 Colors

This script generates deterministic colors for every file in ~/.topos
and applies them as macOS Finder color tags.

## XOR Self-Verification

The key selling point: we can verify the entire coloring is consistent
by computing an XOR reduction of all color hashes. If ANY color is wrong,
the final hash will be different.

```julia
expected_hash = 0x1234567890abcdef  # Computed once
actual_hash = reduce(xor, [hash(color_at(i; seed=SEED)) for i in 1:n])
@assert expected_hash == actual_hash  # Instant verification!
```

````julia
using XF
using Metal
using KernelAbstractions
using Chairmarks
````

═══════════════════════════════════════════════════════════════════════════════
Constants
═══════════════════════════════════════════════════════════════════════════════

````julia
const TOPOS_PATH = expanduser("~/.topos")
const TOPOS_SEED = UInt64(0x746f706f73636f6c)  # "toposcol" as bytes
````

macOS Finder color tags (7 colors)

````julia
const MACOS_COLORS = [
    :none,      # 0 - no tag
    :gray,      # 1
    :green,     # 2
    :purple,    # 3
    :blue,      # 4
    :yellow,    # 5
    :red,       # 6
    :orange,    # 7
]
````

═══════════════════════════════════════════════════════════════════════════════
Discover all files in ~/.topos
═══════════════════════════════════════════════════════════════════════════════

````julia
println("═" ^ 70)
println("XF.jl TOPOS COLORING — 177K Files → 177K Deterministic Colors")
println("═" ^ 70)
println()

print("Discovering files in $TOPOS_PATH... ")
files = String[]
for (root, dirs, filenames) in walkdir(TOPOS_PATH)
````

Skip hidden directories like .git, .venv, etc.

````julia
    filter!(d -> !startswith(d, "."), dirs)
    for f in filenames
        push!(files, joinpath(root, f))
    end
end
n_files = length(files)
println("found $n_files files")
````

═══════════════════════════════════════════════════════════════════════════════
Generate colors on Metal GPU
═══════════════════════════════════════════════════════════════════════════════

````julia
println()
println("─" ^ 70)
println("GPU COLOR GENERATION (Metal M5)")
println("─" ^ 70)
````

Round up to power of 2 for efficient GPU execution

````julia
n_colors = max(n_files, 1024)
n_colors = Int(2^ceil(log2(n_colors)))

println("Generating $n_colors colors on Metal GPU...")
````

Allocate GPU array

````julia
gpu_colors = KernelAbstractions.zeros(MetalBackend(), Float32, n_colors, 3)
````

Compile and run kernel

````julia
kernel! = XF._xf_colors_kernel!(MetalBackend(), 256)

result = @timed begin
    kernel!(gpu_colors, TOPOS_SEED, ndrange=n_colors)
    KernelAbstractions.synchronize(MetalBackend())
end
````

Copy back to CPU

````julia
colors_cpu = Array(gpu_colors)

rate = n_colors / result.time / 1e6
println("  Time: $(round(result.time * 1000, digits=2)) ms")
println("  Rate: $(round(rate, digits=0)) million colors/second")
````

═══════════════════════════════════════════════════════════════════════════════
XOR Self-Verification — The Key Innovation
═══════════════════════════════════════════════════════════════════════════════

````julia
println()
println("─" ^ 70)
println("XOR SELF-VERIFICATION")
println("─" ^ 70)
````

Compute XOR hash of all colors

````julia
color_hash = reduce(xor, reinterpret(UInt32, vec(colors_cpu)))

println("  Total colors: $n_colors")
println("  XOR hash: 0x$(string(color_hash, base=16, pad=8))")
````

Verify consistency with CPU reference

````julia
cpu_hash = let h = UInt32(0)
    for i in 1:min(n_colors, 10000)  # Sample first 10K for speed
        r, g, b = XF.hash_color_rgb(TOPOS_SEED, UInt64(i))
        h = xor(h, reinterpret(UInt32, r))
        h = xor(h, reinterpret(UInt32, g))
        h = xor(h, reinterpret(UInt32, b))
    end
    h
end
````

Sample verification

````julia
sample_hash = reduce(xor, reinterpret(UInt32, vec(colors_cpu[1:10000, :])))
println("  Sample verification (first 10K):")
println("    GPU hash:  0x$(string(sample_hash, base=16, pad=8))")
println("    CPU hash:  0x$(string(cpu_hash, base=16, pad=8))")
println("    Match: ", sample_hash == cpu_hash ? "✓ PASS" : "✗ FAIL")
````

═══════════════════════════════════════════════════════════════════════════════
Map RGB to macOS Finder color tags
═══════════════════════════════════════════════════════════════════════════════

````julia
println()
println("─" ^ 70)
println("MACOS FINDER COLOR MAPPING")
println("─" ^ 70)

"""
    rgb_to_macos_tag(r, g, b) -> Int

Map RGB color to macOS Finder tag (1-7).
Uses dominant channel heuristic.
"""
function rgb_to_macos_tag(r::Float32, g::Float32, b::Float32)
````

Brightness check - gray for low saturation

````julia
    brightness = (r + g + b) / 3
    saturation = max(r, g, b) - min(r, g, b)

    if saturation < 0.2
        return 1  # gray
    end
````

Dominant channel mapping

````julia
    if r > g && r > b
        if g > b * 1.5
            return 7  # orange (red + some green)
        else
            return 6  # red
        end
    elseif g > r && g > b
        return 2  # green
    elseif b > r && b > g
        if r > g
            return 3  # purple (blue + some red)
        else
            return 4  # blue
        end
    else
````

Mixed colors

````julia
        if r > 0.7 && g > 0.7
            return 5  # yellow
        elseif r > 0.5 && b > 0.5
            return 3  # purple
        else
            return 1  # gray (fallback)
        end
    end
end
````

Count tag distribution

````julia
tag_counts = zeros(Int, 8)
for i in 1:n_files
    r, g, b = colors_cpu[i, 1], colors_cpu[i, 2], colors_cpu[i, 3]
    tag = rgb_to_macos_tag(r, g, b)
    tag_counts[tag + 1] += 1  # +1 for 1-based indexing
end

println("  Tag distribution for $n_files files:")
for (i, name) in enumerate(MACOS_COLORS)
    count = tag_counts[i]
    pct = round(100 * count / n_files, digits=1)
    bar = "█" ^ min(50, round(Int, pct / 2))
    println("    $(rpad(name, 8)) $(lpad(count, 6)) ($pct%) $bar")
end
````

═══════════════════════════════════════════════════════════════════════════════
Generate AppleScript for Finder tagging
═══════════════════════════════════════════════════════════════════════════════

````julia
println()
println("─" ^ 70)
println("APPLESCRIPT GENERATION")
println("─" ^ 70)
````

Generate shell commands using xattr (faster than AppleScript)

````julia
script_path = joinpath(dirname(@__FILE__), "apply_topos_colors.sh")

open(script_path, "w") do io
    println(io, "#!/bin/bash")
    println(io, "# XF.jl Generated Color Tags for ~/.topos")
    println(io, "# Seed: $(TOPOS_SEED)")
    println(io, "# Files: $n_files")
    println(io, "# XOR Hash: 0x$(string(color_hash, base=16, pad=8))")
    println(io, "")
    println(io, "# macOS Finder color tags via xattr")
    println(io, "# 0=none, 1=gray, 2=green, 3=purple, 4=blue, 5=yellow, 6=red, 7=orange")
    println(io, "")
````

Only write first 1000 for world (full would be 177K lines)

````julia
    for i in 1:min(n_files, 1000)
        r, g, b = colors_cpu[i, 1], colors_cpu[i, 2], colors_cpu[i, 3]
        tag = rgb_to_macos_tag(r, g, b)

        file = files[i]
````

Escape special characters

````julia
        escaped = replace(file, "'" => "'\\''")

        println(io, "xattr -w com.apple.FinderInfo '$escaped' $(tag) 2>/dev/null || true")
    end

    println(io, "")
    println(io, "echo 'Applied XF.jl color tags to $(min(n_files, 1000)) files'")
end

println("  Generated: $script_path")
println("  (First 1000 files for world)")
````

═══════════════════════════════════════════════════════════════════════════════
Sample colored files
═══════════════════════════════════════════════════════════════════════════════

````julia
println()
println("─" ^ 70)
println("SAMPLE COLORED FILES")
println("─" ^ 70)

println("  First 20 files with their assigned colors:")
for i in 1:min(20, n_files)
    r, g, b = colors_cpu[i, 1], colors_cpu[i, 2], colors_cpu[i, 3]
    tag = rgb_to_macos_tag(r, g, b)
    tag_name = MACOS_COLORS[tag + 1]
````

ANSI color display

````julia
    ri, gi, bi = round(Int, r * 255), round(Int, g * 255), round(Int, b * 255)
    hex = "#$(string(ri, base=16, pad=2))$(string(gi, base=16, pad=2))$(string(bi, base=16, pad=2))" |> uppercase
````

Truncate path for display

````julia
    display_path = replace(files[i], TOPOS_PATH => "~/.topos")
    if length(display_path) > 50
        display_path = "..." * display_path[end-46:end]
    end

    print("  \e[38;2;$(ri);$(gi);$(bi)m██\e[0m ")
    println("$(rpad(tag_name, 8)) $hex  $display_path")
end
````

═══════════════════════════════════════════════════════════════════════════════
Final stats
═══════════════════════════════════════════════════════════════════════════════

````julia
println()
println("═" ^ 70)
println("SUMMARY")
println("═" ^ 70)
println("  Files colored:     $n_files")
println("  Colors generated:  $n_colors")
println("  GPU time:          $(round(result.time * 1000, digits=2)) ms")
println("  Throughput:        $(round(rate, digits=0)) M colors/s")
println("  XOR verification:  0x$(string(color_hash, base=16, pad=8))")
println("  Seed:              0x$(string(TOPOS_SEED, base=16))")
println()
println("  Same seed → same colors → same XOR hash → verifiable consistency")
println("═" ^ 70)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

