# spivak_wiring.jl - Patterns running on matter (Spivak/Libkind)
#
# In the AlgebraicJulia sense:
#   Pattern = wiring diagram describing HOW parallel generators compose
#   Matter  = KernelAbstractions backend (CPU threads, Metal GPU, CUDA, etc.)
#
# The pattern is schedule-invariant (same wiring = same result).
# The matter determines throughput. Effective parallelism measures
# how well a given matter exploits a given pattern.
#
# key types:
#   Box          — a single computation unit (one kernel workitem)
#   Wire         — data dependency between boxes (seed → color)
#   WiringDiagram — directed graph of boxes and wires (the pattern)
#   Matter       — a KernelAbstractions backend (the substrate)
#   Placement    — assignment of boxes to matter (the schedule)

module SpivakWiring

using KernelAbstractions
using Colors: RGB

# Re-use Gay.jl's splitmix64 via module-level constants
const GOLDEN = 0x9e3779b97f4a7c15
const MIX1 = 0xbf58476d1ce4e5b9
const MIX2 = 0x94d049bb133111eb

@inline function splitmix64(x::UInt64)::UInt64
    x += GOLDEN
    x = ((x ⊻ (x >> 30)) * MIX1) % UInt64
    x = ((x ⊻ (x >> 27)) * MIX2) % UInt64
    (x ⊻ (x >> 31)) % UInt64
end

@inline function hash_color(seed::UInt64, index::UInt64)
    h = splitmix64(xor(seed, index * GOLDEN))
    r = Float32(h & 0xFF) / 255.0f0
    g = Float32((h >> 8) & 0xFF) / 255.0f0
    b = Float32((h >> 16) & 0xFF) / 255.0f0
    (r, g, b)
end

# ═══════════════════════════════════════════════════════════════════════════
# Pattern: Wiring Diagram
# ═══════════════════════════════════════════════════════════════════════════

"""
    Box

A computation unit in the wiring diagram.
In Gay.jl, each box is a color generator identified by (seed, index_range).
"""
struct Box
    id::Int
    seed::UInt64
    index_start::UInt64
    index_end::UInt64
end

n_items(b::Box) = Int(b.index_end - b.index_start + 1)

"""
    Wire

A directed data dependency: output of source box feeds into sink box.
In the color pipeline: seed → splitmix64 → RGB extraction.
"""
struct Wire
    source::Int  # box id
    sink::Int    # box id
    label::Symbol
end

"""
    WiringDiagram

The pattern: a directed graph of boxes and wires.
This is the schedule-invariant specification of what to compute.
It says nothing about how many threads, what GPU, etc.
"""
struct WiringDiagram
    boxes::Vector{Box}
    wires::Vector{Wire}
    n_total::Int
    seed::UInt64
end

"""
    flat_diagram(n::Int, seed::UInt64)

Trivially parallel pattern: n independent boxes, no wires.
Every workitem is independent — maximum theoretical parallelism.
This is the pattern behind `ka_colors(n, seed)`.
"""
function flat_diagram(n::Int, seed::UInt64)
    boxes = [Box(i, seed, UInt64(i), UInt64(i)) for i in 1:n]
    WiringDiagram(boxes, Wire[], n, seed)
end

"""
    chunked_diagram(n::Int, seed::UInt64, chunk_size::Int)

Chunked parallel pattern: n/chunk_size boxes, each processing chunk_size items.
Wires connect seed to each chunk. Intra-chunk is sequential, inter-chunk is parallel.
This is the pattern behind `ka_color_sums(n, seed; chunk_size=...)`.
"""
function chunked_diagram(n::Int, seed::UInt64, chunk_size::Int)
    n_chunks = cld(n, chunk_size)
    boxes = Box[]
    for c in 1:n_chunks
        start = UInt64((c - 1) * chunk_size + 1)
        stop = UInt64(min(c * chunk_size, n))
        push!(boxes, Box(c, seed, start, stop))
    end
    # No inter-chunk wires: each chunk reduces independently
    WiringDiagram(boxes, Wire[], n, seed)
end

"""
    triad_diagram(n::Int, seed::UInt64)

GF(3)-balanced pattern: 3 parallel generator streams per logical workitem.
Each workitem spawns MINUS/ERGODIC/PLUS sub-generators.
Pattern has 3n boxes with intra-triad wires for balance checking.
"""
function triad_diagram(n::Int, seed::UInt64)
    boxes = Box[]
    wires = Wire[]
    for i in 1:n
        base_id = (i - 1) * 3
        # Three generators per workitem
        push!(boxes, Box(base_id + 1, seed ⊻ 0x2d2d2d2d2d2d2d2d, UInt64(i), UInt64(i)))  # MINUS
        push!(boxes, Box(base_id + 2, seed ⊻ 0x5f5f5f5f5f5f5f5f, UInt64(i), UInt64(i)))  # ERGODIC
        push!(boxes, Box(base_id + 3, seed ⊻ 0x2b2b2b2b2b2b2b2b, UInt64(i), UInt64(i)))  # PLUS
        # Intra-triad balance wires
        push!(wires, Wire(base_id + 1, base_id + 2, :balance))
        push!(wires, Wire(base_id + 2, base_id + 3, :balance))
        push!(wires, Wire(base_id + 3, base_id + 1, :balance))
    end
    WiringDiagram(boxes, wires, n, seed)
end

"""
    reduction_diagram(n::Int, seed::UInt64, chunk_size::Int)

Two-phase reduction pattern: parallel generation → parallel partial sums → final sum.
This is the pattern behind `ka_color_sums` with its map-reduce structure.
"""
function reduction_diagram(n::Int, seed::UInt64, chunk_size::Int)
    n_chunks = cld(n, chunk_size)
    boxes = Box[]
    wires = Wire[]

    # Phase 1: generation chunks
    for c in 1:n_chunks
        start = UInt64((c - 1) * chunk_size + 1)
        stop = UInt64(min(c * chunk_size, n))
        push!(boxes, Box(c, seed, start, stop))
    end

    # Phase 2: reduction node
    reducer_id = n_chunks + 1
    push!(boxes, Box(reducer_id, seed, UInt64(0), UInt64(0)))
    for c in 1:n_chunks
        push!(wires, Wire(c, reducer_id, :reduce))
    end

    WiringDiagram(boxes, wires, n, seed)
end

# ═══════════════════════════════════════════════════════════════════════════
# Matter: Backend Abstraction
# ═══════════════════════════════════════════════════════════════════════════

"""
    Matter

The physical substrate a pattern runs on.
Wraps a KernelAbstractions backend with resource metadata.
"""
struct Matter
    backend::KernelAbstractions.Backend
    n_threads::Int
    label::String
end

function cpu_matter(; threads::Int = Threads.nthreads())
    Matter(CPU(), threads, "CPU($(threads) threads)")
end

# ═══════════════════════════════════════════════════════════════════════════
# Placement: Pattern × Matter → Execution
# ═══════════════════════════════════════════════════════════════════════════

"""
    Placement

Assignment of a wiring diagram's boxes to a matter's resources.
The oapply of Spivak/Libkind: pattern composed with matter yields computation.
"""
struct Placement
    diagram::WiringDiagram
    matter::Matter
    workgroup_size::Int
end

"""
    oapply(diagram, matter; workgroup=256)

Compose pattern with matter. Returns a Placement ready to execute.
This is the Spivak/Libkind oapply: the operad algebra evaluation
that takes a wiring diagram and fills its boxes with actual computation.
"""
function oapply(diagram::WiringDiagram, matter::Matter; workgroup::Int = 256)
    Placement(diagram, matter, workgroup)
end

# ═══════════════════════════════════════════════════════════════════════════
# Execution: run placement, measure effective parallelism
# ═══════════════════════════════════════════════════════════════════════════

@kernel function _flat_kernel!(colors, @Const(seed::UInt64))
    i = @index(Global)
    r, g, b = hash_color(seed, UInt64(i))
    colors[i, 1] = r
    colors[i, 2] = g
    colors[i, 3] = b
end

"""
    execute(p::Placement) -> (result, time_seconds)

Run the placement and return (output, wall_time).
"""
function execute(p::Placement)
    n = p.diagram.n_total
    backend = p.matter.backend
    colors = zeros(Float32, n, 3)

    # Warmup
    if n > 1000
        warmup = zeros(Float32, min(1000, n), 3)
        k! = _flat_kernel!(backend, p.workgroup_size)
        k!(warmup, p.diagram.seed, ndrange = min(1000, n))
        KernelAbstractions.synchronize(backend)
    end

    t = @elapsed begin
        k! = _flat_kernel!(backend, p.workgroup_size)
        k!(colors, p.diagram.seed, ndrange = n)
        KernelAbstractions.synchronize(backend)
    end

    (colors, t)
end

"""
    execute_sequential(diagram::WiringDiagram) -> (result, time_seconds)

Sequential baseline: single-threaded loop, no KA overhead.
"""
function execute_sequential(diagram::WiringDiagram)
    n = diagram.n_total
    seed = diagram.seed
    colors = zeros(Float32, n, 3)

    t = @elapsed begin
        for i in 1:n
            r, g, b = hash_color(UInt64(seed), UInt64(i))
            colors[i, 1] = r
            colors[i, 2] = g
            colors[i, 3] = b
        end
    end

    (colors, t)
end

# ═══════════════════════════════════════════════════════════════════════════
# Effective Parallelism Measurement
# ═══════════════════════════════════════════════════════════════════════════

"""
    ParallelismReport

Quantifies how well a pattern exploits a given matter.
"""
struct ParallelismReport
    pattern_name::String
    matter_label::String
    n::Int
    t_sequential::Float64
    t_parallel::Float64
    speedup::Float64
    efficiency::Float64       # speedup / n_threads
    throughput::Float64       # colors/second
    spi_verified::Bool        # Strong Parallelism Invariance holds
    theoretical_max::Float64  # Amdahl upper bound
    serial_fraction::Float64  # estimated from Amdahl's law inversion
end

"""
    measure_parallelism(diagram, matter; workgroup=256, verify_spi=true)

The core measurement: how effectively does this pattern run on this matter?

Returns a ParallelismReport with speedup, efficiency, throughput, and SPI status.
"""
function measure_parallelism(diagram::WiringDiagram, matter::Matter;
                              workgroup::Int = 256, verify_spi::Bool = true)
    placement = oapply(diagram, matter; workgroup = workgroup)

    # Sequential baseline
    seq_colors, t_seq = execute_sequential(diagram)

    # Parallel execution
    par_colors, t_par = execute(placement)

    # SPI check: same output regardless of schedule
    spi_ok = verify_spi ? isapprox(seq_colors, par_colors; rtol = 1e-5) : true

    speedup = t_seq / max(t_par, 1e-12)
    n_threads = matter.n_threads
    efficiency = speedup / n_threads
    throughput = diagram.n_total / max(t_par, 1e-12)

    # Amdahl inversion: estimate serial fraction f from observed speedup
    # S = 1 / (f + (1-f)/p)  =>  f = (1/S - 1/p) / (1 - 1/p)
    p = Float64(n_threads)
    f = if speedup > 1.0 && p > 1.0
        (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p)
    else
        1.0
    end
    f = clamp(f, 0.0, 1.0)
    theoretical_max = 1.0 / (f + (1.0 - f) / p)

    # Determine pattern name
    n_wires = length(diagram.wires)
    n_boxes = length(diagram.boxes)
    pattern_name = if n_wires == 0 && n_boxes == diagram.n_total
        "flat"
    elseif n_wires == 0
        "chunked($(n_boxes))"
    elseif all(w -> w.label == :balance, diagram.wires)
        "triad($(diagram.n_total))"
    else
        "reduction($(n_boxes - 1)→1)"
    end

    ParallelismReport(
        pattern_name, matter.label, diagram.n_total,
        t_seq, t_par, speedup, efficiency, throughput,
        spi_ok, theoretical_max, f
    )
end

function Base.show(io::IO, r::ParallelismReport)
    println(io, "═══════════════════════════════════════════════════════════════")
    println(io, "  EFFECTIVE PARALLELISM: pattern on matter")
    println(io, "═══════════════════════════════════════════════════════════════")
    println(io, "  Pattern:           $(r.pattern_name)")
    println(io, "  Matter:            $(r.matter_label)")
    println(io, "  N:                 $(r.n)")
    println(io, "  ───────────────────────────────────────────────────────────")
    println(io, "  T_sequential:      $(round(r.t_sequential * 1000, digits=2)) ms")
    println(io, "  T_parallel:        $(round(r.t_parallel * 1000, digits=2)) ms")
    println(io, "  Speedup:           $(round(r.speedup, digits=2))×")
    println(io, "  Efficiency:        $(round(r.efficiency * 100, digits=1))%")
    println(io, "  Throughput:        $(round(r.throughput / 1e6, digits=1)) M colors/s")
    println(io, "  ───────────────────────────────────────────────────────────")
    println(io, "  SPI verified:      $(r.spi_verified ? "✓" : "✗")")
    println(io, "  Serial fraction:   $(round(r.serial_fraction * 100, digits=1))%")
    println(io, "  Amdahl ceiling:    $(round(r.theoretical_max, digits=2))×")
    println(io, "═══════════════════════════════════════════════════════════════")
end

# ═══════════════════════════════════════════════════════════════════════════
# Convenience: full audit across patterns
# ═══════════════════════════════════════════════════════════════════════════

"""
    audit_patterns(n::Int=100_000, seed::UInt64=UInt64(69);
                   chunk_sizes=[100, 1000, 10000])

Run all wiring diagram patterns on current matter and report effective parallelism.
"""
function audit_patterns(n::Int = 100_000, seed::UInt64 = UInt64(69);
                         chunk_sizes::Vector{Int} = [100, 1000, 10000])
    matter = cpu_matter()
    reports = ParallelismReport[]

    # Flat (trivially parallel)
    push!(reports, measure_parallelism(flat_diagram(n, seed), matter))

    # Chunked at various sizes
    for cs in chunk_sizes
        push!(reports, measure_parallelism(chunked_diagram(n, seed, cs), matter))
    end

    # Triad (GF(3) balanced)
    n_triad = min(n, 10_000)  # triad creates 3× boxes
    push!(reports, measure_parallelism(triad_diagram(n_triad, seed), matter))

    # Reduction
    push!(reports, measure_parallelism(reduction_diagram(n, seed, 10000), matter))

    for r in reports
        show(stdout, r)
        println()
    end

    reports
end

export Box, Wire, WiringDiagram, Matter, Placement
export flat_diagram, chunked_diagram, triad_diagram, reduction_diagram
export cpu_matter, oapply, execute, execute_sequential
export ParallelismReport, measure_parallelism, audit_patterns

end # module SpivakWiring
