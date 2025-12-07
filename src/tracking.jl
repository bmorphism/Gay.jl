# Deterministic Test Tracking with SPI Colors
# =============================================
# Track test results, violations, and metrics using hash-based coloring.
# Each test run, violation, and metric gets a deterministic color fingerprint.

using Colors: RGB
using Dates

export TestTracker, track!, violation!, metric!, render_history, tracker_fingerprint
export TestResult, ViolationType, TrackedMetric
export PASS, FAIL, SKIP, VIOLATION, ANOMALY

# ═══════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════

@enum TestResult PASS FAIL SKIP VIOLATION ANOMALY

@enum ViolationType begin
    SPI_VIOLATION           # Same seed produced different results
    RACE_CONDITION          # Concurrent writes caused data corruption
    FINGERPRINT_MISMATCH    # XOR fingerprints don't match
    DETERMINISM_FAILURE     # Non-deterministic output
    DISTRIBUTION_ANOMALY    # Statistical distribution is wrong
    TIMING_ANOMALY          # Unexpected timing behavior
    MEMORY_CORRUPTION       # Memory was corrupted
    CROSS_SUBSTRATE_MISMATCH # CPU vs GPU mismatch
end

struct TrackedMetric
    name::String
    value::Float64
    unit::String
    timestamp::DateTime
    color::RGB{Float64}
end

struct TrackedViolation
    type::ViolationType
    description::String
    context::Dict{Symbol, Any}
    timestamp::DateTime
    color::RGB{Float64}
    fingerprint::UInt32
end

struct TrackedTest
    name::String
    result::TestResult
    duration::Float64
    iterations::Int
    violations::Vector{TrackedViolation}
    metrics::Vector{TrackedMetric}
    timestamp::DateTime
    color::RGB{Float64}
    fingerprint::UInt32
end

mutable struct TestTracker
    seed::UInt64
    tests::Vector{TrackedTest}
    violations::Vector{TrackedViolation}
    metrics::Vector{TrackedMetric}
    start_time::DateTime
    total_colors::Int
    total_iterations::Int
end

# ═══════════════════════════════════════════════════════════════════════════
# Color Generation for Tracking
# ═══════════════════════════════════════════════════════════════════════════

"""
Generate deterministic color for a string identifier.
"""
function id_color(s::String, seed::UInt64=GAY_SEED)
    # Hash the string to get an index
    h = hash(s, seed)
    r, g, b = hash_color(seed, UInt64(h & 0xFFFFFFFF))
    RGB{Float64}(r, g, b)
end

"""
Generate color for a test result type.
"""
function result_color(r::TestResult)
    if r == PASS
        RGB{Float64}(0.2, 0.8, 0.3)      # Green
    elseif r == FAIL
        RGB{Float64}(0.9, 0.2, 0.2)      # Red
    elseif r == SKIP
        RGB{Float64}(0.6, 0.6, 0.6)      # Gray
    elseif r == VIOLATION
        RGB{Float64}(1.0, 0.5, 0.0)      # Orange
    else  # ANOMALY
        RGB{Float64}(0.8, 0.2, 0.8)      # Purple
    end
end

"""
Generate color for a violation type.
"""
function violation_color(v::ViolationType)
    # Each violation type gets a deterministic color
    idx = Int(v) + 1
    r, g, b = hash_color(UInt64(0xDEADBEEF), UInt64(idx * 1000))
    RGB{Float64}(r, g, b)
end

"""
Compute fingerprint for a tracked item.
"""
function compute_fingerprint(name::String, timestamp::DateTime, seed::UInt64)
    h = hash(name, seed)
    h = hash(string(timestamp), h)
    UInt32(h & 0xFFFFFFFF)
end

# ═══════════════════════════════════════════════════════════════════════════
# Tracker API
# ═══════════════════════════════════════════════════════════════════════════

"""
Create a new test tracker.
"""
function TestTracker(; seed::UInt64=GAY_SEED)
    TestTracker(
        seed,
        TrackedTest[],
        TrackedViolation[],
        TrackedMetric[],
        now(),
        0,
        0
    )
end

# Global tracker
const GLOBAL_TRACKER = Ref{TestTracker}()

"""
Get or create global tracker.
"""
function get_tracker()
    if !isassigned(GLOBAL_TRACKER)
        GLOBAL_TRACKER[] = TestTracker()
    end
    GLOBAL_TRACKER[]
end

"""
Track a test result.
"""
function track!(name::String, result::TestResult;
                duration::Float64=0.0,
                iterations::Int=0,
                colors::Int=0,
                violations::Vector{TrackedViolation}=TrackedViolation[],
                metrics::Vector{TrackedMetric}=TrackedMetric[],
                tracker::TestTracker=get_tracker())
    
    ts = now()
    fp = compute_fingerprint(name, ts, tracker.seed)
    color = id_color(name, tracker.seed)
    
    test = TrackedTest(
        name, result, duration, iterations,
        violations, metrics, ts, color, fp
    )
    
    push!(tracker.tests, test)
    append!(tracker.violations, violations)
    append!(tracker.metrics, metrics)
    tracker.total_colors += colors
    tracker.total_iterations += iterations
    
    test
end

"""
Track a violation.
"""
function violation!(type::ViolationType, description::String;
                    context::Dict{Symbol, Any}=Dict{Symbol, Any}(),
                    tracker::TestTracker=get_tracker())
    
    ts = now()
    fp = compute_fingerprint(string(type), ts, tracker.seed)
    color = violation_color(type)
    
    v = TrackedViolation(type, description, context, ts, color, fp)
    push!(tracker.violations, v)
    
    v
end

"""
Track a metric.
"""
function metric!(name::String, value::Float64, unit::String="";
                 tracker::TestTracker=get_tracker())
    
    ts = now()
    color = id_color(name, tracker.seed)
    
    m = TrackedMetric(name, value, unit, ts, color)
    push!(tracker.metrics, m)
    
    m
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

"""
ANSI color code from RGB.
"""
function ansi_fg(c::RGB)
    r = round(Int, clamp(c.r, 0, 1) * 255)
    g = round(Int, clamp(c.g, 0, 1) * 255)
    b = round(Int, clamp(c.b, 0, 1) * 255)
    "\e[38;2;$(r);$(g);$(b)m"
end

const RESET = "\e[0m"

"""
Render a colored block.
"""
function color_block(c::RGB, width::Int=2)
    "$(ansi_fg(c))$("█" ^ width)$(RESET)"
end

"""
Render tracker history to terminal.
"""
function render_history(tracker::TestTracker=get_tracker())
    println()
    println("╔══════════════════════════════════════════════════════════════════════╗")
    println("║  SPI TEST TRACKER: Deterministic Coloring                           ║")
    println("╚══════════════════════════════════════════════════════════════════════╝")
    println()
    
    # Summary
    n_pass = count(t -> t.result == PASS, tracker.tests)
    n_fail = count(t -> t.result == FAIL, tracker.tests)
    n_violations = length(tracker.violations)
    
    println("  Seed: 0x$(string(tracker.seed, base=16))")
    println("  Tests: $(length(tracker.tests)) ($(n_pass) pass, $(n_fail) fail)")
    println("  Violations: $(n_violations)")
    println("  Total colors: $(round(tracker.total_colors / 1e6, digits=2))M")
    println("  Total iterations: $(tracker.total_iterations)")
    println()
    
    # Test timeline
    if !isempty(tracker.tests)
        println("  Test Timeline:")
        println("  ─────────────────────────────────────────────────────────────────")
        
        for (i, t) in enumerate(tracker.tests)
            result_c = result_color(t.result)
            test_c = t.color
            block = color_block(test_c)
            status = color_block(result_c)
            
            fp_str = "0x$(string(t.fingerprint, base=16, pad=8))"
            duration_str = t.duration > 0 ? " $(round(t.duration, digits=2))s" : ""
            iter_str = t.iterations > 0 ? " $(t.iterations) iters" : ""
            
            println("  $block $status $(rpad(t.name, 30))$duration_str$iter_str [$fp_str]")
        end
        println()
    end
    
    # Violations
    if !isempty(tracker.violations)
        println("  Violations:")
        println("  ─────────────────────────────────────────────────────────────────")
        
        for (i, v) in enumerate(tracker.violations)
            block = color_block(v.color)
            fp_str = "0x$(string(v.fingerprint, base=16, pad=8))"
            
            println("  $block $(rpad(string(v.type), 25)) $(v.description) [$fp_str]")
        end
        println()
    end
    
    # Metrics
    if !isempty(tracker.metrics)
        println("  Metrics:")
        println("  ─────────────────────────────────────────────────────────────────")
        
        for m in tracker.metrics
            block = color_block(m.color)
            value_str = if m.value >= 1e9
                "$(round(m.value / 1e9, digits=2))B"
            elseif m.value >= 1e6
                "$(round(m.value / 1e6, digits=2))M"
            elseif m.value >= 1e3
                "$(round(m.value / 1e3, digits=2))K"
            else
                "$(round(m.value, digits=2))"
            end
            
            println("  $block $(rpad(m.name, 25)) $(value_str) $(m.unit)")
        end
        println()
    end
    
    # Color fingerprint of entire tracker state
    fp = tracker_fingerprint(tracker)
    println("  Tracker Fingerprint: 0x$(string(fp, base=16, pad=8))")
    println("═══════════════════════════════════════════════════════════════════════")
end

"""
Compute fingerprint of entire tracker state.
"""
function tracker_fingerprint(tracker::TestTracker=get_tracker())
    fp = UInt32(0)
    
    for t in tracker.tests
        fp = xor(fp, t.fingerprint)
    end
    
    for v in tracker.violations
        fp = xor(fp, v.fingerprint)
    end
    
    fp
end

# ═══════════════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════════════

"""
Export tracker to JSON-compatible dict.
"""
function tracker_to_dict(tracker::TestTracker=get_tracker())
    Dict(
        "seed" => tracker.seed,
        "start_time" => string(tracker.start_time),
        "total_colors" => tracker.total_colors,
        "total_iterations" => tracker.total_iterations,
        "fingerprint" => tracker_fingerprint(tracker),
        "tests" => [
            Dict(
                "name" => t.name,
                "result" => string(t.result),
                "duration" => t.duration,
                "iterations" => t.iterations,
                "timestamp" => string(t.timestamp),
                "fingerprint" => t.fingerprint,
                "color" => [t.color.r, t.color.g, t.color.b]
            ) for t in tracker.tests
        ],
        "violations" => [
            Dict(
                "type" => string(v.type),
                "description" => v.description,
                "timestamp" => string(v.timestamp),
                "fingerprint" => v.fingerprint,
                "color" => [v.color.r, v.color.g, v.color.b]
            ) for v in tracker.violations
        ],
        "metrics" => [
            Dict(
                "name" => m.name,
                "value" => m.value,
                "unit" => m.unit,
                "timestamp" => string(m.timestamp),
                "color" => [m.color.r, m.color.g, m.color.b]
            ) for m in tracker.metrics
        ]
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Convenience Macros
# ═══════════════════════════════════════════════════════════════════════════

"""
    @tracked "Test Name" begin
        # test code
    end

Run a test block and automatically track the result.
"""
macro tracked(name, block)
    quote
        local _name = $(esc(name))
        local _start = time()
        local _result = PASS
        local _violations = TrackedViolation[]
        
        try
            $(esc(block))
        catch e
            _result = FAIL
            if e isa AssertionError
                push!(_violations, violation!(SPI_VIOLATION, string(e)))
            end
            rethrow()
        finally
            local _duration = time() - _start
            track!(_name, _result; duration=_duration, violations=_violations)
        end
    end
end
