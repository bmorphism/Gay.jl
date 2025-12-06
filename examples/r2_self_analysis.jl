# Gay.jl: r2 analyzing itself with deterministic SPI coloring
#
# Demonstrates Strong Parallelism Invariance (SPI) for binary analysis:
# - Same seed → same colors (reproducibility)
# - XOR parity for xrefs (checkerboard decomposition)
# - Parallel sublattice streams for even/odd analysis
#
# Data from: r2 -A /Users/bob/.topos/.flox/run/aarch64-darwin.effective-topos.run/bin/r2

using LispSyntax
using Colors

# Load Gay.jl (assumes we're in the package directory)
include("../src/Gay.jl")
using .Gay

# ═══════════════════════════════════════════════════════════════════════════
# r2 function data from live analysis (1155 functions found)
# ═══════════════════════════════════════════════════════════════════════════

const R2_FUNCTIONS = [
    # Core SDB functions (r2's internal key-value store)
    (0x10001caa0, "sym._sdb_new", 902),
    (0x10001d774, "sym._sdb_free", 148),
    (0x10001dd20, "sym._sdb_set", 24),
    (0x10001dc50, "sym._sdb_get", 56),
    (0x10001dbe8, "sym._sdb_unset", 48),
    (0x10001dc18, "sym._sdb_const_get", 24),
    
    # Hash table implementations
    (0x10000e1c0, "sym._ht_uu_new0", 108),
    (0x10000e22c, "sym._ht_uu_free", 276),
    (0x10000fd14, "sym._ht_up_new_opt", 28),
    (0x10000fe5c, "sym._ht_up_free", 344),
    (0x100011c3c, "sym._ht_pp_new", 152),
    (0x100011cd4, "sym._ht_pp_new0", 164),
    
    # CWISS (Swiss table implementation)
    (0x10000eca0, "sym._CWISS_RawTable_PrepareInsert", 312),
    (0x10000f078, "sym._CWISS_RawTable_Resize", 736),
    (0x10000f358, "sym._CWISS_RawTable_DropDeletesWithoutResize", 964),
    (0x10000f860, "sym._CWISS_RawTable_iter_at", 308),
    (0x10000f994, "sym._CWISS_RawTable_find", 452),
    
    # List operations
    (0x100018994, "sym._ls_new", 88),
    (0x1000189ec, "sym._ls_merge_sort", 116),
    (0x100018d18, "sym._ls_delete", 208),
    (0x100018f34, "sym._ls_destroy", 232),
    (0x10001901c, "sym._ls_free", 92),
    (0x100019078, "sym._ls_append", 148),
    
    # Main entry points
    (0x100000960, "main", 2092),
    (0x10000c380, "sym._sdb_main", 3728),
    (0x100007694, "sym._sdb_tool", 3524),
    
    # JSON handling
    (0x10001714c, "sym._sdb_json_indent", 840),
    (0x100017754, "sym._sdb_json_get", 316),
    (0x100017c28, "sym._sdb_json_set", 1500),
    
    # Query system
    (0x10001ab78, "sym._sdb_query", 260),
    (0x10001ad2c, "sym._sdb_querys", 5716),
    (0x10001c7a0, "sym._sdb_query_file", 480),
]

# Xrefs to sym._sdb_new (from live r2 analysis)
const XREFS_TO_SDB_NEW = [
    (0x100007ce0, 0x10001caa0, "CALL"),  # sdb_tool → sdb_new
    (0x100008150, 0x10001caa0, "CALL"),  # sdb_tool → sdb_new
    (0x10000c878, 0x10001caa0, "CALL"),  # (nofunc) → sdb_new
    (0x10000c958, 0x10001caa0, "CALL"),  # sdb_main → sdb_new
    (0x10000c96c, 0x10001caa0, "CALL"),  # sdb_main → sdb_new
    (0x10000ca64, 0x10001caa0, "CALL"),  # sdb_main → sdb_new
    (0x10000cbe4, 0x10001caa0, "CALL"),  # sdb_main → sdb_new
    (0x10000d288, 0x10001caa0, "CALL"),  # sdb_dump → sdb_new
    (0x10000d888, 0x10001caa0, "CALL"),  # synchronize → sdb_new
    (0x10000dbac, 0x10001caa0, "CALL"),  # createdb → sdb_new
    (0x10001ca9c, 0x10001caa0, "CODE"),  # sdb_new0 → sdb_new (tail call)
]

# Xrefs to sym._sdb_set (heavily used)
const XREFS_TO_SDB_SET = [
    (0x1000050b8, 0x10001dd20, "CALL"),  # sdb_array_insert
    (0x100005434, 0x10001dd20, "CALL"),  # sdb_array_set
    (0x100005574, 0x10001dd20, "CALL"),  # sdb_array_set
    (0x100005d80, 0x10001dd20, "CALL"),  # sdb_array_add_sorted_num
    (0x100005e78, 0x10001dd20, "CALL"),  # sdb_array_append
    (0x10000601c, 0x10001dd20, "CALL"),  # sdb_array_append_num
    (0x100006634, 0x10001dd20, "CALL"),  # sdb_array_prepend
    (0x1000068a0, 0x10001dd20, "CALL"),  # sdb_array_pop_head
    (0x10000833c, 0x10001dd20, "CALL"),  # sdb_tool
    (0x100015e6c, 0x10001dd20, "CALL"),  # sdb_journal_load
    (0x100017e34, 0x10001dd20, "CALL"),  # sdb_json_set
    (0x10001a574, 0x10001dd20, "CALL"),  # sdb_num_set
    (0x10001a670, 0x10001dd20, "CALL"),  # sdb_num_inc
    (0x10001a74c, 0x10001dd20, "CALL"),  # sdb_num_dec
    (0x10001a790, 0x10001dd20, "CALL"),  # sdb_num_dec
    (0x10001a860, 0x10001dd20, "CALL"),  # sdb_num_min
    (0x10001a934, 0x10001dd20, "CALL"),  # sdb_num_max
    (0x10001a9b0, 0x10001dd20, "CODE"),  # sdb_bool_set (tail)
    (0x10001aa68, 0x10001dd20, "CALL"),  # sdb_ptr_set
    (0x10001b594, 0x10001dd20, "CALL"),  # sdb_querys
    (0x10001be2c, 0x10001dd20, "CALL"),  # sdb_querys
    (0x100021ec8, 0x10001dd20, "CALL"),  # load_process_line
]

# ═══════════════════════════════════════════════════════════════════════════
# Determinism Tests
# ═══════════════════════════════════════════════════════════════════════════

function test_determinism()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║  Gay.jl + r2: Deterministic SPI Coloring                     ║")
    println("║  r2 analyzing itself: /bin/r2 (1155 functions)               ║")
    println("╚══════════════════════════════════════════════════════════════╝\n")
    
    # Test 1: Same seed → same colors
    println("═══ Test 1: Reproducibility (same seed = same colors) ═══\n")
    
    seeds = [UInt64(0xDEADBEEF), UInt64(0xCAFEBABE), UInt64(0x42424242)]
    
    for seed in seeds
        println("  Seed: 0x$(string(seed, base=16, pad=16))")
        colored1 = color_functions(R2_FUNCTIONS[1:5]; seed=seed)
        colored2 = color_functions(R2_FUNCTIONS[1:5]; seed=seed)
        
        match = all(c1.color == c2.color for (c1, c2) in zip(colored1, colored2))
        println("    Pass 1 == Pass 2: $(match ? "✓" : "✗")")
        
        for (i, f) in enumerate(colored1[1:3])
            println("    ", render_colored_function(f))
        end
        println()
    end
    
    # Test 2: Different seeds → different colors
    println("═══ Test 2: Different Seeds = Different Trajectories ═══\n")
    
    func = (0x10001caa0, "sym._sdb_new", 902)
    for seed in seeds
        c = color_function(UInt64(func[1]), func[2], func[3]; seed=seed)
        println("  ", render_colored_function(c), " (seed=0x$(string(seed, base=16)))")
    end
    println()
    
    # Test 3: XOR parity for xrefs (checkerboard)
    println("═══ Test 3: XOR Parity Coloring (Checkerboard Xrefs) ═══\n")
    
    seed = UInt64(0xDEADBEEF)
    colored_xrefs = color_xrefs(XREFS_TO_SDB_NEW; seed=seed)
    
    even_count = count(x -> x.parity == 0, colored_xrefs)
    odd_count = count(x -> x.parity == 1, colored_xrefs)
    
    println("  Xrefs to sym._sdb_new:")
    println("    Even parity (●): $even_count")
    println("    Odd parity  (○): $odd_count")
    println()
    
    for x in colored_xrefs[1:6]
        println("    ", render_colored_xref(x))
    end
    println("    ...")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# SPI Parallel Sublattice Demo
# ═══════════════════════════════════════════════════════════════════════════

function demo_spi_sublattices()
    println("═══ SPI Parallel Sublattices (Even/Odd Decomposition) ═══\n")
    
    seed = UInt64(0x7232636f6c6f7273)  # "r2colors"
    il = GayInterleaver(seed, 2)
    
    println("  Interleaver with 2 sublattices (checkerboard):")
    println("  → Even sites can be analyzed in parallel")
    println("  → Then odd sites in parallel")
    println("  → Preserves detailed balance for Monte Carlo")
    println()
    
    # Color functions by sublattice
    println("  Functions by address parity:")
    for (addr, name, size) in R2_FUNCTIONS[1:10]
        parity = Int(addr) & 1
        color = gay_sublattice(il, parity)
        rgb = convert(RGB, color)
        r = round(Int, clamp(rgb.r, 0, 1) * 255)
        g = round(Int, clamp(rgb.g, 0, 1) * 255)
        b = round(Int, clamp(rgb.b, 0, 1) * 255)
        
        symbol = parity == 0 ? "●" : "○"
        fg = "\e[38;2;$(r);$(g);$(b)m"
        R = "\e[0m"
        
        println("    $(fg)$(symbol)$(R) 0x$(string(addr, base=16, pad=8)) $(fg)$(name)$(R)")
    end
    println()
    
    # Show the XOR coloring for call graph edges
    println("  XOR-colored call graph edges (from → to):")
    for x in XREFS_TO_SDB_SET[1:8]
        from, to, _ = x
        xor_parity = Int((from ⊻ to) & 1)
        color = gay_sublattice(il, xor_parity)
        rgb = convert(RGB, color)
        r = round(Int, clamp(rgb.r, 0, 1) * 255)
        g = round(Int, clamp(rgb.g, 0, 1) * 255)
        b = round(Int, clamp(rgb.b, 0, 1) * 255)
        
        symbol = xor_parity == 0 ? "═" : "─"
        fg = "\e[38;2;$(r);$(g);$(b)m"
        R = "\e[0m"
        
        println("    0x$(string(from, base=16)) $(fg)$(symbol)→$(R) 0x$(string(to, base=16))")
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Colored Disassembly Demo
# ═══════════════════════════════════════════════════════════════════════════

function demo_colored_disasm()
    println("═══ Colored Disassembly (r2 main function) ═══\n")
    
    # Representative ARM64 instructions from main
    disasm = [
        (0x100000960, "stp", "x28, x27, [sp, #-0x60]!"),
        (0x100000964, "stp", "x26, x25, [sp, #0x10]"),
        (0x100000968, "stp", "x24, x23, [sp, #0x20]"),
        (0x10000096c, "stp", "x22, x21, [sp, #0x30]"),
        (0x100000970, "stp", "x20, x19, [sp, #0x40]"),
        (0x100000974, "stp", "x29, x30, [sp, #0x50]"),
        (0x100000978, "add", "x29, sp, #0x50"),
        (0x10000097c, "sub", "sp, sp, #0x650"),
        (0x100000980, "mov", "x25, x1"),
        (0x100000984, "mov", "w24, w0"),
        (0x100000988, "adrp", "x8, 0x100025000"),
        (0x10000098c, "ldr", "x8, [x8, #0x8]"),
        (0x100000990, "ldr", "x8, [x8]"),
        (0x100000994, "str", "x8, [x29, #-0x58]"),
        (0x100000998, "bl", "sym.imp.getuid"),
    ]
    
    seed = UInt64(0xDEADBEEF)
    println(render_colored_disasm(disasm; seed=seed))
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# S-expression Coloring (from semiosis.jl)
# ═══════════════════════════════════════════════════════════════════════════

function demo_sexpr_binary()
    println("═══ S-expression View of Binary Structure ═══\n")
    
    # Represent call graph as S-expression
    # (defmodule sdb
    #   (exports sdb_new sdb_free sdb_set sdb_get)
    #   (imports ht_uu_new ls_new malloc))
    
    sdb_module = [:defmodule, :sdb,
        [:exports, :sdb_new, :sdb_free, :sdb_set, :sdb_get, :sdb_query],
        [:imports, :ht_uu_new, :ls_new, :malloc, :free],
        [:callgraph,
            [:sdb_new, [:calls, :ht_uu_new, :ls_new, :malloc]],
            [:sdb_free, [:calls, :ls_free, :ht_uu_free, :free]],
            [:sdb_set, [:calls, :sdb_get, :malloc]]]]
    
    seed = UInt64(0x7232636f6c6f7273)
    gs = gay_magnetized_sexpr(sdb_module, seed)
    
    println("  Magnetized S-expression of SDB module:")
    println()
    println("  ", gay_render_sexpr(gs))
    println()
    
    # Compute magnetization
    M = gay_sexpr_magnetization(gs)
    println("  Magnetization ⟨M⟩ = $(round(M, digits=4))")
    
    depth_stats = gay_sexpr_depth_spins(gs)
    println("  Depth statistics:")
    for (d, (up, down, m)) in sort(collect(depth_stats))
        println("    d=$d: ↑$up ↓$down  ⟨M⟩=$(round(m, digits=2))")
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

function r2_semiosis()
    println()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  r2 Semiosis: Signs in Binary Analysis                       ║")
    println("║  σημείωσις - The process by which signs acquire meaning      ║")
    println("║                                                              ║")
    println("║  r2 analyzing itself with Gay.jl deterministic coloring      ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    
    test_determinism()
    demo_spi_sublattices()
    demo_colored_disasm()
    demo_sexpr_binary()
    
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  \"The colored address is the sign;                           ║")
    println("║   the XOR parity is the semiosis.\"                           ║")
    println("╚══════════════════════════════════════════════════════════════╝")
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    r2_semiosis()
end

export r2_semiosis, test_determinism, demo_spi_sublattices
export R2_FUNCTIONS, XREFS_TO_SDB_NEW, XREFS_TO_SDB_SET
