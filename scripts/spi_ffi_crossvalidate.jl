# Cross-validate Gay.jl's `spi_*` kernel against spi-race's Zig `libspi.dylib`
# at the C-ABI boundary via ccall. This is the -1/coplay refutation leg:
# an independent binary reference, not Gay's own test pins.
#
#   JULIA_DEPOT_PATH=~/worlds/.julia_depot julia --project=. \
#       scripts/spi_ffi_crossvalidate.jl
#
# Prints PASS/FAIL per category and up to 5 concrete mismatches each.
# Exits nonzero if any category diverges.

using Gay

const LIB = get(ENV, "SPI_LIB",
    "/Users/dietrich/worlds/b/spi-race/zig-out/lib/libspi.dylib")
isfile(LIB) || error("libspi.dylib not found at $LIB (build b/spi-race first)")

# --- Zig C-ABI (from libspi.zig) ---
z_color(seed, idx)        = ccall((:spi_color_at, LIB), UInt32, (UInt64, UInt64), seed, idx)
z_fp(seed, start, count)  = ccall((:spi_xor_fingerprint, LIB), UInt64, (UInt64, UInt64, UInt64), seed, start, count)
z_fp_par(seed, n, thr)    = ccall((:spi_xor_fingerprint_parallel, LIB), UInt64, (UInt64, UInt64, UInt32), seed, n, thr)
z_trit(seed, idx)         = ccall((:spi_trit, LIB), Int8, (UInt64, UInt64), seed, idx)
z_trit_sum(seed, s, c)    = ccall((:spi_trit_sum, LIB), Int32, (UInt64, UInt64, UInt64), seed, s, c)

const SEEDS = UInt64[42, 1069, 0x8b449cd3828014dd, 7]
const IDXS  = UInt64[0:255..., 69, 1000, 999983]
const WINS  = [(0,0),(0,1),(0,7),(0,8),(0,100),(5,95),(123,45678),(0,1_000_000),(0,10_000_000)]

struct Cat; name::String; fails::Vector{String}; total::Ref{Int}; end
Cat(n) = Cat(n, String[], Ref(0))
function check!(c::Cat, ok::Bool, msg)
    c.total[] += 1
    ok || (length(c.fails) < 5 && push!(c.fails, msg))
    ok
end

cats = Cat[]

# 1. color_at (packed u32)
c = Cat("spi_color_u32 vs spi_color_at"); push!(cats, c)
for s in SEEDS, i in IDXS
    j = Gay.spi_color_u32(s, i); k = z_color(s, i)
    check!(c, j == k, "seed=$(repr(s)) idx=$i julia=$(repr(j)) zig=$(repr(k))")
end

# 2. single trit
c = Cat("spi_trit vs spi_trit"); push!(cats, c)
for s in SEEDS, i in IDXS
    j = Gay.spi_trit(s, i); k = z_trit(s, i)
    check!(c, j == k, "seed=$(repr(s)) idx=$i julia=$j zig=$k")
end

# 3. xor fingerprint (single-threaded)
c = Cat("spi_xor_fingerprint vs spi_xor_fingerprint"); push!(cats, c)
for s in SEEDS, (st, cnt) in WINS
    j = Gay.spi_xor_fingerprint(s, st, cnt); k = z_fp(s, st, cnt)
    check!(c, j == k, "seed=$(repr(s)) win=($st,$cnt) julia=$(repr(j)) zig=$(repr(k))")
end

# 4. parallel fingerprint (Julia chunks) vs Zig parallel over [0,n)
c = Cat("spi_xor_fingerprint_parallel vs spi_xor_fingerprint_parallel"); push!(cats, c)
for s in SEEDS, n in (1000, 1_000_000, 10_000_000)
    j = Gay.spi_xor_fingerprint_parallel(s, n; chunks=4)
    k = z_fp_par(s, UInt64(n), UInt32(0))   # 0 => all cores in Zig
    check!(c, j == k, "seed=$(repr(s)) n=$n julia=$(repr(j)) zig=$(repr(k))")
end

# 5. trit sum over windows
c = Cat("spi_trit_sum vs spi_trit_sum"); push!(cats, c)
for s in SEEDS, (st, cnt) in [(0,0),(0,3),(0,100),(7,50),(0,1000),(0,1069)]
    j = Gay.spi_trit_sum(s, st, cnt); k = z_trit_sum(s, st, cnt)
    check!(c, Int(j) == Int(k), "seed=$(repr(s)) win=($st,$cnt) julia=$j zig=$k")
end

println("=== spi-race FFI cross-validation (Julia ccall -> libspi.dylib) ===")
println("LIB=$LIB")
allok = Ref(true)
for c in cats
    nf = length(c.fails)
    status = isempty(c.fails) ? "PASS" : "FAIL"
    isempty(c.fails) || (allok[] = false)
    println("[$status] $(c.name): $(c.total[]) checks, $(isempty(c.fails) ? 0 : "≥$(nf)") mismatch")
    for f in c.fails
        println("    MISMATCH $f")
    end
end
println(allok[] ? "RESULT: ALL CATEGORIES CONSISTENT" : "RESULT: DIVERGENCE FOUND")
flush(stdout)
exit(allok[] ? 0 : 1)
