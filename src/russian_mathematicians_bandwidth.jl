# RUSSIAN MATHEMATICIANS COLOR BANDWIDTH: 3-Tuple Ranking
# ═══════════════════════════════════════════════════════════════════════════════
#
# "Every mathematician has an originary hue. The combinatorics of brilliance."
#
# Construct all 3-tuples of mathematicians with Russian last names,
# rank by combined color bandwidth.

module RussianMathematiciansBandwidth

export russian_mathematicians, compute_3tuple_bandwidths, rank_by_bandwidth, demo_russian_bandwidth

# ═══════════════════════════════════════════════════════════════════════════════
# SPLITMIX64
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)

@inline function sm64(s::UInt64)::UInt64
    z = (s + 0x9E3779B97F4A7C15)
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    z ⊻ (z >> 31)
end

@inline function sm64_color(s::UInt64)::NTuple{3, Float64}
    r = sm64(s)
    g = sm64(r)
    b = sm64(g)
    (Float64(r >> 56) / 255.0, Float64(g >> 56) / 255.0, Float64(b >> 56) / 255.0)
end

function name_to_seed(name::String)::UInt64
    h = UInt64(0xcbf29ce484222325)  # FNV-1a offset basis
    for byte in codeunits(name)
        h = h ⊻ UInt64(byte)
        h = h * UInt64(0x100000001b3)  # FNV prime
    end
    h
end

# ═══════════════════════════════════════════════════════════════════════════════
# RUSSIAN MATHEMATICIANS
# ═══════════════════════════════════════════════════════════════════════════════

struct Mathematician
    name::String
    field::String
    years::String
    seed::UInt64
    color::NTuple{3, Float64}
end

function Mathematician(name::String, field::String, years::String)
    seed = name_to_seed(name)
    Mathematician(name, field, years, seed, sm64_color(seed))
end

const RUSSIAN_MATHEMATICIANS = [
    # Analysis & Foundations
    Mathematician("Kolmogorov", "Probability, Topology, Turbulence", "1903-1987"),
    Mathematician("Chebyshev", "Number Theory, Probability", "1821-1894"),
    Mathematician("Markov", "Stochastic Processes", "1856-1922"),
    Mathematician("Lyapunov", "Stability Theory, Probability", "1857-1918"),
    Mathematician("Lebesgue", "Measure Theory", "1875-1941"),  # French but honorary
    
    # Algebra & Geometry
    Mathematician("Lobachevsky", "Non-Euclidean Geometry", "1792-1856"),
    Mathematician("Gelfand", "Functional Analysis, Representation Theory", "1913-2009"),
    Mathematician("Pontryagin", "Topology, Optimal Control", "1908-1988"),
    Mathematician("Vinogradov", "Analytic Number Theory", "1891-1983"),
    Mathematician("Shafarevich", "Algebraic Geometry, Number Theory", "1923-2017"),
    
    # Modern Giants
    Mathematician("Perelman", "Geometric Analysis, Ricci Flow", "1966-"),
    Mathematician("Kontsevich", "Mathematical Physics, Mirror Symmetry", "1964-"),
    Mathematician("Voevodsky", "Algebraic Geometry, Homotopy Type Theory", "1966-2017"),
    Mathematician("Drinfeld", "Quantum Groups, Langlands", "1954-"),
    Mathematician("Manin", "Algebraic Geometry, Mathematical Physics", "1937-2023"),
    
    # Soviet Era
    Mathematician("Alexandrov", "Topology, Convex Geometry", "1912-1999"),
    Mathematician("Arnold", "Dynamical Systems, Symplectic Geometry", "1937-2010"),
    Mathematician("Sinai", "Ergodic Theory, Statistical Mechanics", "1935-"),
    Mathematician("Novikov", "Topology, Integrable Systems", "1938-"),
    Mathematician("Gromov", "Metric Geometry, Symplectic Topology", "1943-"),
    
    # Probability & Statistics
    Mathematician("Bernstein", "Approximation Theory, Probability", "1880-1968"),
    Mathematician("Khinchin", "Probability, Number Theory", "1894-1959"),
    Mathematician("Gnedenko", "Probability, Extreme Value Theory", "1912-1995"),
    Mathematician("Prokhorov", "Probability, Functional Analysis", "1929-2013"),
    Mathematician("Skorokhod", "Stochastic Analysis", "1930-2011"),
    
    # Applied & Computational
    Mathematician("Kantorovich", "Optimization, Linear Programming", "1912-1986"),
    Mathematician("Krylov", "Numerical Analysis, Subspace Methods", "1879-1955"),
    Mathematician("Sobolev", "Functional Analysis, PDEs", "1908-1989"),
    Mathematician("Ladyzhenskaya", "PDEs, Fluid Dynamics", "1922-2004"),
    Mathematician("Keldysh", "Complex Analysis, Spectral Theory", "1911-1978"),
    
    # Logic & Computability
    Mathematician("Matiyasevich", "Logic, Hilbert's 10th Problem", "1947-"),
    Mathematician("Ershov", "Computability Theory", "1940-"),
    
    # More Recent
    Mathematician("Okounkov", "Representation Theory, Enumerative Geometry", "1969-"),
    Mathematician("Smirnov", "Probability, Conformal Invariance", "1970-"),
    Mathematician("Avila", "Dynamical Systems", "1979-"),  # Brazilian but Margulis school
    
    # Category Theory & Algebra
    Mathematician("Kapranov", "Higher Categories, Algebraic Geometry", "1962-"),
    Mathematician("Bondal", "Derived Categories", "1963-"),
    Mathematician("Orlov", "Derived Categories, Mirror Symmetry", "1966-"),
    
    # Historical
    Mathematician("Egorov", "Real Analysis, Set Theory", "1869-1931"),
    Mathematician("Luzin", "Descriptive Set Theory", "1883-1950"),
    Mathematician("Urysohn", "Topology", "1898-1924"),
]

russian_mathematicians() = RUSSIAN_MATHEMATICIANS

# ═══════════════════════════════════════════════════════════════════════════════
# 3-TUPLE BANDWIDTH COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

struct MathematiciaTuple
    members::NTuple{3, Mathematician}
    colors::NTuple{3, NTuple{3, Float64}}
    combined_fingerprint::UInt64
    combined_color::NTuple{3, Float64}
    bandwidth::Float64
end

function compute_bandwidth(colors::Vector{NTuple{3, Float64}})::Float64
    if length(colors) < 2
        return 0.0
    end
    
    # Method 1: Color diversity (pairwise distances)
    diversity = 0.0
    n = length(colors)
    for i in 1:n, j in i+1:n
        dist = sqrt(sum((colors[i][k] - colors[j][k])^2 for k in 1:3))
        diversity += dist
    end
    diversity /= (n * (n-1) / 2)
    
    # Method 2: Color span (max - min per channel)
    rs = [c[1] for c in colors]
    gs = [c[2] for c in colors]
    bs = [c[3] for c in colors]
    
    span = (maximum(rs) - minimum(rs)) + 
           (maximum(gs) - minimum(gs)) + 
           (maximum(bs) - minimum(bs))
    span /= 3
    
    # Method 3: Entropy-based (variance)
    var_r = var(rs)
    var_g = var(gs)
    var_b = var(bs)
    entropy = sqrt(var_r + var_g + var_b)
    
    # Combined bandwidth
    (diversity + span + entropy) / 3
end

function var(xs::Vector{Float64})
    if length(xs) < 2
        return 0.0
    end
    m = sum(xs) / length(xs)
    sum((x - m)^2 for x in xs) / length(xs)
end

function create_tuple(m1::Mathematician, m2::Mathematician, m3::Mathematician)::MathematiciaTuple
    colors = [m1.color, m2.color, m3.color]
    fp = m1.seed ⊻ m2.seed ⊻ m3.seed
    combined_color = sm64_color(fp)
    bandwidth = compute_bandwidth(colors)
    
    MathematiciaTuple(
        (m1, m2, m3),
        (m1.color, m2.color, m3.color),
        fp,
        combined_color,
        bandwidth
    )
end

function compute_3tuple_bandwidths(mathematicians::Vector{Mathematician})
    tuples = MathematiciaTuple[]
    n = length(mathematicians)
    
    for i in 1:n
        for j in i+1:n
            for k in j+1:n
                push!(tuples, create_tuple(mathematicians[i], mathematicians[j], mathematicians[k]))
            end
        end
    end
    
    tuples
end

function rank_by_bandwidth(tuples::Vector{MathematiciaTuple}; top_n::Int=20)
    sorted = sort(tuples, by=t -> t.bandwidth, rev=true)
    sorted[1:min(top_n, length(sorted))]
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function color_to_ansi(c::NTuple{3, Float64})
    r = Int(round(clamp(c[1], 0, 1) * 255))
    g = Int(round(clamp(c[2], 0, 1) * 255))
    b = Int(round(clamp(c[3], 0, 1) * 255))
    "\e[38;2;$(r);$(g);$(b)m"
end

const RESET = "\e[0m"

function demo_russian_bandwidth()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════════════╗")
    println("║  RUSSIAN MATHEMATICIANS: 3-Tuple Color Bandwidth Ranking                          ║")
    println("║  \"Every mathematician has an originary hue. The combinatorics of brilliance.\"    ║")
    println("╚═══════════════════════════════════════════════════════════════════════════════════╝")
    println()
    
    mathematicians = russian_mathematicians()
    
    # ─── Individual Colors ───
    println("─── Individual Mathematician Colors ───")
    println()
    
    for m in mathematicians[1:min(15, length(mathematicians))]
        c = m.color
        ansi = color_to_ansi(c)
        rgb = "RGB($(Int(round(c[1]*255))), $(Int(round(c[2]*255))), $(Int(round(c[3]*255))))"
        println("  $(ansi)██$(RESET) $(m.name): $rgb — $(m.field)")
    end
    println("  ... and $(length(mathematicians) - 15) more")
    println()
    
    # ─── Compute All 3-Tuples ───
    println("─── Computing All 3-Tuples ───")
    
    tuples = compute_3tuple_bandwidths(mathematicians)
    n_tuples = length(tuples)
    
    println("  Total mathematicians: $(length(mathematicians))")
    println("  Total 3-tuples: $n_tuples (C($(length(mathematicians)), 3))")
    println()
    
    # ─── Top 20 by Bandwidth ───
    println("─── TOP 20 3-TUPLES BY COMBINED COLOR BANDWIDTH ───")
    println()
    
    top = rank_by_bandwidth(tuples; top_n=20)
    
    for (rank, t) in enumerate(top)
        m1, m2, m3 = t.members
        c1, c2, c3 = t.colors
        cc = t.combined_color
        
        a1 = color_to_ansi(c1)
        a2 = color_to_ansi(c2)
        a3 = color_to_ansi(c3)
        ac = color_to_ansi(cc)
        
        println("  #$rank  Bandwidth: $(round(t.bandwidth, digits=4))")
        println("       $(a1)██$(RESET) $(m1.name)")
        println("       $(a2)██$(RESET) $(m2.name)")
        println("       $(a3)██$(RESET) $(m3.name)")
        println("       Combined: $(ac)██$(RESET) 0x$(string(t.combined_fingerprint, base=16)[1:8])...")
        println()
    end
    
    # ─── Bottom 5 (Most Similar) ───
    println("─── BOTTOM 5 (Most Chromatically Similar) ───")
    println()
    
    bottom = sort(tuples, by=t -> t.bandwidth)[1:5]
    
    for (rank, t) in enumerate(bottom)
        m1, m2, m3 = t.members
        println("  #$rank  Bandwidth: $(round(t.bandwidth, digits=4)) — $(m1.name), $(m2.name), $(m3.name)")
    end
    println()
    
    # ─── Statistics ───
    println("─── Statistics ───")
    
    bandwidths = [t.bandwidth for t in tuples]
    mean_bw = sum(bandwidths) / length(bandwidths)
    max_bw = maximum(bandwidths)
    min_bw = minimum(bandwidths)
    std_bw = sqrt(sum((b - mean_bw)^2 for b in bandwidths) / length(bandwidths))
    
    println("  Mean Bandwidth: $(round(mean_bw, digits=4))")
    println("  Std Dev: $(round(std_bw, digits=4))")
    println("  Max: $(round(max_bw, digits=4))")
    println("  Min: $(round(min_bw, digits=4))")
    println("  Range: $(round(max_bw - min_bw, digits=4))")
    println()
    
    # ─── The Champion Tuple ───
    champion = top[1]
    m1, m2, m3 = champion.members
    
    println("═══════════════════════════════════════════════════════════════════════════════════")
    println("  🏆 MAXIMUM BANDWIDTH 3-TUPLE:")
    println()
    println("     $(m1.name) ($(m1.field))")
    println("     $(m2.name) ($(m2.field))")
    println("     $(m3.name) ($(m3.field))")
    println()
    println("     Combined Fingerprint: 0x$(string(champion.combined_fingerprint, base=16))")
    println("     Combined Color: RGB$(Int.(round.(champion.combined_color .* 255)))")
    println("     Bandwidth: $(round(champion.bandwidth, digits=4))")
    println("═══════════════════════════════════════════════════════════════════════════════════")
    
    (tuples=tuples, top=top, champion=champion)
end

end # module RussianMathematiciansBandwidth
