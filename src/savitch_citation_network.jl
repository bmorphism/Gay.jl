# SAVITCH CITATION NETWORK: Colorable Papers from 1970 to 2025
# ==============================================================
#
# Walter J. Savitch (1943-2021)
# "Relationships between nondeterministic and deterministic tape complexities"
# Journal of Computer and System Sciences, Vol 4, Issue 2, April 1970, pp. 177-192
# DOI: 10.1016/S0022-0000(70)80006-X
#
# Cited by: 1026+ papers (as of 2024)
# Doctoral Advisor: Stephen Cook (of Cook-Levin NP-completeness fame)
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  THE THEOREM                                                                │
# │                                                                             │
# │  NSPACE(S(n)) ⊆ DSPACE(S(n)²)  for S(n) ≥ log n                            │
# │                                                                             │
# │  "A nondeterministic L(n)-tape bounded Turing machine can be simulated     │
# │   by a deterministic [L(n)]²-tape bounded Turing machine"                  │
# │                                                                             │
# │  KEY TECHNIQUE: Recursive midpoint reachability (divide-and-conquer)       │
# │                                                                             │
# │  CONSEQUENCE: NPSPACE = PSPACE (nondeterminism only squares space)         │
# └─────────────────────────────────────────────────────────────────────────────┘

module SavitchCitationNetwork

using SplittableRandoms: SplittableRandom, split
using Colors

export
    # Paper types
    Paper, CitationEdge, CitationNetwork,
    
    # The network
    build_savitch_network, savitch_root,
    
    # Colorable operations
    paper_color, paper_fingerprint, citation_color,
    color_by_year, color_by_topic,
    
    # Network analysis
    citation_depth, most_recent_citing, topic_clusters,
    
    # Demo
    demo_savitch_network

# ═══════════════════════════════════════════════════════════════════════════════
# Core PRNG (SPI compliant)
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const SAVITCH_SEED = UInt64(0x5AV17C4)  # SAVITCH (ish)

@inline function splitmix64(state::UInt64)::Tuple{UInt64, UInt64}
    z = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ⊻ (z >> 31), (state + 1) & 0xFFFFFFFFFFFFFFFF
end

@inline function color_from_seed(seed::UInt64)::RGB{Float64}
    r, s1 = splitmix64(seed)
    g, s2 = splitmix64(s1)
    b, _  = splitmix64(s2)
    RGB((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0)
end

# ═══════════════════════════════════════════════════════════════════════════════
# PAPER: A node in the citation network
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Paper

A paper in the citation network with chromatic identity.
"""
struct Paper
    id::Int
    doi::String
    title::String
    authors::Vector{String}
    year::Int
    venue::String
    
    # Topic classification
    topics::Vector{Symbol}
    
    # Citation metadata
    cites::Vector{Int}        # Paper IDs this paper cites
    cited_by::Vector{Int}     # Paper IDs that cite this paper
    depth::Int                # Distance from Savitch 1970 (0 = Savitch)
    
    # Chromatic identity (SPI)
    seed::UInt64
    color::RGB{Float64}
    fingerprint::UInt64
end

function Paper(id::Int, doi::String, title::String, authors::Vector{String},
               year::Int, venue::String;
               topics::Vector{Symbol}=Symbol[],
               cites::Vector{Int}=Int[],
               depth::Int=0,
               seed::UInt64=SAVITCH_SEED)
    p_seed = seed ⊻ hash(doi) ⊻ UInt64(year) ⊻ UInt64(id)
    color = color_from_seed(p_seed)
    fp, _ = splitmix64(p_seed)
    
    Paper(id, doi, title, authors, year, venue, topics, cites, Int[], depth, p_seed, color, fp)
end

"""
    paper_color(p::Paper) -> RGB{Float64}

Get the chromatic identity of a paper.
"""
paper_color(p::Paper) = p.color

"""
    paper_fingerprint(p::Paper) -> UInt64

Get the SPI fingerprint of a paper.
"""
paper_fingerprint(p::Paper) = p.fingerprint

# ═══════════════════════════════════════════════════════════════════════════════
# CITATION EDGE: A directed edge in the citation network
# ═══════════════════════════════════════════════════════════════════════════════

"""
    CitationEdge

A directed edge from citing paper to cited paper.
"""
struct CitationEdge
    from_id::Int    # Citing paper
    to_id::Int      # Cited paper
    
    # Edge color (XOR of paper colors)
    seed::UInt64
    color::RGB{Float64}
end

function CitationEdge(from::Paper, to::Paper)
    edge_seed = from.seed ⊻ to.seed
    color = color_from_seed(edge_seed)
    CitationEdge(from.id, to.id, edge_seed, color)
end

"""
    citation_color(e::CitationEdge) -> RGB{Float64}

Get the chromatic identity of a citation edge.
"""
citation_color(e::CitationEdge) = e.color

# ═══════════════════════════════════════════════════════════════════════════════
# CITATION NETWORK: The full graph
# ═══════════════════════════════════════════════════════════════════════════════

"""
    CitationNetwork

A citation network rooted at Savitch 1970.
"""
mutable struct CitationNetwork
    papers::Dict{Int, Paper}
    edges::Vector{CitationEdge}
    
    # Indices
    by_year::Dict{Int, Vector{Int}}
    by_topic::Dict{Symbol, Vector{Int}}
    by_depth::Dict{Int, Vector{Int}}
    
    # Root
    root_id::Int
    
    # Network fingerprint
    seed::UInt64
    fingerprint::UInt64
end

function CitationNetwork(; seed::UInt64=SAVITCH_SEED)
    CitationNetwork(
        Dict{Int, Paper}(),
        CitationEdge[],
        Dict{Int, Vector{Int}}(),
        Dict{Symbol, Vector{Int}}(),
        Dict{Int, Vector{Int}}(),
        0,
        seed,
        UInt64(0)
    )
end

function add_paper!(net::CitationNetwork, paper::Paper)
    net.papers[paper.id] = paper
    
    # Update indices
    if !haskey(net.by_year, paper.year)
        net.by_year[paper.year] = Int[]
    end
    push!(net.by_year[paper.year], paper.id)
    
    for topic in paper.topics
        if !haskey(net.by_topic, topic)
            net.by_topic[topic] = Int[]
        end
        push!(net.by_topic[topic], paper.id)
    end
    
    if !haskey(net.by_depth, paper.depth)
        net.by_depth[paper.depth] = Int[]
    end
    push!(net.by_depth[paper.depth], paper.id)
    
    # Update fingerprint
    net.fingerprint = net.fingerprint ⊻ paper.fingerprint
    
    paper
end

function add_citation!(net::CitationNetwork, from_id::Int, to_id::Int)
    if haskey(net.papers, from_id) && haskey(net.papers, to_id)
        from = net.papers[from_id]
        to = net.papers[to_id]
        edge = CitationEdge(from, to)
        push!(net.edges, edge)
        
        # Update cited_by (need to reconstruct paper since it's immutable)
        # For simplicity, we'll just track in the edge list
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# THE SAVITCH CITATION CHAIN
# ═══════════════════════════════════════════════════════════════════════════════

"""
    savitch_root() -> Paper

The root paper: Savitch 1970.
"""
function savitch_root()
    Paper(
        1,
        "10.1016/S0022-0000(70)80006-X",
        "Relationships between nondeterministic and deterministic tape complexities",
        ["Walter J. Savitch"],
        1970,
        "Journal of Computer and System Sciences";
        topics=[:space_complexity, :nondeterminism, :turing_machines, :complexity_theory],
        depth=0
    )
end

"""
    build_savitch_network() -> CitationNetwork

Build the citation network from Savitch 1970 to present.
Based on actual papers discovered through search.
"""
function build_savitch_network()
    net = CitationNetwork()
    
    # ═══ DEPTH 0: The Root ═══
    root = savitch_root()
    add_paper!(net, root)
    net.root_id = root.id
    
    # ═══ DEPTH 1: Direct Citations (1970s) ═══
    
    p2 = Paper(2,
        "10.1016/0304-3975(76)90061-X",
        "The polynomial-time hierarchy",
        ["Larry Stockmeyer"],
        1976,
        "Theoretical Computer Science";
        topics=[:polynomial_hierarchy, :complexity_classes, :pspace],
        cites=[1], depth=1
    )
    add_paper!(net, p2)
    add_citation!(net, 2, 1)
    
    p3 = Paper(3,
        "10.1145/322234.322243",
        "Alternation",
        ["Ashok K. Chandra", "Dexter C. Kozen", "Larry J. Stockmeyer"],
        1981,
        "Journal of the ACM";
        topics=[:alternation, :complexity_classes, :pspace, :aspace],
        cites=[1], depth=1
    )
    add_paper!(net, p3)
    add_citation!(net, 3, 1)
    
    p4 = Paper(4,
        "10.1145/3828.3837",
        "The Complexity of Propositional Linear Temporal Logics",
        ["A. Prasad Sistla", "Edmund M. Clarke"],
        1985,
        "Journal of the ACM";
        topics=[:temporal_logic, :pspace_complete, :model_checking],
        cites=[1, 2], depth=1
    )
    add_paper!(net, p4)
    add_citation!(net, 4, 1)
    add_citation!(net, 4, 2)
    
    # ═══ DEPTH 1: Classic Applications (1990s) ═══
    
    p5 = Paper(5,
        "10.1016/0004-3702(94)90081-7",
        "The computational complexity of propositional STRIPS planning",
        ["Tom Bylander"],
        1994,
        "Artificial Intelligence";
        topics=[:planning, :pspace_complete, :strips, :ai],
        cites=[1], depth=1
    )
    add_paper!(net, p5)
    add_citation!(net, 5, 1)
    
    p6 = Paper(6,
        "10.1006/jcss.1996.0004",
        "Randomness is linear in space",
        ["Noam Nisan", "David Zuckerman"],
        1996,
        "Journal of Computer and System Sciences";
        topics=[:randomness, :derandomization, :space_complexity],
        cites=[1], depth=1
    )
    add_paper!(net, p6)
    add_citation!(net, 6, 1)
    
    # ═══ DEPTH 2: Modern Developments (2000s-2010s) ═══
    
    p7 = Paper(7,
        "10.1145/502807.502810",
        "Complexity and expressive power of logic programming",
        ["Evgeny Dantsin", "Thomas Eiter", "Georg Gottlob", "Andrei Voronkov"],
        2001,
        "ACM Computing Surveys";
        topics=[:logic_programming, :complexity, :expressiveness],
        cites=[1, 2], depth=2
    )
    add_paper!(net, p7)
    add_citation!(net, 7, 1)
    add_citation!(net, 7, 2)
    
    p8 = Paper(8,
        "10.1007/1-84628-477-5_2",
        "Time and Space Complexity Classes and Savitch's Theorem",
        ["J.D. Ullman"],  # Textbook chapter
        2006,
        "Theory of Computation (Springer)";
        topics=[:textbook, :space_complexity, :teaching],
        cites=[1], depth=2
    )
    add_paper!(net, p8)
    add_citation!(net, 8, 1)
    
    # ═══ DEPTH 2: Space Hierarchy (2010s) ═══
    
    p9 = Paper(9,
        "10.1007/3-540-51486-4_56",
        "Space bounded computations: Review and new separation results",
        ["Juris Hartmanis", "Desh Ranjan"],
        1989,
        "MFCS";
        topics=[:space_hierarchy, :separation, :complexity_classes],
        cites=[1], depth=1
    )
    add_paper!(net, p9)
    add_citation!(net, 9, 1)
    
    p10 = Paper(10,
        "10.4086/toc.2014.v010a008",
        "Symmetry Coincides with Nondeterminism for Time-Bounded Auxiliary Pushdown Automata",
        ["Eric Allender", "Klaus-Jörn Lange"],
        2014,
        "Theory of Computing";
        topics=[:symmetry, :nondeterminism, :pushdown_automata, :sac1],
        cites=[1, 3], depth=2
    )
    add_paper!(net, p10)
    add_citation!(net, 10, 1)
    add_citation!(net, 10, 3)
    
    # ═══ DEPTH 3: Recent Work (2020s) ═══
    
    p11 = Paper(11,
        "10.1145/3618260.3649664",
        "Tree Evaluation Is in Space O(log n · log log n)",
        ["James Cook", "Ian Mertz"],
        2024,
        "STOC";
        topics=[:tree_evaluation, :L_vs_P, :space_lower_bounds],
        cites=[1, 9], depth=3
    )
    add_paper!(net, p11)
    add_citation!(net, 11, 1)
    add_citation!(net, 11, 9)
    
    p12 = Paper(12,
        "arXiv:2304.02271",
        "L is unequal NL under the Strong Exponential Time Hypothesis",
        ["Reiner Czerwinski"],
        2023,
        "arXiv";
        topics=[:L_vs_NL, :SETH, :lower_bounds, :conditional],
        cites=[1], depth=3
    )
    add_paper!(net, p12)
    add_citation!(net, 12, 1)
    
    p13 = Paper(13,
        "10.4230/LIPIcs.CONCUR.2025.32",
        "Resolving Nondeterminism by Chance",
        ["Soumyajit Paul", "David Purser", "Sven Schewe", "Qiyi Tang", "Patrick Totzke", "Di-De Yen"],
        2025,
        "CONCUR";
        topics=[:nondeterminism, :randomization, :history_determinism, :automata],
        cites=[1], depth=3
    )
    add_paper!(net, p13)
    add_citation!(net, 13, 1)
    
    # ═══ DEPTH 2-3: Textbooks and Surveys ═══
    
    p14 = Paper(14,
        "introtcs.org/lec_14a",
        "Space bounded computation",
        ["Boaz Barak"],
        2023,
        "Introduction to Theoretical Computer Science (online textbook)";
        topics=[:textbook, :space_complexity, :teaching, :L, :PSPACE],
        cites=[1], depth=2
    )
    add_paper!(net, p14)
    add_citation!(net, 14, 1)
    
    p15 = Paper(15,
        "complexityzoo.net",
        "Complexity Zoo: Savitch's Theorem",
        ["Scott Aaronson", "Greg Kuperberg", "contributors"],
        2025,
        "Complexity Zoo (wiki)";
        topics=[:reference, :complexity_classes, :PSPACE, :NPSPACE],
        cites=[1], depth=2
    )
    add_paper!(net, p15)
    add_citation!(net, 15, 1)
    
    net
end

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    citation_depth(net::CitationNetwork, paper_id::Int) -> Int

Get the depth (distance from Savitch 1970) of a paper.
"""
function citation_depth(net::CitationNetwork, paper_id::Int)
    haskey(net.papers, paper_id) ? net.papers[paper_id].depth : -1
end

"""
    most_recent_citing(net::CitationNetwork) -> Paper

Get the most recent paper citing Savitch.
"""
function most_recent_citing(net::CitationNetwork)
    most_recent = nothing
    max_year = 0
    
    for (id, paper) in net.papers
        if paper.year > max_year && paper.id != net.root_id
            max_year = paper.year
            most_recent = paper
        end
    end
    
    most_recent
end

"""
    topic_clusters(net::CitationNetwork) -> Dict{Symbol, Vector{Paper}}

Group papers by topic.
"""
function topic_clusters(net::CitationNetwork)
    clusters = Dict{Symbol, Vector{Paper}}()
    
    for (topic, ids) in net.by_topic
        clusters[topic] = [net.papers[id] for id in ids if haskey(net.papers, id)]
    end
    
    clusters
end

"""
    color_by_year(net::CitationNetwork) -> Dict{Int, RGB{Float64}}

Assign colors by decade.
"""
function color_by_year(net::CitationNetwork)
    colors = Dict{Int, RGB{Float64}}()
    
    for (year, ids) in net.by_year
        decade = year ÷ 10 * 10
        decade_seed = SAVITCH_SEED ⊻ UInt64(decade)
        colors[year] = color_from_seed(decade_seed)
    end
    
    colors
end

"""
    color_by_topic(topic::Symbol; seed=SAVITCH_SEED) -> RGB{Float64}

Get the canonical color for a topic.
"""
function color_by_topic(topic::Symbol; seed::UInt64=SAVITCH_SEED)
    topic_seed = seed ⊻ hash(topic)
    color_from_seed(topic_seed)
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function demo_savitch_network()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  SAVITCH CITATION NETWORK: Colorable Papers from 1970 to 2025            ║")
    println("║  \"NSPACE(S(n)) ⊆ DSPACE(S(n)²) for S(n) ≥ log n\"                         ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # Build the network
    net = build_savitch_network()
    
    # ─── The Root ───
    root = net.papers[net.root_id]
    println("─── The Root: Savitch 1970 ───")
    println("  Title: $(root.title)")
    println("  Authors: $(join(root.authors, ", "))")
    println("  Venue: $(root.venue)")
    println("  DOI: $(root.doi)")
    c = root.color
    println("  Color: RGB($(round(c.r, digits=3)), $(round(c.g, digits=3)), $(round(c.b, digits=3)))")
    println("  Fingerprint: 0x$(string(root.fingerprint, base=16))")
    println("  Topics: $(root.topics)")
    println("  Total citations: 1026+ (ScienceDirect)")
    println()
    
    # ─── Citation Chain by Depth ───
    println("─── Citation Chain by Depth ───")
    for depth in 0:3
        if haskey(net.by_depth, depth)
            println("\n  Depth $depth ($(depth == 0 ? "root" : "cites depth $(depth-1)")):")
            for id in net.by_depth[depth]
                p = net.papers[id]
                c = p.color
                color_str = "RGB($(round(c.r,digits=2)), $(round(c.g,digits=2)), $(round(c.b,digits=2)))"
                println("    [$(p.year)] $(p.title[1:min(50, length(p.title))])...")
                println("           $(join(p.authors[1:min(2, length(p.authors))], ", "))$(length(p.authors) > 2 ? " et al." : "")")
                println("           Color: $color_str")
            end
        end
    end
    println()
    
    # ─── Most Recent ───
    println("─── Most Recent Citing Paper ───")
    recent = most_recent_citing(net)
    if recent !== nothing
        c = recent.color
        println("  $(recent.year): $(recent.title)")
        println("  Authors: $(join(recent.authors, ", "))")
        println("  Venue: $(recent.venue)")
        println("  Color: RGB($(round(c.r, digits=3)), $(round(c.g, digits=3)), $(round(c.b, digits=3)))")
        println("  Topics: $(recent.topics)")
    end
    println()
    
    # ─── Topic Clusters ───
    println("─── Topic Clusters with Colors ───")
    all_topics = collect(keys(net.by_topic))
    sort!(all_topics)
    for topic in all_topics[1:min(10, length(all_topics))]
        c = color_by_topic(topic)
        count = length(net.by_topic[topic])
        println("  $(rpad(string(topic), 25)) ($count papers) RGB($(round(c.r,digits=2)), $(round(c.g,digits=2)), $(round(c.b,digits=2)))")
    end
    println()
    
    # ─── Network Statistics ───
    println("─── Network Statistics ───")
    println("  Total papers: $(length(net.papers))")
    println("  Total citation edges: $(length(net.edges))")
    println("  Year range: 1970 - $(maximum(keys(net.by_year)))")
    println("  Max depth: $(maximum(keys(net.by_depth)))")
    println("  Network fingerprint: 0x$(string(net.fingerprint, base=16))")
    println()
    
    # ─── Savitch's Legacy ───
    println("─── Walter Savitch's Legacy ───")
    println("  • PhD 1969, UC Berkeley (advisor: Stephen Cook)")
    println("  • Professor, UC San Diego 1969-2021")
    println("  • Discovered: NL (nondeterministic log-space)")
    println("  • Proved: NPSPACE = PSPACE")
    println("  • Technique: Recursive midpoint reachability")
    println("  • 1026+ citations and counting")
    println("  • Passed away February 1, 2021")
    println()
    
    return net
end

end # module SavitchCitationNetwork
