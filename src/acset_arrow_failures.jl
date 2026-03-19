# ACSET ARROW FAILURES: Categorical Obstruction Taxonomy
# =======================================================
#
# "The arrow that fails to exist tells us more than the arrow that does."
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  GAYACSET SUBSTRATE SPECTRUM                                                │
# │                                                                             │
# │  Maximum Bandwidth Layer (safe parallel with abandon):                      │
# │                                                                             │
# │                         ┌─────────────┐                                     │
# │                         │  GayACSet   │ ← Chromatic parallel reflow         │
# │                         └──────┬──────┘                                     │
# │                    ┌───────────┼───────────┐                                │
# │                    ↓           ↓           ↓                                │
# │            ┌───────────┐ ┌──────────┐ ┌──────────┐                          │
# │            │ObsidianAC │ │DuckDBAC  │ │ GeoACSet │                          │
# │            └─────┬─────┘ └────┬─────┘ └────┬─────┘                          │
# │                  │            │            │                                │
# │                  ↓            ↓            ↓                                │
# │            ┌───────────┐ ┌──────────┐ ┌──────────┐                          │
# │            │ LogseqAC  │ │DuckLakeAC│ │TraceAC   │                          │
# │            └───────────┘ └──────────┘ └──────────┘                          │
# │                                                                             │
# │  ARROW FAILURES (↛) and their categorical obstructions:                    │
# │                                                                             │
# │  ↛ = morphism fails to exist                                               │
# │  ⇸ = morphism exists but not natural                                       │
# │  ⥇ = morphism exists but loses information                                 │
# │                                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘

module ACSetArrowFailures

export
    # ACSet types in the spectrum
    ACSetType, GayACSet, ObsidianACSet, DuckDBACSet, DuckLakeACSet,
    SpatialACSet, GeoACSet, TopoACSet, MetricACSet, UltrametricACSet,
    RelativisticACSet, LogseqACSet, TraceACSet,
    
    # Arrow failure types
    ArrowFailure, FailureType, NonExistence, NonNatural, InformationLoss,
    Obstruction, ObstructionClass,
    
    # Research communities
    ResearchCommunity, CommunityExpertise,
    
    # Failure taxonomy
    FailureTaxonomy, arrow_failures, research_lineage,
    community_for_failure, obstruction_cohomology,
    
    # Stigmergy and telepathy
    StigmergyChannel, CyberneticTelepathy, DreamIngress,
    PersistentDiagramFlow,
    
    # Demo
    demo_arrow_failures

# ═══════════════════════════════════════════════════════════════════════════════
# Core PRNG
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)

@inline function sm64(s::UInt64)::Tuple{UInt64, UInt64}
    z = (s + 0x9E3779B97F4A7C15)
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    (z ⊻ (z >> 31), s + 1)
end

@inline function color_from_seed(seed::UInt64)::NTuple{3, Float64}
    r, s1 = sm64(seed)
    g, s2 = sm64(s1)
    b, _  = sm64(s2)
    ((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0)
end

# ═══════════════════════════════════════════════════════════════════════════════
# ACSET TYPES IN THE SPECTRUM
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ACSetType

Types of ACSets in the Gay substrate spectrum.
"""
@enum ACSetType begin
    GayACSet          # Maximum bandwidth, chromatic parallel reflow
    ObsidianACSet     # Knowledge graph collection layer
    DuckDBACSet       # Analytical database substrate
    DuckLakeACSet     # Data lake with versioning
    SpatialACSet      # Abstract spatial structure (generalizes GeoACSet)
    GeoACSet          # Geometric path finding (concrete SpatialACSet)
    TopoACSet         # Topological structure (concrete SpatialACSet)
    MetricACSet       # Metric space structure (concrete SpatialACSet)
    UltrametricACSet  # Ultrametric: d(x,z) ≤ max(d(x,y), d(y,z)) - trees, p-adic, hierarchies
    RelativisticACSet # Lorentz metric (−,+,+,+), lightcones, proper time, causality
    LogseqACSet       # Outliner knowledge graph
    TraceACSet        # Interaction traces (AbstractInteractionTraceACSet)
end

"""
Properties of each ACSet type.
"""
const ACSET_PROPERTIES = Dict(
    GayACSet => (
        bandwidth = :maximum,
        parallelism = :safe_abandon,
        chromatic = true,
        parent = nothing,
        description = "Maximum physics-allowed parallel reflow with chromatic identity"
    ),
    ObsidianACSet => (
        bandwidth = :high,
        parallelism = :graph_local,
        chromatic = false,
        parent = nothing,
        language = :TypeScript,
        description = "Bidirectional links, block references, knowledge graph (TypeScript/Electron)"
    ),
    DuckDBACSet => (
        bandwidth = :columnar,
        parallelism = :vectorized,
        chromatic = false,
        parent = nothing,
        description = "Analytical SQL, columnar storage, embedded OLAP"
    ),
    DuckLakeACSet => (
        bandwidth = :batch,
        parallelism = :partition,
        chromatic = false,
        parent = nothing,
        description = "Data lake with ACID, time travel, schema evolution"
    ),
    SpatialACSet => (
        bandwidth = :spatial,
        parallelism = :spatial_decomposition,
        chromatic = true,
        parent = nothing,
        description = "Abstract spatial structure: position, distance, neighborhood, path"
    ),
    GeoACSet => (
        bandwidth = :spatial,
        parallelism = :r_tree,
        chromatic = true,
        parent = SpatialACSet,
        description = "Geometric: coordinates, path finding, spatial indexing (R-tree, KD-tree)"
    ),
    TopoACSet => (
        bandwidth = :spatial,
        parallelism = :nerve,
        chromatic = true,
        parent = SpatialACSet,
        description = "Topological: open sets, continuity, homotopy, nerve complexes"
    ),
    MetricACSet => (
        bandwidth = :spatial,
        parallelism = :ball_tree,
        chromatic = true,
        parent = SpatialACSet,
        description = "Metric: distance function, balls, Lipschitz maps, Gromov-Hausdorff"
    ),
    UltrametricACSet => (
        bandwidth = :hierarchical,
        parallelism = :tree_decomposition,
        chromatic = true,
        parent = MetricACSet,
        description = "Ultrametric: d(x,z)≤max(d(x,y),d(y,z)), trees, p-adic, dendrograms, all triangles isosceles"
    ),
    RelativisticACSet => (
        bandwidth = :causal,
        parallelism = :lightcone_parallel,  # Spacelike events parallelize freely
        chromatic = true,
        parent = MetricACSet,
        signature = (-1, +1, +1, +1),  # Lorentz metric signature
        description = "Lorentz metric (−,+,+,+): lightcones, proper time τ, causal structure"
    ),
    LogseqACSet => (
        bandwidth = :outline,
        parallelism = :block_level,
        chromatic = false,
        parent = nothing,
        language = :Clojure,
        equivalent = ObsidianACSet,
        description = "Outliner-first knowledge graph (Clojure/ClojureScript) - Clojure equivalent of Obsidian"
    ),
    TraceACSet => (
        bandwidth = :streaming,
        parallelism = :causal,
        chromatic = true,
        parent = nothing,
        description = "Interaction traces, stigmergy, cybernetic feedback"
    ),
)

"""
SpatialACSet hierarchy and their categorical structure.

┌─────────────────────────────────────────────────────────────────────────────┐
│  SPATIALACSET HIERARCHY                                                     │
│                                                                             │
│                      SpatialACSet (abstract)                                │
│                            │                                                │
│         ┌──────────────────┼──────────────────┐                            │
│         ↓                  ↓                  ↓                            │
│     GeoACSet          TopoACSet          MetricACSet                       │
│     (coordinates)     (open sets)        (distances)                       │
│                                           ┌───┴───┐                        │
│                                           ↓       ↓                        │
│                                   UltrametricAC  RelativisticAC            │
│                                   (strong △)     (Lorentz η)               │
│                                                                             │
│  ULTRAMETRIC SPECIAL PROPERTIES:                                           │
│    • d(x,z) ≤ max(d(x,y), d(y,z))  (strong triangle inequality)           │
│    • Every triangle is isosceles (unequal side is shortest)               │
│    • Every point in a ball is its center                                   │
│    • All balls are clopen (closed AND open)                                │
│    • Balls are either disjoint or nested                                   │
│                                                                             │
│  ULTRAMETRIC ARISES IN:                                                    │
│    • p-adic numbers (number theory)                                        │
│    • Phylogenetic trees (biology)                                          │
│    • Hierarchical clustering / dendrograms (ML)                            │
│    • Spin glasses (physics)                                                │
│    • Parsing trees (linguistics)                                           │
│    • LogseqACSet outline hierarchy (!!)                                    │
│                                                                             │
│  RELATIVISTIC SPECIAL PROPERTIES:                                           │
│    • Lorentz metric: ds² = -c²dt² + dx² + dy² + dz²                        │
│    • Proper time τ = ∫√(-ds²) along timelike worldlines                    │
│    • Lightcone structure: timelike (τ² > 0), spacelike (τ² < 0), null      │
│    • Causal ordering: x ≤ y iff y - x is future-pointing causal            │
│    • Parallel OK for spacelike-separated events (no causal conflict)       │
│                                                                             │
│  RELATIVISTIC ARISES IN:                                                    │
│    • Spacetime physics (Minkowski, Schwarzschild, Kerr)                    │
│    • Causal inference (Pearl's do-calculus, DAGs)                          │
│    • TraceACSet when traces have causal ordering                           │
│    • Distributed systems (Lamport clocks are causal = timelike)            │
│    • Quantum field theory (spacelike = commuting operators)                │
│                                                                             │
│  FORGETFUL FUNCTORS (information loss):                                    │
│    GeoACSet → TopoACSet        (forget coordinates, keep topology)         │
│    MetricACSet → TopoACSet     (forget metric, keep induced topology)      │
│    UltrametricACSet → MetricACSet (forget ultrametric, keep metric)        │
│    RelativisticACSet → MetricACSet (forget Lorentz, keep pseudo-metric)    │
│    RelativisticACSet → TopoACSet (forget metric, keep causal topology)     │
│    GeoACSet → MetricACSet      (forget embedding, keep Euclidean metric)   │
│                                                                             │
│  FREE FUNCTORS (add structure):                                            │
│    TopoACSet → GeoACSet        (choose embedding - non-canonical!)         │
│    TopoACSet → MetricACSet     (choose metric - many choices!)             │
│    MetricACSet → UltrametricACSet  (subdominant ultrametric)               │
│    MetricACSet → RelativisticACSet (choose time direction - observer!)     │
│    TraceACSet → RelativisticACSet  (causal trace ordering!)                │
│    LogseqACSet → UltrametricACSet  (outline depth as ultrametric!)         │
│                                                                             │
│  KEY ARROW FAILURES:                                                       │
│    TopoACSet ↛ GeoACSet        : H¹ (embedding may not exist)              │
│    TopoACSet ↛ MetricACSet     : NonUnique (many compatible metrics)       │
│    MetricACSet ↛ UltrametricACSet : NonExistence (not all metrics lift)    │
│    MetricACSet ↛ RelativisticACSet : NonExistence (no natural time dir)    │
│    RelativisticACSet ↛ GeoACSet : NonNatural (Lorentz → Euclidean loses τ) │
│    UltrametricACSet ↛ RelativisticACSet : NonExistence (trees not causal)  │
│    UltrametricACSet ⥇ GeoACSet : InformationLoss (tree → ℝⁿ loses nesting) │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
const SPATIAL_HIERARCHY = Dict(
    :abstract => SpatialACSet,
    :concrete => [GeoACSet, TopoACSet, MetricACSet, UltrametricACSet, RelativisticACSet],
    :forgetful => [
        (GeoACSet, TopoACSet, :induced_topology),
        (MetricACSet, TopoACSet, :metric_topology),
        (GeoACSet, MetricACSet, :euclidean_metric),
        (UltrametricACSet, MetricACSet, :underlying_metric),
        (RelativisticACSet, MetricACSet, :lorentz_to_pseudometric),
        (RelativisticACSet, TopoACSet, :causal_topology),
    ],
    :free => [
        (TopoACSet, GeoACSet, :embedding_choice),
        (TopoACSet, MetricACSet, :metric_choice),
        (MetricACSet, UltrametricACSet, :subdominant_ultrametric),
        (MetricACSet, RelativisticACSet, :observer_time_choice),
        (TraceACSet, RelativisticACSet, :causal_trace_embedding),
        (LogseqACSet, UltrametricACSet, :outline_depth_ultrametric),
    ],
    :special_isos => [
        # LogseqACSet outline structure IS an ultrametric!
        (LogseqACSet, UltrametricACSet, :outline_tree_isomorphism),
        # TraceACSet causal structure IS relativistic when Lamport-ordered
        (TraceACSet, RelativisticACSet, :lamport_causal_isomorphism),
    ],
)

"""
Knowledge Graph ACSet equivalences across language ecosystems.

┌─────────────────────────────────────────────────────────────────────────────┐
│  KNOWLEDGE GRAPH ACSET EQUIVALENCES                                         │
│                                                                             │
│  ObsidianACSet (TypeScript) ≃ LogseqACSet (Clojure)                        │
│                                                                             │
│  Both provide:                                                              │
│    • Bidirectional links ([[wikilinks]])                                   │
│    • Block references                                                       │
│    • Graph visualization                                                    │
│    • Markdown/Org-mode storage                                              │
│    • Local-first, files on disk                                             │
│                                                                             │
│  Differences (arrow failure sources):                                       │
│                                                                             │
│    ObsidianACSet                    LogseqACSet                             │
│    ─────────────                    ───────────                             │
│    Document-first                   Outliner-first (≅ UltrametricACSet)    │
│    Plugin ecosystem (JS)            Open source core                        │
│    Proprietary sync                 Git-based sync                          │
│    Canvas (spatial)                 Whiteboards                             │
│    Properties (YAML)                Properties (EDN/Clojure)                │
│                                                                             │
│  KEY INSIGHT:                                                               │
│    LogseqACSet's outliner structure gives it UltrametricACSet for FREE     │
│    ObsidianACSet must construct ultrametric via folder hierarchy           │
│                                                                             │
│  ARROW: ObsidianACSet → LogseqACSet                                        │
│    Direction: NonNatural (document → outline loses structure)              │
│    Community: KGraph (Hogan et al.)                                         │
│                                                                             │
│  ARROW: LogseqACSet → ObsidianACSet                                        │
│    Direction: InformationLoss (outline → document loses nesting depth)     │
│    Community: KGraph                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
const KNOWLEDGE_GRAPH_EQUIVALENCES = Dict(
    :typescript_clojure => (ObsidianACSet, LogseqACSet),
    :document_outliner => (:document_first, :outliner_first),
    :ultrametric_natural => (false, true),  # Logseq has natural ultrametric
    :sync_model => (:proprietary, :git),
)

# ═══════════════════════════════════════════════════════════════════════════════
# GAY GÖDEL MACHINE: Self-Deciding Spatial Substrate
# ═══════════════════════════════════════════════════════════════════════════════

"""
Gay Gödel Machine substrate selection for Königsberg-Schrödinger bridges.

┌─────────────────────────────────────────────────────────────────────────────┐
│  KÖNIGSBERG-SCHRÖDINGER BRIDGES PROBLEM                                     │
│                                                                             │
│  Königsberg (1736): Can you cross all bridges exactly once?                 │
│    → Graph connectivity, Eulerian paths, cycles required                    │
│    → UltrametricACSet FAILS: trees have no cycles!                         │
│                                                                             │
│  Schrödinger Bridges (1931): Optimal transport with entropy                 │
│    → Find path between distributions minimizing KL divergence               │
│    → Requires probability flow on graph edges                               │
│                                                                             │
│  Königsberg-Schrödinger: Quantum superposition of bridge crossings          │
│    → Explore ALL paths simultaneously until measurement                     │
│    → Collapse to optimal Eulerian path (if exists)                          │
│                                                                             │
│  GAY GÖDEL MACHINE DECISION PROCEDURE:                                      │
│                                                                             │
│    Input: Problem P with structure S                                        │
│    Output: Optimal ACSet type for P                                         │
│                                                                             │
│    1. DETECT CYCLES:                                                        │
│       has_cycles(S) → ¬UltrametricACSet (trees are cycle-free)             │
│                                                                             │
│    2. DETECT HIERARCHY:                                                     │
│       is_hierarchical(S) → UltrametricACSet preferred                      │
│                                                                             │
│    3. DETECT METRIC STRUCTURE:                                              │
│       has_triangle_inequality(S) → MetricACSet                             │
│       has_strong_triangle(S) → UltrametricACSet                            │
│                                                                             │
│    4. DETECT SUPERPOSITION NEED:                                            │
│       needs_parallel_exploration(P) → QuantumACSet (superposition)         │
│                                                                             │
│    5. DETECT OPTIMAL TRANSPORT:                                             │
│       needs_distribution_flow(P) → SchrödingerBridgeACSet                  │
│                                                                             │
│    6. SELF-MODIFY:                                                          │
│       If current choice suboptimal, Gödel-rewrite to better substrate      │
│                                                                             │
│  SUBSTRATE HIERARCHY FOR KÖNIGSBERG-SCHRÖDINGER:                           │
│                                                                             │
│                         GayACSet (maximum bandwidth)                        │
│                              │                                              │
│              ┌───────────────┼───────────────┐                              │
│              ↓               ↓               ↓                              │
│        SpatialACSet    QuantumACSet   TransportACSet                        │
│              │               │               │                              │
│      ┌───────┼───────┐       │               │                              │
│      ↓       ↓       ↓       ↓               ↓                              │
│   GeoAC  MetricAC  TopoAC  SuperposAC  SchrödingerAC                       │
│              │                               │                              │
│              ↓                               │                              │
│      UltrametricAC ←─────────────────────────┘                              │
│              ↑              (when graph is tree)                            │
│              │                                                              │
│         LogseqAC                                                            │
│                                                                             │
│  DECISION: For Königsberg-Schrödinger bridges:                             │
│                                                                             │
│    IF graph has cycles (Königsberg) AND needs optimal transport:            │
│      → SchrödingerBridgeACSet (NOT UltrametricACSet)                       │
│                                                                             │
│    IF graph is tree AND needs optimal transport:                            │
│      → UltrametricACSet + transport on tree = closed form solution!        │
│                                                                             │
│    IF needs to explore all paths before deciding:                           │
│      → QuantumACSet with deferred collapse                                  │
│                                                                             │
│    GÖDEL SELF-MODIFICATION:                                                 │
│      The Gay Gödel machine can rewrite its own substrate choice            │
│      if it proves a better choice exists. This is safe because:            │
│        • Chromatic identity preserved across substrate changes              │
│        • Galois connection ensures no unaccounted paths                     │
│        • CRDT merge handles concurrent self-modifications                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""

"""
Problem characteristics for substrate selection.
"""
struct ProblemCharacteristics
    has_cycles::Bool              # Königsberg: bridges form cycles
    is_hierarchical::Bool         # Tree-like structure
    needs_transport::Bool         # Schrödinger: optimal transport
    needs_superposition::Bool     # Quantum: explore all paths
    has_metric::Bool              # Distance function exists
    metric_is_ultrametric::Bool   # Strong triangle inequality
    has_causal_ordering::Bool     # Lamport-like event ordering
    needs_lightcone::Bool         # Relativistic: spacelike/timelike distinction
    
    seed::UInt64
    color::NTuple{3, Float64}
end

function ProblemCharacteristics(;
    has_cycles::Bool=false,
    is_hierarchical::Bool=false,
    needs_transport::Bool=false,
    needs_superposition::Bool=false,
    has_metric::Bool=true,
    metric_is_ultrametric::Bool=false,
    has_causal_ordering::Bool=false,
    needs_lightcone::Bool=false,
    seed::UInt64=GAY_SEED
)
    ProblemCharacteristics(
        has_cycles, is_hierarchical, needs_transport, needs_superposition,
        has_metric, metric_is_ultrametric, has_causal_ordering, needs_lightcone,
        seed, color_from_seed(seed)
    )
end

"""
Substrate decision by Gay Gödel machine.
"""
struct SubstrateDecision
    problem::ProblemCharacteristics
    chosen_substrate::ACSetType
    
    # Why this choice
    reasoning::Vector{String}
    
    # Alternatives considered
    alternatives::Vector{Tuple{ACSetType, String}}  # (type, why rejected)
    
    # Self-modification potential
    can_self_modify::Bool
    modification_trigger::String  # Condition that would trigger rewrite
    
    seed::UInt64
    color::NTuple{3, Float64}
end

"""
    decide_substrate(p::ProblemCharacteristics) → SubstrateDecision

Gay Gödel machine decides optimal substrate for problem.
"""
function decide_substrate(p::ProblemCharacteristics)
    reasoning = String[]
    alternatives = Tuple{ACSetType, String}[]
    
    # Decision logic
    chosen = if p.has_causal_ordering && p.needs_lightcone
        # Relativistic: causal structure with lightcone
        push!(reasoning, "Has causal ordering → timelike/spacelike distinction")
        push!(reasoning, "Needs lightcone → Lorentz metric (−,+,+,+)")
        push!(alternatives, (MetricACSet, "Rejected: no signature, loses causality"))
        push!(alternatives, (UltrametricACSet, "Rejected: trees have no lightcones"))
        push!(alternatives, (TraceACSet, "Considered: but continuous spacetime needed"))
        RelativisticACSet
        
    elseif p.has_causal_ordering && !p.needs_lightcone
        # Causal but discrete: TraceACSet with Lamport ordering
        push!(reasoning, "Has causal ordering → Lamport timestamps")
        push!(reasoning, "No continuous lightcone needed → discrete traces")
        push!(alternatives, (RelativisticACSet, "Considered: but discrete is sufficient"))
        TraceACSet
        
    elseif p.has_cycles && p.needs_transport
        # Königsberg-Schrödinger: need graph structure + transport
        push!(reasoning, "Problem has cycles → cannot use tree structure")
        push!(reasoning, "Needs optimal transport → Schrödinger bridge formulation")
        push!(alternatives, (UltrametricACSet, "Rejected: cycles violate ultrametric"))
        push!(alternatives, (RelativisticACSet, "Considered: but cycles may violate causality"))
        push!(alternatives, (GeoACSet, "Considered: but transport not natural"))
        MetricACSet  # Would be SchrödingerBridgeACSet if we had it
        
    elseif p.is_hierarchical && p.metric_is_ultrametric
        # Pure hierarchy: ultrametric is perfect
        push!(reasoning, "Hierarchical structure → tree representation")
        push!(reasoning, "Strong triangle inequality satisfied → ultrametric")
        push!(alternatives, (MetricACSet, "Rejected: loses hierarchical structure"))
        push!(alternatives, (RelativisticACSet, "Rejected: trees not causal"))
        UltrametricACSet
        
    elseif p.needs_superposition
        # Quantum: need to explore all paths
        push!(reasoning, "Needs parallel exploration → superposition")
        push!(reasoning, "Defer collapse until measurement")
        push!(alternatives, (MetricACSet, "Rejected: collapses too early"))
        push!(alternatives, (RelativisticACSet, "Considered: spacelike parallelism"))
        GayACSet  # Maximum parallelism = quantum-like
        
    elseif p.has_metric && !p.metric_is_ultrametric
        # General metric space
        push!(reasoning, "Has metric but not ultrametric")
        push!(alternatives, (UltrametricACSet, "Rejected: triangle inequality too strong"))
        push!(alternatives, (RelativisticACSet, "Rejected: no time direction"))
        MetricACSet
        
    elseif p.is_hierarchical
        # Hierarchical but no explicit metric
        push!(reasoning, "Hierarchical → construct ultrametric from depth")
        UltrametricACSet
        
    else
        # Default to most general
        push!(reasoning, "No strong structure detected → maximum flexibility")
        SpatialACSet
    end
    
    # Self-modification conditions
    can_modify = true
    trigger = if chosen == UltrametricACSet
        "Cycle detected in data → switch to MetricACSet"
    elseif chosen == MetricACSet
        "All triangles isosceles → upgrade to UltrametricACSet; time direction found → upgrade to RelativisticACSet"
    elseif chosen == RelativisticACSet
        "CTCs detected → switch to MetricACSet; discrete sufficient → downgrade to TraceACSet"
    elseif chosen == TraceACSet
        "Continuous proper time needed → upgrade to RelativisticACSet"
    else
        "Better substrate proven → Gödel-rewrite"
    end
    
    d_seed = p.seed ⊻ hash(chosen)
    SubstrateDecision(
        p, chosen, reasoning, alternatives,
        can_modify, trigger, d_seed, color_from_seed(d_seed)
    )
end

"""
Königsberg-Schrödinger specific analysis.
"""
function konigsberg_schrodinger_decision()
    # Königsberg: 7 bridges, 4 land masses, need Eulerian path
    # Schrödinger: optimal transport between start/end distributions
    
    problem = ProblemCharacteristics(
        has_cycles = true,           # Bridges form cycles
        is_hierarchical = false,     # Not a tree
        needs_transport = true,      # Optimal path finding
        needs_superposition = true,  # Explore all crossings
        has_metric = true,           # Distance = number of crossings
        metric_is_ultrametric = false  # Cycles break ultrametric!
    )
    
    decision = decide_substrate(problem)
    
    # Additional Königsberg-specific reasoning
    push!(decision.reasoning, "")
    push!(decision.reasoning, "KÖNIGSBERG SPECIFIC:")
    push!(decision.reasoning, "  • 4 vertices (land masses), 7 edges (bridges)")
    push!(decision.reasoning, "  • All vertices have odd degree → no Eulerian path exists")
    push!(decision.reasoning, "  • Schrödinger formulation: minimize entropy of path distribution")
    push!(decision.reasoning, "  • Gay Gödel insight: problem is UNSOLVABLE, but we can find")
    push!(decision.reasoning, "    the path distribution closest to Eulerian")
    
    decision
end

# Export new types
export ProblemCharacteristics, SubstrateDecision, decide_substrate
export konigsberg_schrodinger_decision

# ═══════════════════════════════════════════════════════════════════════════════
# ARROW FAILURE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    FailureType

How an arrow can fail to exist or be well-behaved.
"""
@enum FailureType begin
    NonExistence      # ↛ Arrow doesn't exist at all
    NonNatural        # ⇸ Arrow exists but isn't natural transformation
    InformationLoss   # ⥇ Arrow exists but loses information (not faithful)
    NonUnique         # Arrow exists but choice is non-canonical
    Discontinuous     # Arrow exists pointwise but not continuous
    NonMonotone       # Arrow doesn't preserve order structure
end

"""
    ObstructionClass

Cohomological obstruction to arrow existence.
"""
@enum ObstructionClass begin
    H0_Obstruction    # No global sections (connectivity failure)
    H1_Obstruction    # Čech obstruction (gluing failure)
    H2_Obstruction    # Gerbe obstruction (higher coherence)
    Pi1_Obstruction   # Fundamental group (path non-equivalence)
    Pi2_Obstruction   # Higher homotopy (sphere non-triviality)
    K_Obstruction     # K-theory (stable equivalence failure)
end

"""
    Obstruction

A specific obstruction to an arrow's existence.
"""
struct Obstruction
    class::ObstructionClass
    degree::Int
    description::String
    
    # Vanishing condition
    vanishes_when::String
    
    seed::UInt64
    color::NTuple{3, Float64}
end

function Obstruction(class::ObstructionClass, description::String, vanishes::String;
                     seed::UInt64=GAY_SEED)
    degree = Int(class) + 1
    o_seed = seed ⊻ hash(class) ⊻ hash(description)
    Obstruction(class, degree, description, vanishes, o_seed, color_from_seed(o_seed))
end

"""
    ArrowFailure

A failure of an arrow between ACSet types.
"""
struct ArrowFailure
    source::ACSetType
    target::ACSetType
    direction::Symbol           # :forward, :backward, :both
    
    failure_type::FailureType
    obstruction::Obstruction
    
    # Which research community has studied this
    primary_community::Symbol
    secondary_communities::Vector{Symbol}
    
    # Key papers/results
    key_results::Vector{String}
    
    seed::UInt64
    color::NTuple{3, Float64}
end

# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH COMMUNITIES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ResearchCommunity

A community that has studied categorical arrow failures.
"""
struct ResearchCommunity
    name::Symbol
    full_name::String
    
    # Key figures
    founders::Vector{String}
    current_leaders::Vector{String}
    
    # What they study
    expertise::Vector{Symbol}
    
    # Relationship to ACSets
    acset_relevance::String
    
    seed::UInt64
    color::NTuple{3, Float64}
end

function ResearchCommunity(name::Symbol, full_name::String, 
                           founders::Vector{String}, leaders::Vector{String},
                           expertise::Vector{Symbol}, relevance::String;
                           seed::UInt64=GAY_SEED)
    c_seed = seed ⊻ hash(name)
    ResearchCommunity(name, full_name, founders, leaders, expertise, relevance,
                      c_seed, color_from_seed(c_seed))
end

# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH COMMUNITY DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

const COMMUNITIES = Dict{Symbol, ResearchCommunity}(
    :ACT => ResearchCommunity(
        :ACT, "Applied Category Theory",
        ["Lawvere", "Mac Lane"],
        ["Spivak", "Fong", "Patterson", "Schultz"],
        [:acsets, :polynomial_functors, :operads, :double_categories],
        "Invented ACSets; schema as category, instance as functor"
    ),
    
    :TDA => ResearchCommunity(
        :TDA, "Topological Data Analysis",
        ["Edelsbrunner", "Carlsson"],
        ["Ghrist", "Oudot", "Bubenik", "Curry"],
        [:persistent_homology, :sheaves, :barcodes, :mapper],
        "Persistent diagrams; failure of exact sequences under filtration"
    ),
    
    :HoTT => ResearchCommunity(
        :HoTT, "Homotopy Type Theory",
        ["Voevodsky", "Awodey"],
        ["Riehl", "Shulman", "Rijke", "Licata"],
        [:univalence, :higher_inductive_types, :cubical, :path_equivalence],
        "Path non-equivalence; univalence axiom failure in classical settings"
    ),
    
    :DBT => ResearchCommunity(
        :DBT, "Database Theory",
        ["Codd", "Abiteboul"],
        ["Buneman", "Libkin", "Vianu", "Kolaitis"],
        [:schema_mappings, :data_exchange, :query_containment, :chase],
        "Schema mapping non-existence; data exchange obstructions"
    ),
    
    :Cybernetics => ResearchCommunity(
        :Cybernetics, "Cybernetics & Systems Theory",
        ["Wiener", "Ashby", "Bateson"],
        ["Beer", "Maturana", "Varela", "Luhmann"],
        [:requisite_variety, :autopoiesis, :stigmergy, :feedback],
        "Variety obstructions; Ashby's law as functor non-existence"
    ),
    
    :Concurrency => ResearchCommunity(
        :Concurrency, "Concurrency Theory",
        ["Petri", "Milner"],
        ["Winskel", "Nielsen", "Sassone", "Pratt"],
        [:event_structures, :true_concurrency, :presheaves, :bisimulation],
        "Non-interleaving concurrency; presheaf semantics of parallel composition"
    ),
    
    :CRDT => ResearchCommunity(
        :CRDT, "Conflict-free Replicated Data Types",
        ["Shapiro", "Preguiça"],
        ["Kleppmann", "Gomes", "Balegas"],
        [:semilattices, :causal_consistency, :merge_functions, :delta_crdts],
        "Merge non-associativity; causal ordering failures"
    ),
    
    :Sheaves => ResearchCommunity(
        :Sheaves, "Sheaf Theory & Topos Theory",
        ["Grothendieck", "Lawvere"],
        ["Johnstone", "Lurie", "Moerdijk"],
        [:sites, :topoi, :geometric_morphisms, :classifying_topoi],
        "Gluing failures; sheaf condition violations"
    ),
    
    :KGraph => ResearchCommunity(
        :KGraph, "Knowledge Graphs & Semantic Web",
        ["Berners-Lee", "Hendler"],
        ["Hogan", "Suchanek", "Weikum"],
        [:rdf, :owl, :sparql, :entity_linking, :knowledge_completion],
        "Link prediction failure; schema alignment obstructions"
    ),
    
    :Dream => ResearchCommunity(
        :Dream, "Dream Research & Oneirology",
        ["Freud", "Jung"],
        ["Hobson", "Domhoff", "Zadra", "Revonsuo"],
        [:rem_sleep, :memory_consolidation, :threat_simulation, :continuity],
        "Dream ingress failures; waking-dream boundary crossings"
    ),
    
    :Relativity => ResearchCommunity(
        :Relativity, "Relativity & Causal Structure",
        ["Einstein", "Minkowski"],
        ["Penrose", "Hawking", "Geroch", "Malament", "Sorkin"],
        [:lightcones, :causal_sets, :proper_time, :lorentz_invariance, :global_hyperbolicity],
        "Causal ordering failures; signature obstruction; CTCs"
    ),
)

# ═══════════════════════════════════════════════════════════════════════════════
# ARROW FAILURE TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════════

"""
Complete taxonomy of arrow failures in the GayACSet spectrum.
"""
const ARROW_FAILURES = [
    # ─── GayACSet ↔ ObsidianACSet ───
    ArrowFailure(
        GayACSet, ObsidianACSet, :backward,
        NonNatural,
        Obstruction(H1_Obstruction, 
            "Obsidian links lack inherent chromatic structure",
            "When links carry explicit color metadata"),
        :KGraph, [:ACT, :Sheaves],
        ["Spivak: 'Ologs' (2011)", "Patterson: 'Categorical Data' (2022)"],
        GAY_SEED ⊻ UInt64(1), color_from_seed(GAY_SEED ⊻ UInt64(1))
    ),
    
    ArrowFailure(
        ObsidianACSet, GayACSet, :forward,
        InformationLoss,
        Obstruction(Pi1_Obstruction,
            "Block reference paths not preserved under chromatic projection",
            "When block structure is homotopy-invariant"),
        :HoTT, [:KGraph, :ACT],
        ["Riehl: 'Categorical Homotopy Theory' (2014)"],
        GAY_SEED ⊻ UInt64(2), color_from_seed(GAY_SEED ⊻ UInt64(2))
    ),
    
    # ─── GayACSet ↔ DuckDBACSet ───
    ArrowFailure(
        GayACSet, DuckDBACSet, :backward,
        NonExistence,
        Obstruction(H0_Obstruction,
            "SQL semantics don't support parallel reflow with abandon",
            "Never (fundamental SQL limitation)"),
        :DBT, [:Concurrency, :ACT],
        ["Hellerstein: 'Declarative Imperative' (2010)"],
        GAY_SEED ⊻ UInt64(3), color_from_seed(GAY_SEED ⊻ UInt64(3))
    ),
    
    ArrowFailure(
        DuckDBACSet, GayACSet, :forward,
        InformationLoss,
        Obstruction(K_Obstruction,
            "Columnar layout loses row identity under chromatic lifting",
            "When columns carry stable equivalence structure"),
        :DBT, [:ACT, :CRDT],
        ["Abadi: 'Column-Stores' (2013)"],
        GAY_SEED ⊻ UInt64(4), color_from_seed(GAY_SEED ⊻ UInt64(4))
    ),
    
    # ─── DuckDBACSet ↔ DuckLakeACSet ───
    ArrowFailure(
        DuckDBACSet, DuckLakeACSet, :forward,
        NonUnique,
        Obstruction(H1_Obstruction,
            "Time travel creates non-canonical version selection",
            "When version graph is totally ordered"),
        :DBT, [:CRDT, :Concurrency],
        ["Armbrust: 'Delta Lake' (2020)"],
        GAY_SEED ⊻ UInt64(5), color_from_seed(GAY_SEED ⊻ UInt64(5))
    ),
    
    ArrowFailure(
        DuckLakeACSet, DuckDBACSet, :backward,
        Discontinuous,
        Obstruction(Pi1_Obstruction,
            "Schema evolution breaks query continuity",
            "When schema changes are backwards-compatible"),
        :DBT, [:HoTT, :ACT],
        ["Buneman: 'Data Provenance' (2001)"],
        GAY_SEED ⊻ UInt64(6), color_from_seed(GAY_SEED ⊻ UInt64(6))
    ),
    
    # ─── GeoACSet ↔ LogseqACSet ───
    ArrowFailure(
        GeoACSet, LogseqACSet, :forward,
        NonNatural,
        Obstruction(Pi1_Obstruction,
            "Spatial paths don't naturally correspond to outline hierarchy",
            "When outline reflects spatial embedding"),
        :TDA, [:KGraph, :HoTT],
        ["Ghrist: 'Elementary Applied Topology' (2014)"],
        GAY_SEED ⊻ UInt64(7), color_from_seed(GAY_SEED ⊻ UInt64(7))
    ),
    
    ArrowFailure(
        LogseqACSet, GeoACSet, :backward,
        NonExistence,
        Obstruction(H2_Obstruction,
            "Outline structure lacks intrinsic spatial embedding",
            "When using force-directed layout with stable embedding"),
        :KGraph, [:TDA, :ACT],
        ["Hogan: 'Knowledge Graphs' (2021)"],
        GAY_SEED ⊻ UInt64(8), color_from_seed(GAY_SEED ⊻ UInt64(8))
    ),
    
    # ─── TraceACSet ↔ GayACSet ───
    ArrowFailure(
        TraceACSet, GayACSet, :forward,
        NonMonotone,
        Obstruction(H1_Obstruction,
            "Interaction traces may violate causal monotonicity under chromatic projection",
            "When traces respect Lamport ordering"),
        :Concurrency, [:CRDT, :Cybernetics],
        ["Winskel: 'Event Structures' (1986)"],
        GAY_SEED ⊻ UInt64(9), color_from_seed(GAY_SEED ⊻ UInt64(9))
    ),
    
    ArrowFailure(
        GayACSet, TraceACSet, :backward,
        InformationLoss,
        Obstruction(Pi2_Obstruction,
            "Chromatic parallel reflow erases trace ordering (by design)",
            "When ordering is recoverable from colorgrade"),
        :Cybernetics, [:Concurrency, :CRDT],
        ["Ashby: 'Requisite Variety' (1956)", "Beer: 'Brain of the Firm' (1972)"],
        GAY_SEED ⊻ UInt64(10), color_from_seed(GAY_SEED ⊻ UInt64(10))
    ),
    
    # ─── RelativisticACSet ↔ MetricACSet ───
    ArrowFailure(
        MetricACSet, RelativisticACSet, :forward,
        NonExistence,
        Obstruction(H1_Obstruction,
            "Euclidean metric has no natural time direction; observer choice required",
            "When metric space has distinguished timelike vector field"),
        :Relativity, [:ACT, :Concurrency],
        ["Malament: 'Causal Theories of Time' (1977)", "Penrose: 'Causal Structure' (1972)"],
        GAY_SEED ⊻ UInt64(20), color_from_seed(GAY_SEED ⊻ UInt64(20))
    ),
    
    ArrowFailure(
        RelativisticACSet, MetricACSet, :backward,
        InformationLoss,
        Obstruction(Pi1_Obstruction,
            "Lorentz signature (−,+,+,+) → Euclidean (+,+,+,+) loses causal structure",
            "Never (signature is fundamental)"),
        :Relativity, [:TDA, :ACT],
        ["Hawking & Ellis: 'Large Scale Structure' (1973)"],
        GAY_SEED ⊻ UInt64(21), color_from_seed(GAY_SEED ⊻ UInt64(21))
    ),
    
    # ─── RelativisticACSet ↔ GeoACSet ───
    ArrowFailure(
        RelativisticACSet, GeoACSet, :forward,
        NonNatural,
        Obstruction(H2_Obstruction,
            "Proper time τ along worldlines doesn't embed naturally in Euclidean distance",
            "When restricted to spacelike hypersurface (Cauchy surface)"),
        :Relativity, [:TDA, :ACT],
        ["Geroch: 'Domain of Dependence' (1970)"],
        GAY_SEED ⊻ UInt64(22), color_from_seed(GAY_SEED ⊻ UInt64(22))
    ),
    
    ArrowFailure(
        GeoACSet, RelativisticACSet, :backward,
        NonExistence,
        Obstruction(H1_Obstruction,
            "Euclidean ℝⁿ has no lightcone structure; cannot distinguish timelike/spacelike",
            "When embedding Minkowski slice at t=const"),
        :Relativity, [:ACT, :TDA],
        ["Minkowski: 'Space and Time' (1908)"],
        GAY_SEED ⊻ UInt64(23), color_from_seed(GAY_SEED ⊻ UInt64(23))
    ),
    
    # ─── RelativisticACSet ↔ UltrametricACSet ───
    ArrowFailure(
        UltrametricACSet, RelativisticACSet, :forward,
        NonExistence,
        Obstruction(H0_Obstruction,
            "Tree structure (ultrametric) has no cycles; lightcones require closed causal curves in general",
            "In globally hyperbolic spacetimes (no CTCs)"),
        :Relativity, [:ACT, :HoTT],
        ["Penrose: 'Gravitational Collapse' (1965)", "Hawking: 'Chronology Protection' (1992)"],
        GAY_SEED ⊻ UInt64(24), color_from_seed(GAY_SEED ⊻ UInt64(24))
    ),
    
    ArrowFailure(
        RelativisticACSet, UltrametricACSet, :backward,
        NonExistence,
        Obstruction(Pi1_Obstruction,
            "Causal diamonds don't satisfy strong triangle inequality",
            "When spacetime is totally ordered (1+0 dimensional)"),
        :Relativity, [:ACT, :TDA],
        ["Sorkin: 'Causal Sets' (1987)"],
        GAY_SEED ⊻ UInt64(25), color_from_seed(GAY_SEED ⊻ UInt64(25))
    ),
    
    # ─── RelativisticACSet ↔ TraceACSet (KEY ISOMORPHISM) ───
    ArrowFailure(
        TraceACSet, RelativisticACSet, :forward,
        NonNatural,  # But BECOMES natural under Lamport ordering!
        Obstruction(H1_Obstruction,
            "Interaction traces need causal ordering to embed in lightcone structure",
            "When traces carry Lamport timestamps (then ≅ causal structure)"),
        :Concurrency, [:Relativity, :CRDT],
        ["Lamport: 'Time, Clocks' (1978)", "Mattern: 'Vector Clocks' (1988)"],
        GAY_SEED ⊻ UInt64(26), color_from_seed(GAY_SEED ⊻ UInt64(26))
    ),
    
    ArrowFailure(
        RelativisticACSet, TraceACSet, :backward,
        InformationLoss,
        Obstruction(Pi2_Obstruction,
            "Continuous spacetime → discrete traces loses proper time resolution",
            "When sampling at Planck scale or using causal set discretization"),
        :Relativity, [:Concurrency, :CRDT],
        ["Bombelli et al: 'Causal Sets' (1987)"],
        GAY_SEED ⊻ UInt64(27), color_from_seed(GAY_SEED ⊻ UInt64(27))
    ),
    
    # ─── RelativisticACSet ↔ GayACSet ───
    ArrowFailure(
        RelativisticACSet, GayACSet, :forward,
        NonMonotone,
        Obstruction(H1_Obstruction,
            "Proper time ordering may conflict with chromatic parallel reflow",
            "When chromatic identity respects lightcone (causal colorgrade)"),
        :Relativity, [:Cybernetics, :CRDT],
        ["Wheeler: 'Spacetime Foam' (1955)"],
        GAY_SEED ⊻ UInt64(28), color_from_seed(GAY_SEED ⊻ UInt64(28))
    ),
    
    ArrowFailure(
        GayACSet, RelativisticACSet, :backward,
        NonExistence,
        Obstruction(H0_Obstruction,
            "Maximum bandwidth parallel reflow has no preferred time direction",
            "When using proper time of fiducial observer"),
        :Cybernetics, [:Relativity, :ACT],
        ["Ashby: 'Requisite Variety' (1956)"],
        GAY_SEED ⊻ UInt64(29), color_from_seed(GAY_SEED ⊻ UInt64(29))
    ),
    
    # ─── Stigmergy Channels ───
    ArrowFailure(
        TraceACSet, ObsidianACSet, :forward,
        NonNatural,
        Obstruction(H1_Obstruction,
            "Stigmergic coordination leaves traces that don't map to explicit links",
            "When stigmergy markers are first-class links"),
        :Cybernetics, [:KGraph, :Concurrency],
        ["Grassé: 'Stigmergy' (1959)", "Theraulaz: 'Swarm Intelligence' (1999)"],
        GAY_SEED ⊻ UInt64(11), color_from_seed(GAY_SEED ⊻ UInt64(11))
    ),
    
    # ─── Dream Ingress ───
    ArrowFailure(
        TraceACSet, GayACSet, :both,
        Discontinuous,
        Obstruction(Pi1_Obstruction,
            "Dream ingress creates discontinuous jumps in persistent diagrams",
            "When REM-wake boundary has smooth chromatic transition"),
        :Dream, [:TDA, :Cybernetics],
        ["Hobson: 'Dreaming Brain' (1988)", "Revonsuo: 'Threat Simulation' (2000)"],
        GAY_SEED ⊻ UInt64(12), color_from_seed(GAY_SEED ⊻ UInt64(12))
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# STIGMERGY, TELEPATHY, AND DREAM INGRESS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    StigmergyChannel

Indirect coordination through environment modification.
Arrow failure: explicit communication arrows don't exist.
"""
struct StigmergyChannel
    source::ACSetType
    target::ACSetType
    medium::Symbol              # :pheromone, :color, :trace, :persistent_diagram
    
    # Why direct arrow fails
    direct_arrow_failure::ArrowFailure
    
    # How stigmergy compensates
    coordination_mechanism::String
    
    seed::UInt64
    color::NTuple{3, Float64}
end

"""
    CyberneticTelepathy

Information flow without explicit message passing.
Ashby's variety principle as functor existence condition.
"""
struct CyberneticTelepathy
    sender::ACSetType
    receiver::ACSetType
    
    # Requisite variety analysis
    sender_variety::Int         # log₂ of possible states
    receiver_variety::Int
    channel_capacity::Int
    
    # Ashby's law: variety(receiver) ≥ variety(perturbation)
    ashby_satisfied::Bool
    
    # Arrow that would exist if Ashby satisfied
    potential_arrow::Union{ArrowFailure, Nothing}
    
    seed::UInt64
    color::NTuple{3, Float64}
end

"""
    DreamIngress

Dream state entering persistent diagram computation.
"""
struct DreamIngress
    waking_acset::ACSetType
    dream_acset::ACSetType
    
    # Persistent diagram properties
    birth_times::Vector{Float64}
    death_times::Vector{Float64}
    
    # Discontinuity at wake-dream boundary
    boundary_obstruction::Obstruction
    
    # Ingress success (did dream features persist?)
    features_persisted::Int
    features_lost::Int
    
    seed::UInt64
    color::NTuple{3, Float64}
end

"""
    PersistentDiagramFlow

Flow of persistent homology through ACSet transformations.
"""
struct PersistentDiagramFlow
    source::ACSetType
    target::ACSetType
    
    # Barcodes that survive the transformation
    surviving_bars::Vector{Tuple{Float64, Float64}}
    
    # Barcodes that die (arrow failure for their features)
    dying_bars::Vector{Tuple{Float64, Float64, ArrowFailure}}
    
    # Bottleneck distance (measures arrow quality)
    bottleneck_distance::Float64
    
    seed::UInt64
    color::NTuple{3, Float64}
end

# ═══════════════════════════════════════════════════════════════════════════════
# QUERY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    arrow_failures(source::ACSetType, target::ACSetType) → Vector{ArrowFailure}

Get all arrow failures between two ACSet types.
"""
function arrow_failures(source::ACSetType, target::ACSetType)
    filter(ARROW_FAILURES) do af
        (af.source == source && af.target == target) ||
        (af.source == target && af.target == source && af.direction == :both)
    end
end

"""
    research_lineage(failure::ArrowFailure) → Vector{ResearchCommunity}

Get research communities that have studied this failure.
"""
function research_lineage(failure::ArrowFailure)
    communities = ResearchCommunity[]
    
    if haskey(COMMUNITIES, failure.primary_community)
        push!(communities, COMMUNITIES[failure.primary_community])
    end
    
    for c in failure.secondary_communities
        if haskey(COMMUNITIES, c)
            push!(communities, COMMUNITIES[c])
        end
    end
    
    communities
end

"""
    community_for_failure(failure_type::FailureType, obstruction::ObstructionClass) → Symbol

Get the primary research community for a specific failure type.
"""
function community_for_failure(failure_type::FailureType, obstruction::ObstructionClass)
    # Map failure types to most relevant communities
    type_map = Dict(
        NonExistence => Dict(
            H0_Obstruction => :DBT,
            H1_Obstruction => :Sheaves,
            H2_Obstruction => :HoTT,
        ),
        NonNatural => Dict(
            H1_Obstruction => :ACT,
            Pi1_Obstruction => :HoTT,
        ),
        InformationLoss => Dict(
            Pi1_Obstruction => :TDA,
            Pi2_Obstruction => :Cybernetics,
            K_Obstruction => :ACT,
        ),
        NonUnique => Dict(
            H1_Obstruction => :CRDT,
        ),
        Discontinuous => Dict(
            Pi1_Obstruction => :Dream,
        ),
        NonMonotone => Dict(
            H1_Obstruction => :Concurrency,
        ),
    )
    
    get(get(type_map, failure_type, Dict()), obstruction, :ACT)
end

"""
    obstruction_cohomology(failures::Vector{ArrowFailure}) → Dict

Compute cohomology groups from arrow failures.
"""
function obstruction_cohomology(failures::Vector{ArrowFailure})
    groups = Dict{ObstructionClass, Vector{ArrowFailure}}()
    
    for f in failures
        class = f.obstruction.class
        if !haskey(groups, class)
            groups[class] = ArrowFailure[]
        end
        push!(groups[class], f)
    end
    
    # Compute "dimension" of each cohomology group
    dims = Dict(class => length(failures) for (class, failures) in groups)
    
    (groups = groups, dimensions = dims, total_obstruction = sum(values(dims)))
end

# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE TAXONOMY STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

"""
    FailureTaxonomy

Complete taxonomy of arrow failures with research lineages.
"""
struct FailureTaxonomy
    failures::Vector{ArrowFailure}
    communities::Dict{Symbol, ResearchCommunity}
    
    # Cohomology summary
    obstruction_summary::Dict{ObstructionClass, Int}
    
    # Community coverage
    community_coverage::Dict{Symbol, Int}  # community → number of failures studied
    
    seed::UInt64
    color::NTuple{3, Float64}
    fingerprint::UInt64
end

function FailureTaxonomy()
    failures = ARROW_FAILURES
    communities = COMMUNITIES
    
    # Count obstructions by class
    obs_summary = Dict{ObstructionClass, Int}()
    for f in failures
        class = f.obstruction.class
        obs_summary[class] = get(obs_summary, class, 0) + 1
    end
    
    # Count coverage by community
    comm_coverage = Dict{Symbol, Int}()
    for f in failures
        comm_coverage[f.primary_community] = get(comm_coverage, f.primary_community, 0) + 1
        for c in f.secondary_communities
            comm_coverage[c] = get(comm_coverage, c, 0) + 1
        end
    end
    
    fp = reduce(⊻, f.seed for f in failures; init=GAY_SEED)
    
    FailureTaxonomy(failures, communities, obs_summary, comm_coverage,
                    GAY_SEED, color_from_seed(fp), fp)
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function demo_arrow_failures()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  ACSET ARROW FAILURES: Categorical Obstruction Taxonomy                   ║")
    println("║  \"The arrow that fails to exist tells us more than the arrow that does.\" ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    taxonomy = FailureTaxonomy()
    
    # ─── ACSet Spectrum ───
    println("─── GayACSet Substrate Spectrum ───")
    for acset in instances(ACSetType)
        props = ACSET_PROPERTIES[acset]
        println("  $(acset):")
        println("    Bandwidth: $(props.bandwidth), Parallelism: $(props.parallelism)")
        println("    Chromatic: $(props.chromatic)")
    end
    println()
    
    # ─── Arrow Failures by Type ───
    println("─── Arrow Failures by Type ───")
    for ft in instances(FailureType)
        failures = filter(f -> f.failure_type == ft, taxonomy.failures)
        if !isempty(failures)
            println("  $(ft) ($(length(failures)) failures):")
            for f in failures[1:min(2, end)]
                println("    $(f.source) ↛ $(f.target): $(f.obstruction.description[1:min(50, end)])...")
            end
        end
    end
    println()
    
    # ─── Obstruction Cohomology ───
    println("─── Obstruction Cohomology ───")
    for (class, count) in sort(collect(taxonomy.obstruction_summary), by=x->Int(x[1]))
        println("  $(class): $(count) failures")
    end
    println("  Total obstruction dimension: $(sum(values(taxonomy.obstruction_summary)))")
    println()
    
    # ─── Research Communities ───
    println("─── Research Communities by Coverage ───")
    sorted = sort(collect(taxonomy.community_coverage), by=x->-x[2])
    for (comm, count) in sorted
        if haskey(taxonomy.communities, comm)
            c = taxonomy.communities[comm]
            color = c.color
            println("  $(comm) ($(count) failures): $(c.full_name)")
            println("    Founders: $(join(c.founders[1:min(2, end)], ", "))")
            println("    Color: RGB($(round(color[1], digits=2)), $(round(color[2], digits=2)), $(round(color[3], digits=2)))")
        end
    end
    println()
    
    # ─── Specific Failure Deep Dive ───
    println("─── Deep Dive: GayACSet ↔ DuckDBACSet ───")
    failures = arrow_failures(GayACSet, DuckDBACSet)
    for f in failures
        println("  $(f.direction == :backward ? "←" : "→") $(f.failure_type)")
        println("    Obstruction: $(f.obstruction.class)")
        println("    Description: $(f.obstruction.description)")
        println("    Vanishes when: $(f.obstruction.vanishes_when)")
        println("    Primary: $(f.primary_community)")
        lineage = research_lineage(f)
        println("    Research lineage: $(join([c.full_name for c in lineage], " → "))")
        if !isempty(f.key_results)
            println("    Key result: $(f.key_results[1])")
        end
    end
    println()
    
    # ─── Stigmergy and Dream Ingress ───
    println("─── Stigmergy, Telepathy, and Dream Ingress ───")
    println("  Stigmergy: Indirect coordination when direct arrows fail")
    println("    TraceACSet ⇸ ObsidianACSet via pheromone-like color markers")
    println("    Community: Cybernetics (Grassé, Theraulaz)")
    println()
    println("  Cybernetic Telepathy: Ashby's requisite variety as functor condition")
    println("    Arrow exists iff Variety(receiver) ≥ Variety(perturbation)")
    println("    Community: Cybernetics (Ashby, Beer)")
    println()
    println("  Dream Ingress: Discontinuous jumps in persistent diagrams")
    println("    TraceACSet ↔ GayACSet at REM-wake boundary")
    println("    Community: Dream Research + TDA (Hobson, Ghrist)")
    println()
    
    # ─── Summary ───
    println("─── Summary: How to Make Failed Arrows Work ───")
    println("  1. NonExistence → Change the categories (different ACSet schema)")
    println("  2. NonNatural → Add structure (chromatic metadata)")
    println("  3. InformationLoss → Use adjoint pairs (shadow bits)")
    println("  4. NonUnique → Pick canonical representative (Galois γ)")
    println("  5. Discontinuous → Smooth via persistent homology")
    println("  6. NonMonotone → Embed in event structure with causal order")
    
    taxonomy
end

end # module ACSetArrowFailures
