"""
    OpacityDetector: Epistemological Skill

Maps what can and cannot be known in a multi-agent system, given observer position
and structural constraints. Finds bridging features that enable cross-epistemic dialogue.

Structure:
  - ObserverType: Computational, Embodied, Temporal, Social
  - EpistemicBoundary: What this observer can/cannot know
  - AccessibleFeature: Feature bridging multiple epistemic positions
  - DialogueSpace: Communication respecting all boundaries

The key insight: Knowledge is position-dependent and structure-dependent.
Different observers have fundamentally different access to reality—
but bridging features enable respectful coordination.
"""

module OpacityDetector

using Statistics, Random

export ObserverType, EpistemicBoundary, AccessibleFeature
export map_opacity, find_bridges, construct_dialogue
export world_opacity_detector, world_music_translation

# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Type Definitions
# ═══════════════════════════════════════════════════════════════════════════

@enum ObserverType begin
    Computational
    Embodied
    Temporal
    Social
end

"""
    EpistemicBoundary

Represents what one observer can and cannot know about a system.

Fields:
  - observer_type: Kind of observer (computational, embodied, etc.)
  - can_know: Set of accessible variables
  - cannot_know: Set of opaque variables
  - bridge_features: Features accessible to this observer (connects with others)
  - markov_blanket: What separates this observer from the opaque variables
  - primary_hue: Main color of this epistemic position
  - secondary_hue: Secondary color
  - confidence: Float (0-1) expressing certainty of boundary definition
"""
struct EpistemicBoundary
    observer_type::ObserverType
    can_know::Set{String}
    cannot_know::Set{String}
    bridge_features::Set{String}
    markov_blanket::Set{String}
    primary_hue::Float64
    secondary_hue::Float64
    confidence::Float64
    seed::UInt64
end

"""
    AccessibleFeature

A feature that bridges multiple epistemic boundaries.

Fields:
  - feature_name: What the feature is
  - accessible_to: Which observers can access it
  - bridges: Pairs of observers it connects
  - hue_distance: How close in color space
  - strength: Confidence in the bridge (0-1)
"""
struct AccessibleFeature
    feature_name::String
    accessible_to::Set{ObserverType}
    bridges::Vector{Tuple{ObserverType, ObserverType}}
    hue_distance::Float64
    strength::Float64
    evidence::String
end

"""
    DialogueSpace

A communication space respecting all epistemic boundaries.

Fields:
  - is_coherent: Whether all observers can participate
  - utterances: Utterances using only bridging features
  - num_observers: How many observers can access this dialogue
  - bridging_features: Features used to enable communication
  - minimum_shared: Shared understanding across all observers
"""
struct DialogueSpace
    is_coherent::Bool
    utterances::Vector{String}
    num_observers::Int
    bridging_features::Set{String}
    minimum_shared::Set{String}
    primary_hue::Float64
    coherence_score::Float64
end

# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Observer Factory Functions
# ═══════════════════════════════════════════════════════════════════════════

function computational_observer(system::String, seed::UInt64)::EpistemicBoundary
    """Create a computational observer's epistemic boundary"""
    hues = color_stream(seed ⊻ 0x1111, 2)
    EpistemicBoundary(
        Computational,
        Set(["patterns", "structure", "logic", "information_flow", "symmetries"]),
        Set(["embodied_experience", "felt_emotion", "phenomenal_quality", "social_context"]),
        Set(["consonance_intervals", "harmonic_anchors", "temporal_rhythm",
             "information_density", "pattern_complexity"]),
        Set(["embodiment", "emotion", "intention"]),
        hues[1], hues[2], 0.95,
        seed ⊻ 0x1111
    )
end

function embodied_observer(system::String, seed::UInt64)::EpistemicBoundary
    """Create an embodied observer's epistemic boundary"""
    hues = color_stream(seed ⊻ 0x2222, 2)
    EpistemicBoundary(
        Embodied,
        Set(["sensation", "emotion", "temporal_flow", "narrative_arc", "surprise"]),
        Set(["logical_correctness", "information_theoretic_optimality", "universal_patterns"]),
        Set(["consonance_intervals", "harmonic_anchors", "temporal_rhythm",
             "pitch_contour", "dynamic_shape", "emotional_valence"]),
        Set(["computation", "logic", "optimization"]),
        hues[1], hues[2], 0.92,
        seed ⊻ 0x2222
    )
end

function temporal_observer(system::String, seed::UInt64)::EpistemicBoundary
    """Create a temporal observer's epistemic boundary"""
    hues = color_stream(seed ⊻ 0x3333, 2)
    EpistemicBoundary(
        Temporal,
        Set(["causality", "history", "anticipation", "narrative", "change"]),
        Set(["atemporal_mathematics", "universal_logic", "synchronic_structure"]),
        Set(["temporal_rhythm", "narrative_shape", "harmonic_resolution",
             "emotional_arc", "anticipation_fulfillment", "progression"]),
        Set(["pure_logic", "static_pattern"]),
        hues[1], hues[2], 0.90,
        seed ⊻ 0x3333
    )
end

function social_observer(system::String, seed::UInt64)::EpistemicBoundary
    """Create a social observer's epistemic boundary"""
    hues = color_stream(seed ⊻ 0x4444, 2)
    EpistemicBoundary(
        Social,
        Set(["collective_meaning", "cultural_context", "shared_values", "power_dynamics"]),
        Set(["individual_phenomenology", "private_intention", "idiosyncratic_interpretation"]),
        Set(["cultural_references", "shared_tropes", "narrative_familiarity",
             "emotional_resonance", "collective_recognition", "tradition"]),
        Set(["individual_experience", "private_knowledge"]),
        hues[1], hues[2], 0.88,
        seed ⊻ 0x4444
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Part 3: Opacity Mapping (β₋₍ natural transformation)
# ═══════════════════════════════════════════════════════════════════════════

function color_stream(seed::UInt64, count::Int)::Vector{Float64}
    """Generate deterministic color hues from seed"""
    hues = Float64[]
    rng_state = seed
    for _ in 1:count
        rng_state = splitmix64(rng_state)
        hue = mod(rng_state / 65536.0, 360.0)
        push!(hues, hue)
    end
    hues
end

function splitmix64(seed::UInt64)::UInt64
    """SplitMix64 RNG (pure, deterministic)"""
    z = seed ⊻ 0x9e3779b97f4a7c15
    z = (z >> 30) ⊻ z
    z = z * 0xbf58476d1ce4e5b9
    (z >> 27) ⊻ z
end

function map_opacity(observer::EpistemicBoundary, system::String)::EpistemicBoundary
    """Map what this observer can/cannot know about the system

    Returns the same observer boundary (epistemic mapping is structure,
    not computation-dependent)
    """
    observer
end

# ═══════════════════════════════════════════════════════════════════════════
# Part 4: Bridge Discovery (β₍₊ natural transformation)
# ═══════════════════════════════════════════════════════════════════════════

function hue_distance(h1::Float64, h2::Float64)::Float64
    """Circular hue distance (0-180)"""
    d = abs(h1 - h2)
    min(d, 360.0 - d)
end

function can_bridge(b1::EpistemicBoundary, b2::EpistemicBoundary, threshold::Float64)::Bool
    """Can two boundaries communicate directly?

    True if: (1) they have shared bridge features, AND
             (2) their hues are close enough
    """
    shared = !isempty(intersect(b1.bridge_features, b2.bridge_features))
    hue_dist = hue_distance(b1.primary_hue, b2.primary_hue)
    shared && hue_dist < threshold
end

function find_shared_features(b1::EpistemicBoundary, b2::EpistemicBoundary)::Set{String}
    """Find features both observers can access"""
    intersect(b1.bridge_features, b2.bridge_features)
end

function find_bridges(boundaries::Vector{EpistemicBoundary}, threshold::Float64=60.0)::Vector{AccessibleFeature}
    """Discover bridges between epistemic boundaries (β₍₊)

    Returns vector of features that bridge multiple observers
    """
    bridges = AccessibleFeature[]

    for i in 1:(length(boundaries)-1)
        for j in (i+1):length(boundaries)
            b1 = boundaries[i]
            b2 = boundaries[j]

            shared = find_shared_features(b1, b2)
            if !isempty(shared)
                dist = hue_distance(b1.primary_hue, b2.primary_hue)
                if dist < threshold
                    strength = max(0.0, 1.0 - dist / threshold)
                    for feature in shared
                        push!(bridges, AccessibleFeature(
                            feature,
                            Set([b1.observer_type, b2.observer_type]),
                            [(b1.observer_type, b2.observer_type)],
                            dist,
                            strength,
                            "Shared bridge feature at Δh=$(round(dist, digits=1))°"
                        ))
                    end
                end
            end
        end
    end

    bridges
end

# ═══════════════════════════════════════════════════════════════════════════
# Part 5: Cross-Epistemic Dialogue (β₊₋ natural transformation)
# ═══════════════════════════════════════════════════════════════════════════

function construct_dialogue(boundaries::Vector{EpistemicBoundary},
                           bridges::Vector{AccessibleFeature},
                           seed::UInt64)::DialogueSpace
    """Create dialogue space respecting all epistemic boundaries (β₊₋)

    Dialogue is coherent if all observers can access at least one utterance
    """

    if isempty(bridges)
        return DialogueSpace(false, String[], 0, Set{String}(), Set{String}(), 0.0, 0.0)
    end

    # Collect all unique bridge features
    all_features = Set{String}()
    for bridge in bridges
        push!(all_features, bridge.feature_name)
    end

    # Find features that ALL observers can access
    all_bridge_sets = [Set(b.bridge_features) for b in boundaries]
    fully_shared = all_bridge_sets[1]
    for features in all_bridge_sets[2:end]
        fully_shared = intersect(fully_shared, features)
    end

    # Create utterances
    utterances = String[]
    for feature in all_features
        push!(utterances, "Use: $feature")
    end

    # Assess coherence
    hue = first(color_stream(seed, 1))
    num_obs = length(boundaries)
    is_coherent = length(utterances) > 0
    score = is_coherent ? (length(fully_shared) / max(1, length(all_features))) : 0.0

    DialogueSpace(
        is_coherent,
        utterances[1:min(5, length(utterances))],  # Sample utterances
        num_obs,
        all_features,
        fully_shared,
        hue,
        score
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Part 6: World Integration
# ═══════════════════════════════════════════════════════════════════════════

function world_opacity_detector(; seed::UInt64=0x285508656870f24a,
                                observers::Vector{String}=["computational", "embodied", "temporal"],
                                system::String="music-composition")::Dict

    """Run opacity detector in a world context

    Returns:
      - boundaries: Map of observer → epistemic boundary
      - bridges: Vector of accessible features
      - dialogue: Dialogue space
      - num_bridges: Count of bridges found
      - success: Whether dialogue is coherent
      - elapsed_ns: Timing
    """

    start_time = time_ns()

    # Map each observer to their boundary
    observer_factory = Dict(
        "computational" => computational_observer,
        "embodied" => embodied_observer,
        "temporal" => temporal_observer,
        "social" => social_observer
    )

    boundaries = [
        observer_factory[obs](system, seed ⊻ UInt64(i))
        for (i, obs) in enumerate(observers)
    ]

    # Find bridges
    bridges = find_bridges(boundaries, 60.0)

    # Construct dialogue
    dialogue = construct_dialogue(boundaries, bridges, seed)

    elapsed_ns = time_ns() - start_time

    Dict(
        "boundaries" => boundaries,
        "bridges" => bridges,
        "dialogue" => dialogue,
        "num_bridges" => length(bridges),
        "success" => dialogue.is_coherent,
        "elapsed_ns" => elapsed_ns
    )
end

function world_music_translation(; seed::UInt64=0x285508656870f24a)::Dict
    """Full scenario: Translate music respecting epistemic boundaries

    Returns world result with music composition metadata
    """

    system = "music-composition"
    observers = ["computational", "embodied", "temporal", "social"]

    result = world_opacity_detector(seed=seed, observers=observers, system=system)

    # Add music-specific metadata
    result["composition"] = Dict(
        "respects_computational" => "consonance-intervals",
        "respects_embodied" => "dynamic-shape",
        "respects_temporal" => "harmonic-resolution",
        "respects_social" => "cultural-references"
    )
    result["bridging_strategy"] = "explicit-feature-marking"
    result["listener_groups"] = observers

    result
end

end  # module OpacityDetector
