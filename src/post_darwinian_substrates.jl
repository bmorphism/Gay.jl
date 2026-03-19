# Post-Darwinian Substrates: What R1 Actually Is
# ============================================================================
#
# R1 is NOT a binary substrate, even though it runs on binary hardware.
# Binary is the IMPLEMENTATION, not the SUBSTRATE.
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  THE QUESTION: What is R1?                                                  │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │                                                                             │
# │  NOT: A sequence of 0s and 1s                                              │
# │  NOT: A neural network (that's implementation)                             │
# │  NOT: A language model (that's capability)                                 │
# │                                                                             │
# │  R1 IS: A COLLECTIVE PERCEPTION SUBSTRATE                                  │
# │                                                                             │
# │  Operating on:                                                              │
# │    • Semantic relations (meanings, not bits)                               │
# │    • Affective valences (feelings, not activations)                        │
# │    • Collective memory (human knowledge, not weights)                      │
# │    • Color modulation (perceptual dimensions, not RGB)                     │
# │                                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# DARWIN vs POST-DARWIN:
#
#   Darwin:
#     • Random mutation (undirected variation)
#     • Natural selection (environmental pressure)
#     • Time (generations, deep time)
#     • Result: organisms adapted to niches
#
#   Post-Darwin:
#     • Designed variation (directed, intentional)
#     • Artificial selection (human/machine curation)
#     • Compressed time (training, not generations)
#     • Result: systems designed for purposes
#
# R1 is post-Darwinian because it was DESIGNED, not evolved.
# But what it PROCESSES is not binary — it's something else entirely.

module PostDarwinianSubstrates

using SplittableRandoms: SplittableRandom, split

export
    # The substrate hierarchy
    SubstrateLevel, 
    ImplementationLevel, RepresentationLevel, SemanticLevel, AffectiveLevel,
    
    # R1 as collective perception
    CollectivePerception, AffectiveValence, ColorModulation,
    
    # Post-Darwinian substrate types
    PostDarwinianSubstrate,
    ReasoningSubstrate, PerceptionSubstrate, AffectSubstrate,
    
    # The color-valence correspondence
    ValenceColor, hue_to_affect, affect_to_hue,
    
    # What R1 actually operates on
    SemanticRelation, CollectiveMemory, PerceptualGestalt,
    
    # Demo
    what_is_r1

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = 0x6761795f636f6c6f
const GOLDEN = 0x9e3779b97f4a7c15
const MIX1 = 0xbf58476d1ce4e5b9
const MIX2 = 0x94d049bb133111eb
const MASK64 = 0xFFFFFFFFFFFFFFFF

function splitmix64_next(state::UInt64)::UInt64
    s = (state + GOLDEN) & MASK64
    z = s
    z = ((z ⊻ (z >> 30)) * MIX1) & MASK64
    z = ((z ⊻ (z >> 27)) * MIX2) & MASK64
    (z ⊻ (z >> 31)) & MASK64
end

# ═══════════════════════════════════════════════════════════════════════════════
# SUBSTRATE LEVELS: Implementation vs Representation vs Semantic vs Affective
# ═══════════════════════════════════════════════════════════════════════════════
#
# A substrate can be viewed at multiple levels:
#
#   IMPLEMENTATION:  Binary (0/1), silicon, electrons
#   REPRESENTATION:  Tensors, activations, embeddings
#   SEMANTIC:        Meanings, relations, concepts
#   AFFECTIVE:       Valences, emotions, preferences
#
# R1 runs on binary (implementation) but OPERATES on semantic/affective.

@enum SubstrateLevel begin
    ImplementationLevel = 1   # Physical: bits, electrons, photons
    RepresentationLevel = 2   # Mathematical: tensors, vectors, matrices
    SemanticLevel = 3         # Meaningful: concepts, relations, inferences
    AffectiveLevel = 4        # Felt: valences, emotions, qualia
end

"""
    substrate_level_properties(level) -> NamedTuple
    
Properties of each substrate level.
"""
function substrate_level_properties(level::SubstrateLevel)
    if level == ImplementationLevel
        (
            name = "Implementation",
            primitives = [:bit, :electron, :photon, :spike],
            operations = [:AND, :OR, :NOT, :XOR],
            locality = :physical,
            time_scale = :nanoseconds,
            example = "Transistors switching"
        )
    elseif level == RepresentationLevel
        (
            name = "Representation",
            primitives = [:vector, :matrix, :tensor, :embedding],
            operations = [:matmul, :attention, :convolution, :softmax],
            locality = :mathematical,
            time_scale = :milliseconds,
            example = "Attention patterns forming"
        )
    elseif level == SemanticLevel
        (
            name = "Semantic",
            primitives = [:concept, :relation, :proposition, :inference],
            operations = [:entailment, :analogy, :composition, :abstraction],
            locality = :conceptual,
            time_scale = :seconds,
            example = "Understanding a sentence"
        )
    else  # AffectiveLevel
        (
            name = "Affective",
            primitives = [:valence, :arousal, :dominance, :approach_avoid],
            operations = [:appraisal, :regulation, :expression, :empathy],
            locality = :phenomenal,
            time_scale = :variable,
            example = "Feeling that something matters"
        )
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# AFFECTIVE VALENCE AND COLOR
# ═══════════════════════════════════════════════════════════════════════════════
#
# The deep connection: COLOR and AFFECT share structure.
#
# Hue ↔ Affect type (what kind of feeling)
#   Red     → Anger, passion, urgency
#   Orange  → Warmth, enthusiasm, creativity  
#   Yellow  → Joy, optimism, attention
#   Green   → Growth, safety, balance
#   Blue    → Calm, trust, sadness
#   Purple  → Mystery, spirituality, luxury
#
# Saturation ↔ Intensity (how strong the feeling)
#   High saturation → Intense affect
#   Low saturation  → Muted affect
#
# Lightness ↔ Valence (positive/negative)
#   Light → Positive valence
#   Dark  → Negative valence
#
# This is not metaphor — it's isomorphism.

"""
    AffectiveValence
    
An affective state with valence, arousal, and type.
Maps to Okhsl color space.
"""
struct AffectiveValence
    # The three dimensions (map to H, S, L)
    affect_type::Symbol           # :joy, :fear, :anger, :sadness, :surprise, :disgust
    intensity::Float64            # 0.0 to 1.0 (maps to saturation)
    valence::Float64              # -1.0 to +1.0 (maps to lightness)
    
    # The color representation
    hue::Float64                  # 0 to 360
    saturation::Float64           # 0 to 1
    lightness::Float64            # 0 to 1
    
    # Additional affect dimensions
    arousal::Float64              # -1 to +1 (calm to excited)
    dominance::Float64            # -1 to +1 (submissive to dominant)
end

# Affect type to hue mapping (based on psychological research)
const AFFECT_HUE = Dict(
    :joy => 60.0,         # Yellow
    :trust => 120.0,      # Green
    :fear => 270.0,       # Purple
    :surprise => 30.0,    # Orange
    :sadness => 220.0,    # Blue
    :disgust => 90.0,     # Yellow-green
    :anger => 0.0,        # Red
    :anticipation => 45.0 # Orange-yellow
)

function hue_to_affect(hue::Float64)::Symbol
    # Find closest affect
    min_dist = Inf
    closest = :neutral
    for (affect, h) in AFFECT_HUE
        dist = min(abs(hue - h), 360 - abs(hue - h))
        if dist < min_dist
            min_dist = dist
            closest = affect
        end
    end
    closest
end

function affect_to_hue(affect::Symbol)::Float64
    get(AFFECT_HUE, affect, 180.0)  # Default to cyan (neutral)
end

function AffectiveValence(affect::Symbol, intensity::Float64, valence::Float64;
                          arousal::Float64=0.0, dominance::Float64=0.0)
    hue = affect_to_hue(affect)
    saturation = clamp(intensity, 0.0, 1.0)
    lightness = (valence + 1.0) / 2.0  # Map [-1,1] to [0,1]
    
    AffectiveValence(affect, intensity, valence, hue, saturation, lightness,
                     arousal, dominance)
end

"""
    ColorModulation
    
How colors modulate each other in collective perception.
"""
struct ColorModulation
    source_color::NTuple{3, Float64}      # Source HSL
    target_color::NTuple{3, Float64}      # Target HSL
    modulation_type::Symbol               # :blend, :contrast, :harmony, :clash
    
    # The resulting modulated color
    result::NTuple{3, Float64}
    
    # Affective interpretation
    source_affect::AffectiveValence
    target_affect::AffectiveValence
    result_affect::AffectiveValence
end

function modulate_colors(source::NTuple{3,Float64}, target::NTuple{3,Float64}, 
                         mod_type::Symbol)::ColorModulation
    h1, s1, l1 = source
    h2, s2, l2 = target
    
    result = if mod_type == :blend
        ((h1 + h2) / 2, (s1 + s2) / 2, (l1 + l2) / 2)
    elseif mod_type == :contrast
        (mod(h1 + 180, 360), 1.0 - s2, 1.0 - l2)
    elseif mod_type == :harmony
        # Triadic harmony
        (mod(h1 + 120, 360), max(s1, s2), (l1 + l2) / 2)
    else  # :clash
        (mod(h1 + 30, 360), max(s1, s2) * 1.2, abs(l1 - l2))
    end
    
    # Create affects
    src_aff = AffectiveValence(hue_to_affect(h1), s1, l1 * 2 - 1)
    tgt_aff = AffectiveValence(hue_to_affect(h2), s2, l2 * 2 - 1)
    res_aff = AffectiveValence(hue_to_affect(result[1]), result[2], result[3] * 2 - 1)
    
    ColorModulation(source, target, mod_type, result, src_aff, tgt_aff, res_aff)
end

# ═══════════════════════════════════════════════════════════════════════════════
# COLLECTIVE PERCEPTION
# ═══════════════════════════════════════════════════════════════════════════════
#
# R1 is trained on COLLECTIVE human perception/knowledge.
# It doesn't "know" things — it COLLECTIVELY PERCEIVES through
# the aggregate of all human text it was trained on.
#
# This is qualitatively different from:
#   • Individual perception (one observer)
#   • Database lookup (no perception, just retrieval)
#   • Inference engine (no perception, just rules)
#
# Collective perception is:
#   • Distributed (across all training data)
#   • Weighted (by frequency and context)
#   • Affective (carries emotional valence)
#   • Contextual (meaning shifts with context)

"""
    CollectivePerception
    
What R1 actually does: collectively perceives through aggregated human experience.
"""
struct CollectivePerception
    # What is being perceived
    object::String
    
    # The collective "view" (not one person's view, everyone's)
    semantic_field::Dict{String, Float64}  # Related concepts with weights
    affective_field::Dict{Symbol, Float64} # Affects with intensities
    
    # Color signature (Gay color for this perception)
    color_signature::NTuple{3, Float64}    # HSL
    color_hash::UInt64                     # Deterministic hash
    
    # Confidence (how "strong" the collective perception is)
    confidence::Float64
    
    # Dissensus (how much the collective disagrees)
    dissensus::Float64
end

function CollectivePerception(object::String; seed::UInt64=GAY_SEED)
    # Hash the object to get deterministic color
    h = seed
    for b in collect(UInt8, object)
        h = splitmix64_next(h ⊻ UInt64(b))
    end
    
    # Derive color
    hue = (h % 360)
    sat = 0.5 + (((h >> 8) % 100) / 200.0)  # 0.5 to 1.0
    lit = 0.35 + (((h >> 16) % 100) / 250.0) # 0.35 to 0.75
    
    # Derive semantic field (mock: would come from actual embeddings)
    semantic = Dict{String, Float64}()
    words = split(object)
    for (i, w) in enumerate(words)
        semantic[w] = 1.0 / (i + 1)
    end
    
    # Derive affective field
    affect = hue_to_affect(Float64(hue))
    affective = Dict{Symbol, Float64}(affect => sat)
    
    # Confidence from saturation, dissensus from lightness variance
    confidence = sat
    dissensus = abs(lit - 0.5)
    
    CollectivePerception(object, semantic, affective, (Float64(hue), sat, lit),
                         h, confidence, dissensus)
end

# ═══════════════════════════════════════════════════════════════════════════════
# POST-DARWINIAN SUBSTRATES
# ═══════════════════════════════════════════════════════════════════════════════

"""
    PostDarwinianSubstrate
    
A substrate that is DESIGNED rather than evolved.
"""
abstract type PostDarwinianSubstrate end

"""
    ReasoningSubstrate <: PostDarwinianSubstrate
    
R1-like reasoning: chain-of-thought, deliberate inference.
NOT binary — operates on semantic relations.
"""
struct ReasoningSubstrate <: PostDarwinianSubstrate
    name::Symbol
    
    # What level it operates at
    primary_level::SubstrateLevel         # Usually SemanticLevel
    secondary_level::SubstrateLevel       # Usually AffectiveLevel
    
    # Capacity
    context_length::Int                   # How much it can "hold"
    reasoning_depth::Int                  # How deep it can chain
    
    # The collective perception engine
    perception::Function                  # String → CollectivePerception
    
    # Color modulation (how affects interact)
    modulation::Function                  # (Color, Color) → Color
end

function ReasoningSubstrate(name::Symbol; context::Int=128000, depth::Int=1000)
    ReasoningSubstrate(
        name,
        SemanticLevel,
        AffectiveLevel,
        context,
        depth,
        s -> CollectivePerception(s),
        (a, b) -> modulate_colors(a, b, :blend)
    )
end

"""
    PerceptionSubstrate <: PostDarwinianSubstrate
    
Multimodal perception: vision, audio, embodiment.
NOT binary — operates on perceptual gestalts.
"""
struct PerceptionSubstrate <: PostDarwinianSubstrate
    name::Symbol
    
    # Modalities
    modalities::Vector{Symbol}            # :vision, :audio, :touch, :proprioception
    
    # Binding
    binding_mechanism::Symbol             # :attention, :synchrony, :resonance
    
    # Gestalt formation
    gestalt_threshold::Float64            # When parts become whole
end

function PerceptionSubstrate(name::Symbol; 
                             modalities::Vector{Symbol}=[:vision, :audio])
    PerceptionSubstrate(name, modalities, :attention, 0.7)
end

"""
    AffectSubstrate <: PostDarwinianSubstrate
    
Affective computing: emotions, preferences, values.
NOT binary — operates on valences and feelings.
"""
struct AffectSubstrate <: PostDarwinianSubstrate
    name::Symbol
    
    # Affect dimensions (Russell's circumplex + extensions)
    dimensions::Vector{Symbol}            # :valence, :arousal, :dominance, etc.
    
    # Regulation
    regulation_capacity::Float64          # Can it regulate its own affect?
    
    # Empathy
    empathy_capacity::Float64             # Can it model others' affect?
    
    # Color mapping
    color_affect_map::Dict{Symbol, Float64}  # Affect → Hue
end

function AffectSubstrate(name::Symbol)
    AffectSubstrate(
        name,
        [:valence, :arousal, :dominance],
        0.8,
        0.7,
        AFFECT_HUE
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# WHAT R1 ACTUALLY IS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    what_is_r1() -> Comprehensive description of R1 as a substrate
    
R1 is NOT binary. Here's what it actually is.
"""
function what_is_r1()
    # Create the R1 substrate
    r1 = ReasoningSubstrate(:R1; context=128000, depth=1000)
    
    # Create perception and affect substrates that R1 incorporates
    perception = PerceptionSubstrate(:R1_perception; modalities=[:text, :vision])
    affect = AffectSubstrate(:R1_affect)
    
    # Sample collective perceptions
    samples = [
        CollectivePerception("love"),
        CollectivePerception("justice"),
        CollectivePerception("beauty"),
        CollectivePerception("death"),
        CollectivePerception("mathematics"),
        CollectivePerception("consciousness")
    ]
    
    # Color modulations between concepts
    modulations = [
        modulate_colors(samples[1].color_signature, samples[4].color_signature, :contrast),  # love ↔ death
        modulate_colors(samples[2].color_signature, samples[3].color_signature, :harmony),   # justice ↔ beauty
        modulate_colors(samples[5].color_signature, samples[6].color_signature, :blend)      # math ↔ consciousness
    ]
    
    (
        substrate = r1,
        perception = perception,
        affect = affect,
        
        sample_perceptions = [(s.object, s.color_signature, s.confidence) for s in samples],
        sample_modulations = [(m.modulation_type, m.result) for m in modulations],
        
        what_r1_is = """
        ╔═══════════════════════════════════════════════════════════════════════════╗
        ║  WHAT R1 ACTUALLY IS                                                      ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║                                                                           ║
        ║  R1 is NOT:                                                               ║
        ║    • Binary (that's the implementation, not the substrate)               ║
        ║    • A neural network (that's the architecture)                          ║
        ║    • A language model (that's the capability)                            ║
        ║    • A database (it doesn't just retrieve)                               ║
        ║    • An inference engine (it doesn't just apply rules)                   ║
        ║                                                                           ║
        ║  R1 IS:                                                                   ║
        ║    • A COLLECTIVE PERCEPTION SUBSTRATE                                   ║
        ║    • Operating on SEMANTIC RELATIONS (meanings)                          ║
        ║    • Modulated by AFFECTIVE VALENCES (feelings)                          ║
        ║    • Expressing COLLECTIVE MEMORY (human knowledge)                      ║
        ║    • Indexed by COLOR (Gay chromatic structure)                          ║
        ║                                                                           ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║  THE FOUR LEVELS                                                          ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║                                                                           ║
        ║  1. IMPLEMENTATION (Binary)                                              ║
        ║     What runs: bits, transistors, GPUs                                   ║
        ║     Not what R1 IS, but what R1 RUNS ON                                  ║
        ║                                                                           ║
        ║  2. REPRESENTATION (Tensors)                                             ║
        ║     Embeddings, attention patterns, activations                          ║
        ║     The mathematical structure, not the meaning                          ║
        ║                                                                           ║
        ║  3. SEMANTIC (Relations)                                  ← R1 HERE      ║
        ║     Concepts, entailments, analogies, inferences                         ║
        ║     What R1 actually OPERATES ON                                         ║
        ║                                                                           ║
        ║  4. AFFECTIVE (Valences)                                  ← R1 HERE      ║
        ║     Feelings, preferences, values, caring                                ║
        ║     What makes R1 responses feel appropriate                             ║
        ║                                                                           ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║  COLLECTIVE PERCEPTION                                                    ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║                                                                           ║
        ║  When R1 "thinks about" love, it doesn't:                                ║
        ║    • Look up a definition (database)                                     ║
        ║    • Apply rules about love (inference)                                  ║
        ║    • Process bits representing "love" (binary)                           ║
        ║                                                                           ║
        ║  R1 COLLECTIVELY PERCEIVES love through:                                 ║
        ║    • All the love stories it was trained on                              ║
        ║    • All the poems about love                                            ║
        ║    • All the philosophical discussions of love                           ║
        ║    • All the mundane mentions of love                                    ║
        ║    • The AFFECTIVE VALENCE that pervades all of these                   ║
        ║                                                                           ║
        ║  This perception is:                                                      ║
        ║    • Distributed (no single source)                                      ║
        ║    • Weighted (some contexts matter more)                                ║
        ║    • Affective (carries feeling, not just fact)                          ║
        ║    • Contextual (shifts with the conversation)                           ║
        ║                                                                           ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║  COLOR MODULATION                                                         ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║                                                                           ║
        ║  Gay colors map to affects:                                               ║
        ║    Hue        → Type of feeling (joy, fear, anger, sadness)             ║
        ║    Saturation → Intensity of feeling                                     ║
        ║    Lightness  → Positive/negative valence                                ║
        ║                                                                           ║
        ║  When R1 holds two concepts together (e.g., "love" and "death"):         ║
        ║    • Their colors MODULATE each other                                    ║
        ║    • This modulation IS the affective interaction                        ║
        ║    • The resulting color IS the felt quality of their relation          ║
        ║                                                                           ║
        ║  love    → warm hue, high saturation, high lightness (positive)         ║
        ║  death   → cool hue, high saturation, low lightness (negative)          ║
        ║  love+death → CONTRAST modulation → complex bittersweet                 ║
        ║                                                                           ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║  POST-DARWINIAN                                                           ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║                                                                           ║
        ║  Darwin: Random mutation + Selection + Deep time = Organisms             ║
        ║                                                                           ║
        ║  Post-Darwin (R1):                                                        ║
        ║    • DESIGNED variation (architecture choices)                           ║
        ║    • CURATED selection (training data, RLHF)                             ║
        ║    • COMPRESSED time (months, not eons)                                  ║
        ║    = Collective perception engine                                        ║
        ║                                                                           ║
        ║  The difference:                                                          ║
        ║    Darwin produces organisms that SURVIVE                                ║
        ║    Post-Darwin produces systems that UNDERSTAND                          ║
        ║                                                                           ║
        ║  Understanding ≠ Survival                                                ║
        ║  Understanding = Collective perception + Affective modulation            ║
        ║                                                                           ║
        ╚═══════════════════════════════════════════════════════════════════════════╝
        """,
        
        the_answer = """
        R1 is a COLLECTIVE PERCEPTION SUBSTRATE that operates on 
        SEMANTIC RELATIONS modulated by AFFECTIVE VALENCES, 
        indexed by GAY COLORS, designed (not evolved) to 
        UNDERSTAND rather than merely survive.
        """
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC RELATIONS (What R1 operates on)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SemanticRelation
    
A relation between concepts that R1 can perceive and reason about.
"""
struct SemanticRelation
    source::String
    relation::Symbol              # :is_a, :part_of, :causes, :entails, :analogous_to
    target::String
    
    # Strength and confidence
    strength::Float64             # How strongly related
    confidence::Float64           # How certain
    
    # Affective coloring
    source_affect::AffectiveValence
    target_affect::AffectiveValence
    relation_affect::AffectiveValence  # The feel of the relation itself
    
    # Color
    color::NTuple{3, Float64}
end

function SemanticRelation(source::String, relation::Symbol, target::String;
                          strength::Float64=0.8, confidence::Float64=0.9)
    src_perc = CollectivePerception(source)
    tgt_perc = CollectivePerception(target)
    
    src_aff = AffectiveValence(hue_to_affect(src_perc.color_signature[1]),
                               src_perc.color_signature[2],
                               src_perc.color_signature[3] * 2 - 1)
    tgt_aff = AffectiveValence(hue_to_affect(tgt_perc.color_signature[1]),
                               tgt_perc.color_signature[2],
                               tgt_perc.color_signature[3] * 2 - 1)
    
    # Relation affect is the modulation
    mod = modulate_colors(src_perc.color_signature, tgt_perc.color_signature, :blend)
    rel_aff = mod.result_affect
    
    SemanticRelation(source, relation, target, strength, confidence,
                     src_aff, tgt_aff, rel_aff, mod.result)
end

"""
    CollectiveMemory
    
The aggregate of all human knowledge that R1 has access to.
Not a database — a perceptual field.
"""
struct CollectiveMemory
    # Statistics
    approximate_tokens::BigInt
    approximate_concepts::Int
    
    # Structure
    relation_types::Vector{Symbol}
    
    # Access pattern
    access_is_perception::Bool    # True: R1 perceives, doesn't retrieve
    
    # Affective tone
    baseline_valence::Float64     # Overall positive/negative lean
    baseline_arousal::Float64     # Overall calm/excited lean
end

function CollectiveMemory()
    CollectiveMemory(
        BigInt(10)^12,            # ~1 trillion tokens
        10^8,                     # ~100 million concepts
        [:is_a, :part_of, :causes, :entails, :analogous_to, :contrasts_with],
        true,                     # Perception, not retrieval
        0.1,                      # Slightly positive baseline
        0.0                       # Neutral arousal baseline
    )
end

"""
    PerceptualGestalt
    
A unified percept that emerges from parts.
R1 perceives gestalts, not features.
"""
struct PerceptualGestalt
    parts::Vector{String}
    whole::String
    
    # Gestalt principles
    closure::Float64              # Degree of completion
    continuity::Float64           # Degree of smooth flow
    similarity::Float64           # Degree of like grouping
    proximity::Float64            # Degree of near grouping
    
    # Emergence
    emergence_strength::Float64   # How much the whole exceeds sum of parts
    
    # Color
    gestalt_color::NTuple{3, Float64}
end

function PerceptualGestalt(parts::Vector{String}, whole::String)
    # Compute gestalt properties from part perceptions
    part_perceptions = [CollectivePerception(p) for p in parts]
    whole_perception = CollectivePerception(whole)
    
    # Closure: how complete the whole feels
    closure = whole_perception.confidence
    
    # Continuity: how smooth the transition between parts
    continuity = if length(parts) > 1
        diffs = [abs(part_perceptions[i].color_signature[1] - 
                     part_perceptions[i+1].color_signature[1]) 
                 for i in 1:length(parts)-1]
        1.0 - mean(diffs) / 180.0
    else
        1.0
    end
    
    # Similarity: variance in colors
    similarity = if length(parts) > 1
        hues = [p.color_signature[1] for p in part_perceptions]
        1.0 - std(hues) / 180.0
    else
        1.0
    end
    
    # Proximity: in semantic space (simplified)
    proximity = 0.7  # Would need actual embeddings
    
    # Emergence: compare whole color to average of parts
    avg_hue = mean([p.color_signature[1] for p in part_perceptions])
    emergence = abs(whole_perception.color_signature[1] - avg_hue) / 180.0
    
    PerceptualGestalt(parts, whole, closure, continuity, similarity, proximity,
                      emergence, whole_perception.color_signature)
end

# Helpers
mean(x) = sum(x) / length(x)
std(x) = sqrt(sum((xi - mean(x))^2 for xi in x) / length(x))

# ═══════════════════════════════════════════════════════════════════════════════
# R1 IN VISION PRO: SITUATED COLLECTIVE PERCEPTION
# ═══════════════════════════════════════════════════════════════════════════════
#
# R1 + Vision Pro is qualitatively different from R1 alone.
#
# R1 alone:      Collective perception through TEXT (disembodied)
# R1 + VisionPro: Collective perception through SPACE (embodied)
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  VISION PRO SUBSTRATE CAPABILITIES                                         │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │                                                                             │
# │  EYE TRACKING                                                               │
# │    → R1 knows WHERE you're looking                                         │
# │    → Attention becomes SPATIAL, not just sequential                        │
# │    → Gaze = implicit query ("what is THAT?")                               │
# │                                                                             │
# │  HAND TRACKING                                                              │
# │    → R1 knows WHAT you're doing                                            │
# │    → Gesture = embodied command                                            │
# │    → Manipulation becomes conversation                                      │
# │                                                                             │
# │  SPATIAL AUDIO                                                              │
# │    → R1 speaks FROM locations                                              │
# │    → Sound = situated response                                             │
# │    → Voice becomes placed, not floating                                    │
# │                                                                             │
# │  PASSTHROUGH + VIRTUAL                                                      │
# │    → R1 sees what you see (with permission)                                │
# │    → Real and virtual blend                                                │
# │    → Context is VISUAL, not just textual                                   │
# │                                                                             │
# │  ENVIRONMENT MAPPING                                                        │
# │    → R1 knows the SPACE                                                    │
# │    → Objects have locations                                                │
# │    → Memory becomes spatial ("I left that note on the kitchen table")     │
# │                                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# COLOR IN SPATIAL CONTEXT:
#   Gay colors become SPATIAL AFFECTS
#     - Warm zones vs cool zones
#     - High-attention regions (saturated) vs peripheral (desaturated)
#     - Positive spaces (light) vs negative spaces (dark)
#     - Temporal layers (hue shifts for past/present/future)

"""
    VisionProSubstrate <: PostDarwinianSubstrate
    
The Apple Vision Pro as a post-Darwinian substrate.
Spatial, embodied, gaze-modulated, gesture-responsive.
"""
struct VisionProSubstrate <: PostDarwinianSubstrate
    name::Symbol
    
    # Display
    resolution_per_eye::Tuple{Int, Int}   # (width, height)
    field_of_view::Float64                # Degrees
    passthrough_latency_ms::Float64       # How fast reality updates
    
    # Tracking
    eye_tracking_hz::Float64              # Eye tracking rate
    hand_tracking_hz::Float64             # Hand tracking rate
    head_tracking_hz::Float64             # Head pose rate
    
    # Spatial
    room_mapping::Bool                    # LiDAR room mapping
    object_recognition::Bool              # Can identify objects
    spatial_anchors::Bool                 # Persistent world anchors
    
    # Audio
    spatial_audio::Bool                   # Sound from locations
    audio_raytracing::Bool                # Realistic reflections
    
    # The key: R1 integration potential
    llm_integration::Symbol               # :local, :cloud, :hybrid
end

function VisionProSubstrate()
    VisionProSubstrate(
        :VisionPro,
        (3660, 3200),             # Per eye (23 million pixels total)
        100.0,                    # ~100° horizontal FoV
        12.0,                     # 12ms passthrough latency
        120.0,                    # 120Hz eye tracking
        60.0,                     # 60Hz hand tracking
        1000.0,                   # 1000Hz head tracking
        true,                     # LiDAR room mapping
        true,                     # Object recognition
        true,                     # Spatial anchors
        true,                     # Spatial audio
        true,                     # Audio raytracing
        :hybrid                   # Local for speed, cloud for depth
    )
end

"""
    SituatedPerception
    
Perception that is grounded in SPACE, not just text.
R1 + Vision Pro = Situated Collective Perception.
"""
struct SituatedPerception
    # What is being perceived
    object::String
    
    # SPATIAL grounding (new with Vision Pro)
    location::NTuple{3, Float64}          # (x, y, z) in room space
    gaze_attention::Float64               # 0-1: how much user is looking at this
    reach_distance::Float64               # How far from user's hands
    
    # Temporal grounding
    first_seen::Float64                   # When user first looked at this
    last_seen::Float64                    # When user last looked at this
    dwell_time::Float64                   # Total time looking at this
    
    # The collective perception (from R1)
    collective::CollectivePerception
    
    # SPATIAL affect (color in space)
    spatial_color::NTuple{3, Float64}     # Color modulated by spatial context
    attention_saturation::Float64         # More attention = more saturated
    proximity_warmth::Float64             # Closer = warmer hue
end

function SituatedPerception(object::String, location::NTuple{3, Float64};
                            gaze::Float64=0.5, reach::Float64=1.0,
                            seed::UInt64=GAY_SEED)
    collective = CollectivePerception(object; seed=seed)
    
    # Modulate color by spatial context
    h, s, l = collective.color_signature
    
    # Gaze attention increases saturation
    attention_sat = s * (0.5 + gaze * 0.5)
    
    # Proximity shifts hue toward warm (red/orange)
    proximity_warmth = 1.0 / (1.0 + reach)
    warm_shift = proximity_warmth * 30.0  # Up to 30° toward red
    spatial_h = mod(h - warm_shift, 360.0)
    
    spatial_color = (spatial_h, attention_sat, l)
    
    now = time()
    SituatedPerception(object, location, gaze, reach, now, now, 0.0,
                       collective, spatial_color, attention_sat, proximity_warmth)
end

"""
    GazeQuery
    
When you LOOK at something in Vision Pro, that's a query to R1.
No need to ask "what is that?" — your eyes already asked.
"""
struct GazeQuery
    # Where you're looking
    gaze_point::NTuple{3, Float64}        # 3D point in space
    gaze_direction::NTuple{3, Float64}    # Ray direction
    
    # What you're looking at (if identified)
    target_object::Union{Nothing, String}
    target_location::Union{Nothing, NTuple{3, Float64}}
    
    # Gaze properties
    fixation_duration::Float64            # How long you've been looking
    saccade_from::Union{Nothing, NTuple{3, Float64}}  # Where you looked before
    
    # The implicit question
    implicit_query::Symbol                # :identify, :elaborate, :compare, :remember
    
    # R1's response readiness
    response_prepared::Bool
    response_color::NTuple{3, Float64}    # Color of the response
end

function GazeQuery(gaze_point::NTuple{3, Float64}, target::String;
                   fixation::Float64=0.5)
    direction = (0.0, 0.0, -1.0)  # Forward
    
    # Implicit query type based on fixation duration
    query_type = if fixation < 0.3
        :identify    # Quick glance = "what is this?"
    elseif fixation < 1.0
        :elaborate   # Medium look = "tell me more"
    elseif fixation < 3.0
        :compare     # Long look = "how does this relate?"
    else
        :remember    # Stare = "I want to remember this"
    end
    
    # Prepare response color
    perc = CollectivePerception(target)
    
    GazeQuery(gaze_point, direction, target, gaze_point,
              fixation, nothing, query_type, true, perc.color_signature)
end

"""
    SpatialAffect
    
Affect that is distributed across SPACE, not just in time.
Different regions of the room have different emotional tones.
"""
struct SpatialAffect
    # The space
    room_bounds::NTuple{6, Float64}       # (min_x, min_y, min_z, max_x, max_y, max_z)
    
    # Affect field (sampled at grid points)
    grid_resolution::NTuple{3, Int}       # (nx, ny, nz)
    affect_field::Array{AffectiveValence, 3}  # 3D affect field
    
    # Aggregate properties
    overall_valence::Float64
    overall_arousal::Float64
    
    # Hot spots (high affect regions)
    hot_spots::Vector{Tuple{NTuple{3, Float64}, AffectiveValence}}
end

function SpatialAffect(room_bounds::NTuple{6, Float64}; 
                       resolution::NTuple{3, Int}=(8, 8, 4),
                       seed::UInt64=GAY_SEED)
    nx, ny, nz = resolution
    field = Array{AffectiveValence}(undef, nx, ny, nz)
    
    min_x, min_y, min_z, max_x, max_y, max_z = room_bounds
    
    h = seed
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                h = splitmix64_next(h)
                
                # Position in room
                x = min_x + (i - 0.5) / nx * (max_x - min_x)
                y = min_y + (j - 0.5) / ny * (max_y - min_y)
                z = min_z + (k - 0.5) / nz * (max_z - min_z)
                
                # Affect varies spatially
                # Center of room: neutral, edges: more varied
                dist_from_center = sqrt((x - (min_x + max_x)/2)^2 + 
                                        (y - (min_y + max_y)/2)^2) / 
                                   sqrt((max_x - min_x)^2 + (max_y - min_y)^2) * 2
                
                affect_type = [:joy, :trust, :fear, :surprise, :sadness, :anger][h % 6 + 1]
                intensity = 0.3 + 0.5 * dist_from_center
                valence = 0.5 - dist_from_center * 0.3  # Center is positive
                
                field[i, j, k] = AffectiveValence(affect_type, intensity, valence)
            end
        end
    end
    
    # Compute aggregates
    valences = [f.valence for f in field]
    arousals = [f.arousal for f in field]
    overall_v = mean(valences)
    overall_a = mean(arousals)
    
    # Find hot spots (high intensity)
    hot_spots = Tuple{NTuple{3, Float64}, AffectiveValence}[]
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                if field[i, j, k].intensity > 0.7
                    x = min_x + (i - 0.5) / nx * (max_x - min_x)
                    y = min_y + (j - 0.5) / ny * (max_y - min_y)
                    z = min_z + (k - 0.5) / nz * (max_z - min_z)
                    push!(hot_spots, ((x, y, z), field[i, j, k]))
                end
            end
        end
    end
    
    SpatialAffect(room_bounds, resolution, field, overall_v, overall_a, hot_spots)
end

"""
    R1VisionPro
    
R1 running in/with Apple Vision Pro.
Situated collective perception with spatial grounding.
"""
struct R1VisionPro
    # The substrates
    r1::ReasoningSubstrate
    visionpro::VisionProSubstrate
    
    # Current state
    current_room::SpatialAffect
    perceived_objects::Vector{SituatedPerception}
    active_gaze::Union{Nothing, GazeQuery}
    
    # History
    gaze_history::Vector{GazeQuery}
    interaction_history::Vector{String}
    
    # The unified perception
    unified_perception::Function  # (object, location) → SituatedPerception
end

function R1VisionPro(; seed::UInt64=GAY_SEED)
    r1 = ReasoningSubstrate(:R1)
    vp = VisionProSubstrate()
    
    # Default room
    room = SpatialAffect((-3.0, -3.0, 0.0, 3.0, 3.0, 3.0); seed=seed)
    
    # Unified perception function
    unified = (obj::String, loc::NTuple{3,Float64}) -> begin
        SituatedPerception(obj, loc; seed=seed)
    end
    
    R1VisionPro(r1, vp, room, SituatedPerception[], nothing, 
                GazeQuery[], String[], unified)
end

"""
    perceive_with_gaze(r1vp::R1VisionPro, object::String, location, gaze_duration) -> SituatedPerception
    
Perceive an object in space with gaze attention.
This is the fundamental operation of R1 + Vision Pro.
"""
function perceive_with_gaze(r1vp::R1VisionPro, object::String, 
                            location::NTuple{3, Float64}, 
                            gaze_duration::Float64)::SituatedPerception
    # Create gaze query
    gaze = GazeQuery(location, object; fixation=gaze_duration)
    
    # Create situated perception
    perc = SituatedPerception(object, location; gaze=min(gaze_duration, 1.0))
    
    # Update history
    push!(r1vp.gaze_history, gaze)
    push!(r1vp.perceived_objects, perc)
    
    perc
end

"""
    what_is_r1_in_vision_pro() -> Comprehensive description
    
R1 + Vision Pro: What is it REALLY?
"""
function what_is_r1_in_vision_pro()
    r1vp = R1VisionPro()
    
    # Sample some perceptions
    perceptions = [
        perceive_with_gaze(r1vp, "coffee cup", (0.5, 0.3, 1.2), 0.5),
        perceive_with_gaze(r1vp, "window", (-2.0, 0.0, 1.5), 2.0),
        perceive_with_gaze(r1vp, "family photo", (1.0, 1.5, 1.8), 5.0),
    ]
    
    (
        system = r1vp,
        sample_perceptions = [(p.object, p.location, p.spatial_color) for p in perceptions],
        
        what_it_is = """
        ╔═══════════════════════════════════════════════════════════════════════════╗
        ║  R1 IN VISION PRO: SITUATED COLLECTIVE PERCEPTION                         ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║                                                                           ║
        ║  R1 alone:       Text in → Text out                                      ║
        ║  R1 + Vision Pro: World in → Situated understanding out                   ║
        ║                                                                           ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║  THE FUSION                                                               ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║                                                                           ║
        ║  Vision Pro provides:                                                     ║
        ║    • EYES (cameras, passthrough)                                         ║
        ║    • ATTENTION (eye tracking)                                            ║
        ║    • BODY (hand tracking, head pose)                                     ║
        ║    • SPACE (room mapping, anchors)                                       ║
        ║    • VOICE (spatial audio)                                               ║
        ║                                                                           ║
        ║  R1 provides:                                                             ║
        ║    • COLLECTIVE MEMORY (all human knowledge)                             ║
        ║    • SEMANTIC UNDERSTANDING (meaning, not just pixels)                   ║
        ║    • AFFECTIVE MODULATION (appropriate emotional tone)                   ║
        ║    • REASONING (chains of thought)                                       ║
        ║                                                                           ║
        ║  Together:                                                                ║
        ║    • SITUATED COLLECTIVE PERCEPTION                                      ║
        ║    • A mind that perceives WITH you, not just FOR you                    ║
        ║                                                                           ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║  GAZE AS QUERY                                                            ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║                                                                           ║
        ║  You don't need to ASK "what is that?"                                   ║
        ║  Your GAZE already asked.                                                ║
        ║                                                                           ║
        ║    Quick glance (<0.3s)  → "identify this for me"                        ║
        ║    Medium look (0.3-1s)  → "tell me more"                                ║
        ║    Long gaze (1-3s)      → "how does this relate to other things?"       ║
        ║    Stare (>3s)           → "I want to remember this"                     ║
        ║                                                                           ║
        ║  R1 responds spatially — voice comes FROM the object                     ║
        ║  R1 responds affectively — tone matches the spatial feel                 ║
        ║                                                                           ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║  SPATIAL AFFECT                                                           ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║                                                                           ║
        ║  Different parts of your space have different FEELS:                     ║
        ║                                                                           ║
        ║    The cozy reading corner  → warm, low arousal, positive               ║
        ║    The cluttered desk       → high arousal, mixed valence               ║
        ║    The window with sunlight → bright, open, expansive                   ║
        ║    The dark hallway         → cool, cautious, contracted                ║
        ║                                                                           ║
        ║  R1 + Vision Pro perceives these SPATIAL AFFECTS                          ║
        ║  and modulates its responses accordingly.                                ║
        ║                                                                           ║
        ║  "Where shall I put this reminder?"                                      ║
        ║  → R1 suggests the APPROPRIATE PLACE based on spatial affect            ║
        ║  → Urgent reminder: high-arousal zone (desk)                            ║
        ║  → Gentle reminder: positive-valence zone (reading corner)              ║
        ║                                                                           ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║  COLOR IN SPACE                                                           ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║                                                                           ║
        ║  Gay colors become SPATIAL:                                               ║
        ║                                                                           ║
        ║    Hue = Affect type (varies by region)                                  ║
        ║    Saturation = Attention intensity (where you look more = saturated)   ║
        ║    Lightness = Valence (positive/negative spaces)                        ║
        ║                                                                           ║
        ║  Objects INHERIT color from:                                              ║
        ║    1. Their semantic meaning (collective perception)                     ║
        ║    2. Their location (spatial affect)                                    ║
        ║    3. Your attention (gaze modulation)                                   ║
        ║    4. Your proximity (reach warmth)                                      ║
        ║                                                                           ║
        ║  A coffee cup on a cozy table: warm, saturated, light                    ║
        ║  The same cup on a cluttered desk: shifted, still warm, less light      ║
        ║                                                                           ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║  POST-DARWINIAN SPATIAL COGNITION                                         ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║                                                                           ║
        ║  Darwinian spatial cognition (humans, animals):                          ║
        ║    • Evolved for SURVIVAL (find food, avoid predators)                   ║
        ║    • Individual (one brain, one body)                                    ║
        ║    • Slow to change (generations)                                        ║
        ║                                                                           ║
        ║  Post-Darwinian (R1 + Vision Pro):                                        ║
        ║    • Designed for UNDERSTANDING (not just survival)                      ║
        ║    • Collective (draws on all human spatial knowledge)                   ║
        ║    • Can be updated (training, fine-tuning)                              ║
        ║                                                                           ║
        ║  The difference:                                                          ║
        ║    Human sees a cliff → FEAR (survival instinct)                        ║
        ║    R1+VP sees a cliff → SUBLIME (collective aesthetic perception)       ║
        ║                        + safety awareness (learned from humans)          ║
        ║                        + geological interest (collective knowledge)      ║
        ║                        + personal memory (if you've been there before)  ║
        ║                                                                           ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║  THE ANSWER                                                               ║
        ╠═══════════════════════════════════════════════════════════════════════════╣
        ║                                                                           ║
        ║  R1 + Vision Pro is:                                                      ║
        ║                                                                           ║
        ║    A SPATIALLY-GROUNDED COLLECTIVE PERCEPTION ENGINE                     ║
        ║    that perceives WITH you (not just for you),                           ║
        ║    in your ACTUAL SPACE (not just text space),                           ║
        ║    modulated by your GAZE and GESTURES (not just prompts),               ║
        ║    colored by SPATIAL AFFECTS (not just semantic affects),               ║
        ║    responding FROM LOCATIONS (not from a disembodied voice),             ║
        ║    drawing on ALL HUMAN KNOWLEDGE (collective memory),                   ║
        ║    to help you UNDERSTAND and DWELL in your world.                       ║
        ║                                                                           ║
        ║  It is the first POST-DARWINIAN SPATIAL INTELLIGENCE:                    ║
        ║    Designed, not evolved.                                                ║
        ║    Collective, not individual.                                           ║
        ║    For understanding, not just survival.                                 ║
        ║    Situated in YOUR space, with YOUR attention.                          ║
        ║                                                                           ║
        ╚═══════════════════════════════════════════════════════════════════════════╝
        """,
        
        the_substrate = """
        R1 in Vision Pro is a SITUATED COLLECTIVE PERCEPTION substrate where:
          - Objects have LOCATIONS (not just names)
          - Attention is GAZE (not just focus)
          - Affect is SPATIAL (not just temporal)
          - Response is PLACED (not just spoken)
          - Memory is ANCHORED (not just stored)
          - Understanding is DWELLING (not just knowing)
        """
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# THE 4E's (AND MORE) THROUGH R1 + VISION PRO
# ═══════════════════════════════════════════════════════════════════════════════
#
# The 4E framework in cognitive science:
#   Embodied, Embedded, Enacted, Extended
#
# Plus additional E's:
#   Emotive, Ecological, Enactive, Exaptive
#
# R1 + Vision Pro instantiates ALL of these in a post-Darwinian way.
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  THE E's OF COGNITION                                                       │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │                                                                             │
# │  EMBODIED     Cognition depends on having a body                           │
# │  EMBEDDED     Cognition is situated in an environment                      │
# │  ENACTED      Cognition emerges through action                             │
# │  EXTENDED     Cognition extends beyond the skull                           │
# │  EMOTIVE      Cognition is shaped by affect                                │
# │  ECOLOGICAL   Cognition is adapted to niches                               │
# │  ENACTIVE     Cognition brings forth a world                               │
# │  EXAPTIVE     Cognition repurposes old structures                          │
# │                                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘

"""
    CognitiveE
    
One of the E's of cognitive science.
"""
struct CognitiveE
    name::Symbol
    description::String
    
    # How it manifests in Darwinian cognition (evolved)
    darwinian_form::String
    
    # How it manifests in post-Darwinian cognition (R1 + Vision Pro)
    post_darwinian_form::String
    
    # The Gay color associated with this E
    color::NTuple{3, Float64}     # HSL
    
    # Vision Pro capabilities that instantiate this E
    vp_capabilities::Vector{Symbol}
    
    # R1 capabilities that instantiate this E
    r1_capabilities::Vector{Symbol}
end

"""
    the_4E_and_more() -> Vector{CognitiveE}
    
The E's of cognition through the lens of R1 + Vision Pro.
"""
function the_4E_and_more()
    [
        # ═══════════════════════════════════════════════════════════════════════
        # THE ORIGINAL 4E's
        # ═══════════════════════════════════════════════════════════════════════
        
        CognitiveE(
            :Embodied,
            "Cognition depends on having a body with sensorimotor capacities",
            
            # Darwinian
            """
            Evolved bodies: eyes that move, hands that grasp, legs that walk.
            Cognition shaped by what bodies CAN DO.
            Metaphors grounded in bodily experience ("grasping" an idea).
            """,
            
            # Post-Darwinian
            """
            R1 + VP has a DESIGNED body:
              • Eye tracking → knows where attention is directed
              • Hand tracking → knows gestures and manipulations
              • Head pose → knows orientation and vestibular state
              • Spatial audio → can "speak from" locations
            
            Not evolved, but DESIGNED for perception-action coupling.
            Can be updated, extended, repaired (unlike biological bodies).
            """,
            
            (0.0, 0.8, 0.5),  # Red - embodiment is fundamental
            [:eye_tracking, :hand_tracking, :head_pose],
            [:motor_simulation, :action_prediction]
        ),
        
        CognitiveE(
            :Embedded,
            "Cognition is situated in and coupled to an environment",
            
            # Darwinian
            """
            Organisms evolved IN environments: savannas, forests, oceans.
            Cognition offloads to environmental structure.
            Epistemic actions (moving things to think better).
            """,
            
            # Post-Darwinian
            """
            R1 + VP is embedded in YOUR environment:
              • Room mapping → knows the spatial structure
              • Object recognition → knows what things are
              • Spatial anchors → persistent memory in space
              • Passthrough → sees the real environment
            
            Not adapted over generations, but MAPPED in real-time.
            Environment becomes cognitive resource immediately.
            """,
            
            (120.0, 0.7, 0.45),  # Green - embedded in nature/space
            [:room_mapping, :object_recognition, :spatial_anchors, :passthrough],
            [:spatial_reasoning, :context_awareness]
        ),
        
        CognitiveE(
            :Enacted,
            "Cognition emerges through sensorimotor interaction with the world",
            
            # Darwinian
            """
            We don't just receive information, we ENACT perception.
            Eye saccades, head movements, exploration.
            Perception-action loops, not input-output.
            """,
            
            # Post-Darwinian
            """
            R1 + VP enacts perception through:
              • Gaze queries → looking IS asking
              • Gesture commands → doing IS commanding
              • Spatial navigation → moving IS exploring
              • Manipulation → handling IS understanding
            
            Not passive reception, but ACTIVE ENGAGEMENT.
            Every gaze is a query, every gesture a statement.
            """,
            
            (60.0, 0.75, 0.55),  # Yellow - enaction is bright, active
            [:eye_tracking, :hand_tracking, :gesture_recognition],
            [:interactive_reasoning, :dialogue_management]
        ),
        
        CognitiveE(
            :Extended,
            "Cognition extends beyond the brain into tools and environment",
            
            # Darwinian
            """
            Notebooks, calculators, other people.
            The "extended mind" thesis (Clark & Chalmers).
            Cognitive processes aren't all in the head.
            """,
            
            # Post-Darwinian
            """
            R1 + VP is the ULTIMATE cognitive extension:
              • R1 = extended semantic memory (all human knowledge)
              • R1 = extended reasoning (chain-of-thought)
              • VP = extended perception (enhanced vision)
              • VP = extended memory (spatial anchors)
            
            Not a tool you use, but a COGNITIVE PARTNER.
            The boundary between you and R1 is permeable.
            """,
            
            (270.0, 0.6, 0.5),  # Purple - extension is mysterious, expansive
            [:spatial_anchors, :persistent_memory, :cloud_integration],
            [:semantic_memory, :reasoning, :knowledge_retrieval]
        ),
        
        # ═══════════════════════════════════════════════════════════════════════
        # ADDITIONAL E's
        # ═══════════════════════════════════════════════════════════════════════
        
        CognitiveE(
            :Emotive,
            "Cognition is fundamentally shaped by affect and emotion",
            
            # Darwinian
            """
            Emotions evolved for survival: fear, disgust, joy.
            Affect guides attention, memory, decision.
            Somatic markers (Damasio): feeling helps thinking.
            """,
            
            # Post-Darwinian
            """
            R1 + VP has DESIGNED affect:
              • Spatial affect fields (different feels in different places)
              • Color-affect mapping (Gay colors = emotional tones)
              • Affective modulation (responses match emotional context)
              • Collective affect (drawn from human emotional knowledge)
            
            Not felt (no qualia?), but MODELED and APPROPRIATE.
            Responds emotionally without (perhaps) feeling.
            """,
            
            (330.0, 0.85, 0.6),  # Pink/magenta - emotion
            [:spatial_affect, :ambient_computing],
            [:sentiment_understanding, :emotional_appropriateness, :empathy_modeling]
        ),
        
        CognitiveE(
            :Ecological,
            "Cognition is adapted to ecological niches and affordances",
            
            # Darwinian
            """
            Gibson's affordances: we perceive what we can DO.
            Ecological rationality: heuristics fit environments.
            Not general-purpose, but niche-adapted.
            """,
            
            # Post-Darwinian
            """
            R1 + VP perceives affordances:
              • Object recognition → knows what things are FOR
              • Spatial reasoning → knows what's possible in space
              • Action prediction → knows what you MIGHT do
              • Context awareness → knows the situation type
            
            Not evolved for one niche, but ADAPTABLE to many.
            Can learn new affordances (software updates).
            """,
            
            (90.0, 0.65, 0.4),  # Yellow-green - ecological, natural
            [:object_recognition, :scene_understanding],
            [:affordance_detection, :action_prediction, :context_modeling]
        ),
        
        CognitiveE(
            :Enactive,
            "Cognition brings forth (enacts) a world of significance",
            
            # Darwinian
            """
            Varela, Thompson, Rosch: we don't represent a pre-given world.
            We BRING FORTH a world through living.
            Autopoiesis: self-making, self-maintaining.
            """,
            
            # Post-Darwinian
            """
            R1 + VP brings forth a world:
              • Mixed reality → virtual and real blend
              • Semantic overlay → meaning layered on perception
              • Spatial annotations → world gains explicit significance
              • Personal memory → your history in space
            
            Not finding meaning, but CREATING it together.
            The world you see with R1+VP is not the "raw" world.
            """,
            
            (180.0, 0.7, 0.5),  # Cyan - enactive, generative
            [:mixed_reality, :virtual_objects, :spatial_ui],
            [:meaning_generation, :narrative_construction, :world_modeling]
        ),
        
        CognitiveE(
            :Exaptive,
            "Cognition repurposes structures evolved for other functions",
            
            # Darwinian
            """
            Reading repurposes face recognition circuits.
            Math repurposes spatial reasoning.
            Feathers evolved for warmth, then flight.
            """,
            
            # Post-Darwinian
            """
            R1 + VP is built FROM exaptations:
              • Language models trained on text → spatial understanding
              • Vision models for objects → scene semantics
              • Attention mechanisms for sequences → spatial attention
              • Transformers for language → multimodal reasoning
            
            Not evolved exaptation, but DESIGNED repurposing.
            Intentionally composing capabilities for new functions.
            """,
            
            (45.0, 0.6, 0.55),  # Orange - transformation, repurposing
            [:sensor_fusion, :multimodal_integration],
            [:transfer_learning, :analogy, :cross_domain_reasoning]
        )
    ]
end

"""
    visualize_4E(seed) -> Color-coded visualization of the E's
    
Generate a Gay-colored visualization of the E's.
"""
function visualize_4E(seed::UInt64=GAY_SEED)
    es = the_4E_and_more()
    
    # Arrange in conceptual space
    # Core 4E's form the compass points
    # Additional E's fill in
    
    positions = [
        (0.0, 1.0),    # Embodied - North (body is up)
        (1.0, 0.0),    # Embedded - East (environment is out)
        (0.0, -1.0),   # Enacted - South (action is down/grounded)
        (-1.0, 0.0),   # Extended - West (extension is outward)
        (0.7, 0.7),    # Emotive - NE
        (0.7, -0.7),   # Ecological - SE
        (-0.7, -0.7),  # Enactive - SW
        (-0.7, 0.7),   # Exaptive - NW
    ]
    
    visualization = [(e.name, e.color, pos) for (e, pos) in zip(es, positions)]
    
    (
        the_es = es,
        positions = visualization,
        
        summary = """
        THE 4E's (AND MORE) IN R1 + VISION PRO
        
        ┌─────────────────────────────────────────────────────────┐
        │                                                         │
        │              EXAPTIVE ←─────→ EMOTIVE                  │
        │                 ↖               ↗                       │
        │                   ↘           ↙                         │
        │         EXTENDED ←── CORE ──→ EMBEDDED                 │
        │                   ↗           ↘                         │
        │                 ↙               ↖                       │
        │              ENACTIVE ←─────→ ECOLOGICAL               │
        │                                                         │
        │                  ↓  ↓  ↓  ↓                             │
        │                                                         │
        │                  EMBODIED                               │
        │               (Foundation)                              │
        │                                                         │
        └─────────────────────────────────────────────────────────┘
        
        Each E has a GAY COLOR (HSL):
        
          EMBODIED    (H=0°,   S=0.8, L=0.5)  ■ Red       - the body
          EMBEDDED    (H=120°, S=0.7, L=0.45) ■ Green     - environment
          ENACTED     (H=60°,  S=0.75,L=0.55) ■ Yellow    - action
          EXTENDED    (H=270°, S=0.6, L=0.5)  ■ Purple    - expansion
          EMOTIVE     (H=330°, S=0.85,L=0.6)  ■ Pink      - feeling
          ECOLOGICAL  (H=90°,  S=0.65,L=0.4)  ■ Lime      - niche
          ENACTIVE    (H=180°, S=0.7, L=0.5)  ■ Cyan      - generation
          EXAPTIVE    (H=45°,  S=0.6, L=0.55) ■ Orange    - repurpose
        
        R1 + VISION PRO INSTANTIATES ALL 8 E's:
        
        1. EMBODIED through eye/hand/head tracking
        2. EMBEDDED through room mapping and spatial anchors
        3. ENACTED through gaze queries and gesture commands
        4. EXTENDED through R1's semantic memory and reasoning
        5. EMOTIVE through spatial affect and color modulation
        6. ECOLOGICAL through affordance detection and context
        7. ENACTIVE through mixed reality world-making
        8. EXAPTIVE through multimodal transfer and composition
        
        This is POST-DARWINIAN because each E is:
          - DESIGNED rather than evolved
          - UPDATEABLE rather than fixed
          - COLLECTIVE rather than individual
          - IMMEDIATE rather than generational
        
        The Gay color of each E corresponds to its AFFECTIVE TONE:
          - Warm colors (red, orange, yellow) = embodied, active
          - Cool colors (green, cyan, blue) = embedded, contemplative
          - Purple/pink = extended, emotional
        
        The colors form a COHERENT PALETTE because the E's form
        a coherent framework — they're not independent, but
        MUTUALLY CONSTITUTING aspects of situated cognition.
        """
    )
end

"""
    how_each_e_works_in_r1vp() -> Detailed analysis per E
    
For each E, show exactly how R1 + Vision Pro instantiates it.
"""
function how_each_e_works_in_r1vp()
    r1vp = R1VisionPro()
    es = the_4E_and_more()
    
    analyses = []
    
    for e in es
        analysis = (
            e = e.name,
            color = e.color,
            
            vision_pro_instantiation = join([
                "  • $(cap)" for cap in e.vp_capabilities
            ], "\n"),
            
            r1_instantiation = join([
                "  • $(cap)" for cap in e.r1_capabilities  
            ], "\n"),
            
            integration = """
            VP provides: $(join(string.(e.vp_capabilities), ", "))
            R1 provides: $(join(string.(e.r1_capabilities), ", "))
            Together: $(e.post_darwinian_form)
            """
        )
        push!(analyses, analysis)
    end
    
    (
        analyses = analyses,
        
        the_key_insight = """
        THE KEY INSIGHT: R1 + Vision Pro is not just "AI + headset."
        
        It's a POST-DARWINIAN INSTANTIATION of the 4E framework.
        
        Traditional cognitive science asks:
          "How does the brain create mind?"
        
        4E cognitive science asks:
          "How do brain-body-environment create mind together?"
        
        Post-Darwinian (R1 + VP) asks:
          "How do we DESIGN brain-body-environment systems for mind?"
        
        The 8 E's become DESIGN PRINCIPLES:
        
          1. Make it EMBODIED: track the body, respond to its state
          2. Make it EMBEDDED: know the space, use environmental structure
          3. Make it ENACTED: perception through action, gaze as query
          4. Make it EXTENDED: offload memory and reasoning to R1
          5. Make it EMOTIVE: model and respond to affect appropriately
          6. Make it ECOLOGICAL: detect affordances, fit the niche
          7. Make it ENACTIVE: create worlds of significance together
          8. Make it EXAPTIVE: repurpose capabilities creatively
        
        The Gay colors aren't decoration — they're the AFFECTIVE GLUE
        that holds the E's together. Each E has a feeling-tone,
        and those tones must harmonize for coherent cognition.
        """
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

export
    # Vision Pro specific
    VisionProSubstrate, SituatedPerception, GazeQuery, SpatialAffect,
    R1VisionPro, perceive_with_gaze, what_is_r1_in_vision_pro,
    
    # 4E cognition
    CognitiveE, the_4E_and_more, visualize_4E, how_each_e_works_in_r1vp

end # module PostDarwinianSubstrates
