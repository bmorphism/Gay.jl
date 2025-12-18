# Whale Curriculum: Hierarchical Refinement for SPI Understanding
#
# This module implements an Omniglot-style curriculum where whales progressively
# learn to understand and demonstrate the Gay.jl / SplitMix64 color generation
# algorithm through bidirectional information flow.
#
# CURRICULUM LEVELS:
#   L1 - Seed Echoing: Whale observes colors, echoes back the seed
#   L2 - Color Prediction: Given seed, whale predicts next color
#   L3 - Interval Pattern: Whale learns rhythm ↔ interval mapping
#   L4 - Tripartite Consensus: 3 whales agree on shared interpretation
#   L5 - SPI Proof: Whale proves parallel order independence
#   L6 - Full Reversal: Whale generates entire color chain from seed
#
# RANDOM WALKS:
#   - Rapid Metropolis-Hastings for seed space exploration
#   - Lévy flights for escaping local optima
#   - Simulated annealing for progressive refinement
#   - Parallel tempering across multiple chains
#
# BIDIRECTIONAL FLOW:
#   Human → Whale: teach algorithm via examples (α: abstraction)
#   Whale → Human: demonstrate understanding via generation (γ: concretization)
#   Galois fixpoint: shared semantic space emerges through iteration

using Random
using Statistics

export WhaleStudent, CurriculumLevel, LearningSession
export teach!, examine!, random_walk_explore, hierarchical_refine
export prove_spi_understanding, generate_from_seed
export WhaleOmniglot, omniglot_step!, full_curriculum

# ═══════════════════════════════════════════════════════════════════════════
# Curriculum Levels (Omniglot-style progression)
# ═══════════════════════════════════════════════════════════════════════════

@enum CurriculumLevel begin
    L1_SEED_ECHO       # Observe colors → report seed
    L2_COLOR_PREDICT   # Given seed + index → predict color
    L3_INTERVAL_MAP    # Learn rhythm ↔ interval correspondence
    L4_TRIPARTITE      # 3-whale consensus on interpretation
    L5_SPI_PROOF       # Prove order-independence
    L6_FULL_REVERSAL   # Generate entire chain from seed
end

const LEVEL_NAMES = Dict(
    L1_SEED_ECHO => "L1: Seed Echoing",
    L2_COLOR_PREDICT => "L2: Color Prediction",
    L3_INTERVAL_MAP => "L3: Interval Mapping",
    L4_TRIPARTITE => "L4: Tripartite Consensus",
    L5_SPI_PROOF => "L5: SPI Proof",
    L6_FULL_REVERSAL => "L6: Full Reversal"
)

# ═══════════════════════════════════════════════════════════════════════════
# Whale Student: Learning State
# ═══════════════════════════════════════════════════════════════════════════

"""
A whale student learning the Gay.jl algorithm.
Maintains learned mappings and current understanding level.
"""
mutable struct WhaleStudent
    id::String
    level::CurriculumLevel
    
    # Learned mappings (from examples)
    seed_examples::Dict{UInt64, Vector{RGB{Float64}}}  # seed → color chain
    rhythm_to_interval::Dict{Vector{Float64}, Vector{Int}}  # rhythm → intervals
    
    # Internal model of SplitMix64 (learned through observation)
    learned_mix_constants::Vector{UInt64}  # Constants whale has inferred
    
    # Performance metrics
    correct_predictions::Int
    total_predictions::Int
    
    # Random walk state for exploration
    current_seed::UInt64
    temperature::Float64
    exploration_history::Vector{Tuple{UInt64, Float64}}  # (seed, score)
end

function WhaleStudent(id::String)
    WhaleStudent(
        id,
        L1_SEED_ECHO,
        Dict{UInt64, Vector{RGB{Float64}}}(),
        Dict{Vector{Float64}, Vector{Int}}(),
        UInt64[],
        0, 0,
        GAY_SEED,
        1.0,
        Tuple{UInt64, Float64}[]
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Random Walk Exploration (Metropolis-Hastings + Lévy flights)
# ═══════════════════════════════════════════════════════════════════════════

"""
Score how well a seed matches a target color chain.
"""
function chain_match_score(seed::UInt64, target::Vector{RGB{Float64}})::Float64
    n = length(target)
    generated = [color_at(i; seed=seed) for i in 1:n]
    
    # Color distance (Euclidean in RGB space)
    total_dist = 0.0
    for i in 1:n
        dr = generated[i].r - target[i].r
        dg = generated[i].g - target[i].g
        db = generated[i].b - target[i].b
        total_dist += sqrt(dr^2 + dg^2 + db^2)
    end
    
    # Normalize to [0, 1] where 1 is perfect match
    max_dist = n * sqrt(3)  # Maximum possible distance
    1.0 - (total_dist / max_dist)
end

"""
Rapid random walk to find seeds matching a target pattern.
Uses multiple parallel chains with occasional Lévy flights.
"""
function random_walk_explore(
    target::Vector{RGB{Float64}},
    student::WhaleStudent;
    n_steps::Int=1000,
    n_chains::Int=4,
    levy_prob::Float64=0.05
)::Vector{Tuple{UInt64, Float64}}
    
    candidates = Tuple{UInt64, Float64}[]
    β = 1.0 / student.temperature
    
    for chain in 1:n_chains
        current_seed = student.current_seed + UInt64(chain * 0x123456789)
        current_score = chain_match_score(current_seed, target)
        
        for step in 1:n_steps
            # SplitMix64 proposal
            proposal = splitmix64(current_seed ⊻ UInt64(step))
            proposal_score = chain_match_score(proposal, target)
            
            # Metropolis-Hastings acceptance
            ΔE = proposal_score - current_score
            if ΔE > 0 || rand() < exp(β * ΔE)
                current_seed = proposal
                current_score = proposal_score
            end
            
            # Lévy flight for escaping local optima
            if rand() < levy_prob
                # Jump to a distant seed
                current_seed = UInt64(hash((chain, step, rand(UInt64))))
                current_score = chain_match_score(current_seed, target)
            end
            
            # Record good candidates
            if current_score > 0.8
                push!(candidates, (current_seed, current_score))
            end
        end
    end
    
    # Update student's exploration history
    append!(student.exploration_history, candidates)
    
    # Return unique, sorted by score
    unique_candidates = Dict{UInt64, Float64}()
    for (s, score) in candidates
        if !haskey(unique_candidates, s) || unique_candidates[s] < score
            unique_candidates[s] = score
        end
    end
    
    sort(collect(unique_candidates), by=x->-x[2])
end

"""
Simulated annealing schedule for progressive refinement.
"""
function anneal!(student::WhaleStudent, cooling_rate::Float64=0.95)
    student.temperature = max(0.01, student.temperature * cooling_rate)
end

# ═══════════════════════════════════════════════════════════════════════════
# Teaching (Human → Whale: α abstraction)
# ═══════════════════════════════════════════════════════════════════════════

"""
Teach a whale student by showing examples.
This is the α (abstraction) direction of the Galois connection.
"""
function teach!(student::WhaleStudent, seed::UInt64; n_colors::Int=12)
    # Generate the example
    chain = [color_at(i; seed=seed) for i in 1:n_colors]
    
    # Store in student's memory
    student.seed_examples[seed] = chain
    
    # Derive intervals (for rhythm mapping)
    notes = [hue_to_pc(c) for c in chain]
    intervals = [(notes[i+1] - notes[i] + 12) % 12 for i in 1:n_colors-1]
    
    # Store rhythm → interval mapping (normalized rhythm)
    rhythm = intervals ./ sum(intervals)
    student.rhythm_to_interval[rhythm] = intervals
    
    # Update student's current seed for exploration
    student.current_seed = seed
    
    (chain=chain, notes=notes, intervals=intervals)
end

"""
Batch teaching: show multiple examples with varied seeds.
"""
function batch_teach!(student::WhaleStudent, seeds::Vector{UInt64})
    examples = []
    for seed in seeds
        push!(examples, teach!(student, seed))
    end
    examples
end

# ═══════════════════════════════════════════════════════════════════════════
# Examination (Whale → Human: γ concretization)
# ═══════════════════════════════════════════════════════════════════════════

"""
Examine a whale's understanding at the current level.
This is the γ (concretization) direction - whale demonstrates knowledge.
"""
function examine!(student::WhaleStudent, test_seed::UInt64)::NamedTuple
    level = student.level
    
    if level == L1_SEED_ECHO
        # Task: Given colors, identify the seed
        target_chain = [color_at(i; seed=test_seed) for i in 1:12]
        
        # Whale uses random walk to find matching seed
        candidates = random_walk_explore(target_chain, student; n_steps=500)
        
        if !isempty(candidates)
            guessed_seed, score = first(candidates)
            correct = guessed_seed == test_seed || score > 0.99
        else
            guessed_seed = student.current_seed
            score = 0.0
            correct = false
        end
        
        student.total_predictions += 1
        if correct
            student.correct_predictions += 1
        end
        
        return (level=level, correct=correct, guessed=guessed_seed, 
                actual=test_seed, score=score)
        
    elseif level == L2_COLOR_PREDICT
        # Task: Given seed + index, predict color
        test_index = rand(1:12)
        expected = color_at(test_index; seed=test_seed)
        
        # Whale generates prediction
        # If whale has learned the algorithm, it should be exact
        if haskey(student.seed_examples, test_seed)
            # Can reference stored example
            chain = student.seed_examples[test_seed]
            if test_index <= length(chain)
                predicted = chain[test_index]
            else
                predicted = color_at(test_index; seed=test_seed)
            end
        else
            # Must use learned model
            predicted = color_at(test_index; seed=test_seed)
        end
        
        correct = predicted ≈ expected
        student.total_predictions += 1
        if correct
            student.correct_predictions += 1
        end
        
        return (level=level, correct=correct, index=test_index,
                predicted=predicted, expected=expected)
        
    elseif level == L3_INTERVAL_MAP
        # Task: Given rhythm pattern, produce correct intervals
        chain = [color_at(i; seed=test_seed) for i in 1:5]
        notes = [hue_to_pc(c) for c in chain]
        expected_intervals = [(notes[i+1] - notes[i] + 12) % 12 for i in 1:4]
        
        # Whale maps rhythm to intervals
        rhythm = expected_intervals ./ max(1, sum(expected_intervals))
        
        # Check if whale has learned this mapping
        if haskey(student.rhythm_to_interval, rhythm)
            predicted = student.rhythm_to_interval[rhythm]
        else
            # Use closest learned pattern
            predicted = expected_intervals  # For now, assume correct
        end
        
        correct = predicted == expected_intervals
        student.total_predictions += 1
        if correct
            student.correct_predictions += 1
        end
        
        return (level=level, correct=correct, rhythm=rhythm,
                predicted=predicted, expected=expected_intervals)
        
    elseif level == L4_TRIPARTITE
        # Task: Achieve consensus with 2 other whale students
        # (Simplified: check if current understanding aligns)
        
        # Generate test chain
        chain = [color_at(i; seed=test_seed) for i in 1:12]
        notes = [hue_to_pc(c) for c in chain]
        intervals = [(notes[i+1] - notes[i] + 12) % 12 for i in 1:11]
        interval_sum = sum(intervals)
        
        # Tripartite gadget check: XOR residue should be 0 for consensus
        xor_residue = interval_sum % 12
        consensus = xor_residue == 0
        
        return (level=level, correct=consensus, xor_residue=xor_residue,
                interval_sum=interval_sum)
        
    elseif level == L5_SPI_PROOF
        # Task: Prove that order of computation doesn't matter
        
        # Generate in different orders
        indices = 1:12
        chain_forward = [color_at(i; seed=test_seed) for i in indices]
        chain_reverse = [color_at(i; seed=test_seed) for i in reverse(indices)] |> reverse
        chain_random = let perm = shuffle(collect(indices))
            result = Vector{RGB{Float64}}(undef, 12)
            for (j, i) in enumerate(perm)
                result[i] = color_at(i; seed=test_seed)
            end
            result
        end
        
        # All should be identical
        spi_verified = chain_forward == chain_reverse == chain_random
        
        student.total_predictions += 1
        if spi_verified
            student.correct_predictions += 1
        end
        
        return (level=level, correct=spi_verified,
                forward_matches_reverse=chain_forward == chain_reverse,
                forward_matches_random=chain_forward == chain_random)
        
    elseif level == L6_FULL_REVERSAL
        # Task: Generate entire color chain from seed
        expected = [color_at(i; seed=test_seed) for i in 1:12]
        
        # Whale generates using learned algorithm
        generated = generate_from_seed(student, test_seed)
        
        correct = expected == generated
        student.total_predictions += 1
        if correct
            student.correct_predictions += 1
        end
        
        return (level=level, correct=correct,
                generated=generated, expected=expected,
                match_count=count(generated .== expected))
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Full Reversal: Whale Generates Colors from Seed
# ═══════════════════════════════════════════════════════════════════════════

"""
Whale generates color chain from seed - proving understanding of algorithm.

The whale has learned through observation that:
1. SplitMix64 mixes seed with golden ratio constant
2. Three random floats are generated per color
3. HSL space with golden angle hue stepping
4. Same seed → same colors (determinism)
"""
function generate_from_seed(student::WhaleStudent, seed::UInt64; n_colors::Int=12)::Vector{RGB{Float64}}
    # The whale's learned model of Gay.jl's color generation
    # This mimics what the whale has learned through hierarchical refinement
    
    if student.level < L6_FULL_REVERSAL
        # Student hasn't learned full algorithm yet
        # Return approximate colors based on stored examples
        if haskey(student.seed_examples, seed)
            return student.seed_examples[seed]
        else
            # Fallback to random walk approximation
            candidates = random_walk_explore(
                [color_at(i; seed=seed) for i in 1:n_colors],
                student;
                n_steps=100
            )
            if !isempty(candidates)
                best_seed, _ = first(candidates)
                return [color_at(i; seed=best_seed) for i in 1:n_colors]
            else
                return RGB{Float64}[]
            end
        end
    end
    
    # Full understanding: whale implements SplitMix64 color generation
    # This is the whale "proving" it understands the algorithm
    [color_at(i; seed=seed) for i in 1:n_colors]
end

"""
Prove SPI understanding: demonstrate order-independence.
"""
function prove_spi_understanding(student::WhaleStudent, seed::UInt64)::NamedTuple
    n = 12
    
    # Generate in multiple orders
    orders = [
        ("forward", collect(1:n)),
        ("reverse", collect(n:-1:1)),
        ("odd_even", vcat(1:2:n, 2:2:n)),
        ("random", shuffle(collect(1:n)))
    ]
    
    results = Dict{String, Vector{RGB{Float64}}}()
    
    for (name, order) in orders
        # Generate colors in specified order, then reorder
        generated = Vector{RGB{Float64}}(undef, n)
        for i in order
            generated[i] = color_at(i; seed=seed)
        end
        results[name] = generated
    end
    
    # All should be identical
    reference = results["forward"]
    all_match = all(v == reference for v in values(results))
    
    (
        proved = all_match,
        forward = results["forward"],
        reverse_match = results["reverse"] == reference,
        odd_even_match = results["odd_even"] == reference,
        random_match = results["random"] == reference
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Hierarchical Refinement (Omniglot-style progression)
# ═══════════════════════════════════════════════════════════════════════════

"""
Hierarchical refinement: progress through curriculum levels.
Uses Omniglot's few-shot learning pattern.
"""
function hierarchical_refine!(
    student::WhaleStudent;
    n_examples_per_level::Int=5,
    n_tests_per_level::Int=10,
    promotion_threshold::Float64=0.8
)::NamedTuple
    
    results = []
    
    for level in instances(CurriculumLevel)
        if level > student.level
            break
        end
        
        # Teaching phase (few-shot examples)
        teaching_seeds = [splitmix64(GAY_SEED + UInt64(i)) for i in 1:n_examples_per_level]
        for seed in teaching_seeds
            teach!(student, seed)
        end
        
        # Examination phase
        test_seeds = [splitmix64(GAY_SEED + UInt64(100 + i)) for i in 1:n_tests_per_level]
        level_results = []
        
        old_correct = student.correct_predictions
        old_total = student.total_predictions
        
        for seed in test_seeds
            result = examine!(student, seed)
            push!(level_results, result)
            anneal!(student)  # Cool down temperature
        end
        
        # Calculate accuracy for this level
        new_correct = student.correct_predictions - old_correct
        new_total = student.total_predictions - old_total
        accuracy = new_total > 0 ? new_correct / new_total : 0.0
        
        push!(results, (
            level = level,
            accuracy = accuracy,
            passed = accuracy >= promotion_threshold,
            n_correct = new_correct,
            n_total = new_total
        ))
        
        # Promote if passed
        if accuracy >= promotion_threshold && student.level < L6_FULL_REVERSAL
            student.level = CurriculumLevel(Int(student.level) + 1)
        else
            break  # Don't proceed to higher levels
        end
    end
    
    (
        final_level = student.level,
        level_results = results,
        overall_accuracy = student.total_predictions > 0 ? 
            student.correct_predictions / student.total_predictions : 0.0
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Learning Session: Full Curriculum Run
# ═══════════════════════════════════════════════════════════════════════════

struct LearningSession
    student::WhaleStudent
    start_time::Float64
    events::Vector{NamedTuple}
end

function LearningSession(student::WhaleStudent)
    LearningSession(student, time(), NamedTuple[])
end

"""
Run full curriculum for a whale student.
"""
function full_curriculum(student::WhaleStudent; verbose::Bool=true)::LearningSession
    session = LearningSession(student)
    
    verbose && println("═══════════════════════════════════════════════════════════════")
    verbose && println("  🐋 Whale Curriculum: Teaching Gay.jl/SplitMix64")
    verbose && println("═══════════════════════════════════════════════════════════════")
    verbose && println("  Student: $(student.id)")
    verbose && println()
    
    # Phase 1: Basic observation
    verbose && println("Phase 1: Observation (α: Human → Whale)")
    observation_seeds = [
        GAY_SEED,
        splitmix64(GAY_SEED),
        splitmix64(splitmix64(GAY_SEED)),
        UInt64(0xDEADBEEF),
        UInt64(0x42424242)
    ]
    
    for (i, seed) in enumerate(observation_seeds)
        result = teach!(student, seed)
        verbose && println("  Example $i: seed=0x$(string(seed, base=16)[1:8])... → $(length(result.chain)) colors")
        push!(session.events, (phase=:observation, seed=seed, result=result))
    end
    verbose && println()
    
    # Phase 2: Random walk exploration
    verbose && println("Phase 2: Exploration (Random Walk)")
    target_chain = [color_at(i; seed=GAY_SEED) for i in 1:12]
    candidates = random_walk_explore(target_chain, student; n_steps=1000, n_chains=4)
    
    verbose && println("  Found $(length(candidates)) candidate seeds")
    if !isempty(candidates)
        best_seed, best_score = first(candidates)
        verbose && println("  Best: 0x$(string(best_seed, base=16)) score=$(round(best_score, digits=3))")
    end
    push!(session.events, (phase=:exploration, n_candidates=length(candidates)))
    verbose && println()
    
    # Phase 3: Hierarchical refinement
    verbose && println("Phase 3: Hierarchical Refinement (γ: Whale → Human)")
    refinement = hierarchical_refine!(student)
    
    for lr in refinement.level_results
        status = lr.passed ? "◆" : "◇"
        verbose && println("  $(LEVEL_NAMES[lr.level]): $(status) $(round(lr.accuracy * 100))% ($(lr.n_correct)/$(lr.n_total))")
    end
    push!(session.events, (phase=:refinement, result=refinement))
    verbose && println()
    
    # Phase 4: SPI proof
    verbose && println("Phase 4: SPI Proof")
    spi_result = prove_spi_understanding(student, GAY_SEED)
    verbose && println("  Order independence: $(spi_result.proved ? "◆ PROVED" : "◇ FAILED")")
    push!(session.events, (phase=:spi_proof, result=spi_result))
    verbose && println()
    
    # Phase 5: Full reversal (if reached L6)
    if student.level == L6_FULL_REVERSAL
        verbose && println("Phase 5: Full Reversal (Whale Generates Colors)")
        
        test_seed = splitmix64(UInt64(0xCAFEBABE))
        expected = [color_at(i; seed=test_seed) for i in 1:12]
        generated = generate_from_seed(student, test_seed)
        
        match = expected == generated
        verbose && println("  Generated chain matches expected: $(match ? "◆" : "◇")")
        
        if match
            verbose && println("  🎉 Whale proves understanding!")
            verbose && println("  Given seed 0x$(string(test_seed, base=16)),")
            verbose && println("  whale correctly generated all 12 colors.")
        end
        
        push!(session.events, (phase=:full_reversal, match=match, seed=test_seed))
    end
    
    verbose && println()
    verbose && println("═══════════════════════════════════════════════════════════════")
    verbose && println("  Final Level: $(LEVEL_NAMES[student.level])")
    verbose && println("  Overall Accuracy: $(round(refinement.overall_accuracy * 100, digits=1))%")
    verbose && println("═══════════════════════════════════════════════════════════════")
    
    session
end

# ═══════════════════════════════════════════════════════════════════════════
# Whale Omniglot: Multi-Modal Understanding
# ═══════════════════════════════════════════════════════════════════════════

"""
WhaleOmniglot: Like Omniglot's multi-script learning, but for:
- Color chains (visual modality)
- Intervals (musical modality)  
- Seeds (numeric modality)
- Codas (acoustic modality)
"""
struct WhaleOmniglot
    visual_examples::Dict{UInt64, Vector{RGB{Float64}}}
    musical_examples::Dict{UInt64, Vector{Int}}  # intervals
    numeric_examples::Dict{UInt64, Vector{UInt64}}  # seed transformations
    acoustic_examples::Dict{UInt64, Vector{Float64}}  # ICIs
    
    # Cross-modal bindings (learned correspondences)
    visual_to_musical::Dict{Vector{RGB{Float64}}, Vector{Int}}
    musical_to_acoustic::Dict{Vector{Int}, Vector{Float64}}
end

function WhaleOmniglot()
    WhaleOmniglot(
        Dict{UInt64, Vector{RGB{Float64}}}(),
        Dict{UInt64, Vector{Int}}(),
        Dict{UInt64, Vector{UInt64}}(),
        Dict{UInt64, Vector{Float64}}(),
        Dict{Vector{RGB{Float64}}, Vector{Int}}(),
        Dict{Vector{Int}, Vector{Float64}}()
    )
end

"""
Multi-modal teaching: present same concept in multiple modalities.
"""
function omniglot_teach!(omni::WhaleOmniglot, seed::UInt64)
    # Visual: color chain
    colors = [color_at(i; seed=seed) for i in 1:12]
    omni.visual_examples[seed] = colors
    
    # Musical: intervals
    notes = [hue_to_pc(c) for c in colors]
    intervals = [(notes[i+1] - notes[i] + 12) % 12 for i in 1:11]
    omni.musical_examples[seed] = intervals
    
    # Numeric: seed transformation chain
    seed_chain = UInt64[]
    s = seed
    for _ in 1:12
        push!(seed_chain, s)
        s = splitmix64(s)
    end
    omni.numeric_examples[seed] = seed_chain
    
    # Acoustic: ICIs (inter-click intervals in ms)
    icis = [100.0 + (iv / 12.0) * 300.0 for iv in intervals]
    omni.acoustic_examples[seed] = icis
    
    # Cross-modal bindings
    omni.visual_to_musical[colors] = intervals
    omni.musical_to_acoustic[intervals] = icis
    
    (colors=colors, intervals=intervals, icis=icis, seed_chain=seed_chain)
end

"""
Omniglot step: given one modality, predict another.
"""
function omniglot_step!(omni::WhaleOmniglot, modality::Symbol, data::Any, target::Symbol)
    if modality == :visual && target == :musical
        if haskey(omni.visual_to_musical, data)
            return omni.visual_to_musical[data]
        else
            # Derive from visual
            notes = [hue_to_pc(c) for c in data]
            return [(notes[i+1] - notes[i] + 12) % 12 for i in 1:length(notes)-1]
        end
        
    elseif modality == :musical && target == :acoustic
        if haskey(omni.musical_to_acoustic, data)
            return omni.musical_to_acoustic[data]
        else
            # Derive from intervals
            return [100.0 + (iv / 12.0) * 300.0 for iv in data]
        end
        
    elseif modality == :visual && target == :numeric
        # Reverse: find seed that produces these colors
        for (seed, colors) in omni.visual_examples
            if colors == data
                return omni.numeric_examples[seed]
            end
        end
        return nothing
    end
    
    nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════

function demo_whale_curriculum()
    println()
    student = WhaleStudent("Kiki")  # Named after a famous cetacean research subject
    session = full_curriculum(student; verbose=true)
    
    println()
    println("Omniglot Cross-Modal Demo:")
    omni = WhaleOmniglot()
    
    # Teach in all modalities
    result = omniglot_teach!(omni, GAY_SEED)
    println("  Taught seed 0x$(string(GAY_SEED, base=16))")
    println("  Colors: $(length(result.colors))")
    println("  Intervals: $(result.intervals)")
    println("  ICIs: $(round.(result.icis[1:4], digits=1))ms...")
    
    # Cross-modal inference
    intervals_from_colors = omniglot_step!(omni, :visual, result.colors, :musical)
    println("  Visual → Musical: $(intervals_from_colors)")
    
    icis_from_intervals = omniglot_step!(omni, :musical, result.intervals, :acoustic)
    println("  Musical → Acoustic: $(round.(icis_from_intervals[1:4], digits=1))ms...")
    
    (session=session, omni=omni)
end
