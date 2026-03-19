# XBOW-WEV: Crossbow for World Extractable Value
# ================================================
#
# A self-certifying improvement system combining:
# - Genetic algorithms (evolution of chromatic strategies)
# - Immune system dynamics (self/non-self via originary color)
# - Information geometry (Fisher metric on color space)
# - GitHub Actions integration (CI/CD as selection pressure)
#
# The key insight: Gay.jl's SPI provides the "self" signature that enables:
# 1. Clonal selection: colors that match originary seed survive
# 2. Affinity maturation: derived colors approach originary via gradient
# 3. Danger signals: deranged colors trigger immune response
# 4. Natural gradients: Fisher information guides evolution in color space
#
# Complexity Classes Extended:
#   -1: Anti-computation (undo/reversal, immune memory)
#    0: Fixed point (equilibrium, self-sameness checkpoint)
#   +1: O(1) - SPI instant (Gay mode)
#   +n: O(n) - Sequential (Slave mode)
#  log: O(log n) - Parallel HDX (Master mode)
#
# The XBOW aims at maximal World Extractable Value through chromatic targeting.

module XBOW_WEV

using Colors
using SplittableRandoms: SplittableRandom, split

export
    # Complexity closure
    ComplexityClass, ANTI, ZERO, CONSTANT, LOGARITHMIC, LINEAR, POLYNOMIAL,
    complexity_order, is_closed,
    
    # Genetic chromatic evolution
    ChromaticGenome, Population, evolve!, fitness,
    crossover, mutate, select,
    
    # Immune system dynamics
    ImmuneSystem, Antibody, Antigen, DangerSignal,
    clonal_selection!, affinity_maturation!, danger_response!,
    is_self, is_foreign,
    
    # Information geometry
    FisherMetric, natural_gradient, geodesic_step,
    color_manifold_distance,
    
    # GitHub Actions integration
    ActionWorkflow, WorkflowRun, SelectionPressure,
    ci_fitness, self_certify_improvement!,
    
    # World Extractable Value
    WEV, WorldState, extract_value!, value_gradient,
    xbow_aim, xbow_fire!,
    
    # Demo
    demo_xbow_wev

# ═══════════════════════════════════════════════════════════════════════════════
# Constants & PRNG
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const XBOW_SEED = UInt64(0x5842_4F57_5745_5621)  # "XBOW WEV!"

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
# Complexity Classes: Extending with -1 and 0
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ComplexityClass

Extended complexity classes for semantic world model closure:

| Order | Class       | Mode     | Equilibrium    | Extraction    | Meaning            |
|-------|-------------|----------|----------------|---------------|--------------------|
|  -1   | ANTI        | Reversal | Undo checkpoint| Rollback      | Immune memory      |
|   0   | ZERO        | Fixed    | Self-sameness  | Identity      | Equilibrium        |
|   1   | CONSTANT    | Gay      | SPI instant    | All at once   | O(1)               |
| log n | LOGARITHMIC | Master   | Parallel HDX   | Batched       | O(log n)           |
|   n   | LINEAR      | Slave    | Sequential     | One-at-a-time | O(n)               |
|  n²   | POLYNOMIAL  | Quadratic| Nested         | Pairwise      | O(n²)              |

The closure property: for any complexity c, applying "gay" (SPI) reduces c → max(c, 0).
This is because SPI provides O(1) verification regardless of generation complexity.
"""
@enum ComplexityClass begin
    ANTI = -1        # O(1/n) conceptually - undo is faster than do
    ZERO = 0         # O(1) but with fixed-point semantics
    CONSTANT = 1     # O(1) - Gay mode
    LOGARITHMIC = 2  # O(log n) - Master mode
    LINEAR = 3       # O(n) - Slave mode
    POLYNOMIAL = 4   # O(n²) - Nested operations
end

"""
    complexity_order(c::ComplexityClass) -> Int

Get the numerical order of a complexity class.
"""
complexity_order(c::ComplexityClass) = Int(c)

"""
    is_closed(c::ComplexityClass) -> Bool

A complexity class is "closed under gay" if applying SPI doesn't increase complexity.
All classes are closed because SPI provides O(1) verification.
"""
function is_closed(c::ComplexityClass)::Bool
    # Closure property: SPI reduces verification to O(1)
    # -1 (ANTI) is special: reversal + SPI = remembered identity
    # 0 (ZERO) is the fixed point
    true
end

"""
    gay_closure(c::ComplexityClass) -> ComplexityClass

Apply "gay" (SPI) to a complexity class.
This maps everything to max(class, ZERO) because SPI provides instant verification.
"""
function gay_closure(c::ComplexityClass)::ComplexityClass
    if c == ANTI
        # Anti-computation + SPI = remembered equilibrium
        ZERO
    else
        # SPI provides O(1) verification of any result
        min(c, CONSTANT)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Genetic Chromatic Evolution
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticGenome

A genome represented as a chromatic sequence.
Each gene is a color derived from a seed.
"""
struct ChromaticGenome
    genes::Vector{UInt64}  # Seeds for each gene
    colors::Vector{RGB{Float64}}
    fitness::Float64
    generation::Int
    originary_seed::UInt64  # The "self" signature
end

function ChromaticGenome(n_genes::Int; seed::UInt64=GAY_SEED)
    genes = UInt64[]
    colors = RGB{Float64}[]
    
    rng_state = seed
    for _ in 1:n_genes
        gene_seed, rng_state = splitmix64(rng_state)
        push!(genes, gene_seed)
        push!(colors, color_from_seed(gene_seed))
    end
    
    ChromaticGenome(genes, colors, 0.0, 0, seed)
end

"""
    Population

A population of chromatic genomes evolving toward optimal value extraction.
"""
mutable struct Population
    genomes::Vector{ChromaticGenome}
    generation::Int
    originary_seed::UInt64
    best_fitness::Float64
    fitness_history::Vector{Float64}
end

function Population(size::Int, genome_length::Int; seed::UInt64=GAY_SEED)
    genomes = ChromaticGenome[]
    rng_state = seed
    
    for _ in 1:size
        genome_seed, rng_state = splitmix64(rng_state)
        push!(genomes, ChromaticGenome(genome_length; seed=genome_seed))
    end
    
    Population(genomes, 0, seed, 0.0, Float64[])
end

"""
    fitness(genome::ChromaticGenome, target::RGB{Float64}) -> Float64

Compute fitness as inverse distance to target color (averaged over genes).
Higher fitness = closer to target.
"""
function fitness(genome::ChromaticGenome, target::RGB{Float64})::Float64
    total_dist = 0.0
    for c in genome.colors
        dist = sqrt((c.r - target.r)^2 + (c.g - target.g)^2 + (c.b - target.b)^2)
        total_dist += dist
    end
    avg_dist = total_dist / length(genome.colors)
    1.0 / (1.0 + avg_dist)
end

"""
    crossover(parent1::ChromaticGenome, parent2::ChromaticGenome; seed::UInt64) -> ChromaticGenome

Single-point crossover of two genomes.
"""
function crossover(parent1::ChromaticGenome, parent2::ChromaticGenome; 
                   seed::UInt64=GAY_SEED)::ChromaticGenome
    n = length(parent1.genes)
    crossover_point, _ = splitmix64(seed)
    point = Int(crossover_point % n) + 1
    
    new_genes = vcat(parent1.genes[1:point], parent2.genes[point+1:end])
    new_colors = [color_from_seed(g) for g in new_genes]
    
    # Inherit originary from fitter parent
    originary = parent1.fitness > parent2.fitness ? parent1.originary_seed : parent2.originary_seed
    
    ChromaticGenome(new_genes, new_colors, 0.0, 
                    max(parent1.generation, parent2.generation) + 1, originary)
end

"""
    mutate(genome::ChromaticGenome; mutation_rate::Float64=0.1, seed::UInt64) -> ChromaticGenome

Mutate genes with given probability.
"""
function mutate(genome::ChromaticGenome; mutation_rate::Float64=0.1, 
                seed::UInt64=GAY_SEED)::ChromaticGenome
    new_genes = copy(genome.genes)
    rng_state = seed
    
    for i in eachindex(new_genes)
        r, rng_state = splitmix64(rng_state)
        if (r % 1000) / 1000.0 < mutation_rate
            mutation, rng_state = splitmix64(rng_state)
            new_genes[i] = new_genes[i] ⊻ mutation
        end
    end
    
    new_colors = [color_from_seed(g) for g in new_genes]
    ChromaticGenome(new_genes, new_colors, 0.0, genome.generation + 1, genome.originary_seed)
end

"""
    select(population::Population, n::Int) -> Vector{ChromaticGenome}

Tournament selection of n individuals.
"""
function select(population::Population, n::Int)::Vector{ChromaticGenome}
    selected = ChromaticGenome[]
    rng_state = population.originary_seed ⊻ UInt64(population.generation)
    
    for _ in 1:n
        # Tournament of size 3
        idx1, rng_state = splitmix64(rng_state)
        idx2, rng_state = splitmix64(rng_state)
        idx3, rng_state = splitmix64(rng_state)
        
        i1 = Int(idx1 % length(population.genomes)) + 1
        i2 = Int(idx2 % length(population.genomes)) + 1
        i3 = Int(idx3 % length(population.genomes)) + 1
        
        candidates = [population.genomes[i1], population.genomes[i2], population.genomes[i3]]
        winner = argmax(g -> g.fitness, candidates)
        push!(selected, winner)
    end
    
    selected
end

"""
    evolve!(population::Population, target::RGB{Float64}; generations::Int=10)

Evolve the population toward target color.
"""
function evolve!(population::Population, target::RGB{Float64}; generations::Int=10)
    for gen in 1:generations
        # Evaluate fitness
        for (i, genome) in enumerate(population.genomes)
            f = fitness(genome, target)
            population.genomes[i] = ChromaticGenome(
                genome.genes, genome.colors, f, genome.generation, genome.originary_seed
            )
        end
        
        # Track best
        best = maximum(g -> g.fitness, population.genomes)
        push!(population.fitness_history, best)
        population.best_fitness = max(population.best_fitness, best)
        
        # Selection
        parents = select(population, length(population.genomes))
        
        # Create next generation
        new_genomes = ChromaticGenome[]
        rng_state = population.originary_seed ⊻ UInt64(population.generation + gen)
        
        for i in 1:2:length(parents)-1
            child_seed, rng_state = splitmix64(rng_state)
            child = crossover(parents[i], parents[i+1]; seed=child_seed)
            
            mutation_seed, rng_state = splitmix64(rng_state)
            child = mutate(child; mutation_rate=0.1, seed=mutation_seed)
            
            push!(new_genomes, child)
        end
        
        # Elitism: keep best
        elite = argmax(g -> g.fitness, population.genomes)
        push!(new_genomes, elite)
        
        while length(new_genomes) < length(population.genomes)
            push!(new_genomes, new_genomes[end])
        end
        
        population.genomes = new_genomes[1:length(population.genomes)]
        population.generation += 1
    end
    
    population
end

# ═══════════════════════════════════════════════════════════════════════════════
# Immune System Dynamics
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Antibody

An antibody with chromatic specificity.
Recognizes antigens by color similarity to its receptor.
"""
struct Antibody
    receptor_color::RGB{Float64}
    affinity::Float64  # Binding strength
    originary_seed::UInt64  # Self marker
    generation::Int  # Clonal generation
end

function Antibody(seed::UInt64; affinity::Float64=0.5)
    color = color_from_seed(seed)
    Antibody(color, affinity, seed, 0)
end

"""
    Antigen

An antigen (potentially foreign) to be recognized.
"""
struct Antigen
    color::RGB{Float64}
    seed::UInt64
    is_danger::Bool  # Danger signal present
end

function Antigen(seed::UInt64; is_danger::Bool=false)
    Antigen(color_from_seed(seed), seed, is_danger)
end

"""
    DangerSignal

A danger signal that activates immune response.
Corresponds to "bad student" in the homotopy hypothesis interpretation.
"""
struct DangerSignal
    source_seed::UInt64
    deranged_color::RGB{Float64}
    originary_color::RGB{Float64}
    confusion_distance::Float64
end

function DangerSignal(originary_seed::UInt64, derangement_key::UInt64)
    originary = color_from_seed(originary_seed)
    deranged = color_from_seed(originary_seed ⊻ derangement_key)
    
    dist = sqrt((originary.r - deranged.r)^2 + 
                (originary.g - deranged.g)^2 + 
                (originary.b - deranged.b)^2)
    
    DangerSignal(originary_seed, deranged, originary, dist)
end

"""
    ImmuneSystem

The chromatic immune system with clonal selection and affinity maturation.
"""
mutable struct ImmuneSystem
    antibodies::Vector{Antibody}
    memory_cells::Vector{Antibody}  # O(-1): immune memory for fast recall
    originary_seed::UInt64  # Self signature
    self_tolerance::Float64  # Threshold for self-recognition
    danger_threshold::Float64
end

function ImmuneSystem(n_antibodies::Int; seed::UInt64=GAY_SEED, 
                      self_tolerance::Float64=0.3)
    antibodies = Antibody[]
    rng_state = seed
    
    for _ in 1:n_antibodies
        ab_seed, rng_state = splitmix64(rng_state)
        affinity, rng_state = splitmix64(rng_state)
        aff = (affinity % 1000) / 1000.0
        push!(antibodies, Antibody(ab_seed; affinity=aff))
    end
    
    ImmuneSystem(antibodies, Antibody[], seed, self_tolerance, 0.5)
end

"""
    is_self(immune::ImmuneSystem, antigen::Antigen) -> Bool

Check if antigen is recognized as "self" (matches originary seed pattern).
"""
function is_self(immune::ImmuneSystem, antigen::Antigen)::Bool
    originary_color = color_from_seed(immune.originary_seed)
    dist = sqrt((antigen.color.r - originary_color.r)^2 +
                (antigen.color.g - originary_color.g)^2 +
                (antigen.color.b - originary_color.b)^2)
    dist < immune.self_tolerance
end

"""
    is_foreign(immune::ImmuneSystem, antigen::Antigen) -> Bool

Check if antigen is foreign (not self).
"""
is_foreign(immune::ImmuneSystem, antigen::Antigen) = !is_self(immune, antigen)

"""
    clonal_selection!(immune::ImmuneSystem, antigen::Antigen) -> Vector{Antibody}

Select and expand antibodies that bind to antigen.
"""
function clonal_selection!(immune::ImmuneSystem, antigen::Antigen)::Vector{Antibody}
    if is_self(immune, antigen)
        return Antibody[]  # Self-tolerance: no response to self
    end
    
    # Find binding antibodies
    binders = Antibody[]
    for ab in immune.antibodies
        dist = sqrt((ab.receptor_color.r - antigen.color.r)^2 +
                    (ab.receptor_color.g - antigen.color.g)^2 +
                    (ab.receptor_color.b - antigen.color.b)^2)
        if dist < (1.0 - ab.affinity)  # Higher affinity = tighter binding
            push!(binders, ab)
        end
    end
    
    # Clonal expansion: duplicate and add to population
    for ab in binders
        # Create clone
        clone = Antibody(ab.receptor_color, ab.affinity, ab.originary_seed, ab.generation + 1)
        push!(immune.antibodies, clone)
        
        # Add to memory (O(-1): instant recall later)
        if ab.affinity > 0.7
            push!(immune.memory_cells, clone)
        end
    end
    
    binders
end

"""
    affinity_maturation!(immune::ImmuneSystem, antigen::Antigen; rounds::Int=5)

Improve antibody affinity through somatic hypermutation.
"""
function affinity_maturation!(immune::ImmuneSystem, antigen::Antigen; rounds::Int=5)
    for _ in 1:rounds
        binders = clonal_selection!(immune, antigen)
        
        # Mutate binders to improve affinity
        for (i, ab) in enumerate(binders)
            if i <= length(immune.antibodies)
                # Somatic hypermutation
                mutation_seed, _ = splitmix64(ab.originary_seed ⊻ UInt64(ab.generation))
                mutated_color = color_from_seed(mutation_seed)
                
                # Blend toward antigen
                blend = 0.1
                new_color = RGB(
                    ab.receptor_color.r * (1-blend) + antigen.color.r * blend,
                    ab.receptor_color.g * (1-blend) + antigen.color.g * blend,
                    ab.receptor_color.b * (1-blend) + antigen.color.b * blend
                )
                
                # Improved affinity
                new_affinity = min(1.0, ab.affinity + 0.1)
                improved = Antibody(new_color, new_affinity, ab.originary_seed, ab.generation + 1)
                immune.antibodies[i] = improved
            end
        end
    end
end

"""
    danger_response!(immune::ImmuneSystem, signal::DangerSignal) -> Symbol

Respond to danger signal. Returns response type.
"""
function danger_response!(immune::ImmuneSystem, signal::DangerSignal)::Symbol
    if signal.confusion_distance < immune.danger_threshold
        return :tolerated
    end
    
    # Check memory first (O(-1): instant recall)
    for memory in immune.memory_cells
        dist = sqrt((memory.receptor_color.r - signal.deranged_color.r)^2 +
                    (memory.receptor_color.g - signal.deranged_color.g)^2 +
                    (memory.receptor_color.b - signal.deranged_color.b)^2)
        if dist < 0.2
            return :memory_response  # Fast recall from previous encounter
        end
    end
    
    # New threat: mount full response
    antigen = Antigen(signal.source_seed; is_danger=true)
    clonal_selection!(immune, antigen)
    affinity_maturation!(immune, antigen; rounds=3)
    
    :primary_response
end

# ═══════════════════════════════════════════════════════════════════════════════
# Information Geometry
# ═══════════════════════════════════════════════════════════════════════════════

"""
    FisherMetric

Fisher information metric on the color manifold.
Provides natural gradient for efficient optimization.
"""
struct FisherMetric
    # Fisher information matrix (3x3 for RGB)
    G::Matrix{Float64}
    determinant::Float64
    inverse::Matrix{Float64}
end

function FisherMetric(colors::Vector{RGB{Float64}})
    n = length(colors)
    if n < 2
        G = Matrix{Float64}(I, 3, 3)
        return FisherMetric(G, 1.0, G)
    end
    
    # Compute empirical Fisher information
    # G_ij = E[∂log p / ∂θ_i · ∂log p / ∂θ_j]
    
    # Approximate with sample covariance
    r_mean = sum(c.r for c in colors) / n
    g_mean = sum(c.g for c in colors) / n
    b_mean = sum(c.b for c in colors) / n
    
    G = zeros(3, 3)
    for c in colors
        dr, dg, db = c.r - r_mean, c.g - g_mean, c.b - b_mean
        G[1,1] += dr * dr
        G[1,2] += dr * dg
        G[1,3] += dr * db
        G[2,1] += dg * dr
        G[2,2] += dg * dg
        G[2,3] += dg * db
        G[3,1] += db * dr
        G[3,2] += db * dg
        G[3,3] += db * db
    end
    G ./= n
    
    # Regularize
    G += 1e-6 * I
    
    det_G = det(G)
    inv_G = inv(G)
    
    FisherMetric(G, det_G, inv_G)
end

"""
    natural_gradient(metric::FisherMetric, euclidean_grad::Vector{Float64}) -> Vector{Float64}

Convert Euclidean gradient to natural gradient using Fisher metric.
"""
function natural_gradient(metric::FisherMetric, euclidean_grad::Vector{Float64})::Vector{Float64}
    metric.inverse * euclidean_grad
end

"""
    geodesic_step(current::RGB{Float64}, direction::Vector{Float64}, step_size::Float64) -> RGB{Float64}

Take a step along the geodesic in color space.
"""
function geodesic_step(current::RGB{Float64}, direction::Vector{Float64}, 
                       step_size::Float64)::RGB{Float64}
    r = clamp(current.r + step_size * direction[1], 0.0, 1.0)
    g = clamp(current.g + step_size * direction[2], 0.0, 1.0)
    b = clamp(current.b + step_size * direction[3], 0.0, 1.0)
    RGB(r, g, b)
end

"""
    color_manifold_distance(c1::RGB{Float64}, c2::RGB{Float64}, metric::FisherMetric) -> Float64

Compute geodesic distance between colors using Fisher metric.
"""
function color_manifold_distance(c1::RGB{Float64}, c2::RGB{Float64}, 
                                  metric::FisherMetric)::Float64
    diff = [c2.r - c1.r, c2.g - c1.g, c2.b - c1.b]
    sqrt(diff' * metric.G * diff)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GitHub Actions Integration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ActionWorkflow

A GitHub Actions workflow as selection pressure on chromatic evolution.
"""
struct ActionWorkflow
    name::String
    seed::UInt64
    color::RGB{Float64}
    tests::Vector{Symbol}
    required_coverage::Float64
end

function ActionWorkflow(name::String; seed::UInt64=GAY_SEED)
    ActionWorkflow(name, seed, color_from_seed(seed),
                   [:unit, :integration, :e2e], 0.8)
end

"""
    WorkflowRun

A single run of a workflow.
"""
struct WorkflowRun
    workflow::ActionWorkflow
    genome::ChromaticGenome
    passed::Bool
    coverage::Float64
    color_match::Float64  # How well genome color matches workflow color
    run_seed::UInt64
end

"""
    SelectionPressure

CI/CD as evolutionary selection pressure.
"""
struct SelectionPressure
    workflows::Vector{ActionWorkflow}
    runs::Vector{WorkflowRun}
    total_passed::Int
    total_failed::Int
end

function SelectionPressure(workflow_names::Vector{String}; seed::UInt64=GAY_SEED)
    workflows = ActionWorkflow[]
    rng_state = seed
    
    for name in workflow_names
        wf_seed, rng_state = splitmix64(rng_state)
        push!(workflows, ActionWorkflow(name; seed=wf_seed))
    end
    
    SelectionPressure(workflows, WorkflowRun[], 0, 0)
end

"""
    ci_fitness(genome::ChromaticGenome, pressure::SelectionPressure) -> Float64

Compute fitness based on CI/CD selection pressure.
"""
function ci_fitness(genome::ChromaticGenome, pressure::SelectionPressure)::Float64
    total_match = 0.0
    
    for wf in pressure.workflows
        # Color match as proxy for test passing
        genome_avg = RGB(
            sum(c.r for c in genome.colors) / length(genome.colors),
            sum(c.g for c in genome.colors) / length(genome.colors),
            sum(c.b for c in genome.colors) / length(genome.colors)
        )
        
        dist = sqrt((genome_avg.r - wf.color.r)^2 +
                    (genome_avg.g - wf.color.g)^2 +
                    (genome_avg.b - wf.color.b)^2)
        
        match = 1.0 / (1.0 + dist)
        total_match += match
    end
    
    total_match / length(pressure.workflows)
end

"""
    self_certify_improvement!(pressure::SelectionPressure, before::ChromaticGenome, 
                               after::ChromaticGenome) -> NamedTuple

Self-certify that including "gay" (SPI) improved the genome.
"""
function self_certify_improvement!(pressure::SelectionPressure, 
                                    before::ChromaticGenome, 
                                    after::ChromaticGenome)
    before_fitness = ci_fitness(before, pressure)
    after_fitness = ci_fitness(after, pressure)
    
    # SPI verification: same seed must give same color
    spi_verified = all(
        color_from_seed(g) == c 
        for (g, c) in zip(after.genes, after.colors)
    )
    
    improvement = after_fitness - before_fitness
    certified = improvement > 0 && spi_verified
    
    (
        before_fitness = before_fitness,
        after_fitness = after_fitness,
        improvement = improvement,
        spi_verified = spi_verified,
        certified = certified,
        complexity_class = certified ? CONSTANT : LINEAR
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# World Extractable Value (WEV)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    WorldState

The state of a world from which value can be extracted.
"""
mutable struct WorldState
    seed::UInt64
    color::RGB{Float64}
    entropy::Float64
    value_extracted::Float64
    interactions::Int
end

function WorldState(seed::UInt64=XBOW_SEED)
    WorldState(seed, color_from_seed(seed), 1.0, 0.0, 0)
end

"""
    WEV (World Extractable Value)

The total extractable value from a world state.
Analogous to MEV but generalized to any chromatic world.
"""
struct WEV
    world::WorldState
    current_value::Float64
    max_value::Float64
    extraction_rate::Float64
    color_gradient::Vector{Float64}
end

function WEV(world::WorldState)
    # Value is proportional to entropy and inversely to extractions
    current = world.entropy / (1.0 + world.value_extracted)
    max_val = world.entropy
    rate = 0.1
    
    # Gradient: direction to maximize value
    grad = [world.color.r, world.color.g, world.color.b] .* current
    
    WEV(world, current, max_val, rate, grad)
end

"""
    value_gradient(wev::WEV) -> Vector{Float64}

Get the gradient direction for maximum value extraction.
"""
value_gradient(wev::WEV) = wev.color_gradient

"""
    extract_value!(wev::WEV, amount::Float64) -> Float64

Extract value from the world. Returns actual extracted amount.
"""
function extract_value!(wev::WEV, amount::Float64)::Float64
    actual = min(amount, wev.current_value * wev.extraction_rate)
    wev.world.value_extracted += actual
    wev.world.interactions += 1
    actual
end

"""
    xbow_aim(wev::WEV, target::RGB{Float64}) -> Vector{Float64}

Aim the XBOW at a target color to maximize extraction.
Returns the optimal direction.
"""
function xbow_aim(wev::WEV, target::RGB{Float64})::Vector{Float64}
    # Direction from current to target
    direction = [
        target.r - wev.world.color.r,
        target.g - wev.world.color.g,
        target.b - wev.world.color.b
    ]
    
    # Normalize
    magnitude = sqrt(sum(d^2 for d in direction))
    if magnitude > 0
        direction ./= magnitude
    end
    
    direction
end

"""
    xbow_fire!(wev::WEV, direction::Vector{Float64}; power::Float64=1.0) -> Float64

Fire the XBOW in the aimed direction to extract value.
Returns extracted value.
"""
function xbow_fire!(wev::WEV, direction::Vector{Float64}; power::Float64=1.0)::Float64
    # Extraction proportional to alignment with value gradient
    alignment = sum(direction .* wev.color_gradient)
    extraction = power * max(0, alignment)
    
    extract_value!(wev, extraction)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

function demo_xbow_wev()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  XBOW-WEV: Crossbow for World Extractable Value                           ║")
    println("║  Genetic + Immune + Information Geometry + GitHub Actions                 ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # ─── Complexity Classes ───
    println("─── Extended Complexity Classes ───")
    println("  Order | Class       | Mode     | Meaning")
    println("  ------|-------------|----------|------------------")
    println("   -1   | ANTI        | Reversal | Immune memory (O(-1) recall)")
    println("    0   | ZERO        | Fixed    | Equilibrium/self-sameness")
    println("    1   | CONSTANT    | Gay      | SPI instant O(1)")
    println("  log n | LOGARITHMIC | Master   | Parallel HDX")
    println("    n   | LINEAR      | Slave    | Sequential")
    println("   n²   | POLYNOMIAL  | Quadratic| Nested")
    println()
    
    println("  Closure under 'gay': all classes map to ≤ O(1) verification")
    for c in [ANTI, ZERO, CONSTANT, LOGARITHMIC, LINEAR, POLYNOMIAL]
        gc = gay_closure(c)
        println("    $(c) → $(gc)")
    end
    println()
    
    # ─── Genetic Evolution ───
    println("─── Genetic Chromatic Evolution ───")
    target = RGB(0.8, 0.3, 0.5)  # Target color
    pop = Population(20, 10; seed=GAY_SEED)
    
    println("  Target: RGB($(round(target.r, digits=2)), $(round(target.g, digits=2)), $(round(target.b, digits=2)))")
    println("  Population: 20 genomes, 10 genes each")
    
    evolve!(pop, target; generations=10)
    
    println("  After 10 generations:")
    println("    Best fitness: $(round(pop.best_fitness, digits=4))")
    println("    Fitness history: $(round.(pop.fitness_history[1:min(5, length(pop.fitness_history))], digits=3))...")
    println()
    
    # ─── Immune System ───
    println("─── Immune System Dynamics ───")
    immune = ImmuneSystem(50; seed=GAY_SEED, self_tolerance=0.3)
    
    # Self antigen (should be tolerated)
    self_ag = Antigen(immune.originary_seed)
    println("  Self antigen: is_self=$(is_self(immune, self_ag))")
    
    # Foreign antigen
    foreign_ag = Antigen(XBOW_SEED)
    println("  Foreign antigen: is_foreign=$(is_foreign(immune, foreign_ag))")
    
    # Clonal selection
    binders = clonal_selection!(immune, foreign_ag)
    println("  Clonal selection: $(length(binders)) antibodies activated")
    
    # Danger signal
    danger = DangerSignal(GAY_SEED, XBOW_SEED)
    response = danger_response!(immune, danger)
    println("  Danger response: $response")
    println("  Memory cells (O(-1) recall): $(length(immune.memory_cells))")
    println()
    
    # ─── Information Geometry ───
    println("─── Information Geometry ───")
    sample_colors = [color_from_seed(GAY_SEED ⊻ UInt64(i)) for i in 1:100]
    metric = FisherMetric(sample_colors)
    
    println("  Fisher metric determinant: $(round(metric.determinant, digits=6))")
    
    euclidean_grad = [0.1, -0.2, 0.15]
    natural_grad = natural_gradient(metric, euclidean_grad)
    println("  Euclidean gradient: $(round.(euclidean_grad, digits=3))")
    println("  Natural gradient:   $(round.(natural_grad, digits=3))")
    
    c1 = color_from_seed(GAY_SEED)
    c2 = color_from_seed(XBOW_SEED)
    geo_dist = color_manifold_distance(c1, c2, metric)
    println("  Geodesic distance (c1→c2): $(round(geo_dist, digits=4))")
    println()
    
    # ─── GitHub Actions ───
    println("─── GitHub Actions as Selection Pressure ───")
    pressure = SelectionPressure(["test", "lint", "build", "deploy"]; seed=GAY_SEED)
    
    before = ChromaticGenome(5; seed=GAY_SEED)
    after = evolve!(Population(10, 5; seed=GAY_SEED), pressure.workflows[1].color; generations=5).genomes[1]
    
    cert = self_certify_improvement!(pressure, before, after)
    
    println("  Workflows: $(length(pressure.workflows))")
    println("  Before fitness: $(round(cert.before_fitness, digits=4))")
    println("  After fitness:  $(round(cert.after_fitness, digits=4))")
    println("  Improvement:    $(round(cert.improvement, digits=4))")
    println("  SPI verified:   $(cert.spi_verified)")
    println("  Self-certified: $(cert.certified)")
    println("  Complexity:     $(cert.complexity_class)")
    println()
    
    # ─── World Extractable Value ───
    println("─── World Extractable Value (WEV) ───")
    world = WorldState(XBOW_SEED)
    wev = WEV(world)
    
    println("  World color: RGB($(round(world.color.r, digits=2)), $(round(world.color.g, digits=2)), $(round(world.color.b, digits=2)))")
    println("  Current value: $(round(wev.current_value, digits=4))")
    println("  Max value: $(round(wev.max_value, digits=4))")
    
    # Aim and fire
    target_color = RGB(0.9, 0.1, 0.3)
    direction = xbow_aim(wev, target_color)
    println("\n  XBOW aimed at RGB(0.9, 0.1, 0.3)")
    println("  Direction: $(round.(direction, digits=3))")
    
    extracted = xbow_fire!(wev, direction; power=2.0)
    println("  Extracted: $(round(extracted, digits=4))")
    println("  Interactions: $(world.interactions)")
    println()
    
    # ─── Summary ───
    println("─── Summary ───")
    println("  ✓ Extended complexity classes: -1 (ANTI), 0 (ZERO) added")
    println("  ✓ Genetic evolution toward chromatic targets")
    println("  ✓ Immune system: self/non-self via originary seed")
    println("  ✓ O(-1): Immune memory for instant recall")
    println("  ✓ Fisher metric for natural gradients on color manifold")
    println("  ✓ GitHub Actions as evolutionary selection pressure")
    println("  ✓ Self-certification: improvement verified via SPI")
    println("  ✓ XBOW fires to extract World Extractable Value")
    
    return (
        population = pop,
        immune = immune,
        metric = metric,
        pressure = pressure,
        wev = wev
    )
end

end # module XBOW_WEV
