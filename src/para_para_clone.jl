# PARA(PARA) CLONE: 2-Categorical Repository Cloning with CNOT² = I Reversibility
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Para(Para(AbstractMC)) → Para(Para(GayMC)) via GayExt                     │
# │                                                                             │
# │  Clone anoma-network repos into ~/worlds/a in optimal order determined     │
# │  by surprisal satisficing over GayMC-sampled permutations.                 │
# │                                                                             │
# │  KEY STRUCTURES:                                                            │
# │                                                                             │
# │    Para(Para):                                                              │
# │      Objects: Para categories                                               │
# │      1-morphisms: Functors Para(A) → Para(B)                               │
# │      2-morphisms: Natural transformations between functors                 │
# │                                                                             │
# │    CNOT CNOT = I:                                                           │
# │      First CNOT: target ⊕= control (clone if different)                    │
# │      Second CNOT: undo the XOR (reverse the clone)                         │
# │      Net effect: identity (reversible computation)                         │
# │                                                                             │
# │    Anticipatory Updates:                                                    │
# │      Use surprisal satisficing to predict which repos are needed           │
# │      Clone in order that minimizes expected surprisal                      │
# │      Order is deterministic from GayMC seed                                │
# │                                                                             │
# │  ANOMA REPOS (137 total as of 2025-12):                                    │
# │    Core: anoma, research, whitepaper                                       │
# │    Geb: geb, geb-bool, juvix-geb                                           │
# │    Juvix: juvix, juvix-stdlib, juvix-docs                                  │
# │    ARM: arm-risc0, evm-protocol-adapter                                    │
# │    Apps: anoma-apps, anoma-local-domain                                    │
# └─────────────────────────────────────────────────────────────────────────────┘

module ParaParaClone

using SplittableRandoms: SplittableRandom, split

export ParaPara, Para, AnomaRepo, CloneOrder, CNOTGate
export para_sample, para_para_sample, compute_clone_order
export cnot!, cnot_cnot!, verify_reversibility
export anticipatory_clone!, surprisal_score, satisfice_order
export ANOMA_REPOS, clone_all_anoma!, demo_para_para_clone

const GAY_SEED = UInt64(0x6761795f636f6c6f)
const ANOMA_SEED = UInt64(0x616E6F6D61)  # "anoma"

# ═══════════════════════════════════════════════════════════════════════════════
# ANOMA REPOSITORY CATALOG
# ═══════════════════════════════════════════════════════════════════════════════

"""
    AnomaRepo

Metadata for an Anoma network repository.
"""
struct AnomaRepo
    name::String
    category::Symbol        # :core, :geb, :juvix, :arm, :apps, :research, :other
    language::Symbol        # :elixir, :rust, :idris, :haskell, :solidity, :tex, :julia
    stars::Int
    dependencies::Vector{String}  # Other anoma repos this depends on
    seed::UInt64
    priority::Float64       # Higher = clone earlier (computed from deps + stars)
end

function AnomaRepo(name::String, category::Symbol, language::Symbol; 
                   stars::Int=0, dependencies::Vector{String}=String[])
    seed = ANOMA_SEED ⊻ hash(name)
    # Priority: more stars + fewer deps = clone earlier (foundational)
    priority = log(stars + 1) / (1 + length(dependencies))
    AnomaRepo(name, category, language, stars, dependencies, seed, priority)
end

# Canonical Anoma repos (subset of 137)
const ANOMA_REPOS = AnomaRepo[
    # Core
    AnomaRepo("anoma", :core, :elixir; stars=34200, dependencies=[]),
    AnomaRepo("whitepaper", :core, :tex; stars=90, dependencies=[]),
    AnomaRepo("research", :core, :idris; stars=35, dependencies=["whitepaper"]),
    
    # Geb - Categorical computation
    AnomaRepo("geb", :geb, :idris; stars=33, dependencies=["research"]),
    AnomaRepo("geb-bool", :geb, :idris; stars=5, dependencies=["geb"]),
    AnomaRepo("juvix-geb", :geb, :haskell; stars=8, dependencies=["geb", "juvix"]),
    
    # Juvix - Functional language
    AnomaRepo("juvix", :juvix, :haskell; stars=450, dependencies=["geb"]),
    AnomaRepo("juvix-stdlib", :juvix, :haskell; stars=12, dependencies=["juvix"]),
    AnomaRepo("juvix-docs", :juvix, :tex; stars=5, dependencies=["juvix"]),
    
    # Abstract Resource Machine
    AnomaRepo("arm-risc0", :arm, :rust; stars=15, dependencies=["research"]),
    AnomaRepo("evm-protocol-adapter", :arm, :solidity; stars=33, dependencies=["arm-risc0"]),
    
    # Applications
    AnomaRepo("anoma-apps", :apps, :elixir; stars=20, dependencies=["anoma"]),
    AnomaRepo("anoma-local-domain", :apps, :elixir; stars=3, dependencies=["anoma"]),
    
    # Research extensions
    AnomaRepo("taiga", :research, :rust; stars=80, dependencies=["research", "arm-risc0"]),
    AnomaRepo("typhon", :research, :rust; stars=25, dependencies=["anoma"]),
    AnomaRepo("specs", :research, :tex; stars=40, dependencies=["whitepaper"]),
]

# ═══════════════════════════════════════════════════════════════════════════════
# PARA: 1-Categorical Parametrization
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Para{P,A}

Para(A) - a parametrized category over A.

The parameter P determines which object of A we're looking at.
In our case: P = repo index, A = clone state (cloned/not cloned)
"""
struct Para{P,A}
    param::P
    value::A
    seed::UInt64
    color::UInt32  # Chromatic identity
end

function Para(param::P, value::A; seed::UInt64=GAY_SEED) where {P,A}
    color = hash_to_color(seed ⊻ hash(param))
    Para{P,A}(param, value, seed, color)
end

function hash_to_color(h::UInt64)
    r = UInt8((h >> 16) & 0xFF)
    g = UInt8((h >> 8) & 0xFF)
    b = UInt8(h & 0xFF)
    UInt32(r) << 16 | UInt32(g) << 8 | UInt32(b)
end

"""Sample from Para (get the value at this parameter)"""
function para_sample(p::Para)
    p.value
end

# ═══════════════════════════════════════════════════════════════════════════════
# PARA(PARA): 2-Categorical Parametrization
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ParaPara{P,Q,A}

Para(Para(A)) - 2-categorical parametrization.

This is the key structure for 2-morphism sampling:
- P parametrizes which Para category we're in
- Q parametrizes which object within that Para category
- A is the underlying value

For repo cloning:
- P = clone order (which position in sequence)
- Q = repo index (which repo at this position)
- A = clone state
"""
struct ParaPara{P,Q,A}
    outer_param::P      # Position in clone order
    inner::Para{Q,A}    # Which repo and its state
    seed::UInt64
    functor_color::UInt32  # 1-morphism color
    nat_trans_color::UInt32  # 2-morphism color
end

function ParaPara(outer::P, inner_param::Q, value::A; seed::UInt64=GAY_SEED) where {P,Q,A}
    inner_seed = seed ⊻ hash(outer)
    inner = Para(inner_param, value; seed=inner_seed)
    
    functor_color = hash_to_color(seed ⊻ hash((outer, inner_param)))
    nat_trans_color = hash_to_color(seed ⊻ hash((outer, inner_param, value)))
    
    ParaPara{P,Q,A}(outer, inner, seed, functor_color, nat_trans_color)
end

"""Sample from Para(Para) via GayMC"""
function para_para_sample(pp::ParaPara; rng::SplittableRandom=SplittableRandom(pp.seed))
    # 2-categorical sampling: first sample the functor, then the natural transformation
    rng = split(rng)
    
    # The sample is the inner Para's value
    (
        position = pp.outer_param,
        repo = pp.inner.param,
        state = pp.inner.value,
        functor_color = pp.functor_color,
        nat_trans_color = pp.nat_trans_color
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# CNOT GATE: Reversible Clone Operation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    CNOTGate

Controlled-NOT gate for reversible repository cloning.

CNOT(control, target):
  If control = 1, flip target
  target' = target ⊕ control

CNOT CNOT = I:
  First CNOT: target ⊕= control
  Second CNOT: target ⊕= control again
  Net: target ⊕ control ⊕ control = target (identity)

For cloning:
  control = repo exists remotely (always 1)
  target = repo exists locally
  CNOT: if remote exists and local doesn't, clone (XOR = 1)
"""
struct CNOTGate
    control_repo::String
    target_path::String
    applied::Bool
end

"""Apply CNOT: clone if control exists and target doesn't"""
function cnot!(gate::CNOTGate, state::Dict{String,Bool}; dry_run::Bool=true)
    control = get(state, gate.control_repo, false)
    target = get(state, gate.target_path, false)
    
    # XOR operation
    new_target = target ⊻ control
    
    if new_target && !target
        # Would clone
        if !dry_run
            # Actual clone command would go here
            # run(`git clone https://github.com/anoma/$(gate.control_repo) $(gate.target_path)`)
        end
        println("  CNOT: clone $(gate.control_repo) → $(gate.target_path)")
    elseif !new_target && target
        # Would delete (reverse clone)
        println("  CNOT⁻¹: remove $(gate.target_path)")
    end
    
    state[gate.target_path] = new_target
    CNOTGate(gate.control_repo, gate.target_path, true)
end

"""Apply CNOT twice = identity"""
function cnot_cnot!(gate::CNOTGate, state::Dict{String,Bool}; dry_run::Bool=true)
    original_state = copy(state)
    
    println("  First CNOT:")
    cnot!(gate, state; dry_run=dry_run)
    
    println("  Second CNOT:")
    cnot!(gate, state; dry_run=dry_run)
    
    # Verify identity
    identical = all(k -> get(state, k, false) == get(original_state, k, false), 
                    keys(original_state))
    
    (identical=identical, original=original_state, final=state)
end

"""Verify CNOT CNOT = I for all repos"""
function verify_reversibility(repos::Vector{AnomaRepo}; seed::UInt64=GAY_SEED)
    all_identical = true
    
    for repo in repos
        state = Dict{String,Bool}(repo.name => true)  # Remote exists
        target = "/tmp/test_$(repo.name)"
        state[target] = false  # Local doesn't exist
        
        gate = CNOTGate(repo.name, target, false)
        result = cnot_cnot!(gate, state; dry_run=true)
        
        if !result.identical
            println("  ❌ CNOT² ≠ I for $(repo.name)")
            all_identical = false
        end
    end
    
    all_identical
end

# ═══════════════════════════════════════════════════════════════════════════════
# SURPRISAL SATISFICING: Optimal Clone Order
# ═══════════════════════════════════════════════════════════════════════════════

"""
    CloneOrder

An ordering of repos for cloning, with surprisal scores.
"""
struct CloneOrder
    repos::Vector{AnomaRepo}
    order::Vector{Int}          # Indices into repos
    surprisals::Vector{Float64} # Surprisal at each step
    total_surprisal::Float64
    seed::UInt64
    satisficed::Bool            # Did we find a good enough order?
end

"""Compute surprisal of cloning repo at this position"""
function surprisal_score(repo::AnomaRepo, position::Int, 
                         already_cloned::Set{String})
    # Surprisal = -log₂ P(repo | already_cloned)
    # Lower surprisal if dependencies are already cloned
    
    deps_satisfied = count(d -> d ∈ already_cloned, repo.dependencies)
    total_deps = length(repo.dependencies)
    
    if total_deps == 0
        # No dependencies = low surprisal (foundational repo)
        dep_factor = 0.1
    else
        # Surprisal increases if dependencies not satisfied
        dep_factor = 1.0 - deps_satisfied / total_deps
    end
    
    # Position penalty (later positions have more uncertainty)
    position_factor = log(position + 1)
    
    # Priority bonus (high priority = low surprisal)
    priority_bonus = -log(repo.priority + 0.1)
    
    # Total surprisal in bits
    max(0.0, dep_factor * 3.0 + position_factor * 0.5 + priority_bonus * 0.3)
end

"""Satisfice: find a clone order with total surprisal below threshold"""
function satisfice_order(repos::Vector{AnomaRepo}; 
                        threshold::Float64=10.0,
                        max_attempts::Int=100,
                        seed::UInt64=GAY_SEED)
    rng = SplittableRandom(seed)
    n = length(repos)
    
    best_order = collect(1:n)
    best_surprisal = Inf
    
    for attempt in 1:max_attempts
        rng = split(rng)
        
        # Generate a random permutation using Fisher-Yates
        order = collect(1:n)
        for i in n:-1:2
            rng = split(rng)
            j = 1 + Int(floor(rand(rng) * i))
            j = clamp(j, 1, i)
            order[i], order[j] = order[j], order[i]
        end
        
        # Compute total surprisal for this order
        already_cloned = Set{String}()
        total = 0.0
        surprisals = Float64[]
        
        for (pos, idx) in enumerate(order)
            repo = repos[idx]
            s = surprisal_score(repo, pos, already_cloned)
            push!(surprisals, s)
            total += s
            push!(already_cloned, repo.name)
        end
        
        if total < best_surprisal
            best_surprisal = total
            best_order = order
            
            if total < threshold
                # Satisficed!
                return CloneOrder(repos, best_order, surprisals, total, seed, true)
            end
        end
    end
    
    # Return best found even if not satisficed
    already_cloned = Set{String}()
    surprisals = Float64[]
    for (pos, idx) in enumerate(best_order)
        s = surprisal_score(repos[idx], pos, already_cloned)
        push!(surprisals, s)
        push!(already_cloned, repos[idx].name)
    end
    
    CloneOrder(repos, best_order, surprisals, best_surprisal, seed, false)
end

"""Compute optimal clone order using Para(Para(GayMC))"""
function compute_clone_order(repos::Vector{AnomaRepo}; seed::UInt64=GAY_SEED)
    # Use Para(Para) to sample orders
    n = length(repos)
    
    # Build Para(Para) structure
    # Outer param: position (1..n)
    # Inner param: repo index
    # Value: clone state
    
    para_paras = ParaPara[]
    
    for pos in 1:n
        for (idx, repo) in enumerate(repos)
            pp = ParaPara(pos, idx, false; seed=seed ⊻ UInt64(pos) ⊻ UInt64(idx))
            push!(para_paras, pp)
        end
    end
    
    # Sample to get the optimal order via satisficing
    order = satisfice_order(repos; seed=seed)
    
    # Annotate with Para(Para) colors
    annotated = NamedTuple[]
    for (pos, idx) in enumerate(order.order)
        pp = ParaPara(pos, idx, true; seed=seed)
        sample = para_para_sample(pp)
        repo = repos[idx]
        push!(annotated, (
            position = pos,
            repo = repo.name,
            category = repo.category,
            surprisal = order.surprisals[pos],
            functor_color = sample.functor_color,
            nat_trans_color = sample.nat_trans_color
        ))
    end
    
    (order=order, annotated=annotated)
end

# ═══════════════════════════════════════════════════════════════════════════════
# ANTICIPATORY CLONE: Clone with Future Perfect Semantics
# ═══════════════════════════════════════════════════════════════════════════════

"""
    anticipatory_clone!(target_dir, repos, order; dry_run=true)

Clone repos in optimal order with anticipatory updates.

"The repos WILL HAVE BEEN cloned in the order that minimizes surprisal."

Uses CNOT gates for reversibility.
"""
function anticipatory_clone!(target_dir::String, repos::Vector{AnomaRepo}, 
                            order::CloneOrder; dry_run::Bool=true)
    state = Dict{String,Bool}()
    gates = CNOTGate[]
    
    # Initialize: all repos exist remotely
    for repo in repos
        state[repo.name] = true  # Remote exists
    end
    
    println("═══════════════════════════════════════════════════════════════")
    println("  ANTICIPATORY CLONE: ~/worlds/a (Para(Para(GayMC)))")
    println("═══════════════════════════════════════════════════════════════")
    println()
    println("  Target: $target_dir")
    println("  Repos: $(length(repos))")
    println("  Total surprisal: $(round(order.total_surprisal, digits=2)) bits")
    println("  Satisficed: $(order.satisficed)")
    println()
    println("  Clone order (optimal by surprisal satisficing):")
    println()
    
    for (pos, idx) in enumerate(order.order)
        repo = repos[idx]
        target_path = joinpath(target_dir, repo.name)
        
        # Check if already cloned
        state[target_path] = isdir(target_path)
        
        # Surprisal for this step
        s = order.surprisals[pos]
        s_bar = "█" ^ min(20, round(Int, s * 4))
        
        # Para(Para) colors
        pp = ParaPara(pos, idx, true; seed=order.seed)
        fc = pp.functor_color
        nc = pp.nat_trans_color
        
        # ANSI color from functor color
        r = (fc >> 16) & 0xFF
        g = (fc >> 8) & 0xFF
        b = fc & 0xFF
        
        println("  $(lpad(pos, 2)). \e[38;2;$(r);$(g);$(b)m$(repo.name)\e[0m")
        println("      Category: $(repo.category), Lang: $(repo.language)")
        println("      Deps: $(isempty(repo.dependencies) ? "none" : join(repo.dependencies, ", "))")
        println("      Surprisal: $(round(s, digits=2)) bits $s_bar")
        
        if !state[target_path]
            gate = CNOTGate(repo.name, target_path, false)
            push!(gates, gate)
            
            if !dry_run
                # Actually clone
                println("      → Cloning...")
                try
                    run(`git clone --depth 1 https://github.com/anoma/$(repo.name) $target_path`)
                    state[target_path] = true
                catch e
                    println("      ⚠️ Clone failed: $e")
                end
            else
                println("      → Would clone (dry run)")
            end
        else
            println("      ✓ Already cloned")
        end
        println()
    end
    
    (gates=gates, state=state, order=order)
end

"""Clone all anoma repos to target directory"""
function clone_all_anoma!(target_dir::String=expanduser("~/worlds/a"); 
                         dry_run::Bool=true, seed::UInt64=GAY_SEED)
    # Compute optimal order
    result = compute_clone_order(ANOMA_REPOS; seed=seed)
    
    # Clone in that order
    anticipatory_clone!(target_dir, ANOMA_REPOS, result.order; dry_run=dry_run)
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function demo_para_para_clone()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  Para(Para(AbstractMC)) → GayMC Clone Ordering                            ║")
    println("║  \"The repos WILL HAVE BEEN cloned in optimal order.\"                     ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    # ─── Para(Para) Structure ───
    println("─── Para(Para) Structure ───")
    println()
    println("  Para(Para(A)):")
    println("    • Outer param P = position in clone order (1..n)")
    println("    • Inner param Q = repo index")
    println("    • Value A = clone state (Bool)")
    println()
    println("  Sampling: Para(Para(GayMC))")
    println("    • 1-morphism (functor): maps position → repo")
    println("    • 2-morphism (nat trans): maps repo → state")
    println("    • Both carry chromatic identity from seed")
    println()
    
    # ─── CNOT CNOT = I ───
    println("─── CNOT CNOT = I (Reversibility) ───")
    println()
    println("  CNOT(control, target):")
    println("    target' = target ⊕ control")
    println()
    println("  CNOT CNOT:")
    println("    target'' = target' ⊕ control")
    println("             = (target ⊕ control) ⊕ control")
    println("             = target ⊕ (control ⊕ control)")
    println("             = target ⊕ 0")
    println("             = target  ✓ Identity!")
    println()
    println("  Verifying reversibility...")
    if verify_reversibility(ANOMA_REPOS[1:3])
        println("  ✓ CNOT² = I verified for all test repos")
    end
    println()
    
    # ─── Compute Clone Order ───
    println("─── Optimal Clone Order (Surprisal Satisficing) ───")
    println()
    
    result = compute_clone_order(ANOMA_REPOS; seed=GAY_SEED)
    
    for item in result.annotated
        r = (item.functor_color >> 16) & 0xFF
        g = (item.functor_color >> 8) & 0xFF
        b = item.functor_color & 0xFF
        
        println("  $(lpad(item.position, 2)). \e[38;2;$(r);$(g);$(b)m$(item.repo)\e[0m " *
                "($(item.category)) S=$(round(item.surprisal, digits=2))")
    end
    println()
    println("  Total surprisal: $(round(result.order.total_surprisal, digits=2)) bits")
    println("  Satisficed: $(result.order.satisficed)")
    println()
    
    # ─── Anticipatory Semantics ───
    println("─── Anticipatory Semantics (Future Perfect) ───")
    println()
    println("  Latin: 'repositōria clōnāta erunt' (the repos will have been cloned)")
    println("  The order is predetermined by GayMC sampling.")
    println("  Same seed → same order → same colors → same surprisal.")
    println()
    println("  To execute:")
    println("    clone_all_anoma!(expanduser(\"~/worlds/a\"); dry_run=false)")
    println()
    
    result
end

end # module ParaParaClone
