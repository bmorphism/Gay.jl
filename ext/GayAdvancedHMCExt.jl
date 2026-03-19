# GayAdvancedHMCExt: Extension for AdvancedHMC.jl Integration
# ============================================================
#
# This extension provides full AdvancedHMC.jl integration with Gay.jl's
# chromatic parallelism and affect-driven prime lattice adaptation.
#
# Activated when AdvancedHMC is loaded alongside Gay.

module GayAdvancedHMCExt

using Gay
using Gay.GayAdvancedHMC: ChromaticHMCState, ChromaticMassMatrix, AffectHMCConfig,
    GayWorld, ZAHN, JULES, FABRIZ, WORLD_SEED, WORLD_EMOJI,
    affect_hmc_config, prime_to_step_size, prime_to_n_leapfrog,
    compute_affect_from_color, splitmix64, next_color,
    GAY_SEED, ZAHN_SEED, JULES_SEED, FABRIZ_SEED

using AdvancedHMC
using AdvancedHMC: Hamiltonian, HMCKernel, NUTS, Leapfrog, DiagEuclideanMetric
using AdvancedHMC: sample, MHStep, Adaptation, StanHMCAdaptor, WindowedAdaptation
using LogDensityProblems
using LogDensityProblemsAD
using ForwardDiff

export
    GayHamiltonian, GayNUTS, GayLeapfrog,
    GayDiagMetric, GayDenseMetric,
    ChromaticLogDensity, ChromaticADGradient,
    gay_hmc_sample, gay_nuts_sample,
    affect_adapted_nuts, world_specific_metric,
    GayHMCSampler, run_gay_hmc!

# ═══════════════════════════════════════════════════════════════════════════════════
# CHROMATIC LOG DENSITY WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════════

struct ChromaticLogDensity{F}
    log_density_fn::F
    dim::Int
    world::GayWorld
    seed::UInt64
    color::Tuple{Float64,Float64,Float64}
end

function ChromaticLogDensity(log_density_fn, dim::Int; 
                             world::GayWorld=ZAHN, 
                             seed::UInt64=GAY_SEED)
    world_seed = WORLD_SEED[world] ⊻ seed
    _, color = next_color(world_seed)
    ChromaticLogDensity(log_density_fn, dim, world, world_seed, color)
end

LogDensityProblems.dimension(ℓ::ChromaticLogDensity) = ℓ.dim
LogDensityProblems.capabilities(::Type{<:ChromaticLogDensity}) = 
    LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.logdensity(ℓ::ChromaticLogDensity, θ) = ℓ.log_density_fn(θ)

struct ChromaticADGradient{L<:ChromaticLogDensity}
    ℓ::L
end

function ChromaticADGradient(log_density_fn, dim::Int; 
                             world::GayWorld=ZAHN,
                             seed::UInt64=GAY_SEED)
    ℓ = ChromaticLogDensity(log_density_fn, dim; world=world, seed=seed)
    ChromaticADGradient(ℓ)
end

LogDensityProblems.dimension(∇ℓ::ChromaticADGradient) = ∇ℓ.ℓ.dim
LogDensityProblems.capabilities(::Type{<:ChromaticADGradient}) = 
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(∇ℓ::ChromaticADGradient, θ) = 
    LogDensityProblems.logdensity(∇ℓ.ℓ, θ)

function LogDensityProblems.logdensity_and_gradient(∇ℓ::ChromaticADGradient, θ)
    result = ForwardDiff.gradient(θ -> ∇ℓ.ℓ.log_density_fn(θ), θ)
    (∇ℓ.ℓ.log_density_fn(θ), result)
end

# ═══════════════════════════════════════════════════════════════════════════════════
# WORLD-SPECIFIC METRICS
# ═══════════════════════════════════════════════════════════════════════════════════

function world_specific_metric(dim::Int; world::GayWorld=ZAHN, seed::UInt64=GAY_SEED)
    world_seed = WORLD_SEED[world] ⊻ seed
    
    variances = ones(Float64, dim)
    
    for i in 1:dim
        _, color = next_color(world_seed ⊻ UInt64(i))
        
        if world == ZAHN
            variances[i] = 0.5 + color[1]
        elseif world == JULES
            variances[i] = 0.5 + color[2]
        else  # FABRIZ
            variances[i] = 0.5 + color[3]
        end
    end
    
    DiagEuclideanMetric(variances)
end

function GayDiagMetric(dim::Int; world::GayWorld=ZAHN, seed::UInt64=GAY_SEED)
    world_specific_metric(dim; world=world, seed=seed)
end

function GayDenseMetric(dim::Int; world::GayWorld=ZAHN, seed::UInt64=GAY_SEED)
    world_seed = WORLD_SEED[world] ⊻ seed
    
    M = Matrix{Float64}(I, dim, dim)
    
    for i in 1:dim
        for j in i:dim
            _, color = next_color(world_seed ⊻ UInt64(i * dim + j))
            if i == j
                M[i, j] = 0.5 + color[1]
            else
                coupling = (color[1] - 0.5) * 0.3
                M[i, j] = coupling
                M[j, i] = coupling
            end
        end
    end
    
    M = M * M'
    
    AdvancedHMC.DenseEuclideanMetric(M)
end

# ═══════════════════════════════════════════════════════════════════════════════════
# AFFECT-ADAPTED INTEGRATORS
# ═══════════════════════════════════════════════════════════════════════════════════

function affect_adapted_leapfrog(config::AffectHMCConfig; 
                                  world::GayWorld=ZAHN,
                                  seed::UInt64=GAY_SEED)
    world_seed = WORLD_SEED[world] ⊻ seed
    _, color = next_color(world_seed)
    
    affect = compute_affect_from_color(color, config.initial_prime)
    step_size = prime_to_step_size(config.initial_prime, affect)
    
    Leapfrog(step_size)
end

function affect_adapted_nuts(config::AffectHMCConfig;
                             world::GayWorld=ZAHN,
                             seed::UInt64=GAY_SEED,
                             max_depth::Int=10)
    integrator = affect_adapted_leapfrog(config; world=world, seed=seed)
    NUTS(integrator; max_depth=max_depth)
end

function GayLeapfrog(; world::GayWorld=ZAHN, 
                      seed::UInt64=GAY_SEED,
                      prime::Int=23)
    config = affect_hmc_config(initial_prime=prime)
    affect_adapted_leapfrog(config; world=world, seed=seed)
end

function GayNUTS(; world::GayWorld=ZAHN,
                  seed::UInt64=GAY_SEED,
                  prime::Int=23,
                  max_depth::Int=10)
    config = affect_hmc_config(initial_prime=prime)
    affect_adapted_nuts(config; world=world, seed=seed, max_depth=max_depth)
end

# ═══════════════════════════════════════════════════════════════════════════════════
# GAY HMC SAMPLER
# ═══════════════════════════════════════════════════════════════════════════════════

mutable struct GayHMCSampler
    world::GayWorld
    seed::UInt64
    config::AffectHMCConfig
    metric::DiagEuclideanMetric
    integrator::Leapfrog
    proposal::NUTS
    fingerprint::UInt64
    samples::Vector{Vector{Float64}}
    colors::Vector{Tuple{Float64,Float64,Float64}}
end

function GayHMCSampler(dim::Int;
                       world::GayWorld=ZAHN,
                       seed::UInt64=GAY_SEED,
                       config::AffectHMCConfig=affect_hmc_config())
    metric = world_specific_metric(dim; world=world, seed=seed)
    integrator = affect_adapted_leapfrog(config; world=world, seed=seed)
    proposal = NUTS(integrator)
    
    world_seed = WORLD_SEED[world] ⊻ seed
    _, color = next_color(world_seed)
    
    GayHMCSampler(world, world_seed, config, metric, integrator, proposal,
                  world_seed, Vector{Float64}[], [color])
end

function run_gay_hmc!(sampler::GayHMCSampler,
                      log_density_fn,
                      initial_θ::Vector{Float64},
                      n_samples::Int;
                      n_adapts::Int=1000)
    dim = length(initial_θ)
    
    ∇ℓ = ChromaticADGradient(log_density_fn, dim; 
                              world=sampler.world, 
                              seed=sampler.seed)
    
    hamiltonian = Hamiltonian(sampler.metric, ∇ℓ)
    
    adaptor = StanHMCAdaptor(
        MassMatrixAdaptor(sampler.metric),
        StepSizeAdaptor(sampler.config.target_acceptance, sampler.integrator)
    )
    
    samples, stats = sample(hamiltonian, sampler.proposal, initial_θ, 
                            n_samples + n_adapts, adaptor, n_adapts;
                            progress=true)
    
    for θ in samples[n_adapts+1:end]
        push!(sampler.samples, θ)
        
        sampler.seed, color = next_color(sampler.seed)
        push!(sampler.colors, color)
        sampler.fingerprint = sampler.fingerprint ⊻ sampler.seed
    end
    
    (samples=sampler.samples, 
     colors=sampler.colors, 
     fingerprint=sampler.fingerprint,
     stats=stats)
end

# ═══════════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL SAMPLING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════

function gay_hmc_sample(log_density_fn,
                        initial_θ::Vector{Float64},
                        n_samples::Int;
                        world::GayWorld=ZAHN,
                        seed::UInt64=GAY_SEED,
                        n_adapts::Int=1000,
                        prime::Int=23)
    dim = length(initial_θ)
    config = affect_hmc_config(initial_prime=prime)
    
    sampler = GayHMCSampler(dim; world=world, seed=seed, config=config)
    run_gay_hmc!(sampler, log_density_fn, initial_θ, n_samples; n_adapts=n_adapts)
end

function gay_nuts_sample(log_density_fn,
                         initial_θ::Vector{Float64},
                         n_samples::Int;
                         world::GayWorld=ZAHN,
                         seed::UInt64=GAY_SEED,
                         n_adapts::Int=1000,
                         max_depth::Int=10,
                         prime::Int=23)
    gay_hmc_sample(log_density_fn, initial_θ, n_samples;
                   world=world, seed=seed, n_adapts=n_adapts, prime=prime)
end

function parallel_gay_hmc(log_density_fn,
                          initial_θ::Vector{Float64},
                          n_samples::Int;
                          seed::UInt64=GAY_SEED,
                          n_adapts::Int=1000)
    dim = length(initial_θ)
    
    samplers = [
        GayHMCSampler(dim; world=ZAHN, seed=ZAHN_SEED ⊻ seed),
        GayHMCSampler(dim; world=JULES, seed=JULES_SEED ⊻ seed),
        GayHMCSampler(dim; world=FABRIZ, seed=FABRIZ_SEED ⊻ seed)
    ]
    
    results = Vector{Any}(undef, 3)
    
    Threads.@sync begin
        for (i, sampler) in enumerate(samplers)
            Threads.@spawn begin
                results[i] = run_gay_hmc!(sampler, log_density_fn, initial_θ,
                                          n_samples ÷ 3; n_adapts=n_adapts)
            end
        end
    end
    
    combined_fp = reduce(⊻, [r.fingerprint for r in results]; init=GAY_SEED)
    
    (results=results, 
     samplers=samplers,
     combined_fingerprint=combined_fp)
end

end # module
