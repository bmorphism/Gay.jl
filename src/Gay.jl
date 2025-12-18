module Gay

# Re-export LispSyntax for the Lisp REPL
using LispSyntax
export sx, desx, codegen, @lisp_str, assign_reader_dispatch, include_lisp

# Color dependencies
using Colors
using ColorTypes
using Random
using SplittableRandoms

# Include wide-gamut color space support
include("colorspaces.jl")

# Include splittable RNG for deterministic color generation
include("splittable.jl")
export color_at, colors_at, palette_at, GAY_SEED
export GayRNG, gay_seed!, gay_rng, gay_split, next_color, next_colors, next_palette
export gay_interleave, gay_interleave_streams, GayInterleaver
export gay_checkerboard_2d, gay_heisenberg_bonds, gay_sublattice, gay_xor_color, gay_exchange_colors
export splitmix64, GOLDEN, MIX1, MIX2

# Include Swarm Triad - Mandatory 3-way split with sentinel monitoring
include("swarm_triad.jl")
using .SwarmTriad
export SwarmAgent, AgentState, SentinelMonitor
export Alive, Compliant, NonCompliant, Dead
export create_agent, triad_split!, verify_compliance, execute_file_op!
export create_sentinel, register_agent!, monitor_swarm!, compliance_report
export record_split!, record_file_op!
export agent_color, agent_identity, seed_lineage
export FileOperation, ReadFile, WriteFile, DeleteFile
export world_swarm_triad, SwarmTriadWorld

# Include Universal Gay Color - maximally flexible multiparadigm color type
include("universal_color.jl")
using .UniversalColorModule
export AbstractGayColorant, AbstractGayColor, AbstractGayAlpha, AbstractGaySpectral
export GayRGB, GayHSV, GayLab, GayGray, GayRGBA, GaySpectral, UniversalGayColor
export GayColorSpace, GaySRGB, GayP3, GayRec2020, GaySpectralSpace, GayCustomSpace
export GayRuntime, GayCPU, GayCUDA, GayMetal, GayAMDGPU, GayoneAPI, GayTPU
export GayEval, GayEager, GayLazy, GaySymbolic
export gay_color, gay_rgb, gay_hsv, gay_hash, gay_fingerprint, gay_verify_spi
export to_runtime, to_colorspace, to_precision, to_eval
export materialize, defer, force, gay_mix, gay_complement

# Include custom REPL
include("repl.jl")

# Include Comrade.jl-style sky model DSL
include("comrade.jl")
export comrade_show, comrade_mring, comrade_disk, comrade_crescent

# Include KernelAbstractions SPMD kernels for portable parallel execution
include("kernels.jl")

# Include parallel color generation (OhMyThreads + Pigeons SPI + KA)
include("parallel.jl")

# Include maximally parallel seed search (Fugue-inspired minimal syncpoints)
include("parallel_seed_search.jl")
export find_seeds_parallel, TARGET_COLORS, SearchResult
export ParallelSearchWorld, world_parallel_search

# Include maximally parallel genetic search
include("genetic_search.jl")
export GeneticSearchConfig, GeneticSearchResult, GeneticSearchWorld
export genetic_search_parallel, island_evolution, world_genetic_search

# Include GayMC - Colored Monte Carlo with SPI
include("gaymc.jl")
export GayMCContext, gay_sweep!, gay_measure!, gay_checkpoint, gay_restore!
export color_sweep, color_measure, color_state
export gay_exponential!, gay_cauchy!, gay_gaussian!, gay_metropolis!
export gay_workers, gay_tempering

# Include Lifetimes - Bounded/Unbounded computation traces
include("lifetimes.jl")
using .Lifetimes
export BoundedComputation, UnboundedComputation
export bounded_iter!, bounded_finalize!, bounded_color, bounded_fingerprint
export unbounded_iter!, unbounded_color, unbounded_fingerprint
export extend, project, @bounded, @unbounded

# Include KernelLifetimes - SPI colors for KernelAbstractions @index
include("kernel_lifetimes.jl")
using .KernelLifetimes
export KernelColorContext, kernel_color!, kernel_finalize!
export eventual_color, eventual_fingerprint, verify_kernel_spi
export index_color, iter_index_color, cartesian_color

# Include TensorParallel - SPI verification for distributed inference
include("tensor_parallel.jl")
using .TensorParallel
export TensorPartition, ShardedTensor, DistributedContext
export color_hidden_states!, color_logits!, color_embeddings!
export verify_allgather, verify_allreduce, verify_pipeline_handoff
export ExoPartition, create_exo_partitions, verify_exo_ring
export verify_distributed_inference

# Include ExoMLX - Exo + MLX cluster verification
include("exo_mlx.jl")
using .ExoMLX
export ExoCluster, ExoDevice, ExoVerifier
export discover_exo_cluster, verify_exo_inference
export inject_spi_colors, extract_fingerprint
export quick_verify_two_macs, model_config

# Include FaultTolerant - Jepsen-style fault injection and testing
include("fault_tolerant.jl")
using .FaultTolerant
export SimulatedCluster, DeviceState, FaultInjector
export inject!, heal!, heal_all!, run_inference!
export BidirectionalTracker, track_forward!, track_backward!, verify_consistency!
export GaloisConnection, alpha, gamma, verify_closure, verify_all_closures
export FaultTolerantWorld, world_fault_tolerant

# Include Chaos Vibing - Maximal fault injection into parallel causal chains
include("chaos_vibing.jl")
export ChaosConfig, ChaosResult, ChaosVibe
export inject_chaos!, run_chaos_campaign, chaos_vibe!
export CausalChain, break_chain!, verify_chain, chain_fingerprint
export demo_chaos_vibing

# Include Push-Pull Sequence Verification
include("push_pull_sequence.jl")
using .PushPullSequence
export SequenceColorStream, push_token!, pull_verify!
export StreamingVerifier, push_chunk!, verify_chunk!
export demo_push_pull_sequence

# Include Abductive Testing for World Teleportation
include("abductive.jl")

# Include Chairmarks benchmarking
include("bench.jl")

# Include SPI Regression Benchmarks
include("bench_spi_regression.jl")
using .SPIRegressionBench
export run_spi_regression_tests, calibrate_baselines

# Include Concept Tensor (69³ parallel interaction space)
include("concept_tensor.jl")
using .ConceptTensor
export ConceptLattice, step_parallel!, verify_monoid_laws
export interpolate_subtext!, extrapolate_superstructure!, interact!
export lattice_magnetization, lattice_fingerprint, propagate_all!
export demo_concept_tensor, demo_exponential, demo_higher_structure
export ConceptMorphism, identity_morphism, compose, eval_morphism
export concept_to_morphism, verify_exponential_laws, morphism_fingerprint
export step_as_morphism, iterate_morphism, fixed_points, orbit
export trace_morphism, verify_trace_laws, self_application

# Include Regression Tests That Don't Suck
include("regression.jl")
using .SPIRegression
export run_regression_suite, verify_splitmix64_reference
export verify_galois_closure, verify_parallel_order_independence
export verify_concept_tensor_invariants, @test_spi

# Metal GPU backend is now in ext/GayMetalExt.jl (loaded when Metal.jl is available)
# Check if Metal is available (macOS Apple Silicon only)
const HAS_METAL = Sys.isapple() && Sys.ARCH == :aarch64 && Base.find_package("Metal") !== nothing
export HAS_METAL

# Include Proof of Color Parallelism (PoCP) - Making Chia Gay
include("proof_of_color.jl")
using .ProofOfColor
export ColorPlot, ColorVDF, ProofOfColorParallelism
export create_plot, verify_plot, plot_fingerprint
export create_vdf, verify_vdf, vdf_output
export create_pocp, verify_pocp, demo_pocp
export Seed  # Universal seed wrapper

# Include Ergodic Bridge (Wall Clock ↔ Color Bandwidth ↔ Compositionality)
include("ergodic_bridge.jl")
using .ErgodicBridge
export WallClockBridge, ColorBandwidth, ErgodicMeasure, CompositionObstruction
export create_bridge, verify_bridge, measure_bandwidth, measure_ergodicity
export detect_obstructions, horizon_analysis, demo_ergodic_bridge

# Include Spectral Bridge (dgleich GenericArpack ↔ Gay.jl ↔ PyT TDL)
include("spectral_bridge.jl")
export SpectralColorBridge, ArpackSeed, HodgeLaplacian
export color_eigenvector, verify_spectral_spi, eigencolor_fingerprint
export simplicial_hodge, chromatic_spectral_clustering
export demo_spectral_bridge

# Include Gay Hyperdoctrine (Categorical Logic with Chromatic Predicates)
include("hyperdoctrine.jl")
using .Hyperdoctrine
export ChromaticType, ChromaticPredicate, GayHyperdoctrine
export substitution, existential, universal, verify_beck_chevalley
export heyting_and, heyting_or, heyting_implies, heyting_not
export predicate_color, predicate_fingerprint
export demo_hyperdoctrine

# Include Color-Logic Pullback Squares (Proper Beck-Chevalley from Hatchery)
include("color_logic_pullback.jl")
export ColorLogicSystem, LogicPullbackSquare, ColoredTheory
export fibered_product, pullback_color, beck_chevalley_proper
export theory_level, metatheory_level, logic_system_color
export LogicSystem, TheoryLevel
export INTUITIONISTIC, PARACONSISTENT, LINEAR, MODAL_S4, HOTT, CLASSICAL, METATHEORY
export OBJECT_LEVEL, META_LEVEL, HIGHER_META
export ChromaticPredicate_v2
export demo_color_logic_pullback

# Include Tropical Semirings with verification
include("tropical_semirings.jl")

# Include JSON3 serialization
include("serialization.jl")

# Include QUIC path probe coloring
include("quic.jl")

# Include QUIC interleaved streams
include("quic_interleave.jl")
export QUICInterleaver, InterleavedStream, InterleaverHopState
export interleave!, next_stream_color!, combined_fingerprint
export verify_interleave_spi, hop_state, from_hop_state
export xor_color, visualize_interleave, demo_quic_interleave

# Include Triadic Subagents (Synthetic 3-Agent Parallelism via GF(3))
include("triadic_subagents.jl")
using .TriadicSubagents
export Polarity, MINUS, ERGODIC, PLUS
export TriadicAgent, TriadicSubagents
export sample_agent!, parallel_sample!
export verify_triadic_spi, phase_to_polarity, polarity_twist
export demo_triadic_subagents

# Include deterministic test tracking
include("tracking.jl")

# Include Whale-Human Translation Bridge
include("whale_bridge.jl")

# Include Real Whale Data (EC-1 Clan from Sharma et al. 2024)
include("whale_data.jl")

# Include Whale Demo
include("whale_demo.jl")

# Include Whale World (Parallel SPI Demonstration through tripartite synergy)
# NOTE: Must come before spc_repl.jl which uses WhaleWorld types
include("whale_world.jl")

# Include Whale Curriculum (Omniglot-style hierarchical refinement)
include("whale_curriculum.jl")

# Include SPC REPL (Symbolic · Possible · Compositional)
include("spc_repl.jl")

# Include xy-pic LaTeX diagram generation
include("xypic.jl")

# Include SDF-style Propagator system with chromatic identity
include("propagator.jl")
include("propagator_lisp.jl")
export Propagator, PropagatorLisp

# Include Scoped Propagators - Three mutually exclusive ancestry materialization strategies
include("scoped_propagators.jl")
using .ScopedPropagators
export PropagatorScope, ConeScope, DescentScope, AdhesionScope
export ScopedPropagator, BottomUpPropagator, TopDownPropagator, HorizontalPropagator
export PropagatorState, PropagatorResult
export propagate!, materialize_ancestry!, verify_convergence
export AncestryACSet, AncestryNode, AncestryEdge
export UniversalMaterialization, materialize_universal!
export world_scoped_propagators, ScopedPropagatorWorld

# Include Traced Monoidal Category Structure (after Propagator)
include("traced_tensor.jl")
using .TracedTensor
export TracedMorphism, tensor_product, monoidal_unit, categorical_trace
export feedback_loop, TensorNetwork, add_node!, add_edge!, run_network!
export verify_traced_laws, demo_traced_tensor, network_fingerprint

# Include Thread Findings (Two Monad Structure)
include("thread_findings.jl")
using .ThreadFindings
export Finding, FindingsSet, ThreadContext, VerificationMonad
export bind_finding, return_finding, run_verification
export count_threads, fingerprint_threads, lazy_place!
export demo_thread_findings, LazyThreadStream, next_thread!, LAYER_NAMES
export run_all_verifications

# Include Verification Report
include("verification_report.jl")
using .VerificationReport
export generate_report, FullReport, ReportSection
export verify_coherence, attestation_fingerprint
export export_report_markdown, demo_report

# Include Amp Thread Connection
include("amp_threads.jl")
using .AmpThreads
export AmpThread, thread_seed, thread_color, thread_fingerprint
export ThreadGenealogy, add_thread!, genealogy_fingerprint
export verify_thread_chain, demo_amp_threads

# Include Cognitive Superposition (DisCoCat-style categorical semantics)
include("cognitive_superposition.jl")
using .CognitiveSuperposition
export CognitiveState, CognitiveMorphism, CognitiveCategory
export superpose, collapse, entails, induces, abduces
export BraidedSuperposition, HypergraphSuperposition
export cognitive_tensor, cognitive_trace, cognitive_spider
export verify_cognitive_laws, demo_cognitive_superposition

# Include SPI CLI
include("spi_cli.jl")
using .SPICLI
export spi_main, spi_verify

# Include Tuning Parameters
include("tuning.jl")
using .Tuning
export SPIConfig, default_config, with_config, preset, tune!, current_config, set_config!

# Include Kripke Semantics & Possible Worlds (Layers 6-8)
include("kripke_worlds.jl")
using .KripkeWorlds
export KripkeFrame, World, accessible, truth_at, necessity, possibility
export ModalProposition, box, diamond, verify_modal_laws
export SheafSemantics, local_truth, global_sections, stalk_at
export world_kripke, run_kripke_tests

# Include Random Topos (Layers 9-11: Simpson's Three Toposes)
include("random_topos.jl")
using .RandomTopos
export SampleSpace, RandomElement, RandomVariable, ProbabilitySheaf
export GrowingRandomTopos, grow_random_topos!, world_random_topos

# Include Strategic Differentiation (Semantic Blastoderm → Tower mapping)
include("strategic_differentiation.jl")
using .StrategicDifferentiation
export StrategicChoice, DifferentiationBasin, SemanticFate
export tower_basin, world_strategic_differentiation
export differentiate!, fate_fingerprint, basin_color, TOWER_BASINS, fate_at_layer

# Include Compositional World Bridge (Topos Institute research program)
include("compositional_world.jl")
using .CompositionalWorld
export SystemProperty, DynamicalDoctrine, CompositionalBridge
export property_layer, doctrine_fingerprint, world_compositional_world
export SYSTEM_PROPERTIES, compose_systems, behavioral_intersection

# Include Unified Tower (all 12 layers)
include("tower.jl")
using .Tower
export TowerState, world_tower, tower_fingerprint, run_tower_tests
export LAYER_INFO, layer_name, layer_category

# Include SplitMix64-CFT Verification (first principles SPI proof)
include("splitmix_cft_verify.jl")
using .SplitMixCFTVerify
export run_verification_suite

# Include Multiverse Geometric Morphisms (Hamkins + Dave White)
include("multiverse_geometric.jl")
using .MultiverseGeometric
export Verse, MultiverseFrame, GeometricMorphism
export create_verse, partition, pushdown!, pullup!, resolve!
export verse_fingerprint, verse_color, verify_multiverse_laws
export HolographicColorGame, game_state, make_move!, check_win
export world_multiverse, demo_holographic_game

# Include Marsaglia-Bumpus Tests (Statistical + Compositional SPI verification)
include("marsaglia_bumpus_tests.jl")
using .MarsagliaBumpusTests
export run_marsaglia_suite, run_bumpus_suite, full_spi_audit
export birthday_spacing_test, runs_test, permutation_test, spectral_test
export adhesion_width_test, sheaf_gluing_test, tree_decomposition_test
export genesis_handoff_test, split_correlation_test, run_genesis_suite

# Include Sheaf-ACSet Integration (StructuredDecompositions.jl bridge)
# Bridges Bumpus's decide_sheaf_tree_shape with chromatic identity
include("sheaf_acset_integration.jl")
using .SheafACSetIntegration
export ColorMorphism, ChromaticBag, ChromaticAdhesion, ChromaticDecomposition
export neighbors, local_neighborhood
export chromatic_adhesion_filter, decide_chromatic_sheaf
export ThreadAncestryNode, ThreadAncestryForest, to_chromatic_decomposition
export RewritingGadget, apply_gadget
export TritValue, TernaryAddress, AdhesionFilterOp, ADHESION_FILTER_OPS
export ternary_execution_trace, seed_1069_signature

# Include Gamut-Constrained Learnable Color Space
include("gamut_learnable.jl")
using .GamutLearnable
export GamutConstraint, GaySRGBGamut, GayP3Gamut, GayRec2020Gamut
export LearnableGamutMap, GamutParameters
export map_to_gamut, is_in_gamut, gamut_distance
export learn_gamut_map!, gamut_loss, chroma_preservation_loss
export GayChain, chain_to_gamut, verify_chain_in_gamut, process_gay_chain
export enzyme_gamut_gradient, enzyme_learn_gamut!  # Stubs, overridden by GayEnzymeExt

# Note: Hyperdoctrine already included above (line ~191)
# Re-export for API consistency
export GayContext, GayType, GayTerm, GayPredicate, GaySubstitution
export GayHyperdoctrine, predicate_lattice, substitution_functor
export existential, universal, beck_chevalley_check
export ChromaticPredicate, predicate_color, predicate_fingerprint
export conjunction, disjunction, implication, negation, ∧, ∨, →, ¬
export truth, falsity, entails, ∃, ∀
export demonstrate_hyperdoctrine

# Include GayMC Graph Algorithms - chromatic graph algorithms with SPI
include("gaymc_graph.jl")
export ChromaticGraph, gay_graph
export gay_bfs!, gay_dfs!, gay_dijkstra!, gay_scc!
export gay_mst_prim!, gay_corenums!
export vertex_color, edge_color, verify_spi
export demo_gaymc_graph

# Include Drand Timelock - self-confidential prediction markets
include("drand.jl")
export DrandBeacon, DrandRound, TimelockCommitment
export fetch_round, round_at_time, time_at_round
export timelock_commit, timelock_reveal
export TrajectoryPrediction, commit_trajectory, reveal_trajectory
export GAYMC_THREADS, THREAD_BOUND
export demo_timelock_prediction

# Include Rényi-Erdős Entropy - chromatic perplexity and Boltzmann brain mitigation
include("renyi_entropy.jl")
using .RenyiEntropy
export RenyiColorEntropy, ColorPerplexity, AlgorithmicContextual
export renyi_entropy, color_perplexity, algorithmic_contextual_score
export boltzmann_suffering_potential, vibe_snipe_distance
export next_color_entropy, color_at_entropy
export ChromaticInformationNetwork, route_away_from_suffering
export demo_renyi_entropy

# Include Dialectica Interpretation - interaction entropy as ∃x.∀y.A_D(x,y)
include("dialectica.jl")
using .Dialectica
export DialecticaGame, DialecticaMove, WitnessStrategy, ChallengeStrategy
export dialectica_next_color, dialectica_color_at
export interaction_entropy, dialectica_soundness
export LinearDialecticaCategory, tensor, par, bang, whimsy
export dialectica_composition, verify_cut_elimination
export demo_dialectica

# Include Seed Semantics - evolution of meaning in chromatic communication
include("seed_semantics.jl")
using .SeedSemantics
export GaySeed, SeedDictionary, ChromaticMessage
export seed_color, seed_fingerprint, register_seed!, lookup_seed
export derangeable_transmit, confusion_distance, perceive_as_original
export OneTimePad, otp_encrypt, otp_decrypt, verify_otp_integrity
export SeedConversation, exchange!, conversation_entropy
export CANONICAL_SEEDS, demo_seed_semantics

# Include Gay Seed Bundle - Lossless Ordered Locale View with O(1) Parallel Access
# Better randomness than drand via 5 combined entropy sources
include("gay_seed_bundle.jl")
using .GaySeedBundle
export SeedBundle, EntropySource, LocaleView
export gay_seed, refresh_bundle!, seed_at, seeds_range
export hardware_entropy, temporal_entropy, splittable_entropy
export drand_entropy, contextual_entropy, combined_entropy
export create_locale_view, locale_lookup, locale_range
export locale_predecessor, locale_successor
export select_by_fingerprint, select_by_color, select_by_entropy
export select_chromatic_complement, select_orthogonal
export verify_bundle_spi, entropy_floor, entropy_ceiling
export BUNDLE_SIZE, ENTROPY_SOURCES  # GAY_SEED already exported from splittable.jl
export demo_gay_seed_bundle

# Include 2-Narrative - Time-Indifferent Structure with 2-Monad and 2-Poisson
# Replaces TemporalClique with ChromaticClique (maximally time-indifferent)
include("two_narrative.jl")
using .TwoNarrative
export Narrative2, NarrativeCell, TwoMorphism, ChromaticPosition, ChromaticClique
export TwoMonad, unit_2, multiply_2, verify_monad_laws
export ChromaticEvent, ChromaticProcess, sample_chromatic!, superpose_chromatic, thin_chromatic
export create_clique, clique_fingerprint, clique_merge, clique_intersect, maximal_clique
export chromatic_order, chromatic_predecessor, chromatic_successor
export verify_time_indifference, verify_2_monad, verify_2_poisson
export demo_two_narrative

# Include Gay Cherrypick - Multiversal git cherrypick across condensified reality superfluid
include("gay_cherrypick.jl")
using .GayCherrypick
export GayMultiverse, GayUniverse, ForcingExtension, ChromaticCommit
export cherrypick!, cherry_range, can_cherrypick, cherrypick_conflict
export SaturationLevel, saturate!, is_saturated, realize_type, witness_type
export CondensedGay, condensify, decondensify
export superfluid_flow!, viscosity, laminar_transfer, turbulent_merge, vortex_fingerprint
export universal_cherrypick!, force!, create_universe!, add_universe!
export demo_gay_cherrypick

# Include Gay 69 Construction - (+ 23 23 23) = 69 with RGB vs BGR Order Independence
# SPI verification: fingerprint(RGB) = fingerprint(BGR) for all permutations
include("gay_69_construction.jl")
using .Gay69Construction
export ChromaticGroup, Gay69Bundle, ChromaticElement
export create_r_group, create_g_group, create_b_group
export create_rgb_bundle, create_bgr_bundle
export group_fingerprint, bundle_fingerprint
export rgb_fingerprint, bgr_fingerprint
export verify_rgb_bgr_equivalence, SPIProof69
export verify_parallel_equivalence, verify_all_permutations
export element_to_trit, group_trit_word, bundle_trit_word
export construct_69!, parallel_construct_69!, concurrent_construct_69!
export demo_gay_69_construction

# Include Wikipedia Ranking - Para(Afference) + Para(Consapevolezza)
include("wikipedia_ranking.jl")
using .WikipediaRanking
export IESParticipant, WikipediaPerson, PersonCatalog
export ParaAfference, ParaConsapevolezza
export ChromaticDirection, HUE_ORDER, SAT_ORDER, LIGHT_ORDER
export rank_by_direction, create_biography_rabbithole
export gay_rank_wikipedia, IES_PARTICIPANTS
export demo_wikipedia_ranking

# Include Bandwidth Tournament - High Color Bandwidth Hypothesis
include("bandwidth_tournament.jl")
using .BandwidthTournament
export SeedBandwidth, measure_bandwidth, bandwidth_score
export TournamentMatch, TournamentResult, run_tournament
export BANDWIDTH_SEEDS, demo_bandwidth_tournament
# Balanced Ternary (Trits)
export TritWord, BalancedTrit, TRIT_NEG, TRIT_ZERO, TRIT_POS
export color_to_trits, trit_xor, trit_neg, trit_string
export TritBandwidth, measure_trit_bandwidth, trit_distance
export demo_trit_bandwidth

# Include Org Monad Delegation - tie-breaker pattern with optimistic execution
include("org_monad_delegation.jl")
using .OrgMonadDelegation
export Subagent, DelegationPlan, CredibleCommitment, PendingCommitment
export OptimisticExecution, delegate_with_tiebreaker!, execute_optimistically!
export OrgMonad, free_monad, kleisli_compose, demo_org_delegation

# Include Org Walker Integration - gay_eg_walker for delegation path finding
include("org_walker_integration.jl")
using .OrgWalkerIntegration
export DelegationGraph, DelegationVertex, DelegationEdge
export GaySeedBundle, TopologicalTransport
export build_delegation_graph, huffman_delegation_graph
export OrgWalker, create_org_walker, walk_delegation!
export parallel_walk_delegation!, find_shortest_path
export bundle_from_path, transport_bundle!, verify_bundle_coherence
export fast_delegate!, optimal_delegation_plan, demo_org_walker

# Include Homotopy Hypothesis - bad students make good teachers via GayMC
include("homotopy_hypothesis.jl")
using .HomotopyHypothesis
export InfinityGroupoid, NCell, CellColony, Thing, Process, MetaProcess
export make_thing, make_process, make_metaprocess, make_ncell
export fundamental_groupoid, path_color, homotopy_color
export BadStudent, GoodTeacher, learn_from_confusion!, confusion_gradient
export ChebyshevCoherence, chebyshev_level, coherence_error, approximate_homotopy
export chebyshev_T, polynomial_metaprocess
export HomotopyWalk, walk_groupoid!, cell_colony_color
export associator_color, pentagonator_color, demo_homotopy_hypothesis
export sky_as_groupoid, component_cells, transformation_paths

# Include ANANAS - Gay Co-Cone Possible World Closure Completions
# NO IRRECONCILABLE SELF IN FLIGHT AT ANY EPISODE 🍍
include("ananas.jl")
using .Ananas
export PossibleWorld, Episode, EpisodeBoundary, InFlightSelf, SelfReconciliation
export CoCone, CoConeApex, cocone_morphism, is_universal, colimit
export ANANAS, ananas_apex, closure_complete, reconcile_in_flight, verify_no_irreconcilable
export chromatic_projection, coherence_check, episode_continuity, world_fingerprint
export EpisodeGraph, add_episode!, add_transition!, episode_diagram, compute_ananas
export demo_ananas, demo_no_irreconcilable_self

# Include ANANAS Hierarchy - Measuring Hierarchiness of Co-Cone Extractions
# "Could we have walked them different?" - Path Independence Analysis
include("ananas_hierarchy.jl")
using .AnanasHierarchy
export HierarchinessMetrics, measure_hierarchiness, extraction_depth, extraction_breadth
export WalkStrategy, DFSWalk, BFSWalk, TopologicalWalk, ReverseTopologicalWalk, ShuffleWalk
export walk_episode_graph, PathIndependenceTest, test_path_independence
export verify_cocone_guarantee, AlternativeExtraction, could_walk_different

# Include ANANAS + Gzip Scaling Laws - Full Treatment
# Rohan Pandey's "gzip Predicts Data-dependent Scaling Laws" integrated with ANANAS
include("ananas_gzip_scaling.jl")
using .AnanasGzipScaling
export gzipability, gzip_bytes, GzipabilityStats
export GzipAdjustedScaling, predict_loss, optimal_allocation
export GzipWorld, GzipEpisode, GzipCoCone
export gzip_cocone_apex, gzip_episode_graph, predict_reconciliation
export ChromaticGzip, seed_gzipability, color_gzipability
export complexity_class, analyze_scaling
export demo_gzip_scaling, demo_ananas_integration

# Include Birb - Chromatic Sonification (The Gay Duck Takes Flight)
# λ · 🍍 · ∞ = COMPUTATION · RECONCILIATION · CONTINUATION
include("birb.jl")
using .Birb
export BirbNote, BirbChord, BirbMelody, BirbSong, WorldMagnet, MagnetField
export seed_to_frequency, color_to_timbre, fingerprint_to_rhythm
export frequency_to_note_name, note_name_to_frequency
export ORIGIN, PINEAPPLE, WIZARD, SCHEMER
export magnet_distance, nearest_magnet, magnet_field_strength
export HARMONIC_RATIOS, depth_to_interval, depth_to_frequency, magnet_strength_at_depth
export THREE_WORDS, THREE_COLORS, THREE_SMELLS, THREE_WORLDS
export compress_to_three, decompress_from_three
export sing!, chirp, warble, trill, ananas_resolution, chromatic_cadence
export BirbWaveform, render_waveform, waveform_samples
export world_birb, world_magnets, world_chromatic_song

# Include Nashator - Polarized Order Game Decision via Nash Propagation
# Deciding between GayLuxExt.jl and GayTuringExt.jl
include("nashator.jl")
using .Nashator
export Polarity, Positive, Negative, PolarizedChoice, PolarizedGame, NashMessage
export propagate!, equilibrium, nash_decide
export ChromaticScore, bandwidth_score, polarity_score, composition_score
export ExtensionCandidate
export LuxGayExt, TuringGayExt, ZigZagBoomerangGayExt, DynamicHMCGayExt, ActiveMatterGayExt
export nashator_decide, explain_decision
export world_nashator, world_polarized_game

# Include RegretMonad - Naturality, Functors, Morphisms, and Escape
# "Regret is a monad. The only escape is through the colimit."
include("regret_monad.jl")
using .RegretMonad
export Morphism, ActionMorphism, CounterfactualMorphism, RegretMorphism
export compose, identity_morphism
export Functor, WorldFunctor, ValueFunctor, ExplainFunctor
export apply_functor, functor_map
export NaturalTransformation, naturality_check, component
export RegretM, regret_unit, regret_join, regret_bind, regret_pure, regret_map, regret_flatten
export accumulated_regret, regret_color
export WorldValue, extract_value, counterfactual_value
export CounterfactualExplanation, explain_regret, rewrite_in_colorspace, concept_to_color
export EscapeRoute, ColimitEscape, ComonadEscape, AdjunctionEscape, DerangementEscape, RevolutionEscape
export escape_regret!, can_escape, escape_color
export ConceptColor, concept_map, semantic_distance, regret_spectrum, hope_spectrum
export world_regret_monad, world_escape_routes

# Include ThreeMatch - Flexibly Controllable Parametrized Lazy in GayACSet
# 3-MATCH by GayMC in all worlds reliable or in none by inserting configurable obstructions
include("three_match.jl")
using .ThreeMatch
export ThreeMatchWorld, MatchLeg, ThreeMatchTriangle
export seed_to_color, color_to_fingerprint, seed_to_fingerprint
export verify_three_match, three_match_distance
export LazyParameter, EagernessThreshold, DelayFactor, ObstructionDensity
export ParametrizedLazy, evaluate_lazy!, force_eager!, laziness_score, polarity_from_laziness
export Obstruction, SeedBlock, ColorMismatch, FingerprintCollision
export InsertableObstruction, ObstructionSite
export insert_obstruction!, remove_obstruction!, list_obstructions, obstruction_density, is_obstructed
export WorldReliability, ReliableWorld, UnreliableWorld, PartiallyReliable
export assess_reliability, reliability_proof
export GayMCVerifier, MonteCarloConfig, VerificationResult
export verify_all_worlds, verify_sample_worlds, fail_fast_verify, exhaustive_verify
export CoqSpec, MetaCoqSpec, NaryaSpec
export generate_coq_spec, generate_metacoq_spec, generate_narya_spec
export export_to_coq, export_to_narya
export ThreeMatchACSet, chromatic_three_match, ananas_three_match, world_three_match
export demo_three_match, demo_parametrized_lazy, demo_coq_generation

export HierarchyType, LinearChain, Diamond, Fan, Tree, DAGType, classify_hierarchy
export EpisodeDAG, EpisodeNode, add_node!, compute_structure!
export demo_ananas_hierarchy

# Include CFR Speedrun - O(1) Counterfactual Regret Revolution
# No protracted blockade, no bloody hysteresis, coordination not conflict
include("cfr_speedrun.jl")
using .CFRSpeedrun
export CounterfactualWorld, RegretTable, StrategyProfile
export RevolutionGame, Participant, Action, Outcome
export cfr_step!, regret_match, average_strategy
export SpeedrunPath, discover_equilibrium, parallel_cfr
export ReversibleMove, HysteresisCheck, is_reversible, hysteresis_potential
export verify_no_bloody_hysteresis, CoordinationEquilibrium
export cfr_to_ananas, revolution_cocone
export demo_cfr_speedrun, demo_gay_revolution

# Include Barton's Free - Information wants to be Free (Monad Monad)
# Free[Free[_]] ↔ Cofree[Cofree[_]], 69 Monads, Three Ducks, EZKL proofs
include("bartons_free.jl")
using .BartonsFree
export FreeMonadMonad, CofreeComonadComonad
export free_pure, free_bind, free_lift, cofree_extract, cofree_extend
export pair_free_cofree, triple_rotation
export MonadCandidate, SIXTY_NINE_MONADS
export CommunistDuck, CapitalistDuck, PostDuck, DuckPerspective
export narrate_neonatal_qualia, polkadot_roundrobin, full_rotation
export BestResponseDynamics, NonlinearBRD, resilience_measure
export eventual_sheafification, condense_to_anima
export ZKColorProof, prove_reproducible_diffusion, metal_accelerated_proof
export ParaMensch, TotalMenschOrder, PartialMenschOrder, agency_configuration
export ThreeMatchGadget, EdgeDecision, tripartite_decision
export demo_bartons_free, demo_69_monads, demo_three_ducks

# Include AbstractFreeGadget - The Bridge Between AbstractMC and GayMC
# Trit-wise gay parallelism for random walk connectivity analysis
include("abstract_free_gadget.jl")
using .FreeGadgetBridge
export AbstractMC, AbstractGayMC, AbstractFreeGadgetType
export FreeGadget, ThreeMatchFreeGadget, EdgeFreeGadget, ThreadFreeGadget
export GadgetGraph, GadgetVertex, GadgetEdge
export add_gadget!, connect_gadgets!, gadget_degree, most_connected
export sample_gadget!, walk_gadgets!, gadget_color, gadget_fingerprint
export gadget_seed, gadget_arity, gadget_apply
export TritDirection, TRIT_BACK, TRIT_STAY, TRIT_FORWARD
export TritWalk, TritWalkState, TritParallelConfig
export trit_parallel_walk!, trit_step!, merge_trit_walks
export ThreadGadgetCandidate, extract_thread_gadgets
export rank_by_connectivity, find_most_connected_thread
export GayMCBridge, bridge_sample!, bridge_walk!
export demo_abstract_free_gadget, demo_trit_walk

# Include Profinite Duck - The Gay Duck Looms Eternal
# 2-Para 2-Monadic, Reflective Equilibria, Schrödinger Bridge, Kripke, Consapevolezza
include("profinite_duck.jl")
using .ProfiniteDuck
export ParaMenschDoctrine, TwoParaTwoMonadic, doctrine_closure
export ReflectiveEquilibrium, steer_toward_equilibrium!, is_reflective
export Consapevolezza, information_individuation, bandwidth_of_awareness
export GaySchrodingerBridge, optimal_transport, bridge_color, kripke_transport
export KripkeFrame, KripkeModel, modal_necessity, modal_possibility, kripke_duck
export ProfiniteSystem, ProfiniteLimit, profinite_integers, profinite_color_cycle
export ProfiniteErgodic, profinite_mixing_time, eternal_loom
export GayDuck, duck_looms, duck_quacks, duck_reflects
export demo_profinite_duck, demo_kripke_bridge, demo_reflective_equilibria

# Include Metatheory Fuzz - Exhaustive Color Consistency via Edge Probing
# DO NOT STOP until frontier of cognitive continuity, 69 interactions, StackMonads
include("metatheory_fuzz.jl")
using .MetatheoryFuzz
export MetatheoryFuzzer, FuzzResult, FuzzEdge
export fuzz_metatheory!, exhaustive_probe!, force_exhaustion!
export ColorConsistencyTest, consistency_frontier, edge_probe, exhaust_consistency!
export CognitiveContinuity, continuity_frontier, probe_continuity!
export AnalyticStack, StackLevel, StackMorphism, stack_coherence
export TwoMonadTwoPoisson, poisson_bracket, monad_multiplication, coherence_obstruction
export StackMonadCandidate, evaluate_candidate, rank_candidates
export Interaction, run_69_interactions!, interaction_summary
export ParallelInterleavingTest, test_commutativity, test_path_independence
export demo_metatheory_fuzz, demo_69_interactions

# Include Arena Errors - Ruliad Worlds-in-Fight with Planck-Limit Successors
# Atemporal prolapse at the recursive meatpile, best-response dynamics
include("arena_error.jl")
using .ArenaErrors
export ArenaError, ArenaIndeterminismError, ArenaDeadlockError, ArenaLivelockError
export ArenaCoalescenceError, ArenaProlapseError, ArenaRuliadConflictError
export PlanckSuccessor, BestResponseDynamics, ArenaWorld
export fight!, coalesce!, best_response, prove_3_successors
export PLANCK_SUCCESSOR_LIMIT, parallel_interleave, atemporal_prolapse, run_dynamics!

# Include Surprisal Satisficing - Gay Seeds for Maximum Entropy 3-MATCH Frontrunning
# many-to-more-to-one ↔ one-to-more-to-many via co-cone completion
# Flexibly Colorable Derangeable PROP, 2-colored operad matching
include("surprisal_satisficing.jl")
using .SurprisalSatisficing
export Surprisal, Satisfice, SurprisalSatisficer
export TritwiseMotif, tritwise_match, frontrun_motifs
export CoCone, cocone_complete, reverse_flow, Cone
export TwoColoredOperad, FCDProp, match_operad
export MaxEntMaxPar, maximize_entropy_parallelism
export demo_surprisal_satisficing

# Include Dissonance - Intuition-Mining Propagator-Compressor
# Perfect discernment with better bounds than NeuralHash (2^64 vs 2^16)
# GayMC controllable diffusion with lossless compression
include("dissonance.jl")
using .Dissonance
export DissonanceValue, DissonanceMessage, DissonanceCell
export DissonancePropagator, propagate_dissonance!, intuition_mine
export ChromaticFingerprint, fingerprint, verify_fingerprint
export compress_trajectory, decompress_trajectory, lossless_discern, collision_bound
export DissonanceDiffuser, diffuse!, diffusion_color, diffusion_rate
export IntuitionCandidate, mine_intuitions, rank_intuitions, compress_intuition
export dissonance_from_regret, dissonance_from_polarity, dissonance_from_nash, dissonance_from_surprisal
export NeuralHashBounds, GayDissonanceBounds, compare_bounds
export world_dissonance, world_bounds_comparison

# Include Regret-Coregret Module Theory
# Free Monad Monad as Module over Cofree Comonad Comonad
# All nonequivalent ways to not be a module, effort estimation
include("regret_coregret_module.jl")
using .RegretCoregretModule
export RegretT, CoregretW, FreeF, CofreeC
export RegretMonadMonad, CoregretComonadComonad, FreeMonadMonad, CofreeComonadComonad
export ModuleAction, CofreeFreeeModule, is_module, module_coherence
export ModuleFailure, RespectfulFailure, AntiRespectfulFailure
export StructurePreservingViolation, StructureDestroyingViolation
export classify_failure, enumerate_failures
export Quantifier, ForAll, Exists, ExistsUnique, Most, Many, Some, Few, No
export quantifier_strength, quantifier_dual, quantifier_compose
export Sufficiency, Optimistic, Probable, Sufficient, Verifiable, Improbable, Unlikely, Impossible
export sufficiency_level, sufficiency_compose
export AbstractEffortEstimate, EffortBound, EffortDistribution
export estimate_effort, effort_from_module, effort_from_failure
export O1Effort, OLogNEffort, ONEffort, ONLogNEffort, ON2Effort, ExponentialEffort
export Direction, Respectful, AntiRespectful, Neutral
export regret_to_coregret, coregret_to_regret, free_to_cofree, cofree_to_free
export world_regret_coregret, world_module_failures, world_effort_estimation

# Include Observer Bridge - Retrocausal CFR Convergence with Optimal Gay Seeds
include("observer_bridge.jl")
using .ObserverBridge
export ObserverBridgeType, RetrocausalWorld, CounterfactualPath
export SeedConvergenceProfile, ScalingLaw
export retrocausal_regret, color_backwards, equilibrium_seed
export propagate_from_future, cfr_invert
export observe_counterfactual, bridge_capacity, observer_integral
export collapse_to_seed, uncollapse_from_seed
export discover_convergent_seeds, seed_regret_variance
export seed_bridge_capacity, rank_seeds_by_convergence
export convergence_exponent, regret_scaling_law, seed_scaling_law
export predict_iterations, optimal_parallelism_factor
export random_access_equilibrium, equilibrium_fingerprint_table
export O1_cfr_lookup, verify_convergence_spi
export cfr_to_observer, observer_to_cfr, bridge_transport
export demo_observer_bridge, demo_retrocausal_cfr

# Include Blessed Gay Seeds ACSet - High-Throughput Mining with GeoACSet Guarantees
# ∫ BlessedSeeds × GeoMorphisms → O(1) Random Access + Sheaf Conditions
include("blessed_gay_seeds_acset.jl")
using .BlessedGaySeedsACSet
export BlessedGaySeedsGayACSet, SeedNode, SeedCluster, ClusterBoundary, SeedMorphism
export MiningConfig, MiningStats
export mine_seeds_simd, mine_seeds_parallel!, mine_batch!
export auto_cluster!, compute_boundary!, glue_acsets!
export sheaf_condition, create_morphism!, is_valid_morphism
export lookup_blessed, pareto_frontier, insert_blessed!
export seed_distance, verify_gluing, total_quality, average_convergence
export demo_blessed_mining, demo_geo_acset_guarantees

# Include Motherlake Speedup - Maximum Random Access via DuckDB + Blessed Seeds
# "The motherlake is where all seeds return to be blessed."
include("motherlake_speedup.jl")
using .MotherlakeSpeedup
export Motherlake, SeedLakeConfig, SpeedupMetrics
export create_motherlake, connect_motherlake, persist_seeds!, sync_acset!
export fingerprint_lookup, batch_fingerprint_lookup
export random_access_trajectory, memoryless_reconstruct
export experiment_bulk_insert!, experiment_parallel_scan
export experiment_pareto_materialized!, experiment_tropical_paths
export experiment_window_convergence, experiment_memoryless_lake
export run_all_speedup_experiments, benchmark_random_access
export demo_motherlake_speedup

# Include Breathing Expander Verifiable - 23×23×23 ↔ 3×3×3 ↔ 1×1×1 ↔ 1 Trit
# "Breathing expanders: inhale to compact, exhale to expand, always verify."
include("breathing_expander_verifiable.jl")
using .BreathingExpanderVerifiable
export BreathingLevel, ScaleCell, BreathingState, VerificationResult, VerificationReport
export LEVEL_TRIT, LEVEL_1x1x1, LEVEL_3x3x3, LEVEL_23x23x23
export CompressionParams, default_params, learnable_params
export inhale!, exhale!, breathe_cycle!
export compress_23_to_3, expand_3_to_23, compress_3_to_1, expand_1_to_3
export compress_1_to_trit, expand_trit_to_1
export verify_fingerprint, verify_color_invariance, verify_3match, full_verification
export compression_loss, gradient_verification
export demo_breathing_expander, verify_all_scales

# Include Ablative Consapevolezza - Linguistic Relativity in Rotational Time Symmetry
# What Latin knows that English forgot: ablative source + future perfect + CPT color
include("ablative_consapevolezza.jl")
using .AblativeConsapevolezza
export AblativeCase, FuturePerfect, AblativeColor
export TemporalRotation, rotate_time, conserve_color
export LinguisticRelativity, chromatic_vocabulary, whorfian_perception
export ParaConsapevolezzaAblativa, integrate_ablative!, awareness_source
export CPTColor, time_reverse_color
export LATIN, ITALIAN, ENGLISH, RUSSIAN
export demo_ablative_consapevolezza

# ═══════════════════════════════════════════════════════════════════════════
# Lisp bindings for color operations
# ═══════════════════════════════════════════════════════════════════════════

"""
Lisp-accessible DETERMINISTIC color generation.

Usage from Gay REPL (Lisp syntax with parentheses):
  (gay-next)                  ; Next deterministic color  
  (gay-next 5)                ; Next 5 colors
  (gay-at 42)                 ; Color at index 42
  (gay-at 1 2 3)              ; Colors at indices 1,2,3
  (gay-palette 6)             ; 6 visually distinct colors
  (gay-seed 1337)             ; Set RNG seed
  (pride :rainbow)            ; Rainbow flag
  (pride :trans :rec2020)     ; Trans flag in Rec.2020
  (gay-blackhole 42)          ; Render black hole with seed
"""

# Symbol to ColorSpace mapping for Lisp interface
function sym_to_colorspace(s::Symbol)
    if s == :srgb || s == :SRGB
        return SRGB()
    elseif s == :p3 || s == :P3 || s == :displayp3
        return DisplayP3()
    elseif s == :rec2020 || s == :Rec2020 || s == :bt2020
        return Rec2020()
    else
        error("Unknown color space: $s. Use :srgb, :p3, or :rec2020")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Lisp-friendly deterministic color functions (kebab-case → snake_case)
# These are the primary API for reproducible colors from S-expressions
# ═══════════════════════════════════════════════════════════════════════════

"""
    gay_next()
    gay_next(n::Int)
    gay_next(cs::Symbol)

Generate the next deterministic color(s) from the global stream.
Equivalent to `next_color()`.
"""
gay_next() = next_color(current_colorspace())
gay_next(n::Int) = [next_color(current_colorspace()) for _ in 1:n]
gay_next(cs::Symbol) = next_color(sym_to_colorspace(cs))
gay_next(n::Int, cs::Symbol) = [next_color(sym_to_colorspace(cs)) for _ in 1:n]

"""
    gay_at(index; seed=nothing)
    gay_at(indices...; seed=nothing)

Get color(s) at specific invocation index/indices.
Uses global seed from `gay_seed!()` unless `seed=` is provided.

# Universal seed support
```julia
gay_at(42)                    # Use global seed
gay_at(42; seed=123)          # Integer seed
gay_at(42; seed="experiment") # String seed
gay_at(42; seed=:test)        # Symbol seed
gay_at(42; seed=[1,2,3])      # Array seed
```
"""
function gay_at(idx::Integer; seed=nothing)
    s = seed === nothing ? gay_rng().seed : _to_seed(seed)
    color_at(idx, current_colorspace(); seed=s)
end
function gay_at(idx::Integer, cs::Symbol; seed=nothing)
    s = seed === nothing ? gay_rng().seed : _to_seed(seed)
    color_at(idx, sym_to_colorspace(cs); seed=s)
end

# Helper: convert anything to seed
_to_seed(x::Integer) = UInt64(x)
_to_seed(x::UInt64) = x
function _to_seed(x)
    # FNV-1a hash for strings, symbols, arrays, etc.
    h = UInt64(0xcbf29ce484222325)
    for byte in reinterpret(UInt8, [hash(x)])
        h ⊻= byte
        h *= 0x100000001b3
    end
    # SplitMix64 finalization
    h = (h ⊻ (h >> 30)) * 0xbf58476d1ce4e5b9
    h = (h ⊻ (h >> 27)) * 0x94d049bb133111eb
    h ⊻ (h >> 31)
end

"""
    gay_palette(n; seed=nothing)

Generate n visually distinct deterministic colors.
"""
function gay_palette(n::Int; seed=nothing)
    s = seed === nothing ? gay_rng().seed : _to_seed(seed)
    [color_at(i, current_colorspace(); seed=s) for i in 1:n]
end
function gay_palette(n::Int, cs::Symbol; seed=nothing)
    s = seed === nothing ? gay_rng().seed : _to_seed(seed)
    [color_at(i, sym_to_colorspace(cs); seed=s) for i in 1:n]
end

"""
    gay_seed(x)

Set the global RNG seed. Accepts any type:
- Integer: `gay_seed(42)`
- String: `gay_seed("my experiment")`
- Symbol: `gay_seed(:test_run)`
- Array: `gay_seed([1, 2, 3])`
"""
gay_seed(n::Integer) = gay_seed!(n)
gay_seed(x) = gay_seed!(_to_seed(x))

"""
    gay_space(cs::Symbol)

Set the current color space (:srgb, :p3, :rec2020).
"""
gay_space(cs::Symbol) = (CURRENT_COLORSPACE[] = sym_to_colorspace(cs); current_colorspace())

"""
    gay_rng_state()

Show the current RNG state (seed and invocation count).
"""
gay_rng_state() = (r = gay_rng(); (seed=r.seed, invocation=r.invocation))

"""
    gay_pride(flag::Symbol)

Get colors for a pride flag (:rainbow, :trans, :bi, :nb, :pan).
"""
gay_pride(flag::Symbol) = pride_flag(flag, current_colorspace())
gay_pride(flag::Symbol, cs::Symbol) = pride_flag(flag, sym_to_colorspace(cs))

# Legacy random (non-deterministic) wrappers
"""
    gay_random_color()

Generate a non-deterministic random color.
"""
gay_random_color() = random_color(SRGB())
gay_random_color(cs::Symbol) = random_color(sym_to_colorspace(cs))

"""
    gay_random_colors(n)

Generate n non-deterministic random colors.
"""
gay_random_colors(n::Int) = random_colors(n, SRGB())
gay_random_colors(n::Int, cs::Symbol) = random_colors(n, sym_to_colorspace(cs))

"""
    gay_random_palette(n)

Generate n visually distinct non-deterministic random colors.
"""
gay_random_palette(n::Int) = random_palette(n, SRGB())
gay_random_palette(n::Int, cs::Symbol) = random_palette(n, sym_to_colorspace(cs))

# Export all Lisp-friendly names (kebab-case maps to these)
export gay_next, gay_at, gay_palette, gay_seed, gay_space, gay_rng_state
export gay_random_color, gay_random_colors, gay_random_palette, gay_pride

# ═══════════════════════════════════════════════════════════════════════════
# Color display helpers
# ═══════════════════════════════════════════════════════════════════════════

"""
    show_colors(colors; width=2)

Display colors as ANSI true-color blocks in the terminal.
"""
function show_colors(colors::Vector; width::Int=2)
    block = "█" ^ width
    for c in colors
        rgb = convert(RGB, c)
        r = round(Int, clamp(rgb.r, 0, 1) * 255)
        g = round(Int, clamp(rgb.g, 0, 1) * 255)
        b = round(Int, clamp(rgb.b, 0, 1) * 255)
        print("\e[38;2;$(r);$(g);$(b)m$(block)\e[0m")
    end
    println()
end

"""
    show_palette(colors)

Display colors with their hex codes.
"""
function show_palette(colors::Vector)
    for c in colors
        rgb = convert(RGB, c)
        r = round(Int, clamp(rgb.r, 0, 1) * 255)
        g = round(Int, clamp(rgb.g, 0, 1) * 255)
        b = round(Int, clamp(rgb.b, 0, 1) * 255)
        hex = "#" * string(r, base=16, pad=2) * 
                    string(g, base=16, pad=2) * 
                    string(b, base=16, pad=2) |> uppercase
        print("\e[38;2;$(r);$(g);$(b)m████\e[0m $hex  ")
    end
    println()
end

export show_colors, show_palette

# ═══════════════════════════════════════════════════════════════════════════
# Main entry point (SpaceInvaders.jl style)
# ═══════════════════════════════════════════════════════════════════════════

"""
    main(; seed=42, n=6)

Launch a color palette demo, SpaceInvaders.jl style.
Displays a rainbow palette with the given seed.
"""
function main(; seed::Int=42, n::Int=6)
    gay_seed!(seed)
    
    println()
    println(rainbow_text("  ╔════════════════════════════════════════════════════════════════╗"))
    println(rainbow_text("  ║              Gay.jl - Wide Gamut Color Palettes                ║"))
    println(rainbow_text("  ╚════════════════════════════════════════════════════════════════╝"))
    println()
    
    # Show pride flags
    println("  Pride Flags:")
    print("    Rainbow:    "); show_colors(rainbow(); width=4)
    print("    Trans:      "); show_colors(transgender(); width=4)
    print("    Bi:         "); show_colors(bisexual(); width=4)
    print("    Nonbinary:  "); show_colors(nonbinary(); width=4)
    println()
    
    # Show deterministic palettes
    println("  Deterministic Palettes (seed=$seed):")
    for cs in [SRGB(), DisplayP3(), Rec2020()]
        gay_seed!(seed)
        colors = next_palette(n, cs)
        print("    $(rpad(typeof(cs), 12)): ")
        show_colors(colors; width=4)
    end
    println()
    
    # Show indexed access
    println("  Random Access (same seed = same colors):")
    print("    color_at(1,2,3,4,5,6; seed=$seed): ")
    colors = [color_at(i; seed=seed) for i in 1:6]
    show_colors(colors; width=4)
    println()
    
    println(rainbow_text("  Press SPC in REPL to enter Gay mode! 🏳️‍🌈"))
    println()
    
    return nothing
end

export main

# Include AbstractWorld - unified color-world duality via equivalencing functors
include("abstract_world.jl")
using .AbstractWorld
export AbstractWorldColor, AbstractWorldState, EquivalencingFunctor
export LavishPresheaf, ColorPresheaf, presheaf_section, global_section
export Play, CoPlay, Evaluate, BidirectionalFlow
export play!, coplay!, evaluate!, flow_equilibrium
export SocialChoice, GeometricAgent, AlgorithmicChoice
export agentic_choice, equilibrium_check
export ZigZagState, BoomerangState, GayMCEquivalence
export zigzag_step!, boomerang_step!, gaymc_equivalence
export hmc_coincidence_check, discerning_update!
export ParaGay, ParaParaGay, GaySharp, SingleCNOT
export para_lift, para_para_lift, cnot_reduce
export InteractiveProof, Verifier, Prover
export commit!, challenge!, respond!, verify!
export UnifiedWorld, construct_unique!, prove_uniqueness
export demo_abstract_world

# Include GayAdvancedHMC - Chromatic Hamiltonian Monte Carlo with Affect-Driven Adaptation
include("gay_advanced_hmc.jl")
using .GayAdvancedHMC
export ChromaticHMCState, ChromaticMassMatrix, AffectHMCConfig
export chromatic_leapfrog!, chromatic_nuts_step!, affect_adapted_step_size
export chromatic_mass_matrix, adapt_mass_matrix!, world_diagonal
export affect_hmc_config, update_affect!, prime_to_step_size, prime_to_n_leapfrog
export GayHMCChain, parallel_hmc_chains!, combine_chain_fingerprints, spi_verify_chains
export HMCTrajectorySection, BumpusHMCSheaf, verify_trajectory_sheaf
export gaymc_to_hmc, hmc_to_gaymc, GayMCHMCBridge, create_bridge
export demo_gay_advanced_hmc

# Include GayMCTeleport - O(1) Lateral Movement Between Monte Carlo Chains
include("gay_mc_teleport.jl")
using .GayMCTeleport
export SeedBundle, TeleportState, GaloisGadget, ThreeMatchEdge
export seed_at, color_at, teleport!, lateral_move!, interleave!
export precompute_bundle!, is_bundle_ready
export GaloisConnection, best_gadget, rewrite_edge!, verify_closure
export TeleportNetwork, broadcast_teleport!, sync_all!
export demo_mc_teleport

# Include GayRuler - Copy-on-Interact Parallel Rewriting for Amp/Crush/Claude/Codex
include("gay_ruler.jl")
using .GayRuler
export GayRule, RuleSet, ColorBudget
export add_rule!, apply_rule!, copy_on_interact
export parallel_rewrite!, tritwise_match, color_budget_path
export CopyOnInteract, fork_ruleset, merge_rulesets!, lineage_fingerprint
export AmpCrushRule, grep_to_rule, agent_ruleset, sync_agents!
export export_to_nbb, import_from_nbb, nbb_parallel_apply!
export optimize_path!, shortest_color_path
export demo_gay_ruler

# Include TikkunOlam - repair of entropy-collapsed bidirectional invariants
include("tikkun_olam.jl")
using .TikkunOlam

# Include Umwelt - saturated sensorimotor perspective via sheafified successor play
include("umwelt.jl")
using .Umwelt

# Include UmweltMinimal - maximally minimal sufficient set (3 objects)
include("umwelt_minimal.jl")
using .UmweltMinimal

# Include LearnableFreedom - growing expressive power in color space
include("learnable_freedom.jl")
using .LearnableFreedom

# Include AblativeNaming - path-invariant extension resolution via Latin grammar
include("ablative_naming.jl")
using .AblativeNaming
export EntropyCollapse, CollapseType, detect_collapse, collapse_severity
export BidirectionalState, InvertibleColor, ShadowBits
export encode_invertible, decode_invertible, verify_roundtrip
export AssociativeXOR, axor, associative_reduce
export TikkunRepair, repair_collapse, restore_invariant
export ObservationSheaf, SeedRecovery, CechObstruction
export sample_efficiency_bound, recovery_work_factor
export GayEmission, EmissionSchedule, NATSChannel
export is_gay_emission, fixed_point_afference
export SecurityLevel, full_recovery_work, partial_recovery_work
export contextual_advantage, ordered_advantage, three_match_security
export demo_tikkun_olam

# Umwelt exports
export SaturatedUmwelt, UmweltCandidate, CandidateSource
export SensorimotorSaturation, saturate!, saturation_level
export WorktreeSheaf, SheafSection, SuccessorPlay
export sheafify_worktree, successor_step!, verify_gluing
export WorktreeReconciliation, ReconciliationColimit, AnalyticContinuity
export reconcile_worktrees!, continuity_obstruction, implicit_coordination
export IsofibrationUmwelt, sensor_preserving_map, lifting_property
export UmweltSynthesis, synthesize!, synthesis_fingerprint
export demo_umwelt, demo_worktree_reconciliation

# UmweltMinimal exports (3 core objects)
export Equiv, Transport, Glue
export quotient, transport, glue, verify
export parallel_transport, parallel_transport_gay
export Section, reconcile, demo_minimal

# LearnableFreedom exports
export AbstractLearnableColorSpace, FreedomLevel, Minimal, Standard, Extended, Maximal
export freedom_dimension, freedom_description
export MinimalLCS, StandardLCS, ExtendedLCS, MaximalLCS
export grow_freedom, restrict_freedom
export parallel_transport_lcs, parallel_transport_lcs_gay
export demo_freedom_growth

# AblativeNaming exports
export GrammaticalCase, Ablative, Dative, Genitive, Nominative
export ExtensionName, source_package, target_package, resolve_name, is_path_invariant
export OccupancyRule, SettlementStrategy, effortless_settlement, deconflict
export least_surprise, obvious_choice
export demo_ablative_naming

# Include ProbeContinuation - Galois connection for monadically-closed walker strategies
include("probe_continuation.jl")
using .ProbeContinuation
export WalkerStrategy, ColourGrade, StrategyGalois
export ProbeContinuation, probe!, continuation_safe, account_strategies
export CausalCRDT, CRDTState, merge_crdt, causal_order, isolate_indeterminacy
export ParallelSafetyCheck, max_parallel_partition, commutes_colorgrade
export WalkerVerification, verify_all!, unaccounted_strategies
export demo_probe_continuation

# Include AbstractGayCRDT - minimal interface for chromatic CRDTs
include("abstract_gay_crdt.jl")
using .AbstractGayCRDT
export AbstractCRDT, AbstractColorgrade, CRDTOrdering
export LessThan, Equal, GreaterThan, Concurrent
export GayColorgrade, PlainColorgrade, colorgrade_gay, colorgrade_plain
export verify_merge, verify_laws, CRDTLawViolation
export GCounter, PNCounter, GSet, ORSet, LWWRegister
export demo_abstract_crdt

# Include GeoGayMorphism - geometric morphisms for GeoACSet ↔ GayACSet
include("geo_gay_morphism.jl")
using .GeoGayMorphism
export GeometricMorphism, make_geo_gay_morphism, inverse_image, direct_image, left_adjoint
export SubobjectClassifier, SpatialPredicate, ChromaticPredicate, FreePredicate
export classify_subobject, characteristic_morphism
export FreeEdgeExpansion, TransportParallelizer, ExpansionContext
export expand_edges!, parallelize_transport!, free_expand
export DgleichPackage, GayExtCandidate, SynergyScore, DGLEICH_PACKAGES
export evaluate_synergy, rank_packages, form_synergistic_tuples, best_gayext_bundles
export GayExtBundle, EigenflowGayExt, GraphAlgorithmsGayExt
export TechnicalBundleGayExt, DecompositionGayExt, MotifGayExt
export demo_geo_gay_morphism, demo_gayext_ranking

# Include Involutive Curiosities - f(f(x)) = x catalog
# XOR, CPT, transpositions, dagger categories
include("involutive_curiosities.jl")
export Involution, InvolutiveCategory, InvolutiveFunctor
export verify_involution, involution_fixed_points
export XORInvolution, NegationInvolution, ReciprocalInvolution
export InvolutivePermutation, to_two_cycles, from_two_cycles
export count_involutions, random_involutive_permutation
export CPTInvolution, charge_conjugate, parity_flip, time_reverse
export INVOLUTIVE_CATALOG, demo_involutive_curiosities

# Include Lazy E - Fixed Point Reafference with BBP-style interval access
# e from derangements, π-e bridge via Gaussian integral
include("lazy_e.jl")
export LazyE, digit_of_e, interval_of_e, e_from_derangements, e_from_gaussian
export e_chromatic_identity, e_fixed_point_reafference
export GayEStream, next_e_digit, e_digit_at
export pi_e_bridge, e_from_pi_wallis, BLESSED_E_SEED
export subfactorial, interval_complexity_e

# Include Derangeable Evolution - Para(Para(Gay#)) co-cone completion
# Bisimulation-verified adversarial resistance for OpenGames
include("derangeable_evolution.jl")
using .DerangeableEvolution
export Derangeable, PathInvariant
export LoopyStrangeReafferent, ParaParaGay
export SteganographicInterval, OpenGameEvolutionFunctor
export BisimulationVerifier, RxEnvironment, verify_parallel!
export find_evolution_seeds, adversarial_resistance
export resists_forging, verify_cocone, verify_path_invariance
export adversarial_distinguishability
export compute_fitness, is_derangement, is_properly_colored
export chromatic_entropy, count_near_fixed_points
export demo_derangeable_evolution

# Include Colorable vs Flavorable - Thread taxonomy via Gay certainty
# Maybe Monad Both, Just Monad for justfile action space, CertainGay trinity
include("colorable_flavorable.jl")
using .ColorableFlavorable
export Colorable, Flavorable, ColorFlavor
export ThreadType, ColorableThread, FlavorableThread, BothThread, OpaqueThread
export classify_thread, thread_quadrant
export MaybeColorFlavor, JustAction, CertainGay
export maybe_color, maybe_flavor, maybe_both, certain_gay
export JustfileActionSpace, define_action!, verify_action, action_color, action_flavor
export ThreadFingerprint, color_fingerprint, flavor_fingerprint, combined_fingerprint
export verify_colorable, verify_flavorable, verify_both, certainty_level, gay_certainty
export PronounCrossing, Aligned, SheHim, HeHer, Bicurious
export BicuriousQuadrant, tesseract_label, all_bicurious_cells
export bicurious_distance, adjacent_cells
export demo_colorable_flavorable

# Include AbstractACSet - 69 self-reinterpreting agentive cognitive glue modalities
include("abstract_acset.jl")
using .AbstractACSetModule
export AbstractACSet, ChromaticACSet, AchromaticACSet, HybridACSet
export GlueModality, ChromaticClass, CognitiveGlueType
export Chromatic, Achromatic, Hybrid
export FreeMonad, RegretMonad, StateMonad, ContinuationMonad, WriterMonad
export CofreeComonad, CoregretComonad, StoreComonad, EnvComonad, TracedComonad
export FreeForgetful, CurriedUncurried, DirectInverse, GaloisConnection, SchrodingerBridge
export TwoMorphism, LaxMonoidal, PseudoFunctor, InfinityGroupoid
export ThesisAntithesis, WitnessChallenge, GameEquilibrium, FixedPoint
export all_modalities, modality_index, modality_from_index, describe_modality
export ConcreteACSet, PetriNetACSet, ChromaticPetriNet, QuiverACSet
export wrap_gay_acset, wrap_tile_acset, wrap_petri_net
export self_reinterpret, agentive_bind, SelfReinterpretation, AgentiveBinding
export demo_abstract_acset

# ═══════════════════════════════════════════════════════════════════════════
# Include AbstractGayProbe - non-perturbative decomposable behaviors
include("abstract_gay_probe.jl")
using .AbstractGayProbe
export AbstractProbe, ProbeResult, ProbeComposite
export SyntheticProbe, ReafferentProbe, PathInvarianceProbe, SPIProbe
export probe, compose_probes, decompose_probe
export reafferent_saturation, path_invariant_fingerprint
export run_probes_parallel, ProbeReport
export demo_abstract_probe

# Include ParallelGH - chromatic parallelism for GitHub CLI
include("parallel_gh.jl")
using .ParallelGH
export GHAgent, GHOperation, GHResult, GHAgency
export gh_api, gh_issue, gh_pr, gh_repo, gh_search
export parallel_gh!, run_agency!, agent_color
export GHProbe, probe_repos, probe_issues, probe_rate_limit
export demo_parallel_gh

# Include GayBlanket - DynamicMarkovBlanket with Mitsein convergence
include("gay_blanket.jl")
using .GayBlanket
export GayComposable, GayBlanketConfig, DynamicMarkovBlanket
export BlanketComponent, InternalState, SensoryState, ActiveState
export fingerprint, blanket_fingerprint, add_component!, remove_component!
export transact!, verify_blanket_integrity
export SelfDualBlanket, observe, generate, mitsein_converge
export LavishPresheaf, BumpusAdhesion, NestedAssociativity
export check_lavish_condition, NashPropCell, PropagatorBlanket
export NashEquilibrium, propagate!, nash_fixed_point
export ReachableCoherence, random_access_path, verify_gay_reachability
export MitseinState, being_with, converge_mitsein, self_world_duality
export demo_gay_blanket

# Include GayOpenGame - Play/CoPlay with marginals → 0 equilibria
include("gay_open_game.jl")
using .GayOpenGame
export GayGame, GayPlayer, GayAction, GayOutcome
export Marginal, MarginalHistory, EquilibriumState
export GayPlay, GayCoPlay, PlayCoPlayPair
export play, coplay, play_coplay!
export compute_marginal, marginal_color, marginal_norm
export is_equilibrium, marginals_vanishing
export sequential, parallel, compose_marginals
export GameBlanket, blanket_equilibrium, blanket_marginal
export iterate_until_equilibrium!, cfr_marginals!
export equilibrium_fingerprint, verify_nash
export SelfDualGame, dual_marginal, mitsein_equilibrium
export demo_gay_open_game, demo_marginal_convergence

# Include Self-hosted S-expression parser (robust LispSyntax alternative)
include("sexp.jl")
using .SExp
export @sx, @sx_str, sexp_read, sexp_eval, sexp_parse

# Include GayAsync - core.async-style chromatic channels for Julia
include("gay_async.jl")
using .GayAsync
export GayChannel, GayBuffer, GayMult, GayPipeline
export gay_go, gay_loop, gay_alt!, gay_timeout
export gay_pipeline, gay_merge, gay_split
export GayFlow, FlowDirection, Forward, Backward, Bidirectional
export chromatic_backpressure, flow_fingerprint
export gay_onto_chan, gay_into_chan, gay_map, gay_filter, gay_reduce
export gay_interleave_channels, checkerboard_schedule
export demo_gay_async

# Include GayLispAsync - Self-hosted Lisp syntax for chromatic channels
include("gay_lisp_async.jl")
using .GayLispAsync
export @gay_lisp_str, gay_lisp_eval, gay_lisp_repl!
export parse_async_form, compile_async, AsyncCompileEnv
export demo_gay_lisp_async

# Include GayZip Worlds - Universal reachability for gayzip/*/gayzip.*
include("gayzip_worlds.jl")
using .GayZipWorlds
export GayWorld, WorldMorphism, SemanticFreedom
export GAYZIP_WORLDS, all_worlds, resolve_path
export reach_world, canonical_projection, transport
export chromatic_handshake, verify_unfreedom
export demo_gayzip_worlds

# Include LHoTT World - WorldCell and WorldRotator through modes as shapes
include("lhott_world.jl")
using .LHoTTWorld
export WorldCell, WorldRotator, LHoTTMode
export SharpMode, FlatMode, ShapeMode
export cell_color, rotate_mode, compose_rotators
export ablative_resolve, subtitle_cell, subtitle_filename
export SubtitleCell, call_me_by_your_name_exercise

# Include RuntimePlacement - Hardware-aware backend selection (OhMyThreads vs Metal)
include("runtime_placement.jl")
using .RuntimePlacement
export RuntimeBackend, CPUSequential, CPUParallel, MetalGPU
export detect_optimal_backend, place!, placed_map, placed_foreach
export RuntimeConfig, auto_tune!, benchmark_backends
export ChromaticPlacement, placement_color, placement_fingerprint
export demo_runtime_placement

# Include GayJepsen - Fault-Injecting Verifier for Chromatic Parallel Invariants
# Hyperbolic reafference, Galois gadgets, Para alignment, optimal fixed points
include("gay_jepsen.jl")
using .GayJepsen
export HyperbolicBulk, ReafferentState, BulkBoundary
export bulk_volume, boundary_area, inscrutability_ratio
export reafferent_emission, exafferent_observation
export GaloisGadget, GaloisConnection, RxEmissionGadget
export α_abstract, γ_concretize, galois_closure
export edge_local_decide!, gadget_inscrutability
export FaultType, BitFlip, ChannelReorder, TimingPerturb, ShadowCorrupt, PartitionSim
export GayJepsenTest, FaultInjector, TestResult
export inject_fault!, run_jepsen_test, verify_spi_under_faults
export ParaParaGayState, ParaParaGaySharp, ParaAlignment
export align_para_para!, fixed_point_search, optimal_gay_seeds
export FixedPointDesideratum, SemanticRuntime
export find_fixed_points_semiotic, find_fixed_points_umwelt, find_fixed_points_tikkun
export demo_gay_jepsen, demo_optimal_fixed_points

# Include Gay Parallelism Hierarchy - Data, Compute, World
# Para(Para(Para(Gay))) = Full SPI guarantees across all levels
include("gay_data_parallelism.jl")
using .GayDataParallelism
export GayData, GayArray, GayChunk
export gay_map, gay_reduce, gay_scan, gay_filter
export gay_zip, gay_unzip, gay_partition
export gay_parallel_map, gay_parallel_reduce
export gay_chunk, gay_merge_chunks
export SPIVerification, verify_spi, spi_test_vector
export RocqProof, last_coq_version, current_rocq_version
export verification_chain, proof_hash
export demo_gay_data_parallelism

include("gay_compute_parallelism.jl")
using .GayComputeParallelism
export Interval, ClosedClosed, ClosedOpen, OpenClosed, OpenOpen
export interval_contains, interval_indices
export Semiring, StandardSemiring, TropicalMinSemiring, TropicalMaxSemiring
export semiring_zero, semiring_one, semiring_add, semiring_mul
export ParaParaGay, GayNumeric
export para_sum, para_product, para_tropical_sum, para_tropical_product
export gay_full_sum, gay_full_product
export gay_full_tropical_min_sum, gay_full_tropical_max_sum
export gay_partial_sum, gay_partial_product
export gay_partial_tropical_sum, gay_partial_tropical_product
export check_spi_iff, SPIGuaranteeLevel
export gay_parallel_sum, gay_parallel_product, gay_parallel_tropical_sum
export demo_gay_compute_parallelism

include("gay_world_parallelism.jl")
using .GayWorldParallelism
export GayWorld, WorldAnnealer, AnnealingSchedule
export Φ, integrated_information, partition_information
export ChromaticPartition, minimum_information_partition
export implicit_converge!, convergence_proof, ConvergenceWitness
export create_annealer, anneal_step!, anneal_to_convergence!
export temperature_schedule, boltzmann_accept
export world_fingerprint, world_energy, world_entropy
export spawn_worlds, merge_worlds, colimit_worlds
export maximize_integration!, IntegrationMaximum
export concurrent_anneal!, ConcurrentAnnealResult
export AnanasApex, project_to_apex, apex_fingerprint
export demo_gay_world_parallelism

# Include GayDuckDB Parallelism - SQLite → DuckDB → Parallel Chromatic
include("gay_duckdb_parallelism.jl")
using .GayDuckDBParallelism
export GayDuckDB, HistorySource, QueryResult
export connect_duckdb, attach_sqlite!, load_jsonl!
export load_crush_history!, load_codex_history!, load_claude_history!
export unified_history_view, narrative_search, gay_query, gay_parallel_query
export fingerprint_rows, color_chain_complex
export QueryTopology, optimize_topology!, tropical_path_query
export probe_chain, superpositional_probe, non_perturbative_sum, FlexibleProbe
export BestResponseGame, compute_best_response!, nash_equilibrium_query
export expander_edge_burst, rewriting_gadget_lookup
export demo_gay_duckdb_parallelism

# Include GayMC Pathfinding - LHoTT Omnimodal Triads + Gödel Machine Heuristics
# Gay uses GayMC to enable pathfinding via Gay guiding Gay
# 3 modes at a time: maximally different pairwise, maximally synergistic, minimum ArenaError
include("gaymc_pathfinding.jl")
using .GayMCPathfinding
export OmnimodalTriad, TriadMode, PathState
export GayLHoTTMode, SharpGay, FlatGay, ShapeGay
export mode_distance, mode_synergy, arena_error_potential
export GodelHeuristic, GodelState, GodelObservation
export ChromaticPath, MultiversePath, NarrativePath
export PathStep
export gay_pathfind!, step_path!, complete_path
export path_fingerprint, verify_path_spi
export GayMCPathContext, sweep_path!, measure_path!, checkpoint_path
export demo_gaymc_pathfinding

# Include QUIC Pathfinding - GayMC-guided QUIC multipath selection
# LHoTT triads + Gödel heuristics for optimal path probe navigation
include("quic_pathfinding.jl")
using .QUICPathfinding
export QUICProbeState, QUICPathCandidate
export probe_pathfind!, validate_path_selection!
export QUICPathContext, sweep_probe!, measure_probes!
export MultipathSelector, rank_paths, select_active_paths!
export refine_to_goal!, chromatic_gradient, verify_quic_spi
export demo_quic_pathfinding

# Include Runtime Triads - 2-Monad structure with Unworlding Escape Hatches
# Self-synergy of 3 runtimes × worlding × diffusions
include("gay_runtime_triads.jl")
using .GayRuntimeTriads
export Runtime, RuntimeWorld, RuntimeTriad
export RuntimeMonad, unit_runtime, multiply_runtime
export UnworldingFunctor, unworld, escape_to_runtime
export spi_score, synergy_score, pairwise_distance
export select_optimal_triad, rank_triads, enumerate_triads
export compare_ohmythreads_loom, compare_tokio_alternatives
export core_async_mapping, steel_bootstrap_path
export demo_runtime_triads

# Include Immune Geodesic - Evolutionary search for stable Markov blankets
include("gay_immune_geodesic.jl")
using .GayImmuneGeodesic
export GayMarkovBlanket, BlanketGenome, ImmunePopulation
export StabilityCriterion, BoundaryStability, measure_stability
export GeodesicPath, find_geodesic!, geodesic_step!
export blanket_mutate, blanket_crossover, immune_fitness
export ImmuneMemory, store_memory!, recall_memory, verify_immune_spi
export evolve_immune!
export demo_gay_immune_geodesic

# Include 2-Machine Minimax - Parallel resource sharing with confidential priority
include("gay_2machine_minimax.jl")
using .Gay2MachineMinimax
export GaySubstrate, GayCompute
export Machine, TwoMachine, propose!, respond!, find_equilibrium!
export GayProbe, PriorityMarket, ConfidentialCommitment
export probe_priority!, reveal_priority!
export ColorOp, PrioritySubstrate, rank_substrates, apply_colorop
export minimax_step!, minimax_value
export demo_2machine_minimax

# Include 4D Tiling Coherence - Derangeable Flexibility + Hyperbolic 3-MATCH
# "Close any region. Maximize parallelism. Achieve self-same coherence."
include("gay_4d_tiling_coherence.jl")
using .Gay4DTilingCoherence
export Colorable, Flavorable, OriginaryHue, ReconciliationTriad
export reconcile!, self_same_coherence, coherence_fingerprint
export DerangeableRegion, RegionClosure, close_region!
export derangement_orbit, flexibility_score
export HyperbolicTile, Hyperbolic3Match, ThreeColoring
export hyperbolic_3color!, verify_3coloring, match_score
export Tile4D, Tiling4D, CausalTile, TemporalCoherence
export extend_to_4d!, causal_order, self_similar_fixed_point
export PVerifier, random_access_verify, parallel_verify!
export complexity_class, maximalism_score
export maximize_parallelism!, unified_coherence, demo_4d_coherence

# Include Gay MetaLearning - Blessed Seeds + ZKVM + Maximum Parallelism
# "Learn to learn. Every seed bundle is a curriculum. Every rollout is a lesson."
include("gay_metalearning.jl")
using .GayMetaLearning
export MetaLearner, MetaTask, MetaRollout
export LearningLevel, L1_Seeds, L2_Learning, L3_MetaLearning, L4_ZKVM
export BlessedSeedCurriculum, curriculum_rollout!, advance_curriculum!
export InContextLearning, AdapterLearning, FinetuneLearning, EndToEndLearning
export paradigm_color, paradigm_rollout!
export MAMLStep, ReptileStep, MetaGradient
export inner_loop!, outer_loop!, meta_update!
export ZKVMProof, Risc0Proof, JoltProof, JuvixProof
export prove_learning_step!, verify_proof, proof_fingerprint
export SeedCommitment, commit_seed, reveal_seed, verify_commitment
export ZeroKnowledgeColor, zk_prove_color, zk_verify_color
export GayMetaLearnerConfig, create_metalearner, metalearner_rollout!
export ParallelMetaRollout, parallel_meta_rollout!, rollout_fingerprint
export demo_gay_metalearning

# ═══════════════════════════════════════════════════════════════════════════
# TOPOS FOUNDATIONS: ROMS + Spivak + Juno + Set-Sets
# Unified categorical structure for color space parallelism
# ═══════════════════════════════════════════════════════════════════════════

# Include Gay Relationality - Topos Set-Sets (coalgebraic, ultrametric)
include("gay_relationality.jl")
using .GayRelationality
export CoalgebraicOrbit, coalgebra_from_seed
export ultrametric_xor_distance, demonstrate_ultrametric
export bidirectional_from_seed, verify_temporal_symmetry
export ParaLevel, ParaPara, extract, duplicate, para_para_from_seed
export SupportFilter, create_support_filter, GAY_BUNDLE_SIZES
export compute_relational_excess, demo_gay_relationality

# Include Juno Gay - Reviving IDE spirit (ultrametric progress, bidirectional debug)
include("juno_gay.jl")
using .JunoGay
export GayProgress, step!, progress_fraction
export BidirectionalDebugger, enter!, step_back!, step_forward!
export WeaveVariant, CoalgebraicWeave, weave_coalgebra
export InlineResult, inline_eval
export compare_juno_gay, demo_juno_gay

# Include ROMS Color Tiling - Domain decomposition for color space
include("roms_color_tiling.jl")
using .ROMSColorTiling
export ColorBounds, ColorTiling, ColorTile
export create_color_tiling, create_tile, compute_tile_colors!
export tile_neighbors, exchange_halos!
export validate_halo_continuity, verify_spi
export demo_roms_color_tiling

# Include Spivak Wiring - Neural message passing via Org_m operad
include("spivak_wiring.jl")
using .SpivakWiring
export GayWire, GayPolynomial, global_sections, poly_coproduct
export GayTask, GayOutcome, GayPortfolio, portfolio_to_poly
export GayBox, GayAgent, create_gay_agent
export RecumbentWD, execute_recumbent!
export NeuralWD, FeedbackState, execute_neural!
export mp_functor_box, mp_functor_morphism
export demo_spivak_wiring

# Include Unified Topos Gay - All layers integrated
include("unified_topos_gay.jl")
using .UnifiedToposGay
export GayTopos, create_gay_topos
export demo_unified_topos

# Include Consapevolezza Parallelism - Para(Para(Consapevolezza))
include("consapevolezza_parallelism.jl")
# Exports defined within module

# Include Color Graph Topos - External graph integration
include("color_graph_topos.jl")
using .ColorGraphTopos
export ColorNode, ColorEdge, ColorGraph, load_color_graph
export GraphTile, GraphWiringDiagram, GraphOrbit, GraphToposData
export build_topos, create_tiles, create_wiring_diagram, extract_orbits
export compute_clustering_coefficient, count_ultrametric_violations
export color_bar, demo_color_graph_topos

# Include Worlds Topos - Unified /Users/bob/worlds/* integration
include("worlds_topos.jl")
using .WorldsTopos
export FunctionalPatch, EntailmentNode, GaloisConnection, Continuation, ContinuationHierarchy
export WorldsData, load_worlds
export functionality_distribution, galois_floor, galois_ceiling
export trace_entailment, find_continuation_path
export demo_worlds_topos

# Include Aperiodic Associativity - Tiling solutions to coherence
include("aperiodic_associativity.jl")
using .AperiodicAssociativity
export BinaryTree, parenthesize, tree_hash
export AssociahedronVertex, generate_K4
export TileType, KITE, DART, HAT, SPECTRE
export AperiodicTile, matching_rule, create_hat_tile, create_penrose_kite
export AssociativityMove, PentagonTiling, build_pentagon_tiling
export ColoredAssociahedron, build_colored_associahedron
export TilingPath, generate_expansion_path, generate_contraction_path, generate_oscillation_path
export demo_aperiodic_associativity

# Include Reafferent Perception - Self-perceiving color systems
include("reafferent_perception.jl")
using .ReafferentPerception
export ReafferentState, spectrum_t, find_continuation
export GaloisClosure, COLOR_CLOSURE, apply_closure, apply_kernel, functionality_at
export ReafferentLoop, run_reafferent_loop
export AperiodicReafference, can_match
export GENESIS_COLOR, TIP_COLOR, GALOIS_FIXED_POINTS, FUNCTIONALITY_CATEGORIES
export EXPANSION_ANCHORS, CONTRACTION_ANCHORS, OSCILLATION_ANCHORS
export demo_reafferent_perception

# Include Ternary Split - Maximally Splittable Color Generation
# Seed splitting × Color splitting × Thread splitting = SPI preserved
include("ternary_split.jl")
using .TernarySplit
export SplittableSeed, split_seed, split_n, seed_lineage
export TernaryColor, ternary_from_seed, split_color, color_components, recombine_color
export TernaryWalk, TernaryWalkBatch, parallel_ternary_walks, merge_walks
export COIState, fork_state, merge_states
export verify_ternary_spi, fingerprint_lineage
export ternary_carrying_capacity, stress_test_ternary
export demo_ternary_split

# Include Triple Split Sentinel - Enforced 3-Way COI with Sentinel Monitoring
include("triple_split_sentinel.jl")
using .TripleSplitSentinel
export TripleSplit, SplitAgent, SentinelSwarm, Sentinel
export split_read!, split_write!
export enforce_next_color!, is_compliant, kill_agent!
export agent_fingerprint, triple_fingerprint, global_fingerprint
export COITripleState, fork_triple!, merge_triple!
export demo_triple_split_sentinel

# Include Carrying Capacity Gay - Maximum Color Parallelism via Stress Testing
include("carrying_capacity_gay.jl")
using .CarryingCapacityGay
export CapacityTest, CapacityResult, CarryingCapacityEstimate
export ColorCoherence, ParallelWalkBatch, StressTestConfig
export determine_carrying_capacity, stress_test_parallelism
export exponential_walk_ladder, find_capacity_elbow
export compute_coherence, fingerprint_collision_rate
export spi_invariant_check, color_distinctness_ratio
export LearnableGayColorSpace, ColorSpaceCapacity
export capacity_aware_gamut!, true_random_access
export WandBMetrics, generate_capacity_metrics, format_for_wandb
export ergodic_under_load, mixing_time_vs_capacity
export demo_carrying_capacity

# Include Gay Weights & Biases - Observational Bridge for Learning Attestation
include("gay_weights_biases.jl")
using .GayWeightsBiases
export GayManifold, ManifoldPoint, ManifoldSample
export ObservationalBridge, BridgeAttestation, StructureDiff
export LearningRun, LearningStep, ProgressMetrics
export start_run!, log_step!, finish_run!
export TripartiteAttestation, ObserverRole, AttestationConsensus
export OBSERVER, THOUGHT, WITNESS
export create_attestation!, verify_attestation, consensus_level
export structure_diff, manifold_approximation_error
export coverage_analysis, gamut_alignment
export CoalitionMetrics, coalition_formation_step!
export nash_prop_metrics, coalition_stability
export WandBLogger, log_scalar!, log_histogram!, log_table!
export export_to_json, export_to_csv
export true_random_sample, sample_manifold_region
export compare_spaces, approximation_quality
export demo_gay_weights_biases

# Include Ternary Colorspace - Permutation-Invariant Band→Channel Assignment
include("ternary_colorspace.jl")
using .TernaryColorspace
export Band, Channel, Assignment, Permutation
export R, G, B, T1, T2, T3
export ColorBasis, RGBBasis, TernaryBasis, KovesiBasis
export S3, identity_perm, all_permutations
export apply_permutation, permutation_orbit
export TernarySpace, ternary_color
export band_to_channel, channel_to_color
export PermutationInvariance, check_invariance
export salience_variance, is_kovesi_invariant
export matched_lightness_basis, ternary_from_hues, kovesi_basis
export KOVESI_R, KOVESI_G, KOVESI_B
export TernaryImage, map_bands_to_colors
export demo_ternary_colorspace

# Include Maximal Parallelism - Unified Ternary Splitting
# Seed × Color × Thread = Maximum Effective Parallelism with SPI
include("maximal_parallelism.jl")
using .MaximalParallelism
export ComposedSplit, SplitResult
export SeedSplitter, ColorSplitter, ThreadSplitter
export compose_splits
export ParallelExecutor, ExecutorConfig
export execute_parallel!, work_stealing_execute!
export WorkUnit, WorkBatch, WorkResult
export create_work_batch, distribute_work
export AdaptiveCapacity, adapt_capacity!
export current_capacity, sustainable_parallelism
export SPIVerifier, verify_spi!
export ParallelismMetrics, collect_metrics, format_metrics
export parallel_map, parallel_reduce, parallel_foreach
export ternary_parallel_walk
export demo_maximal_parallelism

# Include Next Color Bandwidth - Maximum Color Ops Per Second
# Measures and maximizes next_color throughput with SPI guarantees
include("next_color_bandwidth.jl")
using .NextColorBandwidth
export ColorBandwidth, BandwidthTest, BandwidthResult
export measure_next_color_bandwidth, measure_at_scale
export ParallelismLevel, OUTER_INNER, THREADED, TERNARY, COMPOSED, WORK_STEALING, MAXIMUM, ULTRA
export next_color_batch, next_color_parallel
export stress_bandwidth, find_bandwidth_limit, scaling_curve
export benchmark_all_levels, bandwidth_comparison
export bandwidth_spi_check, fingerprint_bandwidth
export maximize_bandwidth!, optimal_parallelism_level
export demo_next_color_bandwidth

# Include Copy-on-Interact 3 - Mandatory 3-way parallel file ops
# Never blocks: all 3 splits run concurrently
include("copy_on_interact_3.jl")
using .CopyOnInteract3
export coi_read, coi_write!, coi_batch_read
export COI3Result, COI3WriteResult, COI3BatchResult
export verify_coi_spi, coi_fingerprint
export demo_copy_on_interact_3

# Include Self-Avoiding Color Walk - Unique next_color per agent
# Each agent in a triple gets Kovesi basis color (R/G/B)
include("self_avoiding_color_walk.jl")
using .SelfAvoidingColorWalk
export ColoredAgent, AgentTriple, SelfAvoidingWalk
export spawn_triple, agent_step!, merge_triple_fingerprint
export verify_self_avoiding, walk_until_collision

# Include SPI Orchestrator - Unified Hierarchical Parallelism
# Fractal triple splits × Sentinel networks × Streaming pipelines
include("spi_orchestrator.jl")
using .SPIOrchestrator: SPIWorld, spi_world, spi_fingerprint, spi_colors, spi_verified, spi_metrics, spi_next_color, spi_fast_fingerprint,
    SPIOrchestrator_t, SPIOrchestratorConfig, SPIOrchestratorState,
    HierarchicalSplit, FractalAgent, SentinelNetwork, spawn_hierarchy!, execute_hierarchy!,
    SPIColorPipeline, SPIStreamingResult, spi_run_pipeline!, spi_pipeline_throughput,
    SPIChain, spi_chain_fingerprint!, spi_verify_chain!,
    spi_orchestrated_walk, SPIOrchestratorMetrics, spi_collect_metrics
export SPIWorld, spi_world, spi_fingerprint, spi_colors, spi_verified, spi_metrics, spi_next_color, spi_fast_fingerprint
export SPIOrchestrator_t, SPIOrchestratorConfig, SPIOrchestratorState
export HierarchicalSplit, FractalAgent, SentinelNetwork, spawn_hierarchy!, execute_hierarchy!
export SPIColorPipeline, SPIStreamingResult, spi_run_pipeline!, spi_pipeline_throughput
export SPIChain, spi_chain_fingerprint!, spi_verify_chain!
export spi_orchestrated_walk, SPIOrchestratorMetrics, spi_collect_metrics

# Include Reafferent DAO Topos - GAY_SEED as root of succ(DAO)
include("reafferent_dao_topos.jl")
using .ReafferentDAOTopos
export DAOState, DAOProposal, DAOVote, DAOConsensus, GayDAO
export ReafferentDAO, perceive!, synchronize!
export subobject_classifier, internal_hom, power_object
export triad_vote!, execute_proposal!, consensus_reached
export demo_reafferent_dao_topos

# NOTE: Move Aptos integration moved to separate package: GayMove.jl
# Install with: using Pkg; Pkg.develop(path="/Users/bob/ies/rio/GayMove.jl")

# Include GaySplittableRNG - Best-in-class splittable PRNG (Issue #205)
# Access via: Gay.GaySplittableRNG.gay_seed() or import Gay.GaySplittableRNG as GRNG
include("gayrng.jl")
using .GaySplittableRNG: GaySplittableRNG,
    ZobristTable, zobrist_init, zobrist_hash, zobrist_update,
    FingerprintCRDT, crdt_update!, crdt_merge, crdt_query,
    spectral_quality, chi_squared_uniformity, serial_correlation,
    world_gayrng, world_incremental_hashing, world_distributed_fingerprint,
    world_monoidal_coherence, world_statistical_quality
export ZobristTable, zobrist_init, zobrist_hash, zobrist_update
export FingerprintCRDT, crdt_update!, crdt_merge, crdt_query
export spectral_quality, chi_squared_uniformity, serial_correlation
export world_gayrng, world_incremental_hashing, world_distributed_fingerprint
export world_monoidal_coherence, world_statistical_quality

# Include Seed Sonification - HSL/Polarity/XOR → Audio (Issue #190)
include("seed_sonification.jl")
using .SeedSonification
export SeedSound, hue_to_frequency, polarity_to_waveform, xor_to_rhythm
export sonify_seed, world_seed_sonification

# Include Seed Mining - Descent-validated seed discovery (Issue #191)
include("seed_mining.jl")
using .SeedMining
export SeedQuality, spectral_test, mine_seeds, generate_move_registration
export world_seed_mining

# Include Descent Tower - 7-level sheaf decomposition sonification (Issue #192)
include("descent_tower.jl")
using .DescentTower
export DescentLevel, DESCENT_TOWER, depth_to_frequency, branches_to_chord
export sonify_descent_level, world_descent_tower

# Include NashProp Zero-Message Mining - 51 subagents with provable bandwidth
include("nashprop_zero_message.jl")
using .NashPropZeroMessage
export TOTAL_SUBAGENTS, FORWARD_COUNT, BACKWARD_COUNT, COMMITTEE_SIZE
export SubagentRole, Forward, Backward, Committee
export ZeroMessageSubagent, create_subagent, subagent_range
export RangeAssignment, compute_range, range_contains, range_overlap
export ZeroMessageMiner, create_miner, mine_epoch!, verify_work
export Swarm51, create_swarm_51, swarm_mine!, swarm_verify
export NashPropEquilibrium, compute_equilibrium, equilibrium_color
export ColorBandwidth, measure_bandwidth
export BandwidthResource, create_bandwidth_resource
export BandwidthConsumer, create_consumer, consume_bandwidth!, verify_bandwidth
export BandwidthStake, stake_bandwidth!, bandwidth_weighted_range
export bridge_to_gaymove, bridge_from_gaymove
export demo_zero_message_mining

# Module initialization
# ═══════════════════════════════════════════════════════════════════════════

function __init__()
    # Initialize global splittable RNG
    gay_seed!(GAY_SEED)
    
    # Auto-initialize REPL if running interactively
    if isdefined(Base, :active_repl) && Base.active_repl !== nothing
        @async begin
            sleep(0.1)  # Let REPL finish loading
            # Initialize SPC REPL (press SPACE to enter)
            init_spc_repl()
        end
    else
        @info "Gay.jl loaded 🏳️‍🌈 - Wide-gamut colors + splittable determinism"
        @info "In REPL: init_spc_repl() for SPC mode (press SPACE to enter)"
    end
end

end # module Gay
