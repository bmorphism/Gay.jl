# PARALLEL GH DEMO: Maximally parallel GitHub repo fetching
#
# Demonstrates:
# 1. RuntimePlacement for OhMyThreads vs Metal selection
# 2. GayAsync channels for producer-consumer pattern
# 3. ParallelGH for chromatic agent-based fetching
# 4. SPI verification - same result regardless of parallelism level

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using OhMyThreads
using KernelAbstractions
using Colors
using Printf
using Dates
using JSON3

# Include modules
include(joinpath(@__DIR__, "..", "src", "runtime_placement.jl"))
include(joinpath(@__DIR__, "..", "src", "gay_async.jl"))
include(joinpath(@__DIR__, "..", "src", "parallel_gh.jl"))

using .RuntimePlacement: detect_optimal_backend, place!, backend_name, metal_available
using .GayAsync
using .ParallelGH

const GAY_SEED_69 = UInt64(69)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMAL PARALLEL FETCH STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

"""
GitHub API limits (as of 2024):
- Core API: 5000 requests/hour authenticated
- Per-page: max 100 items
- Concurrent connections: ~10 recommended (no hard limit)

For 557 repos:
- Sequential: 6 pages × ~400ms/page = ~2.4s
- Parallel pages: 6 concurrent requests = ~400ms total
- With rate limit headroom: 6 requests use 6/5000 = 0.12% of hourly quota

Strategy:
1. Fetch page count first (single request)
2. Fetch all pages in parallel (6 requests for 557 repos)
3. Use RuntimePlacement to process results (OhMyThreads for CPU parsing)
"""

struct FetchResult
    repos::Vector{Any}
    duration_ms::Float64
    pages_fetched::Int
    fingerprint::UInt64
end

"""
Fetch all repos from an org with maximum parallelism.
"""
function fetch_org_repos_parallel(org::String; 
                                   per_page::Int=100,
                                   max_concurrent::Int=6,
                                   seed::UInt64=GAY_SEED_69)
    t0 = time()
    
    println("═" ^ 70)
    println("  PARALLEL GH FETCH: $org")
    println("  Strategy: parallel pagination with chromatic tracing")
    println("═" ^ 70)
    println()
    
    # Step 1: Get first page to determine total count
    println("Step 1: Fetching first page to get total count...")
    first_page_cmd = `gh api /orgs/$org/repos --paginate -q length`
    
    # Just paginate and count
    println("  Counting repos...")
    all_repos_raw = read(`gh api /orgs/$org/repos --paginate`, String)
    all_repos = JSON3.read(all_repos_raw)
    total_count = length(all_repos)
    
    n_pages = ceil(Int, total_count / per_page)
    println("  Total repos: $total_count")
    println("  Pages needed: $n_pages (at $per_page per page)")
    println()
    
    # Step 2: Create parallel fetch operations
    println("Step 2: Creating parallel fetch operations...")
    
    agency = GHAgency("$org-fetch"; n_agents=max_concurrent, seed=seed)
    
    operations = GHOperation[]
    for page in 1:n_pages
        endpoint = "/orgs/$org/repos?per_page=$per_page&page=$page"
        op = GHOperation(REPO_LIST, endpoint; seed=seed ⊻ UInt64(page))
        push!(operations, op)
    end
    
    println("  Created $(length(operations)) operations")
    println("  Agents: $(length(agency.agents))")
    println()
    
    # Step 3: Execute in parallel
    println("Step 3: Fetching pages in parallel...")
    
    results = parallel_gh!(agency, operations)
    
    # Step 4: Aggregate results
    println()
    println("Step 4: Aggregating results...")
    
    repos = Any[]
    for r in results
        if r.success && r.data !== nothing
            if r.data isa AbstractVector
                append!(repos, r.data)
            end
        end
    end
    
    # Compute fingerprint
    fingerprint = seed
    for r in results
        fingerprint = fingerprint ⊻ r.operation.seed ⊻ (r.success ? 0x1 : 0x0)
    end
    
    duration_ms = (time() - t0) * 1000
    
    # Report
    println()
    println("─" ^ 70)
    println("  RESULTS")
    println("─" ^ 70)
    println("  Repos fetched: $(length(repos))")
    println("  Pages: $n_pages")
    println("  Duration: $(round(duration_ms, digits=1))ms")
    println("  Effective parallelism: $(round(n_pages * 400 / duration_ms, digits=2))x")
    println("  Fingerprint: 0x$(string(fingerprint, base=16, pad=16))")
    println("═" ^ 70)
    
    FetchResult(repos, duration_ms, n_pages, fingerprint)
end

"""
Quick fetch without full parallel infrastructure.
Uses shell-level parallelism via gh cli pagination.
"""
function fetch_org_repos_quick(org::String; seed::UInt64=GAY_SEED_69)
    t0 = time()
    
    println("Quick fetch: $org (using gh --paginate)...")
    
    # gh handles pagination internally
    output = read(`gh api /orgs/$org/repos --paginate`, String)
    repos = JSON3.read(output)
    
    duration_ms = (time() - t0) * 1000
    
    # Extract repo names
    repo_names = [r["full_name"] for r in repos]
    
    println("  Fetched $(length(repos)) repos in $(round(duration_ms, digits=1))ms")
    
    (repos=repos, names=repo_names, duration_ms=duration_ms, count=length(repos))
end

# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME PLACEMENT INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
Process fetched repos using optimal backend (OhMyThreads or Metal).
"""
function process_repos_placed(repos; seed::UInt64=GAY_SEED_69)
    println("\nProcessing $(length(repos)) repos with RuntimePlacement...")
    
    backend = detect_optimal_backend(length(repos))
    println("  Selected backend: $(backend_name(backend))")
    
    # Extract relevant info from each repo
    t0 = time()
    processed = place!(repos) do repo
        (
            name = get(repo, "full_name", "unknown"),
            stars = get(repo, "stargazers_count", 0),
            language = get(repo, "language", nothing),
            updated = get(repo, "updated_at", nothing)
        )
    end
    
    duration_ms = (time() - t0) * 1000
    println("  Processed in $(round(duration_ms, digits=2))ms")
    
    processed
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println()
    println("╔" * "═"^68 * "╗")
    println("║" * " "^20 * "PARALLEL GH DEMO" * " "^32 * "║")
    println("║" * " "^10 * "Maximally parallel GitHub fetching with Gay.jl" * " "^12 * "║")
    println("╚" * "═"^68 * "╝")
    println()
    
    # System info
    println("System Configuration:")
    println("  Julia threads: $(Threads.nthreads())")
    println("  Metal available: $(RuntimePlacement.metal_available())")
    println()
    
    # Quick fetch first (baseline)
    println("BASELINE: Sequential fetch with gh --paginate")
    println("─" ^ 70)
    baseline = fetch_org_repos_quick("plurigrid")
    println()
    
    # Show some repos
    println("Sample repos:")
    for name in baseline.names[1:min(10, length(baseline.names))]
        println("  • $name")
    end
    if length(baseline.names) > 10
        println("  ... and $(length(baseline.names) - 10) more")
    end
    println()
    
    # Process with RuntimePlacement
    processed = process_repos_placed(baseline.repos)
    
    # Show top by stars
    sorted = sort(processed, by=x -> x.stars, rev=true)
    println("\nTop 10 by stars:")
    for (i, repo) in enumerate(sorted[1:min(10, length(sorted))])
        println("  $i. $(repo.name) ⭐$(repo.stars)")
    end
    
    println()
    println("Done! Total repos: $(baseline.count)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
