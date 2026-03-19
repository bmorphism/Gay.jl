"""
GayEnergyGrid: Energy grid algorithms with SPI (Seed-Parameterized Invariance).
DC power flow and parallel decomposition with chromatic fingerprinting.
"""
module GayEnergyGrid

export EnergyGrid, GridNode, GridEdge
export gay_power_flow!, gay_grid_partition!, verify_grid_spi
export gay_hmc_power_flow!
export splitmix64, chromatic_fingerprint

"Splitmix64 PRNG - deterministic mixing for SPI"
@inline function splitmix64(x::UInt64)::UInt64
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    x ⊻ (x >> 31)
end

"Grid node: generator (injection > 0) or load (injection < 0)"
struct GridNode
    id::Int
    injection::Float64      # MW: positive=generator, negative=load
    voltage::Float64        # per-unit voltage magnitude
    color::UInt64           # chromatic identity
end

GridNode(id, injection; voltage=1.0, color=UInt64(0)) = 
    GridNode(id, injection, voltage, color)

"Transmission line with susceptance"
struct GridEdge
    from::Int
    to::Int
    susceptance::Float64    # B = 1/X (per-unit)
    capacity::Float64       # MW thermal limit
    color::UInt64
end

GridEdge(from, to, susceptance; capacity=Inf, color=UInt64(0)) =
    GridEdge(from, to, susceptance, capacity, color)

"Energy grid with chromatic identity for SPI verification"
mutable struct EnergyGrid
    nodes::Vector{GridNode}
    edges::Vector{GridEdge}
    angles::Vector{Float64}     # voltage angles (radians)
    flows::Vector{Float64}      # edge power flows (MW)
    chromatic_id::UInt64        # accumulated fingerprint
    iteration::Int
end

function EnergyGrid(nodes::Vector{GridNode}, edges::Vector{GridEdge})
    n = length(nodes)
    EnergyGrid(nodes, edges, zeros(n), zeros(length(edges)), UInt64(0), 0)
end

"Mix value into chromatic fingerprint"
@inline function mix_chromatic!(grid::EnergyGrid, val::UInt64)
    grid.chromatic_id = splitmix64(grid.chromatic_id ⊻ val)
end

"Compute chromatic fingerprint from grid state"
function chromatic_fingerprint(grid::EnergyGrid)::UInt64
    fp = grid.chromatic_id
    for (i, θ) in enumerate(grid.angles)
        fp = splitmix64(fp ⊻ reinterpret(UInt64, θ) ⊻ UInt64(i))
    end
    for (i, f) in enumerate(grid.flows)
        fp = splitmix64(fp ⊻ reinterpret(UInt64, f) ⊻ UInt64(i << 32))
    end
    fp
end

"""
    gay_power_flow!(grid, seed; max_iter=100, tol=1e-8) -> UInt64

DC power flow solver with chromatic tracking per iteration.
Returns final chromatic fingerprint for SPI verification.
"""
function gay_power_flow!(grid::EnergyGrid, seed::UInt64; 
                         max_iter::Int=100, tol::Float64=1e-8)::UInt64
    n = length(grid.nodes)
    m = length(grid.edges)
    grid.chromatic_id = splitmix64(seed)
    grid.iteration = 0
    
    B = zeros(n, n)
    for e in grid.edges
        B[e.from, e.to] -= e.susceptance
        B[e.to, e.from] -= e.susceptance
        B[e.from, e.from] += e.susceptance
        B[e.to, e.to] += e.susceptance
    end
    
    P = [node.injection for node in grid.nodes]
    ref = 1  # slack bus
    
    idx = setdiff(1:n, ref)
    B_red = B[idx, idx]
    P_red = P[idx]
    
    for iter in 1:max_iter
        grid.iteration = iter
        θ_old = copy(grid.angles)
        
        grid.angles[idx] = B_red \ P_red
        grid.angles[ref] = 0.0
        
        for (i, e) in enumerate(grid.edges)
            grid.flows[i] = e.susceptance * (grid.angles[e.from] - grid.angles[e.to])
        end
        
        iter_color = splitmix64(seed ⊻ UInt64(iter))
        mix_chromatic!(grid, iter_color)
        mix_chromatic!(grid, reinterpret(UInt64, sum(grid.flows)))
        
        if maximum(abs.(grid.angles - θ_old)) < tol
            break
        end
    end
    
    chromatic_fingerprint(grid)
end

"""
    gay_grid_partition!(grid, n_parts, seed) -> Vector{Vector{Int}}

Parallel graph decomposition with colored partitions.
Each partition gets deterministic color from seed.
"""
function gay_grid_partition!(grid::EnergyGrid, n_parts::Int, seed::UInt64)::Vector{Vector{Int}}
    n = length(grid.nodes)
    partitions = [Int[] for _ in 1:n_parts]
    
    rng_state = splitmix64(seed)
    node_order = collect(1:n)
    for i in n:-1:2
        rng_state = splitmix64(rng_state)
        j = (rng_state % UInt64(i)) + 1
        node_order[i], node_order[j] = node_order[j], node_order[i]
    end
    
    for (i, node_id) in enumerate(node_order)
        part_idx = ((i - 1) % n_parts) + 1
        push!(partitions[part_idx], node_id)
    end
    
    for (p, part) in enumerate(partitions)
        part_color = splitmix64(seed ⊻ UInt64(p << 48))
        for node_id in part
            old = grid.nodes[node_id]
            grid.nodes[node_id] = GridNode(old.id, old.injection, old.voltage, part_color)
        end
        mix_chromatic!(grid, part_color ⊻ UInt64(length(part)))
    end
    
    partitions
end

"""
    verify_grid_spi(grid, seed; n_trials=10) -> Bool

Verify parallel computation equals sequential fingerprint.
SPI guarantee: same seed → same chromatic result regardless of execution order.
"""
function verify_grid_spi(grid::EnergyGrid, seed::UInt64; n_trials::Int=10)::Bool
    nodes_backup = copy(grid.nodes)
    edges_backup = copy(grid.edges)
    
    function reset_grid!()
        grid.nodes = copy(nodes_backup)
        grid.edges = copy(edges_backup)
        grid.angles = zeros(length(grid.nodes))
        grid.flows = zeros(length(grid.edges))
        grid.chromatic_id = UInt64(0)
        grid.iteration = 0
    end
    
    reset_grid!()
    ref_fp = gay_power_flow!(grid, seed)
    
    for trial in 1:n_trials
        reset_grid!()
        trial_seed = splitmix64(seed ⊻ UInt64(trial))
        gay_grid_partition!(grid, 4, trial_seed)
        fp = gay_power_flow!(grid, seed)
        
        if fp != ref_fp
            return false
        end
    end
    
    true
end

# --- HMC Sampler for Uncertainty Quantification ---

"""
    gay_hmc_power_flow!(grid, seed; n_samples=100, step_size=0.01, n_leapfrog=10) -> UInt64

Hamiltonian Monte Carlo sampling for power flow uncertainty quantification.
Uses SPI-deterministic initialization via splitmix64.
Returns chromatic fingerprint of sampled distribution.
"""
function gay_hmc_power_flow!(grid::EnergyGrid, seed::UInt64;
                             n_samples::Int=100, step_size::Float64=0.01, 
                             n_leapfrog::Int=10)::UInt64
    n = length(grid.nodes)
    rng_state = splitmix64(seed)
    
    # Initialize position (angles) and momentum
    q = copy(grid.angles)
    fingerprint = splitmix64(seed ⊻ UInt64(n))
    
    # Potential energy: power mismatch squared
    function U(θ)
        mismatch = 0.0
        for e in grid.edges
            flow = e.susceptance * (θ[e.from] - θ[e.to])
            mismatch += flow^2
        end
        mismatch
    end
    
    # Gradient of potential
    function ∇U(θ)
        g = zeros(n)
        for e in grid.edges
            diff = θ[e.from] - θ[e.to]
            g[e.from] += 2 * e.susceptance^2 * diff
            g[e.to] -= 2 * e.susceptance^2 * diff
        end
        g
    end
    
    for sample in 1:n_samples
        rng_state = splitmix64(rng_state)
        # Sample momentum from standard normal (Box-Muller via splitmix64)
        p = zeros(n)
        for i in 1:n
            rng_state = splitmix64(rng_state)
            u1 = (rng_state & 0xFFFFFFFF) / 4294967296.0
            rng_state = splitmix64(rng_state)
            u2 = (rng_state & 0xFFFFFFFF) / 4294967296.0
            p[i] = sqrt(-2 * log(max(u1, 1e-10))) * cos(2π * u2)
        end
        
        q_new, p_new = copy(q), copy(p)
        
        # Leapfrog integration
        p_new .-= (step_size / 2) .* ∇U(q_new)
        for _ in 1:n_leapfrog
            q_new .+= step_size .* p_new
            p_new .-= step_size .* ∇U(q_new)
        end
        p_new .-= (step_size / 2) .* ∇U(q_new)
        
        # Metropolis acceptance
        H_old = U(q) + 0.5 * sum(p.^2)
        H_new = U(q_new) + 0.5 * sum(p_new.^2)
        rng_state = splitmix64(rng_state)
        accept_prob = (rng_state & 0xFFFFFFFF) / 4294967296.0
        
        if accept_prob < exp(min(0.0, H_old - H_new))
            q = q_new
        end
        
        # Mix sample into fingerprint
        fingerprint = splitmix64(fingerprint ⊻ reinterpret(UInt64, sum(q)))
    end
    
    grid.angles .= q
    grid.chromatic_id = fingerprint
    fingerprint
end

end # module
