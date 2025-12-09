#!/usr/bin/env julia
"""
Structure Regression (Ziming Liu, 2023)

Symbol vs Structure:
    Symbolic regression: find f(x) = x² + sin(x) + ...
    Structure regression: find STRUCTURE (independence, modularity, compositionality)

BIMT (Brain-Inspired Modular Training):
    1. Embed neurons in 2D space
    2. Penalize non-local weights: λℓ|w| (locality prior)
    3. Swap neurons to avoid topology issues
    → Modules emerge naturally

This IS Markov blanket detection:
    - Locality penalty encourages conditional independence
    - Modules = sub-blankets within the network
    - Structure = the graph, not the function values

Connection to our framework:
    - BIMT penalty λℓ|w| ↔ Fokker-Planck with spatial diffusion
    - Emerging modules ↔ DMBD partition (s,b,z)
    - Network graph ↔ 2-transducer state category Q
    - Spectral structure ↔ which modules emerge

Color traces structure: same connectivity pattern → same color.
"""

using Gay
using Colors: RGB
using LinearAlgebra
using Random

# ═══════════════════════════════════════════════════════════════════════════
# BIMT: Brain-Inspired Modular Training
# ═══════════════════════════════════════════════════════════════════════════

"""
A neuron with a 2D spatial position (for locality penalty).
"""
struct Neuron
    id::Int
    position::Tuple{Float64, Float64}  # (x, y) in 2D space
    layer::Int
end

"""
A simple feedforward network with spatial embedding.
"""
struct SpatialNetwork
    layers::Vector{Vector{Neuron}}
    weights::Vector{Matrix{Float64}}  # weights[l][i,j] = weight from layer l, neuron j to layer l+1, neuron i
end

function SpatialNetwork(layer_sizes::Vector{Int}; seed::Integer=42)
    Random.seed!(seed)
    
    layers = Vector{Vector{Neuron}}()
    weights = Vector{Matrix{Float64}}()
    
    n_layers = length(layer_sizes)
    
    for (l, size) in enumerate(layer_sizes)
        neurons = Neuron[]
        for i in 1:size
            # Spread neurons vertically within each layer
            x = (l - 1) / (n_layers - 1)  # Layer position [0, 1]
            y = (i - 0.5) / size           # Neuron position within layer [0, 1]
            push!(neurons, Neuron(i, (x, y), l))
        end
        push!(layers, neurons)
        
        # Initialize weights
        if l < n_layers
            W = randn(layer_sizes[l+1], layer_sizes[l]) * 0.5
            push!(weights, W)
        end
    end
    
    SpatialNetwork(layers, weights)
end

"""
Compute the distance between two neurons.
"""
function neuron_distance(n1::Neuron, n2::Neuron)
    dx = n1.position[1] - n2.position[1]
    dy = n1.position[2] - n2.position[2]
    return sqrt(dx^2 + dy^2)
end

"""
BIMT locality penalty: λ∑ℓ|w|
Penalize non-local connections more than local ones.
"""
function locality_penalty(net::SpatialNetwork, λ::Float64)
    penalty = 0.0
    
    for l in 1:length(net.weights)
        W = net.weights[l]
        layer_from = net.layers[l]
        layer_to = net.layers[l+1]
        
        for i in axes(W, 1)      # to neuron
            for j in axes(W, 2)  # from neuron
                ℓ = neuron_distance(layer_from[j], layer_to[i])
                penalty += λ * ℓ * abs(W[i, j])
            end
        end
    end
    
    return penalty
end

"""
Identify modules by clustering strongly-connected neurons.
Returns a coloring of neurons by module membership.
"""
function identify_modules(net::SpatialNetwork; threshold::Float64=0.1)
    # For each layer, find which input neurons each hidden neuron depends on
    # by tracing paths with weight above threshold
    
    # Simplified: just identify which weights are "active"
    active = [abs.(W) .> threshold for W in net.weights]
    return active
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

function ansi(c::RGB)
    r = round(Int, clamp(c.r, 0, 1) * 255)
    g = round(Int, clamp(c.g, 0, 1) * 255)
    b = round(Int, clamp(c.b, 0, 1) * 255)
    "\e[38;2;$(r);$(g);$(b)m"
end

const R = "\e[0m"
const BOLD = "\e[1m"
const DIM = "\e[2m"

"""
Render the network structure with colored connections.
"""
function render_network(net::SpatialNetwork; seed::Integer=42, threshold::Float64=0.1)
    gay_seed!(seed)
    
    n_layers = length(net.layers)
    max_neurons = maximum(length.(net.layers))
    
    # Assign colors to neurons based on layer
    layer_colors = [next_color(SRGB()) for _ in 1:n_layers]
    
    println()
    println("  $(BOLD)Spatial Network Structure$(R)")
    println("  ════════════════════════════════════════════════════")
    println()
    
    # Render layer by layer, top to bottom
    for row in 1:max_neurons
        print("  ")
        for (l, layer) in enumerate(net.layers)
            if row <= length(layer)
                c = layer_colors[l]
                print("$(ansi(c))●$(R)")
            else
                print(" ")
            end
            
            # Draw connections to next layer
            if l < n_layers && row <= length(layer)
                W = net.weights[l]
                has_strong = any(abs(W[:, row]) .> threshold)
                if has_strong
                    print("$(DIM)──$(R)")
                else
                    print("  ")
                end
            else
                print("  ")
            end
        end
        println()
    end
    println()
    
    # Layer labels
    print("  ")
    for (l, layer) in enumerate(net.layers)
        c = layer_colors[l]
        print("$(ansi(c))L$l$(R)")
        if l < n_layers
            print("   ")
        end
    end
    println()
    println()
end

"""
Render three examples of structure from Ziming Liu's blog:
1. Independence (two outputs depend on non-overlapping inputs)
2. Feature sharing (x_i² are intermediate features)
3. Compositionality (sum of squares, then sqrt)
"""
function render_structure_examples(; seed::Integer=42)
    gay_seed!(seed)
    
    println()
    println("  $(BOLD)Three Structures (from Ziming Liu's blog)$(R)")
    println("  ════════════════════════════════════════════════════════════")
    println()
    
    # Colors for each structure type
    c_indep = next_color(SRGB())
    c_share = next_color(SRGB())
    c_comp = next_color(SRGB())
    
    # 1. Independence
    println("  $(ansi(c_indep))1. Independence$(R)")
    println("     y₁ = f(x₁,x₂)    y₂ = g(x₃,x₄)")
    println()
    println("     $(ansi(c_indep))x₁ ─┐$(R)         $(ansi(c_indep))x₃ ─┐$(R)")
    println("         $(ansi(c_indep))├─▶ y₁$(R)        $(ansi(c_indep))├─▶ y₂$(R)")
    println("     $(ansi(c_indep))x₂ ─┘$(R)         $(ansi(c_indep))x₄ ─┘$(R)")
    println()
    println("     $(DIM)Two independent modules (no cross-talk)$(R)")
    println()
    
    # 2. Feature sharing
    println("  $(ansi(c_share))2. Feature Sharing$(R)")
    println("     y = f(x₁², x₂², x₃²)")
    println()
    println("     $(ansi(c_share))x₁ ─▶ x₁² ─┐$(R)")
    println("     $(ansi(c_share))x₂ ─▶ x₂² ─┼─▶ y$(R)")
    println("     $(ansi(c_share))x₃ ─▶ x₃² ─┘$(R)")
    println()
    println("     $(DIM)Shared intermediate features (squares)$(R)")
    println()
    
    # 3. Compositionality
    println("  $(ansi(c_comp))3. Compositionality$(R)")
    println("     y = √(x₁² + x₂² + x₃²)")
    println()
    println("     $(ansi(c_comp))x₁ ─┐$(R)")
    println("     $(ansi(c_comp))x₂ ─┼─▶ Σx² ─▶ √· ─▶ y$(R)")
    println("     $(ansi(c_comp))x₃ ─┘$(R)")
    println()
    println("     $(DIM)Hierarchical composition (sum then sqrt)$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Connection to Markov Blankets
# ═══════════════════════════════════════════════════════════════════════════

function render_bimt_as_blanket(; seed::Integer=42)
    gay_seed!(seed)
    
    c_ext = next_color(SRGB())
    c_bln = next_color(SRGB())
    c_int = next_color(SRGB())
    c_loc = next_color(SRGB())
    
    println()
    println("  $(BOLD)BIMT as Markov Blanket Detection$(R)")
    println("  ════════════════════════════════════════════════════════════")
    println()
    println("  BIMT locality penalty: $(ansi(c_loc))λℓ|w|$(R)")
    println()
    println("  This encourages $(BOLD)conditional independence$(R):")
    println("    • Local connections are cheap (small ℓ)")
    println("    • Non-local connections are expensive (large ℓ)")
    println("    • Network self-organizes into modules")
    println()
    println("  Each module IS a Markov blanket:")
    println()
    println("    $(ansi(c_ext))┌──────────────┐$(R)      $(ansi(c_ext))┌──────────────┐$(R)")
    println("    $(ansi(c_ext))│  Module A    │$(R)      $(ansi(c_ext))│  Module B    │$(R)")
    println("    $(ansi(c_ext))│   (inputs    │$(R)      $(ansi(c_ext))│   (inputs    │$(R)")
    println("    $(ansi(c_ext))│    x₁,x₂)    │$(R)      $(ansi(c_ext))│    x₃,x₄)    │$(R)")
    println("    $(ansi(c_ext))└──────┬───────┘$(R)      $(ansi(c_ext))└──────┬───────┘$(R)")
    println("           $(ansi(c_bln))│$(R)                    $(ansi(c_bln))│$(R)")
    println("           $(ansi(c_bln))▼$(R)                    $(ansi(c_bln))▼$(R)")
    println("    $(ansi(c_bln))╔══════════════╗$(R)      $(ansi(c_bln))╔══════════════╗$(R)")
    println("    $(ansi(c_bln))║   Blanket    ║$(R)      $(ansi(c_bln))║   Blanket    ║$(R)")
    println("    $(ansi(c_bln))║  (boundary)  ║$(R)      $(ansi(c_bln))║  (boundary)  ║$(R)")
    println("    $(ansi(c_bln))╚══════╤═══════╝$(R)      $(ansi(c_bln))╚══════╤═══════╝$(R)")
    println("           $(ansi(c_int))│$(R)                    $(ansi(c_int))│$(R)")
    println("           $(ansi(c_int))▼$(R)                    $(ansi(c_int))▼$(R)")
    println("    $(ansi(c_int))┌──────────────┐$(R)      $(ansi(c_int))┌──────────────┐$(R)")
    println("    $(ansi(c_int))│   Output     │$(R)      $(ansi(c_int))│   Output     │$(R)")
    println("    $(ansi(c_int))│     y₁       │$(R)      $(ansi(c_int))│     y₂       │$(R)")
    println("    $(ansi(c_int))└──────────────┘$(R)      $(ansi(c_int))└──────────────┘$(R)")
    println()
    println("  $(DIM)p(y₁, y₂ | blankets) = p(y₁ | blanket_A) × p(y₂ | blanket_B)$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Connection to 2-Transducers
# ═══════════════════════════════════════════════════════════════════════════

function render_structure_as_transducer(; seed::Integer=42)
    gay_seed!(seed)
    
    println()
    println("  $(BOLD)Structure as 2-Transducer$(R)")
    println("  ════════════════════════════════════════════════════════════")
    println()
    println("  The network structure IS the state category Q:")
    println()
    println("    Input A        State Q (structure)        Output B")
    println("    ─────────      ─────────────────          ─────────")
    println("    x₁,x₂,...  ──▶  graph of neurons    ──▶  y₁,y₂,...")
    println()
    println("  Symbolic regression: cares about activation functions")
    println("  Structure regression: cares only about the graph")
    println()
    println("  This is exactly the 2-transducer insight:")
    println("    • Q is a category (not just a set)")
    println("    • Morphisms in Q = signal flow paths")
    println("    • 2-cells = changes to network structure (pruning, growth)")
    println()
    println("  $(DIM)Structure is more robust than symbols:$(R)")
    println("  $(DIM)  macroscopic behavior depends on relations,$(R)")
    println("  $(DIM)  not details of each unit.$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Connection to Fokker-Planck
# ═══════════════════════════════════════════════════════════════════════════

function render_bimt_as_fokker_planck(; seed::Integer=42)
    gay_seed!(seed)
    
    println()
    println("  $(BOLD)BIMT Training as Fokker-Planck$(R)")
    println("  ════════════════════════════════════════════════════════════")
    println()
    println("  The training dynamics of weights w follow:")
    println()
    println("    dw/dt = -∇L(w) - λℓ·sign(w) + η")
    println()
    println("    $(DIM)-∇L(w)$(R)           = gradient descent (drift toward loss minimum)")
    println("    $(DIM)-λℓ·sign(w)$(R)      = locality penalty (drift toward sparsity)")
    println("    $(DIM)η$(R)                = SGD noise (diffusion)")
    println()
    println("  This is Fokker-Planck in weight space:")
    println()
    println("    ∂ρ/∂t = -∇·(μρ) + D∇²ρ")
    println()
    println("    ρ(w,t)  = distribution over weight configurations")
    println("    μ(w)    = drift toward low-loss + local configurations")
    println("    D       = SGD noise level (learning rate × batch variance)")
    println()
    println("  The stationary distribution ρ_∞ concentrates on")
    println("  $(BOLD)modular, sparse structures$(R) — the blanket emerges!")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Spectre of Structure
# ═══════════════════════════════════════════════════════════════════════════

function render_structure_spectre(; seed::Integer=42)
    gay_seed!(seed)
    
    println()
    println("  $(BOLD)The Spectre of Structure$(R)")
    println("  ════════════════════════════════════════════════════════════")
    println()
    println("  The network Jacobian J = ∂y/∂x has a spectre:")
    println()
    println("    • Eigenvalues λₙ = sensitivity to input directions")
    println("    • Eigenvectors = principal directions of signal flow")
    println("    • Spectral gap = separation of modules")
    println()
    println("  Structure regression finds networks where:")
    println()
    println("    • Spectrum is $(BOLD)sparse$(R) (few dominant modes)")
    println("    • Eigenvectors are $(BOLD)localized$(R) (modules)")
    println("    • Spectral gap is $(BOLD)large$(R) (clear separation)")
    println()
    println("  $(DIM)The spectre reveals the structure.$(R)")
    println("  $(DIM)Color the eigenvectors → see the modules.$(R)")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main(; seed::Integer=42)
    println()
    println("  ╔════════════════════════════════════════════════════════════════════╗")
    println("  ║          $(BOLD)STRUCTURE REGRESSION$(R) (Ziming Liu, 2023)                 ║")
    println("  ║   Symbolic regression? No — Structure regression!                  ║")
    println("  ╚════════════════════════════════════════════════════════════════════╝")
    
    render_structure_examples(seed=seed)
    
    # Create a sample network
    net = SpatialNetwork([4, 6, 4, 2]; seed=seed)
    render_network(net; seed=seed)
    
    render_bimt_as_blanket(seed=seed)
    render_structure_as_transducer(seed=seed)
    render_bimt_as_fokker_planck(seed=seed)
    render_structure_spectre(seed=seed)
    
    println()
    gay_seed!(seed + 100)
    c = next_color(SRGB())
    println("  $(ansi(c))Structure is more universal than symbols.$(R)")
    println("  $(ansi(c))Color traces structure: same pattern → same hue.$(R)")
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
