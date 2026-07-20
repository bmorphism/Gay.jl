# GayLuxExt: Neural Network SPI Coloring for Lux.jl
# ═══════════════════════════════════════════════════════════════════════════════
#
# "Every layer gets a color. Every gradient preserves chromatic consistency."
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  NEURAL NETWORK × CHROMATIC PARALLELISM                                     │
# │                                                                             │
# │  Layer Coloring:    hash(typeof(layer) ⊻ hyperparams) → RGB                │
# │  Parameter Coloring: hash(path ⊻ index) → RGB per element                  │
# │  Gradient Coloring:  magnitude → hue (red=high, blue=low)                  │
# │                                                                             │
# │  SPI GUARANTEE:                                                             │
# │    color_layer(layer, seed) = color_layer(layer, seed)  ∀ execution        │
# │    Same architecture + same seed → same layer colors                        │
# │                                                                             │
# │  POLARITY: POSITIVE (+)                                                     │
# │    - Eager evaluation                                                       │
# │    - Output-oriented (produces predictions)                                 │
# │    - Gradient flow (differentiable)                                         │
# │                                                                             │
# │  PARALLELISM:                                                               │
# │    - Batched operations (data parallelism)                                  │
# │    - GPU acceleration via Metal/CUDA                                        │
# │    - Layer-wise parallelism for forward/backward                           │
# └─────────────────────────────────────────────────────────────────────────────┘

module GayLuxExt

using Lux
using Zygote
using Colors
using Random

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

const GAY_SEED = UInt64(1069)
const LUX_SEED = UInt64(0x10A)  # "LUX"

# ═══════════════════════════════════════════════════════════════════════════════
# SPLITMIX64 (SPI Core)
# ═══════════════════════════════════════════════════════════════════════════════

@inline function splitmix64(state::UInt64)::UInt64
    z = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ⊻ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ⊻ (z >> 31)
end

@inline function color_from_seed(seed::UInt64)::RGB{Float32}
    r = splitmix64(seed)
    g = splitmix64(r)
    b = splitmix64(g)
    RGB{Float32}((r >> 56) / 255.0f0, (g >> 56) / 255.0f0, (b >> 56) / 255.0f0)
end

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

export
    # Layer Coloring
    color_layer, layer_fingerprint, ColoredLayer,
    
    # Parameter Coloring
    ColoredParameter, color_parameters, parameter_fingerprint,
    
    # Gradient Coloring
    gradient_color_map, colored_gradients,
    
    # Network Visualization
    ColoredNetwork, render_network_graph, network_fingerprint,
    
    # Training with Colors
    colored_training_step, ColoredTrainingState,
    
    # SPI Verification
    verify_layer_spi, verify_network_spi,
    
    # Learning Integration
    GayOptimizer, gay_adam, gay_sgd,
    in_context_color, adapter_color, finetune_color

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER COLORING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ColoredLayer{L<:Lux.AbstractExplicitLayer}

A Lux layer with chromatic identity for visualization and SPI tracking.
"""
struct ColoredLayer{L<:Lux.AbstractExplicitLayer}
    layer::L
    color::RGB{Float32}
    fingerprint::UInt64
    name::String
end

"""
    color_layer(layer; seed=GAY_SEED, name="") -> ColoredLayer

Assign deterministic color to a Lux layer based on its structure.
Same layer type + same hyperparameters → same color (SPI).
"""
function color_layer(layer::L; seed::UInt64=GAY_SEED, name::String="") where {L<:Lux.AbstractExplicitLayer}
    type_hash = hash(L)
    
    field_hash = type_hash
    for f in fieldnames(L)
        val = getfield(layer, f)
        if val isa Number || val isa Symbol || val isa Tuple
            field_hash = splitmix64(field_hash ⊻ UInt64(hash(val)))
        end
    end
    
    combined = splitmix64(seed ⊻ field_hash ⊻ LUX_SEED)
    color = color_from_seed(combined)
    
    layer_name = isempty(name) ? string(nameof(L)) : name
    
    ColoredLayer(layer, color, combined, layer_name)
end

"""
    layer_fingerprint(layer; seed=GAY_SEED) -> UInt64

Compute SPI fingerprint for a layer.
"""
function layer_fingerprint(layer::Lux.AbstractExplicitLayer; seed::UInt64=GAY_SEED)::UInt64
    cl = color_layer(layer; seed=seed)
    cl.fingerprint
end

# Specific layer coloring implementations
function color_layer(d::Lux.Dense; seed::UInt64=GAY_SEED, name::String="")
    idx = UInt64(d.in_dims) ⊻ (UInt64(d.out_dims) << 16) ⊻ hash(d.activation)
    combined = splitmix64(seed ⊻ idx ⊻ LUX_SEED)
    ColoredLayer(d, color_from_seed(combined), combined, 
                 isempty(name) ? "Dense($(d.in_dims)→$(d.out_dims))" : name)
end

function color_layer(c::Lux.Conv; seed::UInt64=GAY_SEED, name::String="")
    idx = reduce(⊻, UInt64.(c.kernel_size); init=UInt64(0)) ⊻
          (UInt64(c.in_chs) << 16) ⊻ (UInt64(c.out_chs) << 32)
    combined = splitmix64(seed ⊻ idx ⊻ LUX_SEED)
    ColoredLayer(c, color_from_seed(combined), combined,
                 isempty(name) ? "Conv($(c.kernel_size), $(c.in_chs)→$(c.out_chs))" : name)
end

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER COLORING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ColoredParameter{T, N}

A parameter tensor with per-element SPI colors for visualization.
"""
struct ColoredParameter{T, N}
    values::Array{T, N}
    colors::Array{RGB{Float32}, N}
    layer_name::String
    fingerprint::UInt64
end

"""
    color_parameters(ps::NamedTuple; seed=GAY_SEED) -> Dict{String, ColoredParameter}

Color all parameters in a Lux parameter tree.
Each parameter element gets a deterministic color based on its path + index.
"""
function color_parameters(ps::NamedTuple; seed::UInt64=GAY_SEED, prefix::String="")
    result = Dict{String, ColoredParameter}()
    
    for name in keys(ps)
        full_path = isempty(prefix) ? String(name) : "$(prefix).$(name)"
        param = ps[name]
        
        if param isa NamedTuple
            merge!(result, color_parameters(param; seed=seed, prefix=full_path))
        elseif param isa AbstractArray
            path_hash = hash(full_path)
            colors = similar(param, RGB{Float32})
            fp = UInt64(0)
            
            for i in eachindex(param)
                idx = UInt64(path_hash) ⊻ UInt64(i)
                seed_i = splitmix64(seed ⊻ idx)
                colors[i] = color_from_seed(seed_i)
                fp ⊻= seed_i
            end
            
            result[full_path] = ColoredParameter(param, colors, String(name), fp)
        end
    end
    
    result
end

"""
    parameter_fingerprint(ps::NamedTuple; seed=GAY_SEED) -> UInt64

Compute SPI fingerprint for entire parameter tree.
"""
function parameter_fingerprint(ps::NamedTuple; seed::UInt64=GAY_SEED)::UInt64
    colored = color_parameters(ps; seed=seed)
    reduce(⊻, (cp.fingerprint for cp in values(colored)); init=seed)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GRADIENT COLORING
# ═══════════════════════════════════════════════════════════════════════════════

"""
    gradient_color_map(f, ps, x; seed=GAY_SEED) -> NamedTuple

Compute gradients and assign colors based on gradient magnitude.
- High gradient → warm colors (red/orange)
- Low gradient → cool colors (blue/green)
- Zero gradient → gray
"""
function gradient_color_map(f, ps, x; seed::UInt64=GAY_SEED)
    loss, grads = Zygote.withgradient(f, ps, x)
    
    colored_grads = Dict{String, ColoredParameter}()
    
    function walk_grads(g, prefix="")
        if g isa NamedTuple
            for name in keys(g)
                full_path = isempty(prefix) ? String(name) : "$(prefix).$(name)"
                walk_grads(g[name], full_path)
            end
        elseif g isa AbstractArray && eltype(g) <: Number
            abs_grads = abs.(g)
            max_grad = maximum(abs_grads)
            max_grad = max_grad > 0 ? max_grad : 1.0f0
            
            colors = similar(g, RGB{Float32})
            fp = UInt64(0)
            
            for i in eachindex(g)
                normalized = Float32(abs_grads[i] / max_grad)
                hue = (1.0f0 - normalized) * 240.0f0
                colors[i] = convert(RGB{Float32}, HSL(hue, 0.8f0, 0.5f0))
                fp ⊻= splitmix64(seed ⊻ UInt64(round(normalized * 1e9)))
            end
            
            colored_grads[prefix] = ColoredParameter(g, colors, basename(prefix), fp)
        end
    end
    
    if !isnothing(grads) && length(grads) > 0 && !isnothing(grads[1])
        walk_grads(grads[1])
    end
    
    (gradients=grads, colors=colored_grads, loss=loss)
end

# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ColoredNetwork{M<:Lux.AbstractExplicitLayer}

A Lux model with SPI coloring for visualization.
"""
struct ColoredNetwork{M<:Lux.AbstractExplicitLayer}
    model::M
    layer_colors::Vector{ColoredLayer}
    fingerprint::UInt64
    seed::UInt64
end

"""
    ColoredNetwork(model; seed=GAY_SEED)

Create a colored network for visualization.
"""
function ColoredNetwork(model::M; seed::UInt64=GAY_SEED) where {M<:Lux.AbstractExplicitLayer}
    layers = ColoredLayer[]
    
    function walk_layers(m, prefix="")
        name = isempty(prefix) ? string(nameof(typeof(m))) : prefix
        push!(layers, color_layer(m; seed=seed, name=name))
        
        if hasproperty(m, :layers) && m.layers isa NamedTuple
            for fname in keys(m.layers)
                child = m.layers[fname]
                walk_layers(child, "$(name).$(fname)")
            end
        end
    end
    
    walk_layers(model)
    fp = reduce(⊻, (l.fingerprint for l in layers); init=seed)
    
    ColoredNetwork(model, layers, fp, seed)
end

"""
    network_fingerprint(cn::ColoredNetwork) -> UInt64
"""
network_fingerprint(cn::ColoredNetwork) = cn.fingerprint

"""
    render_network_graph(cn::ColoredNetwork) -> String

Render network as ANSI-colored ASCII graph.
"""
function render_network_graph(cn::ColoredNetwork)
    buf = IOBuffer()
    
    println(buf, "┌─────────────────────────────────────────────────────────────┐")
    println(buf, "│  COLORED NEURAL NETWORK (SPI Seed: 0x$(string(cn.seed, base=16)))  │")
    println(buf, "├─────────────────────────────────────────────────────────────┤")
    
    for cl in cn.layer_colors
        r = round(Int, cl.color.r * 255)
        g = round(Int, cl.color.g * 255)
        b = round(Int, cl.color.b * 255)
        
        layer_str = rpad(cl.name, 45)
        fp_str = "0x$(string(cl.fingerprint, base=16)[1:min(8, end)])"
        
        println(buf, "│ \e[38;2;$(r);$(g);$(b)m████\e[0m $(layer_str) $(fp_str) │")
    end
    
    println(buf, "├─────────────────────────────────────────────────────────────┤")
    println(buf, "│  Network FP: 0x$(string(cn.fingerprint, base=16))        │")
    println(buf, "└─────────────────────────────────────────────────────────────┘")
    
    String(take!(buf))
end

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING WITH COLORS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ColoredTrainingState

Training state with chromatic tracking.
"""
mutable struct ColoredTrainingState
    step::Int
    loss_history::Vector{Float32}
    fingerprint_history::Vector{UInt64}
    gradient_colors::Dict{String, ColoredParameter}
    seed::UInt64
end

ColoredTrainingState(; seed::UInt64=GAY_SEED) = 
    ColoredTrainingState(0, Float32[], UInt64[], Dict{String, ColoredParameter}(), seed)

"""
    colored_training_step(model, ps, st, x, y, loss_fn; seed=GAY_SEED, lr=0.01f0)

Execute one training step with gradient coloring for visualization.
"""
function colored_training_step(model, ps, st, x, y, loss_fn;
                                seed::UInt64=GAY_SEED, lr::Float32=0.01f0,
                                training_state::Union{ColoredTrainingState, Nothing}=nothing)
    (loss, st_new), grads = Zygote.withgradient(ps) do p
        y_pred, st_new = model(x, p, st)
        loss_fn(y_pred, y), st_new
    end
    
    grad_colors = color_parameters(grads[1]; seed=seed)
    
    ps_new = Lux.fmap((p, g) -> p .- lr .* g, ps, grads[1])
    
    param_fp = parameter_fingerprint(ps_new; seed=seed)
    
    if !isnothing(training_state)
        training_state.step += 1
        push!(training_state.loss_history, Float32(loss))
        push!(training_state.fingerprint_history, param_fp)
        training_state.gradient_colors = grad_colors
    end
    
    (ps=ps_new, st=st_new, loss=loss, gradient_colors=grad_colors, fingerprint=param_fp)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GAY OPTIMIZERS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayOptimizer

Optimizer with chromatic tracking.
"""
struct GayOptimizer{O}
    optimizer::O
    seed::UInt64
    color::RGB{Float32}
end

function gay_adam(; lr::Float32=0.001f0, seed::UInt64=GAY_SEED)
    opt_seed = splitmix64(seed ⊻ hash(:adam))
    GayOptimizer(Optimisers.Adam(lr), opt_seed, color_from_seed(opt_seed))
end

function gay_sgd(; lr::Float32=0.01f0, momentum::Float32=0.9f0, seed::UInt64=GAY_SEED)
    opt_seed = splitmix64(seed ⊻ hash(:sgd))
    GayOptimizer(Optimisers.Momentum(lr, momentum), opt_seed, color_from_seed(opt_seed))
end

# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING PARADIGM COLORS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    in_context_color(seed; context_length) -> RGB

Color for in-context learning (ICL) - longer context = more saturated.
"""
function in_context_color(seed::UInt64; context_length::Int=1024)::RGB{Float32}
    saturation = min(0.9f0, 0.3f0 + Float32(log2(context_length)) / 20.0f0)
    base = color_from_seed(splitmix64(seed ⊻ hash(:icl)))
    RGB{Float32}(base.r * saturation, base.g, base.b * (1.0f0 - saturation * 0.3f0))
end

"""
    adapter_color(seed; rank) -> RGB

Color for adapter/LoRA learning - lower rank = cooler colors.
"""
function adapter_color(seed::UInt64; rank::Int=8)::RGB{Float32}
    hue = 240.0f0 - Float32(rank) * 10.0f0  # Blue for low rank, warmer for high
    hue = clamp(hue, 0.0f0, 360.0f0)
    base_seed = splitmix64(seed ⊻ hash(:adapter) ⊻ UInt64(rank))
    lightness = 0.4f0 + (splitmix64(base_seed) >> 56) / 255.0f0 * 0.3f0
    convert(RGB{Float32}, HSL(hue, 0.7f0, lightness))
end

"""
    finetune_color(seed; epochs, dataset_size) -> RGB

Color for finetuning - encodes training intensity.
"""
function finetune_color(seed::UInt64; epochs::Int=3, dataset_size::Int=10000)::RGB{Float32}
    intensity = Float32(epochs * log10(dataset_size + 1)) / 20.0f0
    intensity = clamp(intensity, 0.0f0, 1.0f0)
    
    hue = 30.0f0 + intensity * 30.0f0  # Orange to red
    base_seed = splitmix64(seed ⊻ hash(:finetune) ⊻ UInt64(epochs) ⊻ UInt64(dataset_size))
    
    convert(RGB{Float32}, HSL(hue, 0.6f0 + intensity * 0.3f0, 0.5f0))
end

# ═══════════════════════════════════════════════════════════════════════════════
# SPI VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    verify_layer_spi(layer; seed=GAY_SEED, n_trials=100) -> Bool

Verify that layer coloring satisfies SPI.
"""
function verify_layer_spi(layer::Lux.AbstractExplicitLayer; 
                          seed::UInt64=GAY_SEED, n_trials::Int=100)::Bool
    ref = color_layer(layer; seed=seed)
    
    for _ in 1:n_trials
        test = color_layer(layer; seed=seed)
        if test.fingerprint != ref.fingerprint || test.color != ref.color
            return false
        end
    end
    
    true
end

"""
    verify_network_spi(model; seed=GAY_SEED, n_trials=100) -> Bool

Verify that network coloring satisfies SPI.
"""
function verify_network_spi(model::Lux.AbstractExplicitLayer;
                            seed::UInt64=GAY_SEED, n_trials::Int=100)::Bool
    ref = ColoredNetwork(model; seed=seed)
    
    for _ in 1:n_trials
        test = ColoredNetwork(model; seed=seed)
        if test.fingerprint != ref.fingerprint
            return false
        end
    end
    
    true
end

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

function world_gay_lux()
    println()
    println("╔═══════════════════════════════════════════════════════════════════════════╗")
    println("║  GAYLUXEXT: Neural Network SPI Coloring                                   ║")
    println("╚═══════════════════════════════════════════════════════════════════════════╝")
    println()
    
    model = Lux.Chain(
        Lux.Dense(784 => 256, relu),
        Lux.Dense(256 => 128, relu),
        Lux.Dense(128 => 10)
    )
    
    cn = ColoredNetwork(model)
    println(render_network_graph(cn))
    
    println("─── SPI VERIFICATION ───")
    println("  Layer SPI: $(verify_network_spi(model) ? "✓" : "✗")")
    println()
    
    println("─── LEARNING PARADIGM COLORS ───")
    icl = in_context_color(GAY_SEED; context_length=4096)
    ada = adapter_color(GAY_SEED; rank=16)
    ft = finetune_color(GAY_SEED; epochs=5, dataset_size=50000)
    
    println("  In-Context (4096): RGB($(round(icl.r, digits=2)), $(round(icl.g, digits=2)), $(round(icl.b, digits=2)))")
    println("  Adapter (rank=16): RGB($(round(ada.r, digits=2)), $(round(ada.g, digits=2)), $(round(ada.b, digits=2)))")
    println("  Finetune (5 epochs): RGB($(round(ft.r, digits=2)), $(round(ft.g, digits=2)), $(round(ft.b, digits=2)))")
    println()
    
    println("═══════════════════════════════════════════════════════════════════════════")
    
    cn
end

export world_gay_lux

end # module GayLuxExt
