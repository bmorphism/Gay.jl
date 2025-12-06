# Decapodes.jl extension for Gay.jl
# Deterministic SPI-compliant coloring for physics simulations and DEC

module GayDecapodesExt

using Gay: hash_color_rgb, splitmix64, GAY_SEED
using Decapodes
using CombinatorialSpaces
using Colors: RGB, HSL, convert

export color_mesh, color_field, color_operator
export color_decapode, color_simulation_state
export render_colored_mesh, render_decapode

# ═══════════════════════════════════════════════════════════════════════════
# Mesh Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_mesh(mesh::EmbeddedDeltaSet2D; seed=GAY_SEED) -> NamedTuple

Color mesh elements (vertices, edges, triangles) with SPI colors.

# Example
```julia
mesh = loadmesh(Rectangle_30x10())
colors = color_mesh(mesh)
```
"""
function color_mesh(mesh::EmbeddedDeltaSet2D; seed::UInt64=GAY_SEED)
    n_v = nv(mesh)
    n_e = ne(mesh)
    n_t = ntriangles(mesh)
    
    vertex_colors = [hash_color_rgb(UInt64(v), seed) for v in 1:n_v]
    
    edge_colors = map(1:n_e) do e
        v0 = mesh[e, :∂v0]
        v1 = mesh[e, :∂v1]
        idx = UInt64(v0) ⊻ UInt64(v1)
        hash_color_rgb(idx, seed)
    end
    
    tri_colors = map(1:n_t) do t
        verts = triangle_vertices(mesh, t)
        idx = reduce(⊻, UInt64.(sort(collect(verts))))
        hash_color_rgb(idx, seed)
    end
    
    (vertices=vertex_colors, edges=edge_colors, triangles=tri_colors)
end

"""
    color_mesh(mesh::DeltaSet1D; seed=GAY_SEED) -> NamedTuple

Color 1D mesh elements.
"""
function color_mesh(mesh::DeltaSet1D; seed::UInt64=GAY_SEED)
    n_v = nv(mesh)
    n_e = ne(mesh)
    
    vertex_colors = [hash_color_rgb(UInt64(v), seed) for v in 1:n_v]
    
    edge_colors = map(1:n_e) do e
        v0 = mesh[e, :∂v0]
        v1 = mesh[e, :∂v1]
        idx = UInt64(v0) ⊻ UInt64(v1)
        hash_color_rgb(idx, seed)
    end
    
    (vertices=vertex_colors, edges=edge_colors)
end

# ═══════════════════════════════════════════════════════════════════════════
# Field Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_field(mesh, field::AbstractVector; seed=GAY_SEED, form=0) -> Vector{RGB{Float32}}

Color a discrete form (0-form on vertices, 1-form on edges, 2-form on triangles).

Hue from mesh position (SPI), lightness from field value.

# Example
```julia
mesh = loadmesh(Rectangle_30x10())
temperature = simulate(heat_equation, mesh)
colors = color_field(mesh, temperature; form=0)
```
"""
function color_field(mesh, field::AbstractVector{T}; 
                     seed::UInt64=GAY_SEED, form::Int=0) where T
    if form == 0
        n = nv(mesh)
    elseif form == 1
        n = ne(mesh)
    elseif form == 2
        n = ntriangles(mesh)
    else
        error("Form must be 0, 1, or 2")
    end
    
    @assert length(field) == n "Field length must match form dimension"
    
    fmin, fmax = extrema(real.(field))
    range_val = fmax - fmin
    range_val = range_val > 0 ? range_val : one(T)
    
    map(1:n) do i
        base_color = hash_color_rgb(UInt64(i), seed)
        base_hsl = convert(HSL, base_color)
        
        normalized = (real(field[i]) - fmin) / range_val
        lightness = Float32(0.15 + 0.7 * normalized)
        
        convert(RGB{Float32}, HSL(base_hsl.h, 0.85f0, lightness))
    end
end

"""
    color_field(mesh, field::AbstractVector, phase::AbstractVector; seed=GAY_SEED) -> Vector{RGB{Float32}}

Color a complex field with magnitude → lightness, phase → hue shift.
"""
function color_field(mesh, magnitude::AbstractVector{T}, phase::AbstractVector{P}; 
                     seed::UInt64=GAY_SEED) where {T,P}
    n = length(magnitude)
    @assert length(phase) == n
    
    mmax = maximum(abs, magnitude)
    mmax = mmax > 0 ? mmax : one(T)
    
    map(1:n) do i
        base_color = hash_color_rgb(UInt64(i), seed)
        base_hsl = convert(HSL, base_color)
        
        mag_normalized = abs(magnitude[i]) / mmax
        lightness = Float32(0.2 + 0.6 * mag_normalized)
        
        phase_shift = Float32(mod(phase[i] * 180 / π, 360))
        new_hue = mod(base_hsl.h + phase_shift, 360.0f0)
        
        convert(RGB{Float32}, HSL(new_hue, 0.8f0, lightness))
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Operator Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_operator(op::Symbol; seed=GAY_SEED) -> Function

Get a coloring function for DEC operator matrix entries.

Operators: `:d`, `:δ`, `:Δ`, `:⋆`, `:♭`, `:♯`, `:∂`

Returns a function `(i, j) -> RGB{Float32}` for matrix entry coloring.

# Example
```julia
colorer = color_operator(:Δ)
c = colorer(10, 15)  # Color for Laplacian entry (10, 15)
```
"""
function color_operator(op::Symbol; seed::UInt64=GAY_SEED)
    op_hues = Dict(
        :d => 30.0f0,      # exterior derivative - orange
        :δ => 200.0f0,     # codifferential - cyan
        :Δ => 300.0f0,     # Laplacian - magenta
        :⋆ => 180.0f0,     # Hodge star - teal
        :♭ => 120.0f0,     # flat - green
        :♯ => 150.0f0,     # sharp - sea green
        :∂ => 60.0f0,      # boundary - yellow
    )
    
    base_hue = get(op_hues, op, 0.0f0)
    
    function color_entry(i::Int, j::Int)
        idx = UInt64(i) ⊻ (UInt64(j) << 32)
        h_offset = Float32((splitmix64(seed ⊻ idx) % 30)) - 15.0f0
        h = mod(base_hue + h_offset, 360.0f0)
        convert(RGB{Float32}, HSL(h, 0.7f0, 0.5f0))
    end
    
    color_entry
end

# ═══════════════════════════════════════════════════════════════════════════
# Decapode Structure Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_decapode(d::SummationDecapode; seed=GAY_SEED) -> NamedTuple

Color a Decapode diagram - variables, operators, and summations.

# Example
```julia
HeatEquation = @decapode begin
    C::Form0
    ∂ₜ(C) == Δ(C)
end
colors = color_decapode(HeatEquation)
```
"""
function color_decapode(d::SummationDecapode; seed::UInt64=GAY_SEED)
    n_vars = nparts(d, :Var)
    n_ops = nparts(d, :Op1) + nparts(d, :Op2)
    n_sums = nparts(d, :Σ)
    
    var_colors = [hash_color_rgb(UInt64(v), seed) for v in 1:n_vars]
    
    op_seed = seed ⊻ 0x9e3779b97f4a7c15
    op_colors = [hash_color_rgb(UInt64(o), op_seed) for o in 1:n_ops]
    
    sum_seed = seed ⊻ 0xbf58476d1ce4e5b9
    sum_colors = [hash_color_rgb(UInt64(s), sum_seed) for s in 1:n_sums]
    
    (variables=var_colors, operators=op_colors, summations=sum_colors)
end

# ═══════════════════════════════════════════════════════════════════════════
# Simulation State Coloring
# ═══════════════════════════════════════════════════════════════════════════

"""
    color_simulation_state(mesh, state::NamedTuple; seed=GAY_SEED) -> Dict{Symbol, Vector{RGB{Float32}}}

Color all fields in a simulation state.

# Example
```julia
mesh = loadmesh(Rectangle_30x10())
state = (C=temperature_field, V=velocity_field)
colors = color_simulation_state(mesh, state)
```
"""
function color_simulation_state(mesh, state::NamedTuple; seed::UInt64=GAY_SEED)
    result = Dict{Symbol, Vector{RGB{Float32}}}()
    
    for (name, field) in pairs(state)
        field_seed = seed ⊻ UInt64(hash(name))
        
        n = length(field)
        if n == nv(mesh)
            result[name] = color_field(mesh, field; seed=field_seed, form=0)
        elseif n == ne(mesh)
            result[name] = color_field(mesh, field; seed=field_seed, form=1)
        elseif n == ntriangles(mesh)
            result[name] = color_field(mesh, field; seed=field_seed, form=2)
        else
            result[name] = [hash_color_rgb(UInt64(i), field_seed) for i in 1:n]
        end
    end
    
    result
end

# ═══════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════

"""
    render_colored_mesh(mesh, colors::NamedTuple) -> String

Render mesh with ANSI colors showing structure.
"""
function render_colored_mesh(mesh, colors::NamedTuple)
    lines = String[]
    push!(lines, "Mesh: $(nv(mesh)) vertices, $(ne(mesh)) edges")
    
    if haskey(colors, :triangles)
        push!(lines, "      $(ntriangles(mesh)) triangles")
    end
    
    push!(lines, "\nVertex colors (first 10):")
    for (i, c) in enumerate(colors.vertices[1:min(10, end)])
        r = round(Int, clamp(c.r, 0, 1) * 255)
        g = round(Int, clamp(c.g, 0, 1) * 255)
        b = round(Int, clamp(c.b, 0, 1) * 255)
        push!(lines, "  \e[38;2;$(r);$(g);$(b)m█\e[0m v$i")
    end
    
    join(lines, "\n")
end

"""
    render_decapode(d::SummationDecapode, colors::NamedTuple) -> String

Render a Decapode diagram with ANSI colors.
"""
function render_decapode(d::SummationDecapode, colors::NamedTuple)
    lines = String["Decapode Structure:"]
    
    for (i, c) in enumerate(colors.variables)
        r = round(Int, clamp(c.r, 0, 1) * 255)
        g = round(Int, clamp(c.g, 0, 1) * 255)
        b = round(Int, clamp(c.b, 0, 1) * 255)
        var_name = d[i, :name]
        var_type = d[i, :type]
        push!(lines, "  \e[38;2;$(r);$(g);$(b)m█\e[0m $var_name :: $var_type")
    end
    
    join(lines, "\n")
end

function __init__()
    @info "Gay.jl Decapodes extension loaded - physics simulation coloring available"
end

end # module GayDecapodesExt
