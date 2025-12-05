# Comrade.jl-style Sky Model DSL with Colored S-Expressions
# Maps Lisp parentheses to VLBI imaging primitives with deterministic colors
#
# Inspired by Comrade.jl (EHT black hole imaging) which uses Pigeons.jl
# for sampling - the same SplittableRandoms pattern we use for colors!

using Colors: RGB

export SkyPrimitive, Ring, MRing, Gaussian, Disk, Crescent
export SkyModel, sky_add, sky_stretch, sky_rotate, sky_shift
export sky_show, sky_render
export comrade_ring, comrade_gaussian, comrade_model

# ═══════════════════════════════════════════════════════════════════════════
# Sky Model Primitives (like VLBISkyModels)
# ═══════════════════════════════════════════════════════════════════════════

abstract type SkyPrimitive end

"""
    Ring(radius, width)

A circular ring primitive (like Comrade's MRing convolved with Gaussian).
"""
struct Ring <: SkyPrimitive
    radius::Float64
    width::Float64
    color::RGB  # Deterministic color for visualization
end

"""
    MRing(radius, α, β)

Azimuthal ring with Fourier coefficients α, β (like Comrade's MRing).
"""
struct MRing <: SkyPrimitive
    radius::Float64
    α::Vector{Float64}  # cos coefficients
    β::Vector{Float64}  # sin coefficients
    color::RGB
end

"""
    Gaussian(σx, σy)

Elliptical Gaussian primitive.
"""
struct Gaussian <: SkyPrimitive
    σx::Float64
    σy::Float64
    color::RGB
end

"""
    Disk(radius)

Uniform disk primitive.
"""
struct Disk <: SkyPrimitive
    radius::Float64
    color::RGB
end

"""
    Crescent(r_out, r_in, shift)

Crescent (asymmetric ring) primitive.
"""
struct Crescent <: SkyPrimitive
    r_out::Float64
    r_in::Float64
    shift::Float64
    color::RGB
end

# ═══════════════════════════════════════════════════════════════════════════
# Composite Sky Model
# ═══════════════════════════════════════════════════════════════════════════

"""
    SkyModel

A composite sky model built from primitives with transformations.
Like Comrade's model composition: ring + gaussian, modify(...), etc.
"""
struct SkyModel
    components::Vector{Tuple{SkyPrimitive, NamedTuple}}  # (primitive, transforms)
    total_flux::Float64
end

SkyModel() = SkyModel([], 1.0)

# ═══════════════════════════════════════════════════════════════════════════
# Lisp S-Expression Constructors with Deterministic Colors
# ═══════════════════════════════════════════════════════════════════════════

"""
    (comrade-ring radius width)

Create a ring primitive with deterministic color from Gay.jl RNG.
"""
function comrade_ring(radius::Real, width::Real)
    c = next_color(current_colorspace())
    Ring(Float64(radius), Float64(width), c)
end

"""
    (comrade-mring radius α β)

Create an azimuthal ring with Fourier structure.
"""
function comrade_mring(radius::Real, α::Vector, β::Vector)
    c = next_color(current_colorspace())
    MRing(Float64(radius), Float64.(α), Float64.(β), c)
end

"""
    (comrade-gaussian σx σy)

Create an elliptical Gaussian.
"""
function comrade_gaussian(σx::Real, σy::Real)
    c = next_color(current_colorspace())
    Gaussian(Float64(σx), Float64(σy), c)
end
comrade_gaussian(σ::Real) = comrade_gaussian(σ, σ)

"""
    (comrade-disk radius)

Create a uniform disk.
"""
function comrade_disk(radius::Real)
    c = next_color(current_colorspace())
    Disk(Float64(radius), c)
end

"""
    (comrade-crescent r_out r_in shift)

Create a crescent (asymmetric ring).
"""
function comrade_crescent(r_out::Real, r_in::Real, shift::Real)
    c = next_color(current_colorspace())
    Crescent(Float64(r_out), Float64(r_in), Float64(shift), c)
end

# ═══════════════════════════════════════════════════════════════════════════
# Model Composition (like Comrade's +, modify, etc.)
# ═══════════════════════════════════════════════════════════════════════════

"""
    (sky-add component1 component2 ...)

Combine sky components additively (like Comrade's ring + gaussian).
"""
function sky_add(components::SkyPrimitive...)
    transforms = (flux=1.0, stretch=(1.0, 1.0), rotate=0.0, shift=(0.0, 0.0))
    SkyModel([(c, transforms) for c in components], 1.0)
end

"""
    (sky-stretch model sx sy)

Stretch a model (like Comrade's Stretch modifier).
"""
function sky_stretch(m::SkyModel, sx::Real, sy::Real=sx)
    new_components = [(c, merge(t, (stretch=(Float64(sx), Float64(sy)),))) 
                      for (c, t) in m.components]
    SkyModel(new_components, m.total_flux)
end

"""
    (sky-rotate model angle)

Rotate a model (like Comrade's Rotate modifier).
"""
function sky_rotate(m::SkyModel, angle::Real)
    new_components = [(c, merge(t, (rotate=Float64(angle),))) 
                      for (c, t) in m.components]
    SkyModel(new_components, m.total_flux)
end

"""
    (sky-shift model dx dy)

Shift/translate a model (like Comrade's shifted).
"""
function sky_shift(m::SkyModel, dx::Real, dy::Real)
    new_components = [(c, merge(t, (shift=(Float64(dx), Float64(dy)),))) 
                      for (c, t) in m.components]
    SkyModel(new_components, m.total_flux)
end

# ═══════════════════════════════════════════════════════════════════════════
# Colored S-Expression Display
# ═══════════════════════════════════════════════════════════════════════════

"""
    sky_show(primitive)

Display a sky primitive as a colored S-expression.
The parentheses are colored with the primitive's deterministic color.
"""
function sky_show(p::Ring)
    c = p.color
    r, g, b = round(Int, c.r*255), round(Int, c.g*255), round(Int, c.b*255)
    paren_color = "\e[38;2;$(r);$(g);$(b)m"
    reset = "\e[0m"
    "$(paren_color)($(reset)ring $(p.radius) $(p.width)$(paren_color))$(reset)"
end

function sky_show(p::MRing)
    c = p.color
    r, g, b = round(Int, c.r*255), round(Int, c.g*255), round(Int, c.b*255)
    paren_color = "\e[38;2;$(r);$(g);$(b)m"
    reset = "\e[0m"
    α_str = "[" * join(round.(p.α, digits=2), " ") * "]"
    β_str = "[" * join(round.(p.β, digits=2), " ") * "]"
    "$(paren_color)($(reset)mring $(p.radius) $(α_str) $(β_str)$(paren_color))$(reset)"
end

function sky_show(p::Gaussian)
    c = p.color
    r, g, b = round(Int, c.r*255), round(Int, c.g*255), round(Int, c.b*255)
    paren_color = "\e[38;2;$(r);$(g);$(b)m"
    reset = "\e[0m"
    "$(paren_color)($(reset)gaussian $(p.σx) $(p.σy)$(paren_color))$(reset)"
end

function sky_show(p::Disk)
    c = p.color
    r, g, b = round(Int, c.r*255), round(Int, c.g*255), round(Int, c.b*255)
    paren_color = "\e[38;2;$(r);$(g);$(b)m"
    reset = "\e[0m"
    "$(paren_color)($(reset)disk $(p.radius)$(paren_color))$(reset)"
end

function sky_show(p::Crescent)
    c = p.color
    r, g, b = round(Int, c.r*255), round(Int, c.g*255), round(Int, c.b*255)
    paren_color = "\e[38;2;$(r);$(g);$(b)m"
    reset = "\e[0m"
    "$(paren_color)($(reset)crescent $(p.r_out) $(p.r_in) $(p.shift)$(paren_color))$(reset)"
end

function sky_show(m::SkyModel)
    if isempty(m.components)
        return "(sky-model)"
    end
    
    parts = [sky_show(c) for (c, _) in m.components]
    if length(parts) == 1
        return parts[1]
    end
    
    # Color the + with rainbow
    plus = rainbow_text(" + ")
    return join(parts, plus)
end

# ═══════════════════════════════════════════════════════════════════════════
# ASCII Rendering (like Comrade's intensitymap)
# ═══════════════════════════════════════════════════════════════════════════

"""
    sky_render(model; size=30)

Render a sky model as colored ASCII art.
Each primitive contributes its color to the visualization.
"""
function sky_render(m::SkyModel; size::Int=30)
    # Create intensity map
    img = zeros(RGB{Float64}, size, size)
    cx, cy = size ÷ 2, size ÷ 2
    
    for (prim, transforms) in m.components
        render_primitive!(img, prim, transforms, cx, cy, size)
    end
    
    # Convert to ANSI
    buf = IOBuffer()
    for y in 1:size
        for x in 1:size
            c = img[y, x]
            if c.r + c.g + c.b > 0.1
                ri = round(Int, clamp(c.r, 0, 1) * 255)
                gi = round(Int, clamp(c.g, 0, 1) * 255)
                bi = round(Int, clamp(c.b, 0, 1) * 255)
                print(buf, "\e[48;2;$(ri);$(gi);$(bi)m  \e[0m")
            else
                print(buf, "  ")
            end
        end
        println(buf)
    end
    
    return String(take!(buf))
end

function render_primitive!(img, p::Ring, t, cx, cy, size)
    scale = size / 4
    for y in 1:size, x in 1:size
        dx = (x - cx) / scale
        dy = (y - cy) / scale
        r = sqrt(dx^2 + dy^2)
        
        # Ring intensity profile
        ring_r = p.radius
        ring_w = p.width
        intensity = exp(-((r - ring_r) / ring_w)^2)
        
        if intensity > 0.1
            img[y, x] = RGB(
                img[y, x].r + p.color.r * intensity * t.flux,
                img[y, x].g + p.color.g * intensity * t.flux,
                img[y, x].b + p.color.b * intensity * t.flux
            )
        end
    end
end

function render_primitive!(img, p::Gaussian, t, cx, cy, size)
    scale = size / 4
    for y in 1:size, x in 1:size
        dx = (x - cx) / scale
        dy = (y - cy) / scale
        intensity = exp(-(dx^2 / (2*p.σx^2) + dy^2 / (2*p.σy^2)))
        
        if intensity > 0.1
            img[y, x] = RGB(
                img[y, x].r + p.color.r * intensity * t.flux,
                img[y, x].g + p.color.g * intensity * t.flux,
                img[y, x].b + p.color.b * intensity * t.flux
            )
        end
    end
end

function render_primitive!(img, p::Disk, t, cx, cy, size)
    scale = size / 4
    for y in 1:size, x in 1:size
        dx = (x - cx) / scale
        dy = (y - cy) / scale
        r = sqrt(dx^2 + dy^2)
        
        if r < p.radius
            img[y, x] = RGB(
                img[y, x].r + p.color.r * t.flux,
                img[y, x].g + p.color.g * t.flux,
                img[y, x].b + p.color.b * t.flux
            )
        end
    end
end

function render_primitive!(img, p::Crescent, t, cx, cy, size)
    scale = size / 4
    for y in 1:size, x in 1:size
        dx = (x - cx) / scale - p.shift
        dy = (y - cy) / scale
        r = sqrt(dx^2 + dy^2)
        
        if r < p.r_out && r > p.r_in
            img[y, x] = RGB(
                img[y, x].r + p.color.r * t.flux,
                img[y, x].g + p.color.g * t.flux,
                img[y, x].b + p.color.b * t.flux
            )
        end
    end
end

function render_primitive!(img, p::MRing, t, cx, cy, size)
    # Simplified MRing rendering
    render_primitive!(img, Ring(p.radius, 0.2, p.color), t, cx, cy, size)
end

# ═══════════════════════════════════════════════════════════════════════════
# High-level M87/Sgr A* style model builder
# ═══════════════════════════════════════════════════════════════════════════

"""
    comrade_model(; seed=42, style=:m87)

Build a Comrade-style sky model with deterministic colors.
Returns the model and its colored S-expression representation.

Styles: :m87, :sgra, :custom
"""
function comrade_model(; seed::Integer=42, style::Symbol=:m87)
    gay_seed!(seed)
    
    if style == :m87
        # M87* style: ring + gaussian
        ring = comrade_ring(1.0, 0.3)
        gauss = comrade_gaussian(0.5, 0.3)
        model = sky_add(ring, gauss)
        
    elseif style == :sgra
        # Sgr A* style: crescent + disk
        crescent = comrade_crescent(1.2, 0.6, 0.3)
        disk = comrade_disk(0.4)
        model = sky_add(crescent, disk)
        
    else
        # Custom: ring + mring + gaussian
        ring = comrade_ring(1.0, 0.25)
        mring = comrade_mring(0.8, [0.3, 0.1], [0.2, -0.1])
        gauss = comrade_gaussian(0.6)
        model = sky_add(ring, mring, gauss)
    end
    
    return model
end

"""
    (comrade-show model)

Display a model as colored S-expression + ASCII render.
"""
function comrade_show(m::SkyModel)
    println("\n  Colored S-Expression (parentheses colored by component):")
    println("  ", sky_show(m))
    println("\n  Intensity Map:")
    print(sky_render(m; size=25))
    return m
end
