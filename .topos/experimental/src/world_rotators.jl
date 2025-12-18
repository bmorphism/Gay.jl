# ═══════════════════════════════════════════════════════════════════════════════
# World Rotators: Beyond Euclid with Chromatic Identity
# ═══════════════════════════════════════════════════════════════════════════════
#
# "World rotators are the mathematical objects that describe how to rotate
#  between reference frames in different geometries."
#
# This module provides:
#   1. Euclidean rotations (SO(3), quaternions)
#   2. Hyperbolic rotations (Möbius transformations, Lorentz boosts)
#   3. Spherical/elliptic rotations (great circles, stereographic projection)
#   4. Higher-kinded Gay Types for compositional semantics
#   5. Integration with AnanasACSet for provenance tracking
#
# Key insight: Each rotation style gets a chromatic identity from Gay.jl,
# enabling visual verification across geometric domains.
#
# Seeds 69 and 1069 are compared to show how different "worlds" rotate.
#
# ═══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra

export WorldRotator, EuclideanRotator, HyperbolicRotator, SphericalRotator
export GayType, GayHKT, GayRefinement, GayInductive, GayPROP
export rotate, compose_rotators, rotator_color, rotator_fingerprint
export compare_seeds_69_1069, demo_world_rotators

# ═══════════════════════════════════════════════════════════════════════════════
# Abstract World Rotator Interface
# ═══════════════════════════════════════════════════════════════════════════════

"""
    WorldRotator

Abstract type for rotations in different geometries.
Each geometry provides its own rotation group:
- Euclidean: SO(3) (special orthogonal group)
- Hyperbolic: PSL(2,ℝ) (Möbius transformations)
- Spherical: SO(3) but on the sphere
- Lorentz: SO(3,1) (Minkowski space rotations)
"""
abstract type WorldRotator end

"""
    rotate(rotator::WorldRotator, point) -> rotated_point

Apply rotation to a point in the appropriate geometry.
"""
function rotate end

"""
    compose_rotators(r1::T, r2::T) -> T where T <: WorldRotator

Compose two rotators of the same type.
"""
function compose_rotators end

# ═══════════════════════════════════════════════════════════════════════════════
# Euclidean Rotator (SO(3) via quaternions)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    EuclideanRotator

Quaternion-based rotation in 3D Euclidean space.
q = w + xi + yj + zk, with ||q|| = 1

Avoids gimbal lock and composes naturally.
"""
struct EuclideanRotator <: WorldRotator
    q::NTuple{4, Float64}  # (w, x, y, z)
    seed::UInt64
    color::NTuple{3, Float32}
    fingerprint::UInt64
end

function EuclideanRotator(w::Float64, x::Float64, y::Float64, z::Float64; seed::UInt64 = GAY_SEED)
    # Normalize to unit quaternion
    norm = sqrt(w^2 + x^2 + y^2 + z^2)
    q = (w/norm, x/norm, y/norm, z/norm)
    
    # Color from quaternion components
    fp = splitmix64_mix(reinterpret(UInt64, q[1]) ⊻ reinterpret(UInt64, q[2]) ⊻
                        reinterpret(UInt64, q[3]) ⊻ reinterpret(UInt64, q[4]) ⊻ seed)
    color = hash_color(seed, fp)
    
    EuclideanRotator(q, seed, color, fp)
end

"""
Create rotation from axis-angle representation.
"""
function EuclideanRotator(axis::NTuple{3, Float64}, angle::Float64; seed::UInt64 = GAY_SEED)
    # Normalize axis
    ax, ay, az = axis
    norm = sqrt(ax^2 + ay^2 + az^2)
    ax, ay, az = ax/norm, ay/norm, az/norm
    
    # Convert to quaternion
    half = angle / 2
    w = cos(half)
    s = sin(half)
    x, y, z = s*ax, s*ay, s*az
    
    EuclideanRotator(w, x, y, z; seed=seed)
end

function rotate(r::EuclideanRotator, p::NTuple{3, Float64})
    w, x, y, z = r.q
    px, py, pz = p
    
    # Quaternion rotation: q * p * q^(-1)
    # Using expanded formula for efficiency
    t0 = w*px + y*pz - z*py
    t1 = w*py + z*px - x*pz
    t2 = w*pz + x*py - y*px
    t3 = -x*px - y*py - z*pz
    
    (t0*w - t3*x - t1*z + t2*y,
     t1*w - t3*y - t2*x + t0*z,
     t2*w - t3*z - t0*y + t1*x)
end

function compose_rotators(r1::EuclideanRotator, r2::EuclideanRotator)
    # Quaternion multiplication
    w1, x1, y1, z1 = r1.q
    w2, x2, y2, z2 = r2.q
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    EuclideanRotator(w, x, y, z; seed=r1.seed ⊻ r2.seed)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Hyperbolic Rotator (PSL(2,ℝ) Möbius transformations)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    HyperbolicRotator

Möbius transformation in the Poincaré disk model.
f(z) = (az + b) / (cz + d) where ad - bc = 1

Hyperbolic "rotations" are actually isometries that preserve
the hyperbolic metric ds² = 4(dx² + dy²) / (1 - x² - y²)²
"""
struct HyperbolicRotator <: WorldRotator
    a::Complex{Float64}
    b::Complex{Float64}
    c::Complex{Float64}
    d::Complex{Float64}
    seed::UInt64
    color::NTuple{3, Float32}
    fingerprint::UInt64
end

function HyperbolicRotator(a::Complex{Float64}, b::Complex{Float64}, 
                           c::Complex{Float64}, d::Complex{Float64};
                           seed::UInt64 = GAY_SEED)
    # Normalize to SL(2,ℂ)
    det = a*d - b*c
    scale = sqrt(det)
    a, b, c, d = a/scale, b/scale, c/scale, d/scale
    
    fp = splitmix64_mix(
        reinterpret(UInt64, real(a)) ⊻ reinterpret(UInt64, imag(a)) ⊻
        reinterpret(UInt64, real(b)) ⊻ reinterpret(UInt64, imag(b)) ⊻
        reinterpret(UInt64, real(c)) ⊻ reinterpret(UInt64, imag(c)) ⊻
        reinterpret(UInt64, real(d)) ⊻ reinterpret(UInt64, imag(d)) ⊻ seed
    )
    color = hash_color(seed, fp)
    
    HyperbolicRotator(a, b, c, d, seed, color, fp)
end

"""
Create hyperbolic rotation from boost parameter.
"""
function HyperbolicRotator(boost::Float64; angle::Float64 = 0.0, seed::UInt64 = GAY_SEED)
    # Hyperbolic rotation in Poincaré disk
    # Combines rotation and translation along a geodesic
    e_iθ = exp(im * angle)
    cosh_b = cosh(boost / 2)
    sinh_b = sinh(boost / 2)
    
    a = Complex{Float64}(cosh_b * real(e_iθ), cosh_b * imag(e_iθ))
    b = Complex{Float64}(sinh_b * real(e_iθ), -sinh_b * imag(e_iθ))
    c = conj(b)
    d = conj(a)
    
    HyperbolicRotator(a, b, c, d; seed=seed)
end

function rotate(r::HyperbolicRotator, z::Complex{Float64})
    (r.a * z + r.b) / (r.c * z + r.d)
end

function compose_rotators(r1::HyperbolicRotator, r2::HyperbolicRotator)
    # Matrix multiplication for SL(2,ℂ)
    a = r1.a * r2.a + r1.b * r2.c
    b = r1.a * r2.b + r1.b * r2.d
    c = r1.c * r2.a + r1.d * r2.c
    d = r1.c * r2.b + r1.d * r2.d
    
    HyperbolicRotator(a, b, c, d; seed=r1.seed ⊻ r2.seed)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Spherical Rotator (Great circles on S²)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SphericalRotator

Rotation on the 2-sphere using stereographic projection.
In elliptic geometry, parallel lines meet at antipodal points.
"""
struct SphericalRotator <: WorldRotator
    axis::NTuple{3, Float64}  # Axis through center
    angle::Float64
    seed::UInt64
    color::NTuple{3, Float32}
    fingerprint::UInt64
end

function SphericalRotator(axis::NTuple{3, Float64}, angle::Float64; seed::UInt64 = GAY_SEED)
    ax, ay, az = axis
    norm = sqrt(ax^2 + ay^2 + az^2)
    axis_n = (ax/norm, ay/norm, az/norm)
    
    fp = splitmix64_mix(
        reinterpret(UInt64, axis_n[1]) ⊻ reinterpret(UInt64, axis_n[2]) ⊻
        reinterpret(UInt64, axis_n[3]) ⊻ reinterpret(UInt64, angle) ⊻ seed
    )
    color = hash_color(seed, fp)
    
    SphericalRotator(axis_n, angle, seed, color, fp)
end

function rotate(r::SphericalRotator, p::NTuple{3, Float64})
    # Rodrigues' rotation formula on S²
    ax, ay, az = r.axis
    px, py, pz = p
    
    # Normalize input to sphere
    norm_p = sqrt(px^2 + py^2 + pz^2)
    px, py, pz = px/norm_p, py/norm_p, pz/norm_p
    
    c = cos(r.angle)
    s = sin(r.angle)
    
    # k × p
    kxp = (ay*pz - az*py, az*px - ax*pz, ax*py - ay*px)
    # k · p
    kdp = ax*px + ay*py + az*pz
    
    # Rodrigues formula: p*cos(θ) + (k×p)*sin(θ) + k*(k·p)*(1-cos(θ))
    (px*c + kxp[1]*s + ax*kdp*(1-c),
     py*c + kxp[2]*s + ay*kdp*(1-c),
     pz*c + kxp[3]*s + az*kdp*(1-c))
end

function compose_rotators(r1::SphericalRotator, r2::SphericalRotator)
    # Convert to quaternions, multiply, convert back
    q1 = EuclideanRotator(r1.axis, r1.angle; seed=r1.seed)
    q2 = EuclideanRotator(r2.axis, r2.angle; seed=r2.seed)
    q3 = compose_rotators(q1, q2)
    
    # Extract axis-angle from quaternion
    w, x, y, z = q3.q
    angle = 2 * acos(clamp(w, -1.0, 1.0))
    s = sqrt(1 - w^2)
    if s < 1e-10
        axis = (1.0, 0.0, 0.0)
    else
        axis = (x/s, y/s, z/s)
    end
    
    SphericalRotator(axis, angle; seed=r1.seed ⊻ r2.seed)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Gay Types: Higher-Kinded, Refinement, Inductive, PROP
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayType

Abstract base for Gay type system with chromatic identity.
Each type gets a unique color based on its structure.
"""
abstract type GayType end

"""
    GayHKT{F, A}

Higher-Kinded Type: F[A]
Like Haskell's `* -> *` kinds but with color.

Example: List[Int] has color derived from List's color ⊗ Int's color
"""
struct GayHKT{F, A} <: GayType
    constructor::Symbol      # F
    argument::Symbol         # A
    seed::UInt64
    color::NTuple{3, Float32}
    fingerprint::UInt64
end

function GayHKT(F::Symbol, A::Symbol; seed::UInt64 = GAY_SEED)
    fp = splitmix64_mix(hash(F) ⊻ hash(A) ⊻ seed)
    color = hash_color(seed, fp)
    GayHKT{F, A}(F, A, seed, color, fp)
end

"""
    GayRefinement{T}

Refinement Type: {x : T | φ(x)}
Type with predicate that adds color from the constraint.

Example: {n : Nat | n > 0} is "positive natural" with specific color
"""
struct GayRefinement{T} <: GayType
    base_type::Symbol
    predicate::Symbol        # Symbolic predicate
    predicate_hash::UInt64   # Hash of predicate for color
    seed::UInt64
    color::NTuple{3, Float32}
    fingerprint::UInt64
end

function GayRefinement(T::Symbol, pred::Symbol; seed::UInt64 = GAY_SEED)
    pred_hash = splitmix64_mix(hash(pred))
    fp = splitmix64_mix(hash(T) ⊻ pred_hash ⊻ seed)
    color = hash_color(seed, fp)
    GayRefinement{T}(T, pred, pred_hash, seed, color, fp)
end

"""
    GayInductive

Inductive Type: μX. F[X]
Least fixed point of a type functor.

Example: List = μX. 1 + A × X (nil or cons)
"""
struct GayInductive <: GayType
    name::Symbol
    constructors::Vector{Tuple{Symbol, Vector{Symbol}}}  # Name → argument types
    seed::UInt64
    color::NTuple{3, Float32}
    fingerprint::UInt64
end

function GayInductive(name::Symbol, constructors::Vector{Tuple{Symbol, Vector{Symbol}}}; 
                      seed::UInt64 = GAY_SEED)
    fp = hash(name) ⊻ seed
    for (ctor, args) in constructors
        fp ⊻= splitmix64_mix(hash(ctor))
        for arg in args
            fp ⊻= splitmix64_mix(hash(arg))
        end
    end
    fp = splitmix64_mix(fp)
    color = hash_color(seed, fp)
    GayInductive(name, constructors, seed, color, fp)
end

"""
    GayPROP

PROP (Products and Permutations category) with chromatic identity.
Morphisms are (m → n) where m, n are natural numbers.
Used for compositional semantics of interfaces.

This models string diagrams for monoidal categories.
"""
struct GayPROP <: GayType
    name::Symbol
    inputs::Int              # Number of input wires
    outputs::Int             # Number of output wires
    generators::Vector{Tuple{Symbol, Int, Int}}  # (name, in, out)
    seed::UInt64
    color::NTuple{3, Float32}
    fingerprint::UInt64
end

function GayPROP(name::Symbol, inputs::Int, outputs::Int,
                 generators::Vector{Tuple{Symbol, Int, Int}} = Tuple{Symbol, Int, Int}[];
                 seed::UInt64 = GAY_SEED)
    fp = hash(name) ⊻ UInt64(inputs) ⊻ (UInt64(outputs) << 32) ⊻ seed
    for (gen_name, in_wires, out_wires) in generators
        fp ⊻= splitmix64_mix(hash(gen_name) ⊻ UInt64(in_wires * 1000 + out_wires))
    end
    fp = splitmix64_mix(fp)
    color = hash_color(seed, fp)
    GayPROP(name, inputs, outputs, generators, seed, color, fp)
end

"""
    compose_props(p1::GayPROP, p2::GayPROP) -> GayPROP

Compose two PROPs (sequential composition: p1 ; p2)
Requires p1.outputs == p2.inputs
"""
function compose_props(p1::GayPROP, p2::GayPROP)
    @assert p1.outputs == p2.inputs "Cannot compose: $(p1.outputs) ≠ $(p2.inputs)"
    
    new_name = Symbol("$(p1.name);$(p2.name)")
    generators = vcat(p1.generators, p2.generators)
    
    GayPROP(new_name, p1.inputs, p2.outputs, generators; 
            seed=p1.seed ⊻ p2.seed)
end

"""
    tensor_props(p1::GayPROP, p2::GayPROP) -> GayPROP

Tensor two PROPs (parallel composition: p1 ⊗ p2)
"""
function tensor_props(p1::GayPROP, p2::GayPROP)
    new_name = Symbol("$(p1.name)⊗$(p2.name)")
    generators = vcat(p1.generators, p2.generators)
    
    GayPROP(new_name, p1.inputs + p2.inputs, p1.outputs + p2.outputs, generators;
            seed=p1.seed ⊻ p2.seed)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Seed 69 vs 1069 Comparison
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compare_seeds_69_1069()

Compare Gay types and world rotators using seeds 69 and 1069.
Shows how different "worlds" (seeds) produce different chromatic identities.
"""
function compare_seeds_69_1069()
    println("═" ^ 70)
    println("  SEED COMPARISON: 69 vs 1069")
    println("  Beyond Euclid: How Different Worlds Rotate")
    println("═" ^ 70)
    println()
    
    seed_69 = UInt64(69)
    seed_1069 = UInt64(1069)
    
    # 1. Basic colors
    println("1. BASIC next_color COMPARISON")
    for i in 1:5
        c69 = hash_color(seed_69, UInt64(i))
        c1069 = hash_color(seed_1069, UInt64(i))
        r69, g69, b69 = round.(Int, c69 .* 255)
        r1069, g1069, b1069 = round.(Int, c1069 .* 255)
        println("   [$i] seed=69: RGB($r69,$g69,$b69)  seed=1069: RGB($r1069,$g1069,$b1069)")
    end
    println()
    
    # 2. Euclidean rotators
    println("2. EUCLIDEAN ROTATORS (SO(3) quaternions)")
    rot69 = EuclideanRotator((1.0, 0.0, 0.0), π/4; seed=seed_69)
    rot1069 = EuclideanRotator((1.0, 0.0, 0.0), π/4; seed=seed_1069)
    r69, g69, b69 = round.(Int, rot69.color .* 255)
    r1069, g1069, b1069 = round.(Int, rot1069.color .* 255)
    println("   45° around X-axis:")
    println("     seed=69:   RGB($r69,$g69,$b69)")
    println("     seed=1069: RGB($r1069,$g1069,$b1069)")
    
    # Rotate a point
    p = (1.0, 0.0, 0.0)
    p69 = rotate(rot69, p)
    p1069 = rotate(rot1069, p)
    println("   Point (1,0,0) rotated:")
    println("     seed=69:   $(round.(p69, digits=3))")
    println("     seed=1069: $(round.(p1069, digits=3)) (same geometry, different color)")
    println()
    
    # 3. Hyperbolic rotators
    println("3. HYPERBOLIC ROTATORS (PSL(2,ℝ) Möbius)")
    hyp69 = HyperbolicRotator(0.5; angle=π/6, seed=seed_69)
    hyp1069 = HyperbolicRotator(0.5; angle=π/6, seed=seed_1069)
    r69, g69, b69 = round.(Int, hyp69.color .* 255)
    r1069, g1069, b1069 = round.(Int, hyp1069.color .* 255)
    println("   Boost=0.5, angle=30°:")
    println("     seed=69:   RGB($r69,$g69,$b69)")
    println("     seed=1069: RGB($r1069,$g1069,$b1069)")
    
    z = Complex{Float64}(0.3, 0.2)
    z69 = rotate(hyp69, z)
    z1069 = rotate(hyp1069, z)
    println("   Point 0.3+0.2i (Poincaré disk):")
    println("     seed=69:   $(round(z69, digits=3))")
    println("     seed=1069: $(round(z1069, digits=3))")
    println()
    
    # 4. Spherical rotators
    println("4. SPHERICAL ROTATORS (Great circles on S²)")
    sph69 = SphericalRotator((0.0, 0.0, 1.0), π/3; seed=seed_69)
    sph1069 = SphericalRotator((0.0, 0.0, 1.0), π/3; seed=seed_1069)
    r69, g69, b69 = round.(Int, sph69.color .* 255)
    r1069, g1069, b1069 = round.(Int, sph1069.color .* 255)
    println("   60° around Z (north pole):")
    println("     seed=69:   RGB($r69,$g69,$b69)")
    println("     seed=1069: RGB($r1069,$g1069,$b1069)")
    println()
    
    # 5. Gay Types comparison
    println("5. GAY TYPES (Higher-Kinded, Refinement, Inductive, PROP)")
    
    # HKT
    hkt69 = GayHKT(:List, :Int; seed=seed_69)
    hkt1069 = GayHKT(:List, :Int; seed=seed_1069)
    r69, g69, b69 = round.(Int, hkt69.color .* 255)
    r1069, g1069, b1069 = round.(Int, hkt1069.color .* 255)
    println("   List[Int] (HKT):")
    println("     seed=69:   RGB($r69,$g69,$b69)")
    println("     seed=1069: RGB($r1069,$g1069,$b1069)")
    
    # Refinement
    ref69 = GayRefinement(:Nat, :positive; seed=seed_69)
    ref1069 = GayRefinement(:Nat, :positive; seed=seed_1069)
    r69, g69, b69 = round.(Int, ref69.color .* 255)
    r1069, g1069, b1069 = round.(Int, ref1069.color .* 255)
    println("   {n:Nat | n>0} (Refinement):")
    println("     seed=69:   RGB($r69,$g69,$b69)")
    println("     seed=1069: RGB($r1069,$g1069,$b1069)")
    
    # Inductive
    list_ctors = [(:Nil, Symbol[]), (:Cons, [:A, :List])]
    ind69 = GayInductive(:List, list_ctors; seed=seed_69)
    ind1069 = GayInductive(:List, list_ctors; seed=seed_1069)
    r69, g69, b69 = round.(Int, ind69.color .* 255)
    r1069, g1069, b1069 = round.(Int, ind1069.color .* 255)
    println("   μX. 1 + A×X (Inductive List):")
    println("     seed=69:   RGB($r69,$g69,$b69)")
    println("     seed=1069: RGB($r1069,$g1069,$b1069)")
    
    # PROP
    prop69 = GayPROP(:Circuit, 2, 1, [(:AND, 2, 1), (:OR, 2, 1)]; seed=seed_69)
    prop1069 = GayPROP(:Circuit, 2, 1, [(:AND, 2, 1), (:OR, 2, 1)]; seed=seed_1069)
    r69, g69, b69 = round.(Int, prop69.color .* 255)
    r1069, g1069, b1069 = round.(Int, prop1069.color .* 255)
    println("   Circuit(2→1) PROP:")
    println("     seed=69:   RGB($r69,$g69,$b69)")
    println("     seed=1069: RGB($r1069,$g1069,$b1069)")
    println()
    
    # 6. Composition verification
    println("6. COMPOSITION VERIFICATION (PROP ; PROP)")
    gate1 = GayPROP(:Gate1, 2, 1, [(:NAND, 2, 1)]; seed=seed_69)
    gate2 = GayPROP(:Gate2, 1, 2, [(:FORK, 1, 2)]; seed=seed_69)
    composed = compose_props(gate1, gate2)
    r, g, b = round.(Int, composed.color .* 255)
    println("   NAND(2→1) ; FORK(1→2) = $(composed.name)")
    println("   Inputs: $(composed.inputs), Outputs: $(composed.outputs)")
    println("   Color: RGB($r,$g,$b)")
    println("   Fingerprint: 0x$(string(composed.fingerprint, base=16, pad=16))")
    println()
    
    # 7. Tensor (parallel) composition
    println("7. TENSOR COMPOSITION (PROP ⊗ PROP)")
    tensored = tensor_props(gate1, gate2)
    r, g, b = round.(Int, tensored.color .* 255)
    println("   NAND(2→1) ⊗ FORK(1→2) = $(tensored.name)")
    println("   Inputs: $(tensored.inputs), Outputs: $(tensored.outputs)")
    println("   Color: RGB($r,$g,$b)")
    println()
    
    println("═" ^ 70)
    println("  CONCLUSION: Same structure, different seeds → different colors")
    println("  This enables chromatic verification across parallel worlds!")
    println("═" ^ 70)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

function demo_world_rotators()
    compare_seeds_69_1069()
end

# end of world_rotators.jl
