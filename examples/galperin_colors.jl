# Galperin Billiards → Colors
# Collision counts at each step → deterministic color stream
#
# Uses the :galperin continuation branch from bbp_pi.jl
#
# The physics: Two balls with mass ratio 100^n collide elastically.
# Total collisions = first (n+1) digits of π.
# Each collision event → a color from the splittable stream.

using Gay
using Colors
using SplittableRandoms: SplittableRandom, split

include("bbp_pi.jl")  # For continuation_point, branch_seed

# ═══════════════════════════════════════════════════════════════════════════
# Galperin Physics (from unison-terminus/galperin.jl)
# ═══════════════════════════════════════════════════════════════════════════

struct BilliardState
    x1::Float64      # position of ball 1
    x2::Float64      # position of ball 2
    v1::Float64      # velocity of ball 1
    v2::Float64      # velocity of ball 2
    collisions::Int  # total collision count
    event::Symbol    # :wall, :balls, or :none
end

function initial_state(mass_ratio::Real)
    BilliardState(1.0, 2.0, 0.0, -1.0, 0, :none)
end

function ball_collision(m1, v1, m2, v2)
    v1_new = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
    v2_new = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
    return (v1_new, v2_new)
end

function step(state::BilliardState, m1::Float64, m2::Float64)
    x1, x2, v1, v2, n, _ = state.x1, state.x2, state.v1, state.v2, state.collisions, state.event
    
    # Time to next collision
    t_balls = (v2 < v1) ? (x2 - x1) / (v1 - v2) : Inf
    t_wall = (v1 < 0) ? -x1 / v1 : Inf
    
    if t_balls == Inf && t_wall == Inf
        return BilliardState(x1, x2, v1, v2, n, :none)
    end
    
    if t_wall < t_balls
        x1 += v1 * t_wall
        x2 += v2 * t_wall
        v1 = -v1
        return BilliardState(x1, x2, v1, v2, n + 1, :wall)
    else
        x1 += v1 * t_balls
        x2 += v2 * t_balls
        v1, v2 = ball_collision(m1, v1, m2, v2)
        return BilliardState(x1, x2, v1, v2, n + 1, :balls)
    end
end

function simulate(mass_ratio::Real; max_steps=10^7)
    m1 = 1.0
    m2 = Float64(mass_ratio)
    state = initial_state(mass_ratio)
    history = [state]
    
    for _ in 1:max_steps
        state = step(state, m1, m2)
        push!(history, state)
        state.event == :none && break
    end
    
    return history
end

# ═══════════════════════════════════════════════════════════════════════════
# Collision → Color Mapping (uses :galperin branch)
# ═══════════════════════════════════════════════════════════════════════════

const GALPERIN_SEED = 314159

"""
    collision_color(collision_num::Int; seed=GALPERIN_SEED)

Get a deterministic color for collision number `n`.
Uses the :galperin continuation branch.
"""
function collision_color(collision_num::Int; seed::Integer=GALPERIN_SEED)
    gseed = branch_seed(seed, :galperin)
    return color_at(collision_num, Rec2020(); seed=gseed)
end

"""
    collision_colors(history::Vector{BilliardState}; seed=GALPERIN_SEED)

Get colors for all collisions in a simulation history.
Wall collisions get one hue family, ball collisions another.
"""
function collision_colors(history::Vector{BilliardState}; seed::Integer=GALPERIN_SEED)
    gseed = branch_seed(seed, :galperin)
    colors = RGB[]
    
    for (i, state) in enumerate(history)
        state.event == :none && continue
        
        # Offset by event type for visual distinction
        offset = state.event == :wall ? 0 : 1000
        c = color_at(state.collisions + offset, Rec2020(); seed=gseed)
        push!(colors, c)
    end
    
    return colors
end

"""
    galperin_palette(n_digits::Int; seed=GALPERIN_SEED)

Generate a palette from Galperin simulation with mass ratio 100^(n_digits-1).
Returns colors for each collision event.
"""
function galperin_palette(n_digits::Int; seed::Integer=GALPERIN_SEED)
    mass_ratio = 100^(n_digits - 1)
    history = simulate(mass_ratio)
    return collision_colors(history; seed=seed)
end

# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

function render_galperin_colors(n_digits::Int; seed::Integer=GALPERIN_SEED)
    mass_ratio = 100^(n_digits - 1)
    history = simulate(mass_ratio)
    colors = collision_colors(history; seed=seed)
    
    pi_approx = history[end].collisions / 10^(n_digits - 1)
    
    println("\n  ╔════════════════════════════════════════════════════════════╗")
    println("  ║  Galperin Billiards → Colors (:galperin branch)            ║")
    println("  ╚════════════════════════════════════════════════════════════╝")
    println()
    println("  Mass ratio: 100^$(n_digits-1) = $mass_ratio")
    println("  Collisions: $(history[end].collisions)")
    println("  π approximation: $pi_approx")
    println()
    
    # Show collision sequence as colored blocks
    print("  ")
    for (i, c) in enumerate(colors)
        ri = round(Int, c.r * 255)
        gi = round(Int, c.g * 255)
        bi = round(Int, c.b * 255)
        print("\e[48;2;$(ri);$(gi);$(bi)m \e[0m")
        i % 50 == 0 && print("\n  ")
    end
    println()
    
    # Legend
    println()
    println("  Each collision → one color from :galperin stream")
    println("  Wall collisions and ball collisions use offset indices")
end

function demo_galperin_spi()
    println("\n  ═══════════════════════════════════════════════════════")
    println("  SPI Verification: :galperin branch independence")
    println("  ═══════════════════════════════════════════════════════\n")
    
    # Generate colors from :galperin branch
    gseed = branch_seed(314159, :galperin)
    galperin_colors = [color_at(i, Rec2020(); seed=gseed) for i in 1:5]
    
    # Generate colors from :bbp_pi branch  
    pseed = branch_seed(314159, :bbp_pi)
    pi_colors = [color_at(i, Rec2020(); seed=pseed) for i in 1:5]
    
    println("  :galperin branch (first 5):")
    print("    ")
    for c in galperin_colors
        ri = round(Int, c.r * 255)
        gi = round(Int, c.g * 255)
        bi = round(Int, c.b * 255)
        print("\e[48;2;$(ri);$(gi);$(bi)m    \e[0m ")
    end
    println()
    
    println("  :bbp_pi branch (first 5):")
    print("    ")
    for c in pi_colors
        ri = round(Int, c.r * 255)
        gi = round(Int, c.g * 255)
        bi = round(Int, c.b * 255)
        print("\e[48;2;$(ri);$(gi);$(bi)m    \e[0m ")
    end
    println()
    
    println("\n  Branches are independent: $(galperin_colors != pi_colors ? "◆" : "◇")")
end

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

function main()
    println("\n" * "═"^70)
    println("  Galperin Billiards Colors - :galperin continuation branch")
    println("═"^70)
    
    # Show collisions as colors for 3 digits of π (314 collisions)
    render_galperin_colors(3)
    
    # Verify branch independence
    demo_galperin_spi()
    
    println("\n  Properties:")
    println("  ◆ Each collision maps to deterministic color")
    println("  ◆ :galperin branch independent of :bbp_pi")
    println("  ◆ Same seed → same collision colors")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
