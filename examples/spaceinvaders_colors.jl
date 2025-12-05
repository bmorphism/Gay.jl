# GayInvaders - SpaceInvaders.jl with Deterministic Color Palettes from Gay.jl
# 
# Full interactive game with vibrant, reproducible colors!
#
# Usage:
#   using Gay
#   include(joinpath(pkgdir(Gay), "examples", "spaceinvaders_colors.jl"))
#   GayInvaders.main()
#
# Controls:
#   - Arrow keys or WASD to move
#   - Space bar to fire
#   - Q or Ctrl-C to quit
#
# Features:
#   - Each enemy row gets a deterministic color from the palette
#   - Ship and bullets are colored with pride flag colors
#   - Explosions show rainbow effects
#   - Same seed = same colors every time

module GayInvaders

using Gay
using Gay: rainbow_text, show_color_inline, RAINBOW_COLORS
using Colors: RGB

export main

# ═══════════════════════════════════════════════════════════════════════════
# ANSI Color Helpers
# ═══════════════════════════════════════════════════════════════════════════

function rgb_fg(c::RGB)
    r = round(Int, clamp(c.r, 0, 1) * 255)
    g = round(Int, clamp(c.g, 0, 1) * 255)
    b = round(Int, clamp(c.b, 0, 1) * 255)
    "\e[38;2;$(r);$(g);$(b)m"
end

function rgb_fg(t::Tuple{Int,Int,Int})
    "\e[38;2;$(t[1]);$(t[2]);$(t[3])m"
end

const RESET = "\e[0m"

# ═══════════════════════════════════════════════════════════════════════════
# Keyboard Input (from SpaceInvaders.jl)
# ═══════════════════════════════════════════════════════════════════════════

module Keyboard
    const KEY = Threads.Atomic{UInt64}(0)
    const LIVE = Threads.Atomic{Bool}(false)
    
    function get_key()
        k = KEY[]
        time_mask = (typemax(UInt64) >> 3)
        t0 = k & time_mask
        is_initial = (k & (time_mask + 1)) != 0
        t1 = time_ns() & time_mask
        delta = is_initial ? 5*10^8 : 5*10^7
        if t0 ≤ t1 < t0 + delta
            return Int(k >> 62)
        else
            return 0
        end
    end
    
    function set_key(k)
        old = KEY[]
        t = time_ns() & (typemax(UInt64) >> 3)
        is_initial = k != (old >> 62)
        KEY[] = (UInt64(k) << 62) + t + (UInt64(is_initial) << 61)
    end
    
    function listen(f)
        ret = ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid}, Int32), stdin.handle, true)
        ret == 0 || error("Unable to switch to raw mode")
        LIVE[] = true
        Threads.@spawn _listen()
        try
            f(LIVE, get_key)
        finally
            LIVE[] = false
            ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid}, Int32), stdin.handle, false)
        end
    end
    
    function _listen()
        state = 0
        while LIVE[]
            c = read(stdin, Char)
            if state == 0
                if c == '\e'
                    state = 1
                elseif c == ' ' || c == 'w' || c == 'W'
                    set_key(1)  # Space/W = Fire
                elseif c == 'a' || c == 'A'
                    set_key(2)  # A = Left
                elseif c == 'd' || c == 'D'
                    set_key(3)  # D = Right
                elseif c == 'q' || c == 'Q' || c == '\x03' || c == '\x04'
                    LIVE[] = false
                end
            elseif state == 1
                state = c == '[' ? 2 : 0
            elseif state == 2
                if c == 'A'
                    set_key(1)  # Arrow Up = Fire
                elseif c == 'D'
                    set_key(2)  # Arrow Left
                elseif c == 'C'
                    set_key(3)  # Arrow Right
                end
                state = 0
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Screen Buffer with Colors
# ═══════════════════════════════════════════════════════════════════════════

mutable struct ColorScreen
    chars::Matrix{Char}
    colors::Matrix{String}
    display_chars::Matrix{Char}
    display_colors::Matrix{String}
end

function ColorScreen()
    h, w = displaysize(stdout)
    ColorScreen(
        fill(' ', h, w),
        fill("", h, w),
        fill('\0', h, w),
        fill("", h, w)
    )
end

Base.size(s::ColorScreen) = size(s.chars)
Base.size(s::ColorScreen, i) = size(s.chars, i)

function clear!(s::ColorScreen)
    fill!(s.chars, ' ')
    fill!(s.colors, "")
end

function setchar!(s::ColorScreen, y, x, c::Char, color::String="")
    if 1 ≤ y ≤ size(s, 1) && 1 ≤ x ≤ size(s, 2)
        s.chars[y, x] = c
        s.colors[y, x] = color
    end
end

function render!(s::ColorScreen)
    buf = IOBuffer()
    print(buf, "\e[H")  # Move to top-left
    
    for y in axes(s.chars, 1)
        for x in axes(s.chars, 2)
            if s.chars[y, x] != s.display_chars[y, x] || s.colors[y, x] != s.display_colors[y, x]
                print(buf, "\e[$(y);$(x)H")  # Move cursor
                if !isempty(s.colors[y, x])
                    print(buf, s.colors[y, x], s.chars[y, x], RESET)
                else
                    print(buf, s.chars[y, x])
                end
            end
        end
    end
    
    copyto!(s.display_chars, s.chars)
    copyto!(s.display_colors, s.colors)
    print(stdout, String(take!(buf)))
    flush(stdout)
end

# ═══════════════════════════════════════════════════════════════════════════
# Game Colors
# ═══════════════════════════════════════════════════════════════════════════

struct GameColors
    enemy_rows::Vector{String}
    ship::String
    bullet::String
    explosion::Vector{String}
    border::Vector{String}
    seed::Int
end

function GameColors(; seed::Int=42, num_rows::Int=10)
    gay_seed!(seed)
    
    # Enemy colors - deterministic per row
    enemy_rows = [rgb_fg(color_at(i; seed=seed)) for i in 1:num_rows]
    
    # Ship = trans pride light blue
    trans = transgender()
    ship = rgb_fg(trans[3])
    
    # Bullet = trans pride pink  
    bullet = rgb_fg(trans[1])
    
    # Explosions = rainbow
    explosion = [rgb_fg(c) for c in rainbow()]
    
    # Border = rainbow
    border = [rgb_fg(RAINBOW_COLORS[i]) for i in 1:6]
    
    GameColors(enemy_rows, ship, bullet, explosion, border, seed)
end

# ═══════════════════════════════════════════════════════════════════════════
# Game Logic
# ═══════════════════════════════════════════════════════════════════════════

function draw_frame!(s::ColorScreen, colors::GameColors)
    h, w = size(s)
    
    # Top border
    setchar!(s, 1, 1, '╔', colors.border[1])
    setchar!(s, 1, w, '╗', colors.border[1])
    for x in 2:w-1
        setchar!(s, 1, x, '═', colors.border[mod1(x, 6)])
    end
    
    # Side borders
    for y in 2:h
        setchar!(s, y, 1, '║', colors.border[mod1(y, 6)])
        setchar!(s, y, w, '║', colors.border[mod1(y, 6)])
    end
end

function ticker(delay)
    delay, time() + delay
end

function tick((delay, target))
    sleep(max(0.005, target - time()))
    delay, max(target + delay, time())
end

function play_level!(s::ColorScreen, colors::GameColors, level_spec, live, get_key)
    clear!(s)
    h, w = size(s)
    draw_frame!(s, colors)
    
    # Initialize enemies
    play_h = h - 2
    play_w = w - 2
    x0 = play_w ÷ 2 + 1
    y1 = ceil(Int, play_h * level_spec.height) + 1
    
    enemies = falses(h, w)
    for y in 2:min(y1+1, h-1)
        y_frac = (y - 2) / max(1, y1 - 1)
        r = floor(Int, play_w * 0.5 * level_spec.width * (1 - 0.5 * y_frac^2))
        for x in max(2, x0 - r):min(w - 1, x0 + r)
            enemies[y, x] = true
        end
    end
    
    # Ship position
    ship_x = w ÷ 2
    ship_y = h - 1
    
    # Bullets: (x, y)
    bullets = Tuple{Int,Int}[]
    
    # Explosions: (x, y, age)
    explosions = Tuple{Int,Int,Int}[]
    
    action_cooldown = 0
    bullet_cost = level_spec.bullet_cost
    move_cost = 1
    
    enemy_cooldown = 0
    enemy_direction = rand((-1, 1))
    enemy_cost = level_spec.enemy_cost
    
    t = ticker(level_spec.tick_rate)
    
    while live[]
        k = get_key()
        
        # Clear play area
        for y in 2:h-1, x in 2:w-1
            setchar!(s, y, x, ' ')
        end
        
        # Move enemies
        edy, edx = 0, 0
        if enemy_cooldown <= 0
            enemy_cooldown = enemy_cost - 1
            last_col = enemy_direction == 1 ? w - 1 : 2
            reverse = any(enemies[:, last_col])
            reverse && (enemy_direction *= -1)
            edy, edx = reverse ? 1 : 0, enemy_direction
        else
            enemy_cooldown -= 1
        end
        
        new_enemies = falses(h, w)
        has_enemies = false
        for y in 2:h-1, x in 2:w-1
            if enemies[y, x]
                has_enemies = true
                ny, nx = y + edy, x + edx
                if 2 ≤ ny ≤ h-1 && 2 ≤ nx ≤ w-1
                    new_enemies[ny, nx] = true
                end
            end
        end
        enemies = new_enemies
        
        # Move bullets and check collisions
        new_bullets = Tuple{Int,Int}[]
        for (bx, by) in bullets
            nby = by - 1
            if nby >= 2
                if enemies[nby, bx]
                    enemies[nby, bx] = false
                    push!(explosions, (bx, nby, 0))
                else
                    push!(new_bullets, (bx, nby))
                end
            end
        end
        bullets = new_bullets
        
        # Age explosions
        explosions = [(x, y, age + 1) for (x, y, age) in explosions if age < 6]
        
        # Player action
        if action_cooldown <= 0
            if k == 1 && has_enemies  # Fire
                push!(bullets, (ship_x, ship_y - 1))
                action_cooldown = bullet_cost - 1
            elseif k == 2  # Left
                ship_x > 2 && (ship_x -= 1)
                action_cooldown = move_cost - 1
            elseif k == 3  # Right
                ship_x < w - 1 && (ship_x += 1)
                action_cooldown = move_cost - 1
            end
        else
            action_cooldown -= 1
        end
        
        # Check if enemies hit ship
        if enemies[ship_y, ship_x]
            return :lost
        end
        
        # Draw enemies
        for y in 2:h-1, x in 2:w-1
            if enemies[y, x]
                row_color = colors.enemy_rows[mod1(y, length(colors.enemy_rows))]
                setchar!(s, y, x, '🙯', row_color)
            end
        end
        
        # Draw bullets
        for (bx, by) in bullets
            setchar!(s, by, bx, '🢙', colors.bullet)
        end
        
        # Draw explosions
        for (ex, ey, age) in explosions
            exp_color = colors.explosion[mod1(age + 1, 6)]
            setchar!(s, ey, ex, '✦', exp_color)
        end
        
        # Draw ship
        setchar!(s, ship_y, ship_x, '🙭', colors.ship)
        
        # Redraw border
        draw_frame!(s, colors)
        
        render!(s)
        
        # Check win condition
        !has_enemies && isempty(bullets) && return :won
        
        live[] || break
        t = tick(t)
    end
    
    return :quit
end

function draw_text!(s::ColorScreen, text::String, colors::GameColors)
    lines = split(text, '\n')
    h, w = size(s)
    start_y = h ÷ 2 - length(lines)
    
    for (i, line) in enumerate(lines)
        y = start_y + i * 2
        x = w ÷ 2 - length(line) ÷ 2
        for (j, c) in enumerate(line)
            color = colors.border[mod1(j, 6)]
            setchar!(s, y, x + j - 1, c, color)
        end
    end
end

function intro!(s::ColorScreen, colors::GameColors, live, get_key)
    clear!(s)
    h, w = size(s)
    
    # Animated intro
    for frame in 1:min(w, 60)
        live[] || return
        clear!(s)
        
        # Draw partial border
        for x in 1:min(frame, w)
            setchar!(s, 1, x, x == 1 ? '╔' : (x == w ? '╗' : '═'), colors.border[mod1(x, 6)])
        end
        for y in 2:min(frame ÷ 2, h)
            setchar!(s, y, 1, '║', colors.border[mod1(y, 6)])
            setchar!(s, y, w, '║', colors.border[mod1(y, 6)])
        end
        
        render!(s)
        sleep(0.02)
    end
    
    draw_frame!(s, colors)
    
    # Title
    title_y = h ÷ 3
    title = "GAY INVADERS"
    title_x = w ÷ 2 - length(title) ÷ 2
    for (i, c) in enumerate(title)
        setchar!(s, title_y, title_x + i - 1, c, colors.border[mod1(i, 6)])
    end
    
    subtitle = "Press SPACE to start"
    sub_x = w ÷ 2 - length(subtitle) ÷ 2
    for (i, c) in enumerate(subtitle)
        setchar!(s, title_y + 3, sub_x + i - 1, c, colors.ship)
    end
    
    # Show color palette
    palette_y = title_y + 6
    palette_label = "Colors (seed=$(colors.seed)):"
    setchar!.(Ref(s), palette_y, (w÷2 - length(palette_label)÷2):(w÷2 - length(palette_label)÷2 + length(palette_label) - 1), collect(palette_label), Ref(""))
    
    for i in 1:6
        setchar!(s, palette_y + 2, w ÷ 2 - 6 + i * 2, '●', colors.enemy_rows[i])
    end
    
    render!(s)
    
    # Wait for space bar to start
    while live[]
        k = get_key()
        k == 1 && break  # Space pressed
        sleep(0.05)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

"""
    main(; seed=42, splash=true)

Launch Gay Invaders - Space Invaders with deterministic color palettes!

Controls:
- Arrow keys or WASD to move
- Space bar to fire  
- Q or Ctrl-C to quit

Same seed = same colors every time.
"""
function main(; seed::Int=42, splash::Bool=true)
    colors = GameColors(; seed=seed)
    
    Keyboard.listen() do live, get_key
        s = ColorScreen()
        
        if splash
            intro!(s, colors, live, get_key)
        end
        
        live[] || return
        
        levels = [
            (width=0.3, height=0.3, bullet_cost=4, enemy_cost=4, tick_rate=0.06),
            (width=0.35, height=0.35, bullet_cost=5, enemy_cost=4, tick_rate=0.05),
            (width=0.4, height=0.4, bullet_cost=5, enemy_cost=3, tick_rate=0.04),
            (width=0.45, height=0.45, bullet_cost=5, enemy_cost=3, tick_rate=0.03),
            (width=0.5, height=0.5, bullet_cost=5, enemy_cost=3, tick_rate=0.025),
        ]
        
        for (i, spec) in enumerate(levels)
            result = play_level!(s, colors, spec, live, get_key)
            
            result == :quit && return
            
            if result == :won
                if i == length(levels)
                    clear!(s)
                    draw_frame!(s, colors)
                    draw_text!(s, "YOU\nWIN!", colors)
                    render!(s)
                    sleep(3)
                    return
                else
                    clear!(s)
                    draw_frame!(s, colors)
                    draw_text!(s, "LEVEL $(i+1)", colors)
                    render!(s)
                    sleep(2)
                end
            else
                clear!(s)
                draw_frame!(s, colors)
                draw_text!(s, "GAME\nOVER", colors)
                render!(s)
                sleep(3)
                return
            end
        end
    end
    
    # Clear screen on exit
    print("\e[H\e[2J")
    println(rainbow_text("Thanks for playing Gay Invaders! 🏳️‍🌈"))
    println("Seed was: $seed - use same seed for same colors!")
end

end # module GayInvaders

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    using .GayInvaders
    GayInvaders.main()
end
