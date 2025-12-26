# Gay.jl Semiosis: Sign-making and meaning through color
#
# Semiosis (σημείωσις): The process by which signs acquire meaning.
# In Gay.jl, colors are signs that encode:
#   - Deterministic identity (splittable RNG)
#   - Structural position (depth, parity)
#   - Computational flow (forward/reverse autodiff)
#   - Physical state (Ising spins, SSE operators)
#
# From SICP 4A: "Enzymes attach to expressions, change them, then go away."
# The colored parenthesis is the sign; the enzyme binding is the semiosis.
#
# Usage:
#   semiosis()           - Run all demonstrations
#   semiosis(:colors)    - Color generation
#   semiosis(:enzyme)    - Autodiff visualization
#   semiosis(:sse)       - Monte Carlo integration
#   semiosis(:parallel)  - Fork-safe parallelism

using Random

export semiosis
export SSEOperator, SSEEnzyme, create_sse_enzyme, attach_sse_operator!
export sse_magnetization, render_sse_lattice

# ═══════════════════════════════════════════════════════════════════════════
# SSE Types (moved from examples/enzyme_sse.jl)
# ═══════════════════════════════════════════════════════════════════════════

"""
    SSEOperator

An SSE operator attached to a bond, analogous to EnzymeBinding.
"""
struct SSEOperator
    bond::Tuple{Int, Int, Int, Int}  # (i, j, ni, nj) site indices
    operator_type::Symbol            # :diagonal, :offdiag_plus, :offdiag_minus
    color::RGB                       # Gay.jl color for visualization
    spin::Int                        # σ = ±1 for operator sign
    weight::Float64                  # Boltzmann weight
end

"""
    SSEEnzyme

SSE operator string as an "enzyme" attached to the Hamiltonian S-expression.
Parallels GayEnzyme structure.
"""
struct SSEEnzyme
    hamiltonian::GaySexpr           # Magnetized Hamiltonian S-expression
    operators::Vector{SSEOperator}  # Attached SSE operators
    beta::Float64                   # Inverse temperature
    n_operators::Int                # Current operator string length
    seed::UInt64
end

export SSEOperator, SSEEnzyme

"""
    create_sse_enzyme(Lx::Int, Ly::Int, beta::Float64; seed::UInt64=0x535345)

Create SSE enzyme attachment for 2D Heisenberg model.
"""
function create_sse_enzyme(Lx::Int, Ly::Int, beta::Float64; seed::UInt64=0x535345)
    bonds = Any[:+]
    for i in 1:Lx, j in 1:Ly
        jx = mod1(i + 1, Lx)
        jy = mod1(j + 1, Ly)
        push!(bonds, [:*, -1, [:dot, [:S, i, j], [:S, jx, j]]])
        push!(bonds, [:*, -1, [:dot, [:S, i, j], [:S, i, jy]]])
    end
    
    gs = gay_magnetized_sexpr(bonds, seed)
    SSEEnzyme(gs, SSEOperator[], beta, 0, seed)
end

"""
    attach_sse_operator!(sse::SSEEnzyme, bond, op_type, weight)

Attach an SSE operator to a bond.
"""
function attach_sse_operator!(sse::SSEEnzyme, bond::Tuple{Int,Int,Int,Int}, 
                               op_type::Symbol, weight::Float64)
    il = GayInterleaver(sse.seed, 3)
    stream_idx = op_type == :diagonal ? 0 : (op_type == :offdiag_plus ? 1 : 2)
    color = gay_sublattice(il, stream_idx)
    spin = op_type == :offdiag_minus ? -1 : 1
    
    op = SSEOperator(bond, op_type, color, spin, weight)
    push!(sse.operators, op)
    return op
end

"""
    sse_magnetization(sse::SSEEnzyme)

Compute operator magnetization ⟨M⟩ = Σσ/N.
"""
function sse_magnetization(sse::SSEEnzyme)
    isempty(sse.operators) && return 0.0
    return sum(op.spin for op in sse.operators) / length(sse.operators)
end

export create_sse_enzyme, attach_sse_operator!, sse_magnetization

"""
    render_sse_lattice(sse::SSEEnzyme, Lx::Int, Ly::Int)

Render the SSE operator attachments on the lattice.
"""
function render_sse_lattice(sse::SSEEnzyme, Lx::Int, Ly::Int)
    R = "\e[0m"
    
    bond_ops = Dict{Tuple{Int,Int,Int,Int}, Vector{SSEOperator}}()
    for op in sse.operators
        if !haskey(bond_ops, op.bond)
            bond_ops[op.bond] = SSEOperator[]
        end
        push!(bond_ops[op.bond], op)
    end
    
    println("\n  SSE Operator Attachments (n = $(length(sse.operators))):")
    println()
    
    for j in Ly:-1:1
        print("  ")
        for i in 1:Lx
            print("●")
            jx = mod1(i + 1, Lx)
            bond = (i, j, jx, j)
            if haskey(bond_ops, bond)
                ops = bond_ops[bond]
                c = ops[1].color
                rgb = convert(RGB, c)
                r = round(Int, clamp(rgb.r, 0, 1) * 255)
                g = round(Int, clamp(rgb.g, 0, 1) * 255)
                b = round(Int, clamp(rgb.b, 0, 1) * 255)
                sym = length(ops) > 1 ? "═" : "─"
                print("\e[38;2;$(r);$(g);$(b)m$(sym)$(R)")
            else
                print("─")
            end
        end
        println()
        
        print("  ")
        for i in 1:Lx
            jy = mod1(j + 1, Ly)
            bond = (i, j, i, jy)
            if haskey(bond_ops, bond)
                ops = bond_ops[bond]
                c = ops[1].color
                rgb = convert(RGB, c)
                r = round(Int, clamp(rgb.r, 0, 1) * 255)
                g = round(Int, clamp(rgb.g, 0, 1) * 255)
                b = round(Int, clamp(rgb.b, 0, 1) * 255)
                sym = length(ops) > 1 ? "║" : "│"
                print("\e[38;2;$(r);$(g);$(b)m$(sym)$(R) ")
            else
                print("│ ")
            end
        end
        println()
    end
    
    println()
    println("  ⟨M⟩ = $(round(sse_magnetization(sse), digits=4))")
end

export render_sse_lattice

# ═══════════════════════════════════════════════════════════════════════════
# Demo: SSE meets Enzyme
# ═══════════════════════════════════════════════════════════════════════════

function world_sse_enzyme()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║  Gay.jl: SSE QMC ↔ Enzyme.jl Operator Duality                ║")
    println("╚══════════════════════════════════════════════════════════════╝\n")
    
    Lx, Ly = 4, 4
    beta = 2.0
    seed = UInt64(0xDEADBEEF)
    
    sse = create_sse_enzyme(Lx, Ly, beta; seed=seed)
    
    println("Inserting SSE operators (simulating diagonal update)...")
    
    Random.seed!(42)  # Deterministic for demo
    for _ in 1:20
        i = rand(1:Lx)
        j = rand(1:Ly)
        bond = rand() < 0.5 ? (i, j, mod1(i+1, Lx), j) : (i, j, i, mod1(j+1, Ly))
        op_type = rand([:diagonal, :diagonal, :offdiag_plus, :offdiag_minus])
        weight = rand() * beta
        attach_sse_operator!(sse, bond, op_type, weight)
    end
    
    render_sse_lattice(sse, Lx, Ly)
    
    println()
    println("─────────────────────────────────────────────────────────────────")
    println()
    
    println("Enzyme.jl autodiff on Heisenberg energy:")
    println()
    
    energy = GayFunction(:heisenberg_1d, [:s1, :s2, :s3, :s4],
                        [:+, [:*, -1, [:*, :s1, :s2]],
                             [:*, -1, [:*, :s2, :s3]],
                             [:*, -1, [:*, :s3, :s4]],
                             [:*, -1, [:*, :s4, :s1]]])
    
    spins = [1.0, -1.0, 1.0, -1.0]
    println("  Spin configuration: ", spins)
    println("  Energy E = ", energy(spins...))
    
    ge = show_gradient(energy, spins)
    
    println()
    println("─────────────────────────────────────────────────────────────────")
    println()
    println("The Duality:")
    println()
    println("  SSE Operator String          Enzyme Binding Sites")
    println("  ──────────────────           ────────────────────")
    println("  n = $(length(sse.operators)) operators            $(length(ge.bindings)) binding sites")
    println("  ⟨M⟩_SSE = $(round(sse_magnetization(sse), digits=4))        ⟨M⟩_∇ = $(round(gradient_magnetization(ge), digits=4))")
    println()
    println("  Both represent 'attachments' to the Hamiltonian expression.")
end

# ═══════════════════════════════════════════════════════════════════════════
# Demo: Basic colors
# ═══════════════════════════════════════════════════════════════════════════

function world_colors()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║  Gay.jl: Deterministic Splittable Color Generation           ║")
    println("╚══════════════════════════════════════════════════════════════╝\n")
    
    # Reset seed for reproducibility
    gay_seed!(0x42)
    
    println("Generating 10 deterministic colors (seed=0x42):")
    println()
    
    colors = [next_color() for _ in 1:10]
    show_palette(colors)
    
    println()
    println("Same seed = same colors:")
    gay_seed!(0x42)
    colors2 = [next_color() for _ in 1:10]
    show_palette(colors2)
    
    println()
    println("Random access by index:")
    println("  color_at(1)  = ", color_at(1))
    println("  color_at(42) = ", color_at(42))
    println("  color_at(1)  = ", color_at(1), " (same as before)")
    
    println()
    println("Pride flags:")
    print("  Rainbow: ")
    show_colors(rainbow())
    print("  Trans:   ")
    show_colors(transgender())
    print("  Bi:      ")
    show_colors(bisexual())
end

# ═══════════════════════════════════════════════════════════════════════════
# Demo: Interleaver / Checkerboard
# ═══════════════════════════════════════════════════════════════════════════

function world_interleaver()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║  Gay.jl: XOR Checkerboard Coloring for Parallel Updates      ║")
    println("╚══════════════════════════════════════════════════════════════╝\n")
    
    seed = UInt64(0xDEADBEEF)
    il = GayInterleaver(seed, 2)
    Lx, Ly = 8, 6
    
    println("2D lattice with checkerboard decomposition:")
    println("  (parity = (i + j) mod 2)")
    println()
    
    for j in Ly:-1:1
        print("  ")
        for i in 1:Lx
            parity = (i + j) % 2
            color = gay_sublattice(il, parity)
            rgb = convert(RGB, color)
            r = round(Int, clamp(rgb.r, 0, 1) * 255)
            g = round(Int, clamp(rgb.g, 0, 1) * 255)
            b = round(Int, clamp(rgb.b, 0, 1) * 255)
            symbol = parity == 0 ? "●" : "○"
            print("\e[38;2;$(r);$(g);$(b)m$(symbol)\e[0m ")
        end
        println()
    end
    
    println()
    println("  Legend: ● = even sublattice, ○ = odd sublattice")
    println("  Each sublattice uses independent SPI stream")
    println("  Parallel updates within sublattice preserve detailed balance")
end

# ═══════════════════════════════════════════════════════════════════════════
# Semiosis: The process by which signs acquire meaning
# ═══════════════════════════════════════════════════════════════════════════

"""
    semiosis(; which=:all)

Semiosis (σημείωσις): The process by which signs acquire meaning.

In Gay.jl, colors are signs that encode:
- Deterministic identity (splittable RNG)
- Structural position (depth, parity)
- Computational flow (forward/reverse autodiff)
- Physical state (Ising spins, SSE operators)

# Options
- `:all` - Run all demonstrations
- `:colors` - Color generation (sign creation)
- `:enzyme` - Autodiff visualization (enzyme attachment)
- `:dsl` - S-expression → Julia (symbolic transformation)
- `:sse` - Monte Carlo (physical interpretation)
- `:parallel` - Fork-safe parallelism (distributed meaning)
- `:binary` - Radare2 binary analysis (reverse engineering)
- `:derange` - Derangeable permutations (no fixed points)
- `:abduce` - Abductive inference (effect → cause)
- `:binary` - Binary analysis with AST hashing
"""
function semiosis(which::Symbol=:all)
    println()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Gay.jl Semiosis: Sign-Making Through Color                  ║")
    println("║  σημείωσις - The process by which signs acquire meaning      ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    println()
    
    if which == :all || which == :colors
        world_colors()
    end
    
    if which == :all || which == :parallel
        world_interleaver()
    end
    
    if which == :all || which == :enzyme
        world_enzyme_colors()
    end
    
    if which == :all || which == :dsl
        world_enzyme_dsl()
    end
    
    if which == :all || which == :sse
        world_sse_enzyme()
    end
    
    if which == :binary
        world_radare2_colors()
        world_binary_analysis()
    end
    
    if which == :all || which == :derange
        world_derangeable()
    end
    
    if which == :all || which == :abduce
        world_abduce()
    end
    
    println()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  \"The colored parenthesis is the sign;                       ║")
    println("║   the enzyme binding is the semiosis.\" - SICP 4A             ║")
    println("╚══════════════════════════════════════════════════════════════╝")
end

# ═══════════════════════════════════════════════════════════════════════════
# Demo: Derangeable permutations
# ═══════════════════════════════════════════════════════════════════════════

function world_derangeable()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║  Gay.jl: Derangeable Permutations (No Fixed Points)          ║")
    println("╚══════════════════════════════════════════════════════════════╝\n")
    
    seed = UInt64(0xDEADBEEF)
    
    println("A derangement σ satisfies σ(i) ≠ i for all i.")
    println("Every element moves - useful for color shuffling without repetition.\n")
    
    # Show basic derangement
    d = Derangeable(6; seed=seed)
    println("Original:    1  2  3  4  5  6")
    
    for trial in 1:3
        perm = derange_at(d, trial)
        println("Derange[$trial]: $(join(perm, "  "))")
    end
    
    println()
    
    # Show cycle decomposition
    println("Cycle decomposition (no 1-cycles = no fixed points):")
    for trial in 1:3
        cycles = derange_cycle(d, trial)
        sign = derangement_sign(d, trial)
        sign_str = sign > 0 ? "even" : "odd"
        cycle_strs = ["(" * join(c, " ") * ")" for c in cycles]
        println("  Derange[$trial]: $(join(cycle_strs, " ")) [$sign_str]")
    end
    
    println()
    
    # Color derangement
    println("Color derangement (every color moves):")
    colors = [color_at(i, SRGB(); seed=seed) for i in 1:6]
    shuffled = derange_colors(colors, seed; index=1)
    
    print("  Original: ")
    show_colors(colors)
    print("  Deranged: ")
    show_colors(shuffled)
    
    println()
    
    # Sattolo single-cycle
    println("Sattolo algorithm (single cycle, guaranteed derangement):")
    stream = GayDerangementStream(6; seed=seed, sattolo=true)
    for n in 1:3
        perm = nth_derangement(stream, n)
        println("  Sattolo[$n]: $(join(perm, "  "))")
    end
    
    println()
    println("SPI properties: same seed → same derangement, O(1) random access")
end

# ═══════════════════════════════════════════════════════════════════════════
# Demo: Abductive inference
# ═══════════════════════════════════════════════════════════════════════════

function world_abduce()
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║  Gay.jl: Abductive Inference (Effect → Cause)                ║")
    println("╚══════════════════════════════════════════════════════════════╝\n")
    
    println("Abduction (ἀπαγωγή): Reasoning from effect to cause.")
    println("Given a color, recover the seed and index that produced it.\n")
    
    seed = UInt64(0xDEADBEEF)
    
    # Demo 1: Recover index from color
    println("1. Index Recovery (given seed 0xDEADBEEF):")
    for test_idx in [1, 42, 100]
        c = color_at(test_idx, SRGB(); seed=seed)
        found_idx, dist, exact = abduce_index(c, seed; max_index=200)
        
        rgb = convert(RGB, c)
        r = round(Int, clamp(rgb.r, 0, 1) * 255)
        g = round(Int, clamp(rgb.g, 0, 1) * 255)
        b = round(Int, clamp(rgb.b, 0, 1) * 255)
        color_block = "\e[38;2;$(r);$(g);$(b)m████\e[0m"
        
        status = exact ? "◆" : "≈"
        println("   $color_block → index $found_idx $status (actual: $test_idx)")
    end
    
    println()
    
    # Demo 2: Recover seed from color
    println("2. Seed Recovery (given index 42):")
    for test_seed in [0xDEADBEEF, 0xCAFEBABE]
        c = color_at(42, SRGB(); seed=UInt64(test_seed))
        found_seed, dist, exact = abduce_seed(c, 42)
        
        rgb = convert(RGB, c)
        r = round(Int, clamp(rgb.r, 0, 1) * 255)
        g = round(Int, clamp(rgb.g, 0, 1) * 255)
        b = round(Int, clamp(rgb.b, 0, 1) * 255)
        color_block = "\e[38;2;$(r);$(g);$(b)m████\e[0m"
        
        status = exact ? "◆" : "≈"
        println("   $color_block → seed 0x$(string(found_seed, base=16)) $status")
    end
    
    println()
    
    # Demo 3: Recover derangement inverse
    println("3. Inverse Permutation (undo derangement):")
    d = Derangeable(6; seed=seed)
    perm = derange_indices(d, 1)
    inv = abduce_inverse(perm)
    
    println("   Original:  1  2  3  4  5  6")
    println("   Deranged:  $(join(perm, "  "))")
    println("   Inverse:   $(join(inv, "  "))")
    println("   Verify:    $(join([inv[perm[i]] for i in 1:6], "  ")) (should be 1-6)")
    
    println()
    
    # Demo 4: GayAbducer
    println("4. GayAbducer (infer seed from observations):")
    abducer = GayAbducer()
    for i in 1:5
        c = color_at(i, SRGB(); seed=seed)
        register_observation!(abducer, c; index=i)
    end
    
    inferred = infer_seed(abducer)
    println("   Observed 5 colors with known indices")
    println("   Inferred seed: 0x$(string(inferred, base=16))")
    println("   Confidence: $(round(abducer.confidence * 100, digits=1))%")
    println("   Actual seed: 0x$(string(seed, base=16)) $(inferred == seed ? "◆" : "◇")")
    
    println()
    
    # Demo 5: Structure inference
    println("5. Structure Inference:")
    structure = infer_structure(abducer)
    println("   Pattern: $(structure.pattern)")
    println("   Mean hue: $(round(structure.mean_hue, digits=1))°")
    println("   Magnetization: $(round(structure.magnetization, digits=3))")
    
    println()
    println("Peirce: \"Abduction is the process of forming an explanatory hypothesis.\"")
end
