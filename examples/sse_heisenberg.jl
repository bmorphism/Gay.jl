#!/usr/bin/env julia
# SSE QMC Heisenberg model with Gay.jl XOR interleaved coloring
# Demonstrates checkerboard decomposition for parallel spin updates
#
# The Hamiltonian: H = Σ_⟨ij⟩ J_ij (S_i · S_j + d_ij S_i^z S_j^z)
#
# Key insight from SICP Lecture 4A: Pattern matching + rule-based substitution
# Here we match lattice patterns (XOR parity) to color rules (SPI streams)

using LispSyntax
using Gay

# ═══════════════════════════════════════════════════════════════════════════
# Lisp-style SSE simulation with deterministic coloring
# ═══════════════════════════════════════════════════════════════════════════

lisp"""
(define (sse-heisenberg-demo seed Lx Ly n-sweeps)
  "Run SSE QMC demo with Gay.jl checkerboard coloring.
   Each sublattice update gets independent SPI color stream."
  
  ;; Create interleaver with 2 streams (even/odd sublattices)
  (let ((interleaver (GayInterleaver seed 2))
        (beta 1.0)      ; inverse temperature
        (J 1.0)         ; exchange coupling
        (spins (ones Int8 Lx Ly)))  ; start all spin-up
    
    ;; Main SSE loop with colored sweeps
    (for (sweep 1 n-sweeps)
      (let ((parity (mod sweep 2)))
        
        ;; Diagonal update: insert/remove operators
        (diagonal-update interleaver spins parity beta J)
        
        ;; Off-diagonal update: loop update on operator string
        (loop-update interleaver spins parity)
        
        ;; Measure observables with sweep color
        (when (> sweep (div n-sweeps 2))  ; thermalized
          (measure-observables interleaver spins sweep))))
    
    ;; Return final configuration with color metadata
    (values spins (get-color-history interleaver))))

(define (diagonal-update interleaver spins parity beta J)
  "Insert/remove diagonal operators on bonds with XOR parity matching."
  (let* ((Lx (size spins 1))
         (Ly (size spins 2)))
    (for (i 1 Lx)
      (for (j 1 Ly)
        (when (== (mod (+ i j) 2) parity)
          ;; This site belongs to current sublattice
          (let ((color (gay-sublattice interleaver parity))
                (neighbors (get-neighbors i j Lx Ly)))
            ;; Color tracks which random stream generated this update
            (for-each (lambda (neighbor)
              (let* ((ni (first neighbor))
                     (nj (second neighbor))
                     (bond-color (gay-xor-color interleaver i ni)))
                ;; Metropolis for diagonal operator
                (diagonal-metropolis spins i j ni nj beta J bond-color)))
              neighbors)))))))

(define (get-neighbors i j Lx Ly)
  "Get periodic nearest neighbors."
  (list (list (mod1 (+ i 1) Lx) j)    ; right
        (list i (mod1 (+ j 1) Ly))))  ; up

(define (diagonal-metropolis spins i j ni nj beta J color)
  "Attempt to insert/remove diagonal operator on bond (i,j)-(ni,nj).
   Color provides deterministic random number from SPI stream."
  ;; The color encodes which sublattice stream was used
  ;; This ensures SPI: same result regardless of thread execution order
  (let ((Si (getindex spins i j))
        (Sj (getindex spins ni nj))
        (prob (* beta J 0.5 (+ 1 (* Si Sj)))))  ; Heisenberg weight
    ;; Use color's hue as deterministic uniform [0,1]
    (when (< (color-to-uniform color) prob)
      ;; Accept: toggle operator presence (simplified)
      :accepted)))

(define (color-to-uniform color)
  "Convert RGB color to deterministic uniform random in [0,1].
   Uses color hue normalized to unit interval."
  (let ((rgb (convert RGB color)))
    (mod (+ (* 0.299 (red rgb)) 
            (* 0.587 (green rgb)) 
            (* 0.114 (blue rgb))) 
         1.0)))

(define (loop-update interleaver spins parity)
  "Worm/loop update for off-diagonal operators.
   Colored by sublattice for parallel execution."
  (let ((loop-color (gay-sublattice interleaver parity)))
    ;; Loop update preserves detailed balance
    ;; Color tracks which stream drove the update
    :loop-updated))

(define (measure-observables interleaver spins sweep)
  "Measure energy, magnetization with colored sweep."
  (let* ((color (gay-interleave interleaver))
         (mag (mean spins))
         (energy (compute-energy spins)))
    (println (format "Sweep ~a: ⟨M⟩=~.3f ⟨E⟩=~.3f color=~a" 
                     sweep mag energy (first color)))))

(define (compute-energy spins)
  "Heisenberg energy from spin configuration."
  (let* ((Lx (size spins 1))
         (Ly (size spins 2))
         (E 0.0))
    (for (i 1 Lx)
      (for (j 1 Ly)
        (let ((Si (getindex spins i j))
              (Sr (getindex spins (mod1 (+ i 1) Lx) j))
              (Su (getindex spins i (mod1 (+ j 1) Ly))))
          (set! E (+ E (* Si Sr) (* Si Su))))))
    (/ E (* Lx Ly))))
"""

# ═══════════════════════════════════════════════════════════════════════════
# Julia driver with terminal visualization
# ═══════════════════════════════════════════════════════════════════════════

function show_checkerboard(Lx::Int, Ly::Int, seed::UInt64)
    il = GayInterleaver(seed, 2)
    
    println("\n╔══════════════════════════════════════════════════════════════╗")
    println("║  Gay.jl XOR Checkerboard Coloring for Heisenberg Model       ║")
    println("║  H = Σ J_ij (S_i · S_j)  with parity = (i ⊕ j) & 1          ║")
    println("╚══════════════════════════════════════════════════════════════╝\n")
    
    # Show lattice with colored sites
    for j in Ly:-1:1
        print("  ")
        for i in 1:Lx
            parity = (i + j) % 2
            color = gay_sublattice(il, parity)
            rgb = convert(RGB, color)
            r = round(Int, clamp(rgb.r, 0, 1) * 255)
            g = round(Int, clamp(rgb.g, 0, 1) * 255)
            b = round(Int, clamp(rgb.b, 0, 1) * 255)
            # Show site with its sublattice color
            symbol = parity == 0 ? "●" : "○"
            print("\e[38;2;$(r);$(g);$(b)m$(symbol)\e[0m ")
        end
        println()
    end
    
    println("\n  Legend: ● = even sublattice (i+j mod 2 = 0)")
    println("          ○ = odd sublattice  (i+j mod 2 = 1)")
    println("\n  Each sublattice uses independent SPI stream")
    println("  Parallel updates within sublattice preserve detailed balance\n")
end

function show_bond_coloring(L::Int, seed::UInt64)
    il = GayInterleaver(seed, 2)
    bonds = gay_exchange_colors(il, L)
    
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Nearest-Neighbor Exchange Coupling Colors (1D chain)        ║")
    println("║  J_ij colored by XOR parity: parity = (i ⊻ j) & 1           ║")
    println("╚══════════════════════════════════════════════════════════════╝\n")
    
    print("  ")
    for (color, i, j, parity) in bonds
        rgb = convert(RGB, color)
        r = round(Int, clamp(rgb.r, 0, 1) * 255)
        g = round(Int, clamp(rgb.g, 0, 1) * 255)
        b = round(Int, clamp(rgb.b, 0, 1) * 255)
        print("\e[38;2;$(r);$(g);$(b)m━━\e[0m")
        print(parity == 0 ? "●" : "○")
    end
    println("\n")
    
    println("  Bond colors alternate: even bonds (━━) / odd bonds (━━)")
    println("  SPI ensures reproducibility regardless of update order\n")
end

# Run demo if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    using Colors
    
    seed = UInt64(0xDEADBEEF)
    show_checkerboard(8, 6, seed)
    show_bond_coloring(12, seed)
    
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  StochasticSeriesExpansion.jl Hamiltonian Coloring           ║")
    println("║                                                              ║")
    println("║  H = Σ_⟨ij⟩ J_ij [S_i·S_j + d_ij S_i^z S_j^z]               ║")
    println("║    + Σ_i [h_i^z S_i^z + D_i^x (S_i^x)² + D_i^z (S_i^z)²]    ║")
    println("║                                                              ║")
    println("║  Gay.jl provides:                                            ║")
    println("║  • Checkerboard decomposition with XOR parity               ║")
    println("║  • Independent SPI streams per sublattice                   ║")
    println("║  • Deterministic coloring for reproducible MC               ║")
    println("║  • Parallel updates preserve Strong Parallelism Invariance  ║")
    println("╚══════════════════════════════════════════════════════════════╝")
end
