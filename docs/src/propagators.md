# Propagator Networks

**Constraint propagation with chromatic debugging**

Gay.jl includes a propagator system inspired by Sussman & Hanson's "Software Design for Flexibility" (SDF), with SPI colors for visualization and debugging.

## What Are Propagators?

Propagators are **autonomous agents** that watch cells and propagate information:

```
     ┌─────────┐
A ───┤   add   ├─── C
B ───┤         │
     └─────────┘
     
When A or B changes, add propagates A+B to C
```

Unlike functions (which compute once), propagators:
- Run whenever inputs change
- Support **bidirectional** constraints
- Handle **partial information** gracefully
- Enable **truth maintenance** and hypothetical reasoning

## Basic Usage

```julia
using Gay: Propagator
using Gay.Propagator: make_cell, add_content!, cell_content
using Gay.Propagator: propagator, run!

# Create cells
a = make_cell(:a)
b = make_cell(:b)
c = make_cell(:c)

# Create adder propagator: c = a + b
propagator([a, b], c) do inputs
    a_val, b_val = inputs
    (a_val !== nothing && b_val !== nothing) ? a_val + b_val : nothing
end

# Tell the cells some values
add_content!(a, 3)
add_content!(b, 4)
run!()  # Run propagator network

@assert cell_content(c) == 7
```

## Bidirectional Constraints

Propagators naturally support bidirectional flow:

```julia
# Constraint: c = a + b (bidirectional)
constraint_add(a, b, c)

add_content!(c, 10)  # Tell c
add_content!(a, 3)   # Tell a
run!()

@assert cell_content(b) == 7  # b inferred!
```

## Lisp DSL

Use S-expressions for declarative constraint definition:

```julia
using Gay: PropagatorLisp
using Gay.PropagatorLisp: @prop_str, cell_value

# Define cells and constraints
prop"""
(define-cell x)
(define-cell y)
(define-cell z)

(constraint-add x y z)  ; z = x + y

(tell x 5)
(tell y 3)
"""

@assert cell_value(:z) == 8
```

## Hypothetical Reasoning

Propagators support **premises** — assumptions that can be retracted:

```julia
using Gay.Propagator: hypothetical, mark_premise_in!, mark_premise_out!

# Create a hypothesis
h = hypothetical(:assumption_1)

# Propagate under hypothesis
add_content!(a, 10, h)  # "a is 10 assuming h"
run!()

# Check result
@assert cell_content(c) == 14  # 10 + 4

# Retract hypothesis
mark_premise_out!(h)
run!()

@assert cell_content(c) === nothing  # No longer holds!
```

## Chromatic Debugging

Each cell and propagator gets a deterministic SPI color:

```julia
using Gay: gay_seed!

gay_seed!(42)

a = make_cell(:temperature)
# Cell 'temperature' has color RGB(0.34, 0.67, 0.21)

b = make_cell(:pressure)
# Cell 'pressure' has color RGB(0.89, 0.12, 0.45)
```

Visualize the propagator network:

```julia
using Gay.Propagator: show_network

show_network()
# Displays colored ASCII diagram of cells and propagators
```

## Amb and Search

The `amb` (ambiguous) operator enables search:

```julia
using Gay.PropagatorLisp: prop_amb

prop"""
(define-cell x)
(define-cell y)
(define-cell sum)

(amb x (1 2 3 4 5))     ; x ∈ {1,2,3,4,5}
(amb y (1 2 3 4 5))     ; y ∈ {1,2,3,4,5}

(constraint-add x y sum)
(tell sum 7)            ; Find x,y such that x+y=7
"""

# Solutions: (2,5), (3,4), (4,3), (5,2)
```

## Example: Electrical Circuit

```julia
prop"""
; Ohm's Law: V = I * R
(define-cell voltage)
(define-cell current)
(define-cell resistance)

(constraint-mul current resistance voltage)

; Power: P = V * I
(define-cell power)
(constraint-mul voltage current power)

; Given: 12V battery, 100Ω resistor
(tell voltage 12)
(tell resistance 100)
"""

@assert cell_value(:current) ≈ 0.12
@assert cell_value(:power) ≈ 1.44
```

## Example: Unit Conversion

```julia
prop"""
(define-cell celsius)
(define-cell fahrenheit)

; F = C * 9/5 + 32
(define-cell scaled)
(constraint-mul celsius (constant 1.8) scaled)
(constraint-add scaled (constant 32) fahrenheit)

(tell celsius 100)
"""

@assert cell_value(:fahrenheit) ≈ 212.0
```

## Integration with Gay.jl Colors

Propagator networks can generate color palettes:

```julia
using Gay: next_color
using Gay.Propagator: make_cell, propagator, run!

# Color mixing propagator
r = make_cell(:red)
g = make_cell(:green)
mixed = make_cell(:mixed)

propagator([r, g], mixed) do inputs
    r_val, g_val = inputs
    if r_val !== nothing && g_val !== nothing
        # Mix colors using Gay.jl
        (r_val.r + g_val.r) / 2,
        (r_val.g + g_val.g) / 2,
        (r_val.b + g_val.b) / 2
    end
end

add_content!(r, next_color())
add_content!(g, next_color())
run!()

# mixed now contains the blended color
```

## References

- Sussman & Hanson, "Software Design for Flexibility" (2021)
- Radul, "Propagation Networks" (PhD thesis, 2009)
- de Kleer, "An Assumption-Based TMS" (1986)

See also:
- [Lisp Syntax](splittable_determinism.md) — S-expression foundations
- [Theory](theory.md) — SPI background
