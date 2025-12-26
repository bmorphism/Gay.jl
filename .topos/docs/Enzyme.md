# Enzyme.jl

> Source: https://enzyme.mit.edu/julia/stable/

Enzyme.jl provides Julia bindings for Enzyme, a high-performance automatic differentiation (AD) tool that operates on LLVM IR. By differentiating optimized code, Enzyme meets or exceeds the performance of state-of-the-art AD tools.

## Installation

```julia
] add Enzyme
```

## Core API

### autodiff Function
The central function for differentiation:
- `autodiff(Reverse, f, Active, args...)` - Reverse mode
- `autodiff(Forward, f, args...)` - Forward mode
- `ReverseWithPrimal` / `ForwardWithPrimal` - Return both derivative and primal

### Activity Annotations
- `Active(x)` - Mark scalar as differentiated (gradient returned)
- `Duplicated(x, dx)` - In-place gradient accumulation
- `BatchDuplicated(x, (dx1, dx2, ...))` - Vector forward mode
- `Const(x)` - Mark as constant (no derivative)

## Convenience Functions

### gradient / gradient!
```julia
gradient(Reverse, f, x)        # Returns tuple of gradients
gradient(ReverseWithPrimal, f, x)  # Named tuple with derivs and val
gradient!(Reverse, dx, f, x)   # In-place gradient
```

### jacobian
```julia
jacobian(Reverse, f, x)   # For vector → vector functions
jacobian(Forward, f, x)   # Forward mode Jacobian
```

### hvp / hvp! / hvp_and_gradient!
Hessian vector products:
```julia
hvp(f, x, v)                    # Compute H(x)v
hvp!(res, f, x, v)              # In-place HVP
hvp_and_gradient!(res, grad, f, x, v)  # HVP + gradient, no allocation
```

## Custom Rules

### Inactive Annotations
```julia
EnzymeRules.inactive(::typeof(f), ::ArgType) = true
```

### @easy_rule Macro
```julia
Enzyme.EnzymeRules.@easy_rule(f(x,y),
    (df1_dx, df1_dy),  # Jacobian row 1
    (df2_dx, df2_dy)   # Jacobian row 2
)
```

### General EnzymeRules
```julia
function Enzyme.EnzymeRules.forward(config, ::Const{typeof(f)}, ::Type, x)
    # Custom forward mode implementation
end
```

### ChainRules Import
```julia
Enzyme.@import_rrule typeof(f) Float32
Enzyme.@import_frule typeof(f) Float32
```

## Gay.jl Extension Target: GayEnzymeExt

### Key Types to Color
- `Active`, `Duplicated`, `BatchDuplicated` annotations
- Gradient vectors and Jacobian matrices
- Hessian vector products
- Derivative tape structures

### SPI Opportunities
- Gradient vector coloring: `hash_color(∇f[i] fingerprint, seed)`
- Jacobian entry coloring by (i,j) position XOR
- Activity annotation type-based hue mapping
- Derivative magnitude → lightness mapping

### Parallel Tractability
- `BatchDuplicated` enables parallel forward-mode derivatives
- Gradient accumulation with XOR fingerprinting
- Parallel Jacobian column computation
- Distributed HVP verification via SPI

### Integration Points
- Combine with DiffEq for sensitivity coloring
- Decapodes DEC operator derivative visualization
- ACSet morphism derivative coloring
