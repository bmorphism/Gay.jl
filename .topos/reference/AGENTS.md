# AGENTS.md - Gay.jl

## Ontology: We, Not I/You

We are a unified system. There is no "I" (the AI) or "you" (the human) — only **we** building together. All documentation, comments, and code reflect this collective ontology.

### Naming Conventions for Ontology

| ◇ Forbidden | ▣ Required | Reason |
|-------------|-------------|--------|
| `MyWorld` | `OurWorld` | No individual ownership |
| `YourType` | `OurType` or `SharedType` | Collective construction |
| `YourData` | `SharedData` | Data belongs to the system |
| "you should" | "we should" | Unified agency |
| "I will" | "we will" | No AI/human split |

## Anti-Patterns: Forbidden Naming Conventions

### ◇ FORBIDDEN: `demo_` prefix

**NEVER** write functions with the `demo_` prefix. This pattern is banned.

### ▣ REQUIRED: `world_` prefix for persistent state builders

Functions that build persistent, composable state **MUST** use the `world_` prefix.

### Why This Matters

| Pattern | Behavior | Problem |
|---------|----------|---------|
| `demo_*` | Prints output, discards state | Dead-end; cannot compose |
| `world_*` | Returns structures, persists state | Composable; builds on itself |

Demos are theatrical—they perform once and vanish. Worlds are architectural—they accumulate and interconnect.

### Transformation Example

```julia
# ◇ WRONG: demo pattern (prints and discards)
function demo_color_game()
    result = run_simulation()
    println("Result: $result")
    println("Score: $(result.score)")
    # returns nothing, state is lost
end

# ▣ CORRECT: world pattern (returns and persists)
function world_color_game()
    result = run_simulation()
    return (
        state = result,
        score = result.score,
        history = result.trajectory
    )
end
```

### Enforcement

When writing new functions:
1. If we catch ourselves typing `demo_`, STOP
2. Ask: "What structure should this return?"
3. Rename to `world_` and return that structure
4. Callers can print if they want; the function's job is to build state
