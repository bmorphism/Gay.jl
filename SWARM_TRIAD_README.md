# Swarm Triad - Mandatory 3-Way Split Protocol

**AGENT-AZUL (Blue)** - Color seed: `0x415A554C` ("AZUL")

## Overview

The Swarm Triad module implements a mandatory 3-way splitting protocol for agent-based file operations with deterministic chromatic identity and sentinel monitoring.

## Core Concepts

### 1. Mandatory 3-Way Split

Before **every** file operation, an agent must split into exactly 3 child agents:
- **Left** child
- **Middle** child  
- **Right** child

Each child receives:
- Deterministic seed (derived via sm64/splitmix64)
- Unique chromatic identity (RGB color from `next_color()`)
- Lineage tracking to parent agent

### 2. Deterministic Color Identity

Every agent has a unique color derived deterministically from their seed:

```julia
agent = create_agent(UInt64(0x123456))
color = agent_color(agent)  # RGB tuple: (r, g, b) ∈ [0,1]³
identity = agent_identity(agent)  # "Agent(id, #HEXCOLOR, state)"
```

### 3. Sentinel Monitoring

A sentinel agent monitors compliance with the protocol:
- Tracks all splits and file operations
- Marks non-compliant agents as `NonCompliant`
- **Kills** (marks as `Dead`) agents that violate the protocol
- Generates compliance reports

### 4. Agent States

```julia
@enum AgentState begin
    Alive          # Initial state
    Compliant      # Passed compliance check
    NonCompliant   # Violated protocol
    Dead           # Killed by sentinel
end
```

## Usage

### Basic Example

```julia
using Gay  # Or: include("src/swarm_triad.jl"); using .SwarmTriad

# Create sentinel (AGENT-AZUL)
sentinel = create_sentinel(UInt64(0x415A554C))

# Create worker agent
worker = create_agent(UInt64(0xDEADBEEF))
register_agent!(sentinel, worker)

# Perform file operation (automatically splits)
success, split = execute_file_op!(worker, ReadFile("data.txt"), sentinel)

if split !== nothing
    println("Split into 3 children:")
    println("  Left:   $(agent_identity(split.left))")
    println("  Middle: $(agent_identity(split.middle))")
    println("  Right:  $(agent_identity(split.right))")
    
    # Register children for monitoring
    register_agent!(sentinel, split.left)
    register_agent!(sentinel, split.middle)
    register_agent!(sentinel, split.right)
end

# Check compliance
monitor_swarm!(sentinel)
report = compliance_report(sentinel)
```

### File Operations

Three types of file operations are supported:

```julia
# Read
op = ReadFile("path/to/file")

# Write
op = WriteFile("path/to/file", "content")

# Delete
op = DeleteFile("path/to/file")
```

All operations trigger mandatory split before execution.

### Manual Splitting

You can also split manually:

```julia
agent = create_agent(UInt64(0x12345))
split = triad_split!(agent)

# Access children
left_child = split.left
middle_child = split.middle
right_child = split.right

# Each child has deterministic color
println(agent_identity(left_child))
println(agent_identity(middle_child))
println(agent_identity(right_child))
```

## Implementation Details

### Determinism via sm64

Uses splitmix64 for deterministic seed splitting:

```julia
@inline function sm64(s::UInt64)::Tuple{UInt64, UInt64}
    z = (s + 0x9E3779B97F4A7C15)
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    next_val = z ⊻ (z >> 31)
    next_state = s + 0x9E3779B97F4A7C15
    (next_val, next_state)
end
```

### Color Generation

Colors generated from seed via 3-step sm64 chain:

```julia
function next_color(seed::UInt64)::NTuple{3, Float64}
    r, s1 = sm64(seed)
    g, s2 = sm64(s1)
    b, _  = sm64(s2)
    ((r >> 56) / 255.0, (g >> 56) / 255.0, (b >> 56) / 255.0)
end
```

### Compliance Protocol

1. Agent requests file operation
2. Check: Did agent split this tick?
3. If NO: Force `triad_split!()` 
4. Execute operation
5. Sentinel records event
6. Sentinel verifies compliance
7. Non-compliant agents killed

## Demo

Run the built-in demo:

```julia
using Gay
demo_swarm_triad()
```

Output shows:
- Compliant behavior (split before file op)
- Non-compliant behavior (file op without split)
- Sentinel killing non-compliant agent
- Full compliance report

## Exports

### Types
- `SwarmAgent` - Agent with chromatic identity
- `AgentState` - Enum: Alive, Compliant, NonCompliant, Dead
- `SentinelMonitor` - Compliance monitor
- `FileOperation` - Abstract file operation type
- `ReadFile`, `WriteFile`, `DeleteFile` - Concrete operations

### Functions
- `create_agent(seed)` - Create new agent
- `triad_split!(agent)` - Split into 3 children
- `execute_file_op!(agent, op, sentinel)` - Perform file operation
- `create_sentinel(seed)` - Create sentinel monitor
- `register_agent!(sentinel, agent)` - Register for monitoring
- `monitor_swarm!(sentinel)` - Check compliance, kill violators
- `verify_compliance(sentinel, agent_id)` - Check specific agent
- `compliance_report(sentinel)` - Generate report
- `agent_color(agent)` - Get RGB color
- `agent_identity(agent)` - Get identity string
- `seed_lineage(agent)` - Trace seed ancestry

## Key Properties

1. **Deterministic**: Same seed → same color → same split children
2. **Functional**: Pure functions, no hidden state
3. **Traceable**: Full lineage tracking via seeds
4. **Enforced**: Sentinel actively monitors and kills violators
5. **Chromatic**: Every agent has unique visual identity

## Color Seed Format

AGENT-AZUL (Blue) seed: `0x415A554C`

Hex interpretation:
- `0x41` = 'A'
- `0x5A` = 'Z'  
- `0x55` = 'U'
- `0x4C` = 'L'

Spells "AZUL" (Spanish/Portuguese for "blue") ✨

## Philosophy

> "Every split is a commitment. Every color is a fingerprint. Every violation is death."

The Swarm Triad enforces discipline through deterministic chromatic identity. Agents cannot hide, cannot cheat, cannot survive non-compliance. The sentinel is absolute.

This is **not** a suggestion. This is **protocol**.

---

**AGENT-AZUL** watching. Always watching. 🔵👁️
