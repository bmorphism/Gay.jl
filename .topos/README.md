# gay - SPI Color Ecosystem

Splitmix64-based chromatic identity with Strong Parallelism Invariance.

## Quick Start

```bash
# Generate colors
bb gay 69 8

# Aperiodic hat tiling (Tatham)
bb gay hat 69 12
bb gay tdx 69 8      # FST transducer
bb gay wheel 69      # find cycles

# Peer discovery
bb gay drop 42       # UDP broadcast (airdrop gay.py)
bb gay ts 69         # tailscale mesh
bb gay 2tdx 69 0     # 2-transducer listener (polarity 0=MINUS)
bb gay peer 69       # MLX + exo (requires uv)
```

## Files

| File | Description |
|------|-------------|
| `gay` | Unified CLI (babashka) |
| `gay.py` | Airdrop-ready peer discovery |
| `gay_2tdx.py` | 2-transducer triadic interleave |
| `gay_ts.py` | Tailscale mesh discovery |
| `gay_exo_peer.py` | MLX + exo distributed |
| `gay_hat.bb` | Hat monotile coordinates |
| `gay_hat_tdx.bb` | Hat FST transducer |
| `gay_hat_wheel.bb` | Cycle finder |

## 2-TDX Triadic Protocol

```
Polarity  Symbol  Phases      Role
────────  ──────  ──────────  ──────────
MINUS     −       0,3,6...    Contraction
ERGODIC   _       1,4,7...    Afference  
PLUS      +       2,5,8...    Expansion

Cycle: (−) → (_) → (+) → (−) → ...
```

## SPI Guarantee

Same seed → same colors, regardless of:
- Execution order
- Parallelism
- Machine

XOR fingerprint combines order-independently.

## Peers

```bash
# Machine A (listener)
python3 gay_2tdx.py 69 0

# Machine B (connect)
python3 gay_2tdx.py 42 1 <machine-a-ip>

# Machine C (connect)  
python3 gay_2tdx.py 1337 2 <machine-a-ip>
```

Combined fingerprint verifies all contributions.
