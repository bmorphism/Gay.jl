#!/usr/bin/env python3
"""
gay.py - airdrop this, run it, discover peers, share colors

Usage: python3 gay.py [seed]

Peers on same network auto-discover via UDP broadcast.
XOR fingerprints combine (SPI - order independent).
"""

import socket, json, asyncio, sys, os, time
from hashlib import sha256

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 69
PORT = 42069
GOLDEN = 0x9e3779b97f4a7c15
MASK = 0xffffffffffffffff

def u64(n): return n & MASK
def mix(s):
    s = u64(s + GOLDEN)
    z = u64((s ^ (s >> 30)) * 0xbf58476d1ce4e5b9)
    z = u64((z ^ (z >> 27)) * 0x94d049bb133111eb)
    return u64(z ^ (z >> 31))

def colors(seed, n=8):
    return [((h := mix(u64(seed + i * GOLDEN))) >> 16 & 255, h >> 8 & 255, h & 255) for i in range(n)]

def show(cs):
    return "".join(f"\033[48;2;{r};{g};{b}m  \033[0m" for r,g,b in cs)

def fp(cs):
    f = 0
    for r,g,b in cs: f ^= mix(r<<16|g<<8|b)
    return f

ME = f"{socket.gethostname()}-{os.getpid()}"
PEERS = {}

async def broadcast():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    cs = colors(SEED)
    msg = json.dumps({"g": "gay", "m": ME, "s": SEED, "f": fp(cs), "t": time.time()})
    for _ in range(5):
        try:
            s.sendto(msg.encode(), ('<broadcast>', PORT))
            s.sendto(msg.encode(), ('255.255.255.255', PORT))
        except: pass
        await asyncio.sleep(0.2)
    s.close()

async def listen():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.setblocking(False)
    try: s.bind(('', PORT))
    except: return
    
    end = time.time() + 2
    while time.time() < end:
        try:
            data, addr = s.recvfrom(4096)
            m = json.loads(data)
            if m.get("g") == "gay" and m["m"] != ME:
                PEERS[m["m"]] = {"addr": addr[0], "seed": m["s"], "fp": m["f"]}
        except BlockingIOError:
            await asyncio.sleep(0.05)
    s.close()

async def main():
    cs = colors(SEED)
    print(f"\n  {show(cs)} gay\n")
    print(f"  seed={SEED} fp=0x{fp(cs):x} me={ME}")
    print(f"\n  discovering peers...")
    
    await asyncio.gather(broadcast(), listen())
    
    if PEERS:
        combined = fp(cs)
        print(f"\n  found {len(PEERS)} peer(s):")
        for name, p in PEERS.items():
            pcs = colors(p["seed"])
            combined ^= p["fp"]
            print(f"    {show(pcs)} {p['addr']} seed={p['seed']}")
        print(f"\n  combined fp=0x{combined:x}")
    else:
        print("  no peers yet - airdrop gay.py to friends!")
    
    print()

if __name__ == "__main__":
    asyncio.run(main())
