#!/usr/bin/env python3
# /// script
# dependencies = ["mlx"]
# ///
"""gay exo peer - run with: uv run gay_exo_peer.py"""

import asyncio
import socket
import json
import sys

GAY_SEED = 0x285508656870f24a
GOLDEN = 0x9e3779b97f4a7c15
MASK64 = 0xffffffffffffffff

def u64(n): return n & MASK64
def splitmix64(s):
    s = u64(s + GOLDEN)
    z = u64((s ^ (s >> 30)) * 0xbf58476d1ce4e5b9)
    z = u64((z ^ (z >> 27)) * 0x94d049bb133111eb)
    return u64(z ^ (z >> 31)), s

def color(seed, i):
    h, _ = splitmix64(u64(seed + i * GOLDEN))
    return (h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF

def show(r, g, b): return f"\033[48;2;{r};{g};{b}m  \033[0m"

try:
    import mlx.core as mx
    MLX = True
except:
    MLX = False

class GayPeer:
    def __init__(self, seed, port=42069):
        self.seed = seed
        self.port = port
        self.fp = 0
        self.peers = {}
        self.host = socket.gethostname()
    
    def colors(self, n):
        cs = [color(self.seed, i) for i in range(n)]
        self.fp = 0
        for r, g, b in cs:
            self.fp ^= splitmix64(r << 16 | g << 8 | b)[0]
        return cs
    
    async def broadcast(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setblocking(False)
        msg = json.dumps({"t": "gay", "h": self.host, "s": self.seed, "f": self.fp}).encode()
        try:
            sock.sendto(msg, ('<broadcast>', self.port))
        except: pass
        sock.close()
    
    async def listen(self, timeout=1.0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setblocking(False)
        try:
            sock.bind(('', self.port))
        except:
            return
        
        loop = asyncio.get_event_loop()
        end = loop.time() + timeout
        while loop.time() < end:
            try:
                data, addr = sock.recvfrom(1024)
                m = json.loads(data)
                if m.get("t") == "gay" and m["h"] != self.host:
                    self.peers[addr[0]] = m
            except BlockingIOError:
                await asyncio.sleep(0.1)
        sock.close()
    
    def combined_fp(self):
        fp = self.fp
        for p in self.peers.values():
            fp ^= p.get("f", 0)
        return fp

async def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 69
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    peer = GayPeer(seed)
    cs = peer.colors(n)
    
    print(f"\ngay exo peer seed={seed} mlx={MLX}")
    print("  " + "".join(show(*c) for c in cs) + " gay")
    print(f"  fp=0x{peer.fp:016x}")
    
    await peer.broadcast()
    await peer.listen(0.5)
    
    if peer.peers:
        print(f"\npeers ({len(peer.peers)}):")
        for addr, p in peer.peers.items():
            print(f"  {addr}: {p['h']} fp=0x{p['f']:x}")
        print(f"combined=0x{peer.combined_fp():016x}")
    else:
        print("\nno peers")

if __name__ == "__main__":
    asyncio.run(main())
