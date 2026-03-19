#!/usr/bin/env python3
"""
gay_2tdx.py - 2-Transducer interleaved peer protocol

After Loregian arXiv:2509.06769 "Two-Dimensional Transducers"

Peers interleave contributions via:
- Checkerboard (2 peers): even/odd phases
- Triadic (3 peers): GF(3) phases mod 3
- N-ary: round-robin

XOR fingerprint accumulates across all phases.
"""

import socket, json, asyncio, sys, time

PORT = 42069
GOLDEN = 0x9e3779b97f4a7c15
MASK = 0xffffffffffffffff

# Polarity twists for triadic
MINUS_TWIST = 0x2d2d2d2d2d2d2d2d
ERGODIC_TWIST = 0x5f5f5f5f5f5f5f5f  
PLUS_TWIST = 0x2b2b2b2b2b2b2b2b
TWISTS = [MINUS_TWIST, ERGODIC_TWIST, PLUS_TWIST]
POLARITIES = ["−", "_", "+"]

def u64(n): return n & MASK
def mix(s):
    s = u64(s + GOLDEN)
    z = u64((s ^ (s >> 30)) * 0xbf58476d1ce4e5b9)
    z = u64((z ^ (z >> 27)) * 0x94d049bb133111eb)
    return u64(z ^ (z >> 31))

def colors(seed, n=6):
    return [((h := mix(u64(seed + i * GOLDEN))) >> 16 & 255, h >> 8 & 255, h & 255) for i in range(n)]

def show(cs):
    return "".join(f"\033[48;2;{r};{g};{b}m  \033[0m" for r,g,b in cs)

def fp(cs):
    f = 0
    for r,g,b in cs: f ^= mix(r<<16|g<<8|b)
    return f

class TDXState:
    def __init__(self, seed, polarity=0):
        self.seed = seed
        self.polarity = polarity  # 0=MINUS, 1=ERGODIC, 2=PLUS
        self.phase = 0
        self.fingerprint = 0
        self.contributions = []
        self.peers = {}
    
    def my_turn(self, n_peers):
        """Check if it's our turn based on phase mod n_peers"""
        return (self.phase % max(n_peers, 1)) == self.polarity
    
    def contribute(self, data_hash):
        """Add contribution, advance phase"""
        twisted = u64(self.seed ^ TWISTS[self.polarity])
        cs = colors(twisted)
        contrib_fp = fp(cs) ^ mix(data_hash)
        self.fingerprint ^= contrib_fp
        self.contributions.append({
            "phase": self.phase,
            "polarity": self.polarity,
            "fp": contrib_fp
        })
        self.phase += 1
        return cs, contrib_fp
    
    def receive(self, peer_id, peer_polarity, peer_fp):
        """Receive peer contribution"""
        self.peers[peer_id] = {"polarity": peer_polarity, "fp": peer_fp}
        self.fingerprint ^= peer_fp
        self.phase += 1

ME = socket.gethostname()
STATE = None

async def handle_peer(reader, writer):
    addr = writer.get_extra_info("peername")[0]
    try:
        data = await asyncio.wait_for(reader.read(4096), timeout=5)
        msg = json.loads(data)
        
        if msg.get("t") == "2tdx":
            peer_pol = msg.get("pol", 0)
            peer_fp = msg.get("fp", 0)
            peer_phase = msg.get("phase", 0)
            peer_seed = msg.get("s", 69)
            
            # Show peer contribution
            pcs = colors(u64(peer_seed ^ TWISTS[peer_pol]))
            print(f"  {POLARITIES[peer_pol]} {show(pcs)} {addr} phase={peer_phase}")
            
            # Receive into state
            STATE.receive(addr, peer_pol, peer_fp)
            
            # Respond with our state
            resp = json.dumps({
                "t": "2tdx",
                "m": ME,
                "s": STATE.seed,
                "pol": STATE.polarity,
                "phase": STATE.phase,
                "fp": STATE.fingerprint
            })
            writer.write(resp.encode())
            await writer.drain()
            
            print(f"    combined fp=0x{STATE.fingerprint:x}")
    except Exception as e:
        pass
    writer.close()

async def connect_peer(ip, port=PORT):
    """Connect to peer and exchange state"""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port), timeout=2)
        
        # Our contribution
        cs, cfp = STATE.contribute(mix(int(time.time() * 1000)))
        
        msg = json.dumps({
            "t": "2tdx",
            "m": ME,
            "s": STATE.seed,
            "pol": STATE.polarity,
            "phase": STATE.phase,
            "fp": cfp
        })
        writer.write(msg.encode())
        await writer.drain()
        
        data = await asyncio.wait_for(reader.read(4096), timeout=2)
        resp = json.loads(data)
        
        writer.close()
        await writer.wait_closed()
        
        if resp.get("t") == "2tdx":
            STATE.receive(ip, resp.get("pol", 0), resp.get("fp", 0))
            return resp
    except:
        pass
    return None

async def main():
    global STATE
    
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 69
    polarity = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    peer_ip = sys.argv[3] if len(sys.argv) > 3 else None
    
    STATE = TDXState(seed, polarity % 3)
    
    cs = colors(u64(seed ^ TWISTS[polarity % 3]))
    print(f"\ngay 2-TDX interleave")
    print(f"  {show(cs)} gay")
    print(f"  seed={seed} polarity={POLARITIES[polarity % 3]} phase={STATE.phase}")
    print()
    
    if peer_ip:
        # Client mode - connect to peer
        print(f"connecting to {peer_ip}...")
        resp = await connect_peer(peer_ip)
        if resp:
            pcs = colors(u64(resp["s"] ^ TWISTS[resp["pol"]]))
            print(f"  {POLARITIES[resp['pol']]} {show(pcs)} {peer_ip}")
            print(f"\ncombined fp=0x{STATE.fingerprint:x}")
        else:
            print("  connection failed")
    else:
        # Server mode - listen
        print(f"listening on :{PORT}")
        print(f"peers connect: python3 gay_2tdx.py [seed] [polarity] {socket.gethostname()}")
        print()
        
        server = await asyncio.start_server(handle_peer, "0.0.0.0", PORT)
        async with server:
            await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
