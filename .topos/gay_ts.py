#!/usr/bin/env python3
"""
gay_ts.py - tailscale mesh peer discovery + flox

Usage: python3 gay_ts.py [seed]

Discovers peers via tailscale, connects over mesh VPN.
"""

import socket, json, asyncio, sys, os, subprocess, time

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

def tailscale_peers():
    """Get online tailscale peers"""
    try:
        # Try macOS app path first
        ts = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
        if not os.path.exists(ts):
            ts = "tailscale"
        
        result = subprocess.run([ts, "status", "--json"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return []
        
        data = json.loads(result.stdout)
        peers = []
        self_ip = data.get("Self", {}).get("TailscaleIPs", [None])[0]
        
        for name, peer in data.get("Peer", {}).items():
            if peer.get("Online"):
                ips = peer.get("TailscaleIPs", [])
                if ips:
                    peers.append({
                        "name": peer.get("HostName", name),
                        "ip": ips[0],
                        "os": peer.get("OS", "?"),
                        "user": peer.get("UserID", 0)
                    })
        return peers, self_ip
    except Exception as e:
        return [], None

ME = f"{socket.gethostname()}-{os.getpid()}"
FOUND = {}

async def probe_peer(ip, timeout=0.5):
    """Try to connect to peer on gay port"""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, PORT), timeout=timeout)
        writer.write(json.dumps({"g": "gay", "m": ME, "s": SEED, "f": fp(colors(SEED))}).encode())
        await writer.drain()
        data = await asyncio.wait_for(reader.read(4096), timeout=timeout)
        writer.close()
        await writer.wait_closed()
        return json.loads(data)
    except:
        return None

async def listen_tcp():
    """TCP server for peer connections"""
    async def handle(reader, writer):
        try:
            data = await asyncio.wait_for(reader.read(4096), timeout=2)
            msg = json.loads(data)
            if msg.get("g") == "gay":
                FOUND[msg["m"]] = {"seed": msg["s"], "fp": msg["f"], "addr": writer.get_extra_info('peername')[0]}
                response = json.dumps({"g": "gay", "m": ME, "s": SEED, "f": fp(colors(SEED))})
                writer.write(response.encode())
                await writer.drain()
        except: pass
        finally:
            writer.close()
    
    try:
        server = await asyncio.start_server(handle, '0.0.0.0', PORT)
        await asyncio.sleep(3)
        server.close()
        await server.wait_closed()
    except: pass

async def main():
    cs = colors(SEED)
    print(f"\n  {show(cs)} gay\n")
    print(f"  seed={SEED} fp=0x{fp(cs):x} me={ME}")
    
    # Get tailscale peers
    peers, self_ip = tailscale_peers()
    print(f"\n  tailscale: {self_ip or 'not connected'}")
    
    if peers:
        print(f"  {len(peers)} peer(s) online:")
        for p in peers:
            print(f"    {p['ip']:15} {p['name']:20} {p['os']}")
        
        print(f"\n  probing port {PORT}...")
        
        # Start listener and probe in parallel
        listen_task = asyncio.create_task(listen_tcp())
        
        # Probe all peers in parallel
        probe_tasks = [probe_peer(p['ip']) for p in peers]
        results = await asyncio.gather(*probe_tasks, return_exceptions=True)
        
        for p, r in zip(peers, results):
            if isinstance(r, dict) and r.get("g") == "gay":
                FOUND[r["m"]] = {"seed": r["s"], "fp": r["f"], "addr": p['ip'], "name": p['name']}
        
        await listen_task
        
        if FOUND:
            combined = fp(cs)
            print(f"\n  gay peers ({len(FOUND)}):")
            for name, p in FOUND.items():
                pcs = colors(p["seed"])
                combined ^= p["fp"]
                print(f"    {show(pcs)} {p.get('name', p['addr'])} seed={p['seed']}")
            print(f"\n  combined fp=0x{combined:x}")
        else:
            print("\n  no gay peers responding - run gay_ts.py on other machines!")
    else:
        print("  no tailscale peers online")
    
    print()

if __name__ == "__main__":
    asyncio.run(main())
