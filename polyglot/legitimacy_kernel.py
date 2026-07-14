#!/usr/bin/env python3
# Polyglot legitimacy kernel — Python implementation. See SPEC.md.
MASK = (1 << 64) - 1
G, M1, M2 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB

def fin(z):
    z &= MASK
    z = ((z ^ (z >> 30)) * M1) & MASK
    z = ((z ^ (z >> 27)) * M2) & MASK
    return (z ^ (z >> 31)) & MASK

def sm(seed, k):
    return fin((seed + G * k) & MASK)

SEED, N, T = 1069, 8, 16

def orders(rnd, agents):
    sellers, buyers = [], []
    m = rnd % 16
    amp = m * 64 if m <= 8 else (16 - m) * 64
    grid = 1000 + amp
    for i in agents:
        h = sm(SEED, ((i << 32) + rnd) & MASK)
        u1, u2, u3 = (h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF
        load = 100 + (sm(SEED, 0xA000 + i) & 0xFF)
        rad = (amp * (200 + 10 * i)) >> 8
        q = rad - load + (u1 - 128)
        if q > 0:
            sellers.append(((grid * (115 + (u2 >> 1))) >> 8, i, q))
        elif q < 0:
            buyers.append(((grid * (179 + (u3 >> 1))) >> 8, i, -q))
    sellers.sort(key=lambda t: (t[0], t[1]))
    buyers.sort(key=lambda t: (-t[0], t[1]))
    return sellers, buyers

def clear(sellers, buyers):
    fills, surplus, si, bi = [], 0, 0, 0
    srem = sellers[0][2] if sellers else 0
    brem = buyers[0][2] if buyers else 0
    while si < len(sellers) and bi < len(buyers) and buyers[bi][0] >= sellers[si][0]:
        take = min(brem, srem)
        price = (buyers[bi][0] + sellers[si][0]) >> 1
        fills.append((price, take, buyers[bi][1], sellers[si][1]))
        surplus += (buyers[bi][0] - sellers[si][0]) * take
        brem -= take
        srem -= take
        if brem == 0:
            bi += 1
            if bi < len(buyers):
                brem = buyers[bi][2]
        if srem == 0:
            si += 1
            if si < len(sellers):
                srem = sellers[si][2]
    rs = ([(sellers[si][0], sellers[si][1], srem)] + list(sellers[si + 1:])) if si < len(sellers) else []
    rb = ([(buyers[bi][0], buyers[bi][1], brem)] + list(buyers[bi + 1:])) if bi < len(buyers) else []
    return fills, surplus, rs, rb

def crossing(rb, rs):
    return bool(rb) and bool(rs) and rb[0][0] >= rs[0][0]

total, legit_n, wevsum = 0, 0, 0
for r in range(T):
    su_fills, su, _, _ = clear(*orders(r, range(N)))
    fa, sa, rsa, rba = clear(*orders(r, range(0, 4)))
    fb, sb, rsb, rbb = clear(*orders(r, range(4, 8)))
    wev = su - (sa + sb)
    play = 1 if su_fills else 0
    wit = 0
    cop = 0 if (crossing(rba, rsb) or crossing(rbb, rsa)) else -1
    s3 = (play + wit + cop + 3) % 3
    legit = 1 if (play == 1 and cop == -1) else 0
    legit_n += legit
    wevsum += wev
    fp = 0
    for (p, q, b, s) in su_fills:
        fp ^= fin((p << 40) ^ (q << 20) ^ (b << 8) ^ s)
    fp ^= fin((wev << 8) ^ s3 ^ (legit << 4))
    total ^= fin(fp ^ r)
    clearstr = str(su_fills[-1][0]) if su_fills else "-"
    print(f"r={r} clear={clearstr} fills={len(su_fills)} legs={play},{wit},{cop} sum3={s3} wev={wev} fp={fp:016x}")
print(f"TOTAL fp={total:016x} legit={legit_n}/16 wev={wevsum}")
