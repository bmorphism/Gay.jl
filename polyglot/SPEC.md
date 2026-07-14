# Polyglot legitimacy kernel — legitimacy from correctness from without

One deterministic clearing+audit kernel, implemented independently in five
languages. Every implementation must print byte-identical canonical output.

**Legitimation rule (Beetham, operationalized):** an implementation is NOT
legitimate because its own tests believe in it ("Legitimitätsglaube" —
Weber, assigned, decorative). It is legitimate iff its output can be
justified in terms of the others' outputs: byte-identical agreement with at
least 2 of the other implementations (external quorum). `legitimate.sh`
computes the verdict. Self-testing confers zero legitimacy here.

## Kernel (all integer arithmetic; no floats anywhere)

Constants (SplitMix64 / Stafford mix13):

```
G  = 0x9e3779b97f4a7c15    M1 = 0xbf58476d1ce4e5b9    M2 = 0x94d049bb133111eb
fin(z): z ^= z>>>30; z *= M1; z ^= z>>>27; z *= M2; z ^= z>>>31   (mod 2^64)
sm(seed,k) = fin(seed + G*k)                                       (mod 2^64)
```

Market: SEED = 1069, N = 8 agents, T = 16 rounds, pools A = {0..3}, B = {4..7}.

Per agent i, round r:

```
h    = sm(SEED, i*2^32 + r);  u1,u2,u3 = bytes 16,8,0 of h
load = 100 + byte0(sm(SEED, 0xA000 + i))
m    = r mod 16;  amp = m<=8 ? m*64 : (16-m)*64;   grid = 1000 + amp
rad  = (amp * (200 + 10i)) >> 8
q    = rad - load + (u1 - 128)
q>0  -> seller, qty q,  ask = (grid * (115 + (u2>>1))) >> 8
q<0  -> buyer,  qty -q, bid = (grid * (179 + (u3>>1))) >> 8
q==0 -> abstain
```

Clearing (uniform double auction, fully tie-broken): sellers sorted (ask asc,
i asc), buyers (bid desc, i asc); while best bid >= best ask: take =
min(remainders), price = (bid+ask)>>1, surplus += (bid-ask)*take. Returns
fills, surplus, and residual (unfilled) books.

## The two axes, measured per round

- **justice**: `wev = surplus(unified) - surplus(pool A) - surplus(pool B)`
  >= 0. WEV = the price of fragmentation (PoA - 1 in surplus units).
- **legitimacy legs** (GF(3), measured):
  - play = 1 if unified fills exist else 0            (contact observed)
  - wit  = 0 (conservation; structural in single venue — documented, not dressed up)
  - cop  = -1 if NO cross-pool residual crossing survives fragmented clearing,
    else 0 (live inter-pool arbitrage = Cech obstruction on the cover {A,B})
  - sum3 = (play+wit+cop+3) mod 3;  legit bit = (play==1 and cop==-1)

Rawls asymmetry, observable: rounds can be legit with wev>0
(legitimate-but-unjust); wev==0 with play==0 (nothing to be just about —
Williams's first political question failing before legitimacy applies).

## Canonical output (byte-identical, one line per round)

```
r=<r> clear=<lastprice|-> fills=<n> legs=<p>,<w>,<c> sum3=<s> wev=<w> fp=<hex16>
TOTAL fp=<hex16> legit=<k>/16 wev=<sum>
```

Round fingerprint: fp = XOR over fills of fin((price<<40)^(qty<<20)^(buyer<<8)^seller),
then fp ^= fin((wev<<8)^sum3^(legit<<4)). TOTAL fp = XOR over rounds of fin(fp_r ^ r).
Hex is lowercase, zero-padded to 16.

## Run

```
./legitimate.sh     # runs every available implementation, prints the verdict table
```
