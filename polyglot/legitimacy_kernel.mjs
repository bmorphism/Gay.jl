// Polyglot legitimacy kernel — Node (ESM) implementation. See SPEC.md.
const MASK = (1n << 64n) - 1n;
const G = 0x9e3779b97f4a7c15n, M1 = 0xbf58476d1ce4e5b9n, M2 = 0x94d049bb133111ebn;

const fin = (z) => {
  z &= MASK;
  z = ((z ^ (z >> 30n)) * M1) & MASK;
  z = ((z ^ (z >> 27n)) * M2) & MASK;
  return (z ^ (z >> 31n)) & MASK;
};
const sm = (seed, k) => fin((seed + G * k) & MASK);

const SEED = 1069n, N = 8, T = 16;

function orders(rnd, agents) {
  const sellers = [], buyers = [];
  const m = rnd % 16;
  const amp = m <= 8 ? m * 64 : (16 - m) * 64;
  const grid = 1000 + amp;
  for (const i of agents) {
    const h = sm(SEED, ((BigInt(i) << 32n) + BigInt(rnd)) & MASK);
    const u1 = Number((h >> 16n) & 0xFFn), u2 = Number((h >> 8n) & 0xFFn), u3 = Number(h & 0xFFn);
    const load = 100 + Number(sm(SEED, BigInt(0xA000 + i)) & 0xFFn);
    const rad = (amp * (200 + 10 * i)) >> 8;
    const q = rad - load + (u1 - 128);
    if (q > 0) sellers.push([(grid * (115 + (u2 >> 1))) >> 8, i, q]);
    else if (q < 0) buyers.push([(grid * (179 + (u3 >> 1))) >> 8, i, -q]);
  }
  sellers.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  buyers.sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  return [sellers, buyers];
}

function clear(sellers, buyers) {
  const fills = [];
  let surplus = 0, si = 0, bi = 0;
  let srem = sellers.length ? sellers[0][2] : 0;
  let brem = buyers.length ? buyers[0][2] : 0;
  while (si < sellers.length && bi < buyers.length && buyers[bi][0] >= sellers[si][0]) {
    const take = Math.min(brem, srem);
    const price = (buyers[bi][0] + sellers[si][0]) >> 1;
    fills.push([price, take, buyers[bi][1], sellers[si][1]]);
    surplus += (buyers[bi][0] - sellers[si][0]) * take;
    brem -= take; srem -= take;
    if (brem === 0) { bi += 1; if (bi < buyers.length) brem = buyers[bi][2]; }
    if (srem === 0) { si += 1; if (si < sellers.length) srem = sellers[si][2]; }
  }
  const rs = si < sellers.length ? [[sellers[si][0], sellers[si][1], srem], ...sellers.slice(si + 1)] : [];
  const rb = bi < buyers.length ? [[buyers[bi][0], buyers[bi][1], brem], ...buyers.slice(bi + 1)] : [];
  return [fills, surplus, rs, rb];
}

const crossing = (rb, rs) => rb.length > 0 && rs.length > 0 && rb[0][0] >= rs[0][0];
const hex16 = (z) => z.toString(16).padStart(16, "0");

let total = 0n, legitN = 0, wevsum = 0;
const range = (a, b) => Array.from({ length: b - a }, (_, k) => a + k);
for (let r = 0; r < T; r++) {
  const [suFills, su] = clear(...orders(r, range(0, N)));
  const [, sa, rsa, rba] = clear(...orders(r, range(0, 4)));
  const [, sb, rsb, rbb] = clear(...orders(r, range(4, 8)));
  const wev = su - (sa + sb);
  const play = suFills.length ? 1 : 0;
  const wit = 0;
  const cop = (crossing(rba, rsb) || crossing(rbb, rsa)) ? 0 : -1;
  const s3 = (play + wit + cop + 3) % 3;
  const legit = (play === 1 && cop === -1) ? 1 : 0;
  legitN += legit; wevsum += wev;
  let fp = 0n;
  for (const [p, q, b, s] of suFills) {
    fp ^= fin((BigInt(p) << 40n) ^ (BigInt(q) << 20n) ^ (BigInt(b) << 8n) ^ BigInt(s));
  }
  fp ^= fin((BigInt(wev) << 8n) ^ BigInt(s3) ^ (BigInt(legit) << 4n));
  total ^= fin(fp ^ BigInt(r));
  const clearstr = suFills.length ? String(suFills[suFills.length - 1][0]) : "-";
  console.log(`r=${r} clear=${clearstr} fills=${suFills.length} legs=${play},${wit},${cop} sum3=${s3} wev=${wev} fp=${hex16(fp)}`);
}
console.log(`TOTAL fp=${hex16(total)} legit=${legitN}/16 wev=${wevsum}`);
