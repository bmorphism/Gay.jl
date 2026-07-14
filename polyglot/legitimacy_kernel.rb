#!/usr/bin/env ruby
# Polyglot legitimacy kernel — Ruby implementation. See SPEC.md.
MASK = (1 << 64) - 1
G  = 0x9e3779b97f4a7c15
M1 = 0xbf58476d1ce4e5b9
M2 = 0x94d049bb133111eb

def fin(z)
  z &= MASK
  z = ((z ^ (z >> 30)) * M1) & MASK
  z = ((z ^ (z >> 27)) * M2) & MASK
  (z ^ (z >> 31)) & MASK
end

def sm(seed, k)
  fin((seed + G * k) & MASK)
end

SEED = 1069
N = 8
T = 16

def orders(rnd, agents)
  sellers = []
  buyers = []
  m = rnd % 16
  amp = m <= 8 ? m * 64 : (16 - m) * 64
  grid = 1000 + amp
  agents.each do |i|
    h = sm(SEED, ((i << 32) + rnd) & MASK)
    u1 = (h >> 16) & 0xFF
    u2 = (h >> 8) & 0xFF
    u3 = h & 0xFF
    load = 100 + (sm(SEED, 0xA000 + i) & 0xFF)
    rad = (amp * (200 + 10 * i)) >> 8
    q = rad - load + (u1 - 128)
    if q > 0
      sellers << [(grid * (115 + (u2 >> 1))) >> 8, i, q]
    elsif q < 0
      buyers << [(grid * (179 + (u3 >> 1))) >> 8, i, -q]
    end
  end
  sellers.sort_by! { |t| [t[0], t[1]] }
  buyers.sort_by! { |t| [-t[0], t[1]] }
  [sellers, buyers]
end

def clear(sellers, buyers)
  fills = []
  surplus = 0
  si = 0
  bi = 0
  srem = sellers.empty? ? 0 : sellers[0][2]
  brem = buyers.empty? ? 0 : buyers[0][2]
  while si < sellers.length && bi < buyers.length && buyers[bi][0] >= sellers[si][0]
    take = [brem, srem].min
    price = (buyers[bi][0] + sellers[si][0]) >> 1
    fills << [price, take, buyers[bi][1], sellers[si][1]]
    surplus += (buyers[bi][0] - sellers[si][0]) * take
    brem -= take
    srem -= take
    if brem == 0
      bi += 1
      brem = buyers[bi][2] if bi < buyers.length
    end
    if srem == 0
      si += 1
      srem = sellers[si][2] if si < sellers.length
    end
  end
  rs = si < sellers.length ? [[sellers[si][0], sellers[si][1], srem]] + sellers[(si + 1)..] : []
  rb = bi < buyers.length ? [[buyers[bi][0], buyers[bi][1], brem]] + buyers[(bi + 1)..] : []
  [fills, surplus, rs, rb]
end

def crossing(rb, rs)
  !rb.empty? && !rs.empty? && rb[0][0] >= rs[0][0]
end

total = 0
legit_n = 0
wevsum = 0
(0...T).each do |r|
  su_fills, su, _, _ = clear(*orders(r, (0...N).to_a))
  _, sa, rsa, rba = clear(*orders(r, (0..3).to_a))
  _, sb, rsb, rbb = clear(*orders(r, (4..7).to_a))
  wev = su - (sa + sb)
  play = su_fills.empty? ? 0 : 1
  wit = 0
  cop = (crossing(rba, rsb) || crossing(rbb, rsa)) ? 0 : -1
  s3 = (play + wit + cop + 3) % 3
  legit = (play == 1 && cop == -1) ? 1 : 0
  legit_n += legit
  wevsum += wev
  fp = 0
  su_fills.each do |(p, q, b, s)|
    fp ^= fin((p << 40) ^ (q << 20) ^ (b << 8) ^ s)
  end
  fp ^= fin((wev << 8) ^ s3 ^ (legit << 4))
  total ^= fin(fp ^ r)
  clearstr = su_fills.empty? ? "-" : su_fills[-1][0].to_s
  puts "r=#{r} clear=#{clearstr} fills=#{su_fills.length} legs=#{play},#{wit},#{cop} sum3=#{s3} wev=#{wev} fp=#{format('%016x', fp)}"
end
puts "TOTAL fp=#{format('%016x', total)} legit=#{legit_n}/16 wev=#{wevsum}"
