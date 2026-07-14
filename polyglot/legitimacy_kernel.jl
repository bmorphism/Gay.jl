# Polyglot legitimacy kernel — Julia implementation. See SPEC.md.
const G  = 0x9e3779b97f4a7c15
const M1 = 0xbf58476d1ce4e5b9
const M2 = 0x94d049bb133111eb

fin(z::UInt64) = begin
    z = (z ⊻ (z >> 30)) * M1
    z = (z ⊻ (z >> 27)) * M2
    z ⊻ (z >> 31)
end
sm(seed::UInt64, k::UInt64) = fin(seed + G * k)

const SEED = UInt64(1069); const N = 8; const T = 16

function orders(rnd::Int, agents)
    sellers = NTuple{3,Int}[]; buyers = NTuple{3,Int}[]
    m = rnd % 16
    amp = m <= 8 ? m * 64 : (16 - m) * 64
    grid = 1000 + amp
    for i in agents
        h = sm(SEED, (UInt64(i) << 32) + UInt64(rnd))
        u1 = Int((h >> 16) & 0xFF); u2 = Int((h >> 8) & 0xFF); u3 = Int(h & 0xFF)
        load = 100 + Int(sm(SEED, UInt64(0xA000 + i)) & 0xFF)
        rad = (amp * (200 + 10 * i)) >> 8
        q = rad - load + (u1 - 128)
        if q > 0
            push!(sellers, ((grid * (115 + (u2 >> 1))) >> 8, i, q))
        elseif q < 0
            push!(buyers, ((grid * (179 + (u3 >> 1))) >> 8, i, -q))
        end
    end
    sort!(sellers, by = t -> (t[1], t[2]))
    sort!(buyers,  by = t -> (-t[1], t[2]))
    sellers, buyers
end

function clear(sellers, buyers)
    fills = NTuple{4,Int}[]; surplus = 0; si = 1; bi = 1
    srem = isempty(sellers) ? 0 : sellers[1][3]
    brem = isempty(buyers)  ? 0 : buyers[1][3]
    while si <= length(sellers) && bi <= length(buyers) && buyers[bi][1] >= sellers[si][1]
        take = min(brem, srem)
        price = (buyers[bi][1] + sellers[si][1]) >> 1
        push!(fills, (price, take, buyers[bi][2], sellers[si][2]))
        surplus += (buyers[bi][1] - sellers[si][1]) * take
        brem -= take; srem -= take
        if brem == 0
            bi += 1
            bi <= length(buyers) && (brem = buyers[bi][3])
        end
        if srem == 0
            si += 1
            si <= length(sellers) && (srem = sellers[si][3])
        end
    end
    rs = si <= length(sellers) ? vcat([(sellers[si][1], sellers[si][2], srem)], sellers[si+1:end]) : NTuple{3,Int}[]
    rb = bi <= length(buyers)  ? vcat([(buyers[bi][1], buyers[bi][2], brem)],  buyers[bi+1:end])  : NTuple{3,Int}[]
    fills, surplus, rs, rb
end

crossing(rb, rs) = !isempty(rb) && !isempty(rs) && rb[1][1] >= rs[1][1]

function main()
    total = UInt64(0); legit_n = 0; wevsum = 0
    for r in 0:(T-1)
        su_fills, su, _, _ = clear(orders(r, 0:(N-1))...)
        _, sa, rsa, rba = clear(orders(r, 0:3)...)
        _, sb, rsb, rbb = clear(orders(r, 4:7)...)
        wev = su - (sa + sb)
        play = isempty(su_fills) ? 0 : 1
        wit = 0
        cop = (crossing(rba, rsb) || crossing(rbb, rsa)) ? 0 : -1
        s3 = mod(play + wit + cop + 3, 3)
        legit = (play == 1 && cop == -1) ? 1 : 0
        legit_n += legit; wevsum += wev
        fp = UInt64(0)
        for (p, q, b, s) in su_fills
            fp ⊻= fin((UInt64(p) << 40) ⊻ (UInt64(q) << 20) ⊻ (UInt64(b) << 8) ⊻ UInt64(s))
        end
        fp ⊻= fin((UInt64(wev) << 8) ⊻ UInt64(s3) ⊻ (UInt64(legit) << 4))
        total ⊻= fin(fp ⊻ UInt64(r))
        clearstr = isempty(su_fills) ? "-" : string(su_fills[end][1])
        println("r=$(r) clear=$(clearstr) fills=$(length(su_fills)) legs=$(play),$(wit),$(cop) sum3=$(s3) wev=$(wev) fp=$(string(fp, base=16, pad=16))")
    end
    println("TOTAL fp=$(string(total, base=16, pad=16)) legit=$(legit_n)/16 wev=$(wevsum)")
end

main()
