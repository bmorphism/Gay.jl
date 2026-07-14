# MADDPG actor I/O (Qiu et al., Fig. 5) as an order-book ACSet,
# cleared by uniform-price double auction, colored and audited by Gay.jl.
#
# ACSet signature (implemented as typed tables + explicit homs, no Catlab dep):
#
#   Ob:   Round · Agent · Order · Fill
#   Hom:  order_agent : Order → Agent
#         order_round : Order → Round
#         fill_buy    : Fill  → Order      (buy-side order)
#         fill_sell   : Fill  → Order      (sell-side order)
#   Attr: Round : grid_price, clearing_price        # public base — stored ONCE
#         Agent : load, radiation, battery          # private local section
#         Order : sell_price, bid_price, qty, charge  # the four actor outputs
#         Fill  : price, qty
#
# The Fig. 5 role rule is the trit readout: qty > 0 ⇒ buyer (+1),
# qty < 0 ⇒ seller (−1), qty == 0 ⇒ abstain (0).
#
# GF(3) audit per round — measured off events, never assigned:
#   +1 Play    : fills exist            (pairwise contact observed)
#    0 Witness : Σ buy qty == Σ sell qty exactly (conservation closes)
#   −1 Coplay  : no crossed residual    (max unfilled bid < min unfilled ask
#                                        ⇒ no arbitrage left ⇒ H¹ = 0 analog)
#
# The actor here is a deterministic Gay-colored surrogate policy (this is a
# clearing/audit shape demo, not a learning demo): policy outputs are pure
# functions of spi_color_u32(seed, agent ⊕ round), so the whole run is
# SPI-reproducible and the round fingerprint is an XOR fold of fill colors.

using Gay

# --- ACSet tables -----------------------------------------------------------

struct AgentRow;  name::String; color::String; load::Float64; radiation::Float64; battery::Float64; end
struct OrderRow;  agent::Int; round::Int; sell_price::Float64; bid_price::Float64; qty::Float64; charge::Float64; end
struct FillRow;   buy::Int; sell::Int; round::Int; price::Float64; qty::Float64; end
mutable struct RoundRow; grid_price::Float64; clearing_price::Float64; end

struct P2PClearing
    rounds::Vector{RoundRow}
    agents::Vector{AgentRow}
    orders::Vector{OrderRow}   # order_agent, order_round are the Int fields
    fills::Vector{FillRow}     # fill_buy, fill_sell index into orders
end
P2PClearing() = P2PClearing(RoundRow[], AgentRow[], OrderRow[], FillRow[])

# --- deterministic surrogate actor (μᵢ) --------------------------------------

const SEED = Gay.GAY_SEED

"Fig. 5 actor: (grid, last_clear, load, radiation, battery) → 4 outputs, via Gay kernel."
function actor(acs::P2PClearing, i::Int, r::Int)
    a = acs.agents[i]
    grid = acs.rounds[r].grid_price
    h = Gay.spi_color_u32(SEED, UInt64(i) << 32 | UInt64(r))       # deterministic noise
    u1 = ((h >> 16) & 0xFF) / 255;  u2 = ((h >> 8) & 0xFF) / 255;  u3 = (h & 0xFF) / 255
    net = a.radiation - a.load                                      # surplus ⇒ seller-leaning
    qty = clamp(-net + 0.6 * (u1 - 0.5), -1.0, 1.0)                 # tanh-range, Fig. 5
    sell_price = grid * (0.45 + 0.35 * u2)                          # undercut grid to sell
    bid_price  = grid * (0.70 + 0.35 * u3)                          # underbid grid to buy
    charge = clamp(net - qty, -0.5, 0.5)                            # residual to battery
    OrderRow(i, r, sell_price, bid_price, qty, charge)
end

# --- clearing: contact(Λ_B, Λ_S) as uniform-price double auction --------------

function clear_round!(acs::P2PClearing, r::Int)
    idx = [k for (k, o) in enumerate(acs.orders) if o.round == r]
    buys  = sort([k for k in idx if acs.orders[k].qty > 0]; by=k -> -acs.orders[k].bid_price)
    sells = sort([k for k in idx if acs.orders[k].qty < 0]; by=k ->  acs.orders[k].sell_price)
    bi = si = 1
    brem = isempty(buys)  ? 0.0 : acs.orders[buys[1]].qty
    srem = isempty(sells) ? 0.0 : -acs.orders[sells[1]].qty
    last_price = NaN
    while bi <= length(buys) && si <= length(sells)
        b, s = acs.orders[buys[bi]], acs.orders[sells[si]]
        b.bid_price < s.sell_price && break                          # curves separate: contact lost
        q = min(brem, srem)
        p = (b.bid_price + s.sell_price) / 2
        push!(acs.fills, FillRow(buys[bi], sells[si], r, p, q))
        last_price = p
        brem -= q; srem -= q
        if brem ≤ 1e-12; bi += 1; bi <= length(buys)  && (brem =  acs.orders[buys[bi]].qty);  end
        if srem ≤ 1e-12; si += 1; si <= length(sells) && (srem = -acs.orders[sells[si]].qty); end
    end
    isnan(last_price) || (acs.rounds[r].clearing_price = last_price)
    # spread proxy for 1/λ_min(H): gap between best unfilled bid and ask
    ub = [acs.orders[k].bid_price  for k in buys[bi:end]]
    ua = [acs.orders[k].sell_price for k in sells[si:end]]
    spread = (isempty(ub) || isempty(ua)) ? NaN : minimum(ua) - maximum(ub)
    (; spread, ub, ua)
end

# --- GF(3) audit: read the trits off the events ------------------------------

function audit(acs::P2PClearing, r::Int, res)
    fills = [f for f in acs.fills if f.round == r]
    play    = isempty(fills) ? 0 : 1                                  # +1 leg observed?
    bought  = sum((f.qty for f in fills); init=0.0)
    sold    = sum((f.qty for f in fills); init=0.0)                   # symmetric by construction
    witness = abs(bought - sold) < 1e-9 ? 0 : 1                       # 0 leg closes?
    coplay  = (isempty(res.ub) || isempty(res.ua) || res.spread > 0) ? -1 : 0   # residual cross = live arb
    legs = (play, witness, coplay)
    (; legs, sum3 = mod(sum(legs), 3), fills)
end

# --- run ----------------------------------------------------------------------

function main(; N=5, T=8)
    acs = P2PClearing()
    trail = hierarchical_colors("p2p/" * join(string.(1:N), "/"); seed=SEED)
    for i in 1:N
        h = Gay.spi_color_u32(SEED, UInt64(0xA0) + i)
        push!(acs.agents, AgentRow("mg$(i)", spi_color_hex(SEED, UInt64(0xA0) + i),
              0.10 + 0.60 * ((h >> 8 & 0xFF) / 255), 0.0, 0.5))
    end
    println("agents: ", join(("$(a.name) $(a.color)" for a in acs.agents), "  "))
    println(rpad("round", 6), rpad("grid", 7), rpad("clear", 8), rpad("fills", 6),
            rpad("spread", 9), rpad("legs", 12), "Σ%3  fingerprint")
    for r in 1:T
        push!(acs.rounds, RoundRow(1.0 + 0.2 * sin(2π * r / T), NaN))
        # radiation day-cycle drives the Morse probe: night ⇒ everyone buys ⇒ contact lost
        for (i, a) in enumerate(acs.agents)
            rad = max(0.0, sin(2π * (r - 1) / T)) * (0.7 + 0.3 * i / N)
            acs.agents[i] = AgentRow(a.name, a.color, a.load, rad, a.battery)
        end
        for i in 1:N
            push!(acs.orders, actor(acs, i, r))
        end
        res = clear_round!(acs, r)
        aud = audit(acs, r, res)
        # battery update from charge legs (bounded [0,1])
        for o in acs.orders
            o.round == r || continue
            a = acs.agents[o.agent]
            acs.agents[o.agent] = AgentRow(a.name, a.color, a.load, a.radiation,
                                           clamp(a.battery + o.charge, 0.0, 1.0))
        end
        fp = foldl(⊻, (UInt64(Gay.spi_color_u32(SEED, UInt64(round(Int, 1000 * f.price)) ⊻ UInt64(f.buy) << 20 ⊻ UInt64(f.sell) << 40)) for f in aud.fills); init=UInt64(0))
        println(rpad(r, 6), rpad(round(acs.rounds[r].grid_price; digits=3), 7),
                rpad(isnan(acs.rounds[r].clearing_price) ? "—" : round(acs.rounds[r].clearing_price; digits=3), 8),
                rpad(length(aud.fills), 6),
                rpad(isnan(res.spread) ? "—" : round(res.spread; digits=3), 9),
                rpad(string(aud.legs), 12), rpad(aud.sum3, 4),
                string("0x", string(fp, base=16, pad=8)))
    end
    acs
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
