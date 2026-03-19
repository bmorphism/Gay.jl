# OBLIGATION CLEARING: Sardex/Fleischman Mutual Credit as GF(3) Conservation
# =============================================================================
#
# "Clearing IS conservation. Zero-sum IS trit balance."
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  OBLIGATION CLEARING VIA CYCLE DECOMPOSITION                                │
# │                                                                             │
# │  STRUCTURE (Fleischman, Dini, Littera 2020):                                │
# │    ObligationNetwork : directed weighted graph of invoices                  │
# │    ClearingCycle     : directed cycle where debts cancel                    │
# │    NettingSolution   : set of clearing cycles → max debt reduction          │
# │    MutualCreditUnit  : complementary liquidity (Sardex, WIR)               │
# │                                                                             │
# │  ISOMORPHISMS:                                                              │
# │    zero-sum (credits = debits) ≅ GF(3) conservation (Σtrits ≡ 0 mod 3)    │
# │    clearing cycle finding ≅ decide_sheaf_tree_shape on obligation bags     │
# │    set-off (erasure) ≅ Landauer kT ln 2 per bit                           │
# │    creditor(+1) / clearer(0) / debtor(-1) ≅ generator/ergodic/absorber    │
# │                                                                             │
# │  DATA (Sardex 2019):                                                        │
# │    3,199 firms, 138,378 transactions, EUR 31M                              │
# │    obligation-clearing alone → 25% debt reduction                           │
# │    + mutual credit → 50% debt reduction                                     │
# │    Slovenia TCT: running since 1991, 7.58% GDP at 1992 peak                │
# │                                                                             │
# │  PRIVACY (Cycles Protocol, Buchman et al. 2025):                            │
# │    No party sees full graph. Each node verifies local SPI.                  │
# │    XOR fingerprint = order-independent conservation check.                  │
# │    arXiv:2507.22309                                                         │
# └─────────────────────────────────────────────────────────────────────────────┘

module ObligationClearing

using SplittableRandoms: SplittableRandom, split

export ObligationNetwork, Obligation, ClearingCycle, NettingSolution,
       MutualCreditUnit, ClearingResult,
       add_obligation!, find_clearing_cycles, clear!,
       xor_clearing_fingerprint, verify_zero_sum, landauer_clearing_cost,
       trit_role, debt_reduction_ratio

@enum TritRole CREDITOR=1 CLEARER=0 DEBTOR=-1

struct Obligation
    from::Symbol
    to::Symbol
    amount::Float64
end

struct MutualCreditUnit
    name::Symbol
    liquidity::Float64
end

mutable struct ObligationNetwork
    obligations::Vector{Obligation}
    firms::Set{Symbol}
    mutual_credit::Union{Nothing, MutualCreditUnit}
end

ObligationNetwork() = ObligationNetwork(Obligation[], Set{Symbol}(), nothing)

function add_obligation!(net::ObligationNetwork, from::Symbol, to::Symbol, amount::Float64)
    push!(net.obligations, Obligation(from, to, amount))
    push!(net.firms, from)
    push!(net.firms, to)
    net
end

function inject_mutual_credit!(net::ObligationNetwork, name::Symbol, liquidity::Float64)
    net.mutual_credit = MutualCreditUnit(name, liquidity)
    net
end

struct ClearingCycle
    firms::Vector{Symbol}
    amount::Float64
end

struct ClearingResult
    cycles_found::Vector{ClearingCycle}
    total_cleared::Float64
    initial_debt::Float64
    reduction_ratio::Float64
    zero_sum_conserved::Bool
    xor_fingerprint::UInt32
end

function adjacency(net::ObligationNetwork)
    adj = Dict{Symbol, Dict{Symbol, Float64}}()
    for o in net.obligations
        if !haskey(adj, o.from)
            adj[o.from] = Dict{Symbol, Float64}()
        end
        adj[o.from][o.to] = get(adj[o.from], o.to, 0.0) + o.amount
    end
    adj
end

function total_debt(net::ObligationNetwork)::Float64
    sum(o.amount for o in net.obligations)
end

function find_clearing_cycles(net::ObligationNetwork; max_length::Int=6)::Vector{ClearingCycle}
    adj = adjacency(net)
    firms = collect(net.firms)
    cycles = ClearingCycle[]

    for start in firms
        _dfs_cycles!(cycles, adj, [start], start, max_length)
    end

    unique_cycles = ClearingCycle[]
    seen = Set{UInt64}()
    for c in cycles
        h = hash(sort(c.firms))
        if h ∉ seen
            push!(seen, h)
            push!(unique_cycles, c)
        end
    end
    unique_cycles
end

function _dfs_cycles!(cycles, adj, path, target, max_len)
    current = last(path)
    if !haskey(adj, current)
        return
    end
    for (next, amt) in adj[current]
        if next == target && length(path) > 2
            cycle_amt = minimum(
                get(get(adj, path[i], Dict()), path[i == length(path) ? 1 : i+1], 0.0)
                for i in 1:length(path)
            )
            if cycle_amt > 0
                push!(cycles, ClearingCycle(copy(path), cycle_amt))
            end
        elseif next ∉ path && length(path) < max_len
            push!(path, next)
            _dfs_cycles!(cycles, adj, path, target, max_len)
            pop!(path)
        end
    end
end

function clear!(net::ObligationNetwork)::ClearingResult
    initial = total_debt(net)
    cycles = find_clearing_cycles(net)

    cleared = 0.0
    for cycle in cycles
        amt = cycle.amount
        for i in 1:length(cycle.firms)
            from = cycle.firms[i]
            to = cycle.firms[i == length(cycle.firms) ? 1 : i+1]
            for (idx, o) in enumerate(net.obligations)
                if o.from == from && o.to == to && o.amount >= amt
                    net.obligations[idx] = Obligation(from, to, o.amount - amt)
                    break
                end
            end
        end
        cleared += amt * length(cycle.firms)
    end

    filter!(o -> o.amount > 0, net.obligations)
    remaining = total_debt(net)
    ratio = initial > 0 ? (initial - remaining) / initial : 0.0

    fp = xor_clearing_fingerprint(net)
    zs = verify_zero_sum(net)

    ClearingResult(cycles, cleared, initial, ratio, zs, fp)
end

function xor_clearing_fingerprint(net::ObligationNetwork)::UInt32
    fp = UInt32(0)
    for o in net.obligations
        h = hash((o.from, o.to, o.amount))
        fp = xor(fp, UInt32(h & 0xFFFFFFFF))
    end
    fp
end

function verify_zero_sum(net::ObligationNetwork)::Bool
    balances = Dict{Symbol, Float64}()
    for o in net.obligations
        balances[o.from] = get(balances, o.from, 0.0) - o.amount
        balances[o.to] = get(balances, o.to, 0.0) + o.amount
    end
    abs(sum(values(balances))) < 1e-10
end

function landauer_clearing_cost(bits_erased::Int; temperature_K::Float64=300.0)::Float64
    k_B = 1.380649e-23
    bits_erased * k_B * temperature_K * log(2)
end

function trit_role(firm::Symbol, net::ObligationNetwork)::TritRole
    outgoing = sum(o.amount for o in net.obligations if o.from == firm; init=0.0)
    incoming = sum(o.amount for o in net.obligations if o.to == firm; init=0.0)
    balance = incoming - outgoing
    if balance > 0
        CREDITOR
    elseif balance < 0
        DEBTOR
    else
        CLEARER
    end
end

function debt_reduction_ratio(result::ClearingResult)::Float64
    result.reduction_ratio
end

end # module
