# sweep_partitions.jl — jurisdictional sweep over all 35 bipartitions of the
# 8 agents into two pools of 4, per round. Tests the two theorem-shadows from
# the legitimacy/justice formalization (see SPEC.md):
#
#   T1 (arb => injustice):        cop == 0  =>  wev > 0
#   T2 (SWT shadow, Rawls):       wev == 0  =>  cop == -1   (contrapositive of T1)
#   Asymmetry (Rawls):            cop == -1 && wev > 0 exists (legitimate-but-unjust)
#
# A single observed (cop=0, wev=0) cell refutes T1/T2 for this model.
# Rides the externally-legitimated kernel (include, no re-implementation).

include(joinpath(@__DIR__, "legitimacy_kernel.jl"))

# all unordered bipartitions {A,B} of 0:7 with |A|=4: fix agent 0 in A.
function bipartitions()
    parts = Tuple{Vector{Int},Vector{Int}}[]
    for c in Iterators.filter(x -> length(x) == 3, collect(powerset_3(1:7)))
        A = sort(vcat([0], collect(c)))
        B = sort(setdiff(0:7, A))
        push!(parts, (A, B))
    end
    parts
end
# minimal 3-subsets enumerator (no Combinatorics dep)
function powerset_3(xs)
    xs = collect(xs); out = Vector{Vector{Int}}()
    n = length(xs)
    for a in 1:n-2, b in a+1:n-1, c in b+1:n
        push!(out, [xs[a], xs[b], xs[c]])
    end
    out
end

function audit(r::Int, A, B)
    _, su, _, _ = clear(orders(r, 0:7)...)
    _, sa, rsa, rba = clear(orders(r, A)...)
    _, sb, rsb, rbb = clear(orders(r, B)...)
    wev = su - (sa + sb)
    cop = (crossing(rba, rsb) || crossing(rbb, rsa)) ? 0 : -1
    (wev = wev, cop = cop)
end

function main_sweep()
    parts = bipartitions()
    @assert length(parts) == 35
    t1_viol = 0        # cop==0 && wev==0   (refutes T1/T2)
    leg_unjust = 0     # cop==-1 && wev>0   (legitimate-but-unjust)
    arb_unjust = 0     # cop==0  && wev>0
    just_legit = 0     # wev==0  && cop==-1 (just and legitimate)
    println(rpad("round", 6), rpad("wev_min", 9), rpad("wev_max", 9),
            rpad("n(cop=0)", 10), rpad("minwev_cop", 11), "classes")
    for r in 0:15
        res = [audit(r, A, B) for (A, B) in parts]
        wmin, wmax = extrema(x -> x.wev, res)
        imin = findfirst(x -> x.wev == wmin, res)
        n_arb = count(x -> x.cop == 0, res)
        for x in res
            if x.cop == 0 && x.wev == 0
                t1_viol += 1
            elseif x.cop == -1 && x.wev > 0
                leg_unjust += 1
            elseif x.cop == 0 && x.wev > 0
                arb_unjust += 1
            elseif x.wev == 0 && x.cop == -1
                just_legit += 1
            end
        end
        println(rpad(r, 6), rpad(wmin, 9), rpad(wmax, 9), rpad(n_arb, 10),
                rpad(res[imin].cop, 11),
                "arb+unjust=$(count(x -> x.cop == 0 && x.wev > 0, res)) " *
                "legit+unjust=$(count(x -> x.cop == -1 && x.wev > 0, res)) " *
                "just+legit=$(count(x -> x.wev == 0 && x.cop == -1, res))")
    end
    total = 16 * 35
    println("=" ^ 72)
    println("cells: $(total)   T1 violations (cop=0 ∧ wev=0): $(t1_viol)")
    println("legitimate-but-unjust (cop=-1 ∧ wev>0): $(leg_unjust)")
    println("arb-and-unjust        (cop=0  ∧ wev>0): $(arb_unjust)")
    println("just-and-legitimate   (wev=0  ∧ cop=-1): $(just_legit)")
    println(t1_viol == 0 ?
        "T1/T2 HOLD: wev=0 ⇒ cop=-1 in every cell (SWT shadow / Rawls: just ⇒ legitimate)" :
        "T1/T2 REFUTED: $(t1_viol) cells with live arb at zero WEV")
    println(leg_unjust > 0 ?
        "ASYMMETRY OBSERVED: $(leg_unjust) legitimate-but-unjust cells (legitimate ⇏ just)" :
        "ASYMMETRY NOT OBSERVED in this seed")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_sweep()
end
