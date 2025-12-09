# Propagator system inspired by SDF (Sussman & Hanson)
# With Gay.jl chromatic identity for debugging and visualization

module Propagator

using ..Gay: gay_seed!, color_at, next_color, GAY_SEED, GayRNG, gay_split
using SplittableRandoms: SplittableRandom, split

export Cell, make_cell, cell_content, cell_strongest, add_content!
export propagator, primitive_propagator, compound_propagator, constraint_propagator
export Premise, hypothetical, mark_premise_in!, mark_premise_out!, is_premise_in
export SupportSet, support_set, support_set_union, all_premises_in
export TheNothing, TheNothingType, TheContradiction, TheContradictionType, isnothing_prop, is_contradiction
export merge_values, strongest_value
export Scheduler, run!, initialize_scheduler!, alert_propagator!
export p_add, p_mul, p_sub, p_div, c_add, c_mul
export binary_amb, p_amb

# ============================================================
# Special Values (§7.2)
# ============================================================

struct TheNothingType end
const TheNothing = TheNothingType()
isnothing_prop(x) = x isa TheNothingType

struct TheContradictionType
    info::Any
end
TheContradiction() = TheContradictionType(nothing)
TheContradiction(info) = TheContradictionType(info)
is_contradiction(x) = x isa TheContradictionType

# ============================================================
# Premises and Support Sets (§6.4, §7.3)
# ============================================================

mutable struct Premise
    name::Symbol
    is_in::Bool
    is_hypothetical::Bool
    nogoods::Vector{Set{Premise}}
    color::UInt32  # Gay.jl chromatic identity
end

function Premise(name::Symbol; hypothetical::Bool=false)
    # Deterministic color from name hash
    color = UInt32(hash(name) % 0xFFFFFF)
    Premise(name, true, hypothetical, Set{Premise}[], color)
end

hypothetical(name::Symbol) = Premise(name; hypothetical=true)

mark_premise_in!(p::Premise) = (p.is_in = true; p)
mark_premise_out!(p::Premise) = (p.is_in = false; p)
is_premise_in(p::Premise) = p.is_in

struct SupportSet
    premises::Set{Premise}
end

support_set(ps...) = SupportSet(Set(ps))
support_set() = SupportSet(Set{Premise}())

function support_set_union(s1::SupportSet, s2::SupportSet)
    SupportSet(union(s1.premises, s2.premises))
end

function support_set_remove(s::SupportSet, p::Premise)
    SupportSet(setdiff(s.premises, Set([p])))
end

function all_premises_in(s::SupportSet)
    all(is_premise_in, s.premises)
end

# Chromatic identity for support set (XOR of premise colors)
function support_color(s::SupportSet)
    isempty(s.premises) && return 0x000000
    reduce(xor, p.color for p in s.premises)
end

# ============================================================
# Supported Values (§7.4.2)
# ============================================================

struct Supported{T}
    value::T
    support::SupportSet
end

Supported(v) = Supported(v, support_set())
Supported(v, ps::Premise...) = Supported(v, support_set(ps...))

base_value(s::Supported) = s.value
support_layer(s::Supported) = s.support

# ============================================================
# Value Sets (§7.4.3)
# ============================================================

struct ValueSet{T}
    elements::Vector{Supported{T}}
end

ValueSet{T}() where T = ValueSet{T}(Supported{T}[])
value_set(elts...) = ValueSet(collect(elts))

function value_set_adjoin(vs::ValueSet{T}, elt::Supported{T}) where T
    # Don't add if subsumed by existing element
    for old_elt in vs.elements
        if element_subsumes(old_elt, elt)
            return vs
        end
    end
    ValueSet{T}(push!(copy(vs.elements), elt))
end

function element_subsumes(elt1::Supported, elt2::Supported)
    # elt1 subsumes elt2 if:
    # - elt1's value is at least as informative
    # - elt1's support is a subset (smaller = stronger)
    elt1.value == elt2.value && 
    issubset(elt1.support.premises, elt2.support.premises)
end

# ============================================================
# Merging (§7.4)
# ============================================================

"""
    merge_values(content, increment)

Merge two values. Returns TheContradiction if incompatible.
"""
function merge_values(content, increment)
    isnothing_prop(content) && return increment
    isnothing_prop(increment) && return content
    is_contradiction(content) && return content
    is_contradiction(increment) && return increment
    content == increment && return content
    TheContradiction((content, increment))
end

# Merge intervals by intersection
function merge_values(content::Tuple{T,T}, increment::Tuple{T,T}) where T<:Real
    lo = max(content[1], increment[1])
    hi = min(content[2], increment[2])
    lo > hi && return TheContradiction((content, increment))
    (lo, hi)
end

"""
    strongest_value(content)

Extract the strongest fully-supported value from content.
"""
strongest_value(x) = x
strongest_value(::TheNothingType) = TheNothing

function strongest_value(s::Supported)
    all_premises_in(s.support) ? s : TheNothing
end

function strongest_value(vs::ValueSet)
    for elt in vs.elements
        if all_premises_in(elt.support)
            return elt
        end
    end
    TheNothing
end

# ============================================================
# Cells (§7.2.1)
# ============================================================

mutable struct Cell{T}
    name::Symbol
    content::Any  # T | TheNothingType | TheContradictionType | ValueSet{T}
    strongest::Any
    neighbors::Vector{Any}  # Propagators
    color::UInt32  # Gay.jl chromatic identity
    parent::Union{Nothing, Any}  # Parent propagator
end

function make_cell(name::Symbol; parent=nothing)
    color = UInt32(hash(name) % 0xFFFFFF)
    Cell{Any}(name, TheNothing, TheNothing, [], color, parent)
end

cell_content(c::Cell) = c.content
cell_strongest(c::Cell) = c.strongest
cell_neighbors(c::Cell) = c.neighbors

function add_neighbor!(c::Cell, p)
    push!(c.neighbors, p)
end

function add_content!(c::Cell, increment; scheduler=GLOBAL_SCHEDULER[])
    c.content = merge_values(c.content, increment)
    test_content!(c; scheduler)
end

function test_content!(c::Cell; scheduler=GLOBAL_SCHEDULER[])
    new_strongest = strongest_value(c.content)
    if new_strongest == c.strongest
        :content_unchanged
    elseif is_contradiction(new_strongest)
        c.strongest = new_strongest
        handle_contradiction!(c)
        :contradiction
    else
        c.strongest = new_strongest
        for neighbor in c.neighbors
            alert_propagator!(neighbor; scheduler)
        end
        :content_changed
    end
end

function handle_contradiction!(c::Cell)
    # TODO: dependency-directed backtracking (§7.5.1)
    @warn "Contradiction in cell $(c.name)" c.content
end

# ============================================================
# Scheduler (§7.2)
# ============================================================

mutable struct Scheduler
    queue::Vector{Any}
    running::Bool
end

Scheduler() = Scheduler([], false)

const GLOBAL_SCHEDULER = Ref{Scheduler}()

function initialize_scheduler!()
    GLOBAL_SCHEDULER[] = Scheduler()
end

function alert_propagator!(p; scheduler=GLOBAL_SCHEDULER[])
    push!(scheduler.queue, p)
end

function run!(; scheduler=GLOBAL_SCHEDULER[])
    scheduler.running = true
    while !isempty(scheduler.queue) && scheduler.running
        p = popfirst!(scheduler.queue)
        p.activate!()
    end
    scheduler.running = false
    :done
end

# ============================================================
# Propagators (§7.2.2)
# ============================================================

struct PropagatorDef
    name::Symbol
    inputs::Vector{Cell}
    outputs::Vector{Cell}
    activate!::Function
    color::UInt32
    parent::Union{Nothing, Any}
end

function propagator(inputs, outputs, activate!, name::Symbol; parent=nothing)
    color = UInt32(hash(name) % 0xFFFFFF)
    p = PropagatorDef(name, inputs, outputs, activate!, color, parent)
    for cell in inputs
        add_neighbor!(cell, p)
    end
    alert_propagator!(p)
    p
end

function primitive_propagator(f::Function, name::Symbol)
    function make_prop(cells...)
        output = cells[end]
        inputs = collect(cells[1:end-1])
        
        function activate!()
            input_values = [cell_strongest(c) for c in inputs]
            if any(isnothing_prop, input_values) || any(is_contradiction, input_values)
                return :do_nothing
            end
            result = f(input_values...)
            add_content!(output, result)
        end
        
        propagator(inputs, [output], activate!, name)
    end
end

function compound_propagator(inputs, outputs, to_build::Function, name::Symbol)
    built = Ref(false)
    
    function maybe_build()
        if built[]
            return :already_built
        end
        input_values = [cell_strongest(c) for c in inputs]
        if !isempty(inputs) && all(isnothing_prop, input_values)
            return :do_nothing
        end
        to_build()
        built[] = true
        :built
    end
    
    propagator(inputs, outputs, maybe_build, name)
end

function constraint_propagator(to_build::Function, cells, name::Symbol)
    compound_propagator(cells, cells, to_build, name)
end

# ============================================================
# Primitive Propagators
# ============================================================

const p_add = primitive_propagator(+, :p_add)
const p_sub = primitive_propagator(-, :p_sub)
const p_mul = primitive_propagator(*, :p_mul)
const p_div = primitive_propagator(/, :p_div)
const p_sqrt = primitive_propagator(sqrt, :p_sqrt)
const p_square = primitive_propagator(x -> x^2, :p_square)

# ============================================================
# Constraint Propagators (§7.2.2 - c:*)
# ============================================================

function c_add(x, y, sum)
    constraint_propagator([x, y, sum], :c_add) do
        p_add(x, y, sum)
        p_sub(sum, x, y)
        p_sub(sum, y, x)
    end
end

function c_mul(x, y, product)
    constraint_propagator([x, y, product], :c_mul) do
        p_mul(x, y, product)
        p_div(product, x, y)
        p_div(product, y, x)
    end
end

function c_square(x, x2)
    constraint_propagator([x, x2], :c_square) do
        p_square(x, x2)
        p_sqrt(x2, x)  # Note: only positive root
    end
end

# ============================================================
# Choice Propagators (§7.5)
# ============================================================

function binary_amb(cell::Cell)
    true_premise = hypothetical(Symbol("$(cell.name)_true"))
    false_premise = hypothetical(Symbol("$(cell.name)_false"))
    
    # Add both values with their hypothetical supports
    add_content!(cell, Supported(true, true_premise))
    add_content!(cell, Supported(false, false_premise))
    
    function amb_choose()
        # Check nogoods for each premise
        true_blocked = any(all_premises_in, true_premise.nogoods)
        false_blocked = any(all_premises_in, false_premise.nogoods)
        
        if !true_blocked
            mark_premise_in!(true_premise)
            mark_premise_out!(false_premise)
        elseif !false_blocked
            mark_premise_out!(true_premise)
            mark_premise_in!(false_premise)
        else
            mark_premise_out!(true_premise)
            mark_premise_out!(false_premise)
            # TODO: process_contradictions
        end
    end
    
    propagator([cell], [cell], amb_choose, :binary_amb)
end

function p_amb(cell::Cell, values)
    premises = [hypothetical(Symbol("$(cell.name)_$i")) for i in eachindex(values)]
    
    for (v, p) in zip(values, premises)
        add_content!(cell, Supported(v, p))
    end
    
    function amb_choose()
        # Find first unblocked premise
        for p in premises
            blocked = any(all_premises_in, p.nogoods)
            if !blocked
                mark_premise_in!(p)
                for other in premises
                    other !== p && mark_premise_out!(other)
                end
                return
            end
        end
        # All blocked
        for p in premises
            mark_premise_out!(p)
        end
    end
    
    propagator([cell], [cell], amb_choose, :p_amb)
end

# ============================================================
# Visualization with Gay.jl colors
# ============================================================

"""
    propagator_color(p::PropagatorDef)

Get the chromatic identity of a propagator.
"""
propagator_color(p::PropagatorDef) = color_at(p.color)

"""
    cell_color(c::Cell)

Get the chromatic identity of a cell.
"""
cell_color(c::Cell) = color_at(c.color)

"""
    network_palette(cells, propagators)

Generate a consistent color palette for visualizing a propagator network.
"""
function network_palette(cells, propagators)
    seed = reduce(xor, c.color for c in cells; init=GAY_SEED)
    gay_seed!(seed)
    (
        cells = Dict(c.name => cell_color(c) for c in cells),
        propagators = Dict(p.name => propagator_color(p) for p in propagators)
    )
end

end # module Propagator
