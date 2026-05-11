module KripkeWorlds

export KripkeFrame, World, accessible, truth_at, necessity, possibility
export ModalProposition, box, diamond, verify_modal_laws
export SheafSemantics, local_truth, global_sections, stalk_at
export world_kripke, run_kripke_tests

struct World
    name::Symbol
    valuation::Dict{Symbol, Bool}
end

World(name::Symbol; valuation=Dict{Symbol, Bool}()) =
    World(name, Dict{Symbol, Bool}(valuation))

struct KripkeFrame
    worlds::Vector{World}
    relation::Dict{Symbol, Vector{Symbol}}
end

KripkeFrame(worlds::Vector{World}) = KripkeFrame(worlds, Dict{Symbol, Vector{Symbol}}())

struct ModalProposition
    name::Symbol
end

struct SheafSemantics
    frame::KripkeFrame
end

world_name(w::World) = w.name
world_name(w::Symbol) = w

function world_by_name(frame::KripkeFrame, name::Symbol)
    found = findfirst(w -> w.name == name, frame.worlds)
    found === nothing && return nothing
    return frame.worlds[found]
end

function accessible(frame::KripkeFrame, world)
    names = get(frame.relation, world_name(world), Symbol[])
    worlds = World[]
    for name in names
        w = world_by_name(frame, name)
        w === nothing || push!(worlds, w)
    end
    return worlds
end

function truth_at(world::World, prop::Symbol)
    return get(world.valuation, prop, false)
end

truth_at(world::World, prop::ModalProposition) = truth_at(world, prop.name)

function truth_at(frame::KripkeFrame, world, prop)
    w = world isa World ? world : world_by_name(frame, world_name(world))
    w === nothing && return false
    return truth_at(w, prop)
end

function box(frame::KripkeFrame, world, prop)
    worlds = accessible(frame, world)
    isempty(worlds) && return true
    return all(w -> truth_at(frame, w, prop), worlds)
end

function diamond(frame::KripkeFrame, world, prop)
    return any(w -> truth_at(frame, w, prop), accessible(frame, world))
end

necessity(frame::KripkeFrame, world, prop) = box(frame, world, prop)
possibility(frame::KripkeFrame, world, prop) = diamond(frame, world, prop)

local_truth(sheaf::SheafSemantics, world, prop) = truth_at(sheaf.frame, world, prop)
global_sections(sheaf::SheafSemantics, prop) =
    [w for w in sheaf.frame.worlds if truth_at(w, prop)]
stalk_at(sheaf::SheafSemantics, world) =
    world isa World ? world.valuation : truth_at(sheaf.frame, world, :__missing__)

function verify_modal_laws()
    w = World(:w; valuation=Dict(:p => true))
    frame = KripkeFrame([w], Dict(:w => [:w]))
    return box(frame, w, :p) && diamond(frame, w, :p)
end

function world_kripke()
    w = World(:root; valuation=Dict(:true => true))
    return KripkeFrame([w], Dict(:root => [:root]))
end

run_kripke_tests() = verify_modal_laws()

end
