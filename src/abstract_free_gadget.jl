# Auto-generated stub for abstract_free_gadget.jl
module FreeGadgetBridge

export AbstractMC, AbstractGayMC, AbstractFreeGadgetType, FreeGadget, ThreeMatchFreeGadget, EdgeFreeGadget, ThreadFreeGadget, GadgetGraph, GadgetVertex, GadgetEdge, add_gadget!, connect_gadgets!, gadget_degree, most_connected, sample_gadget!, walk_gadgets!, gadget_color, gadget_fingerprint, gadget_seed, gadget_arity, gadget_apply, TritDirection, TRIT_BACK, TRIT_STAY, TRIT_FORWARD, TritWalk, TritWalkState, TritParallelConfig, trit_parallel_walk!, trit_step!, merge_trit_walks, ThreadGadgetCandidate, extract_thread_gadgets, rank_by_connectivity, find_most_connected_thread, GayMCBridge, bridge_sample!, bridge_walk!, demo_abstract_free_gadget, demo_trit_walk

# Stub definitions
struct AbstractMC end
struct AbstractGayMC end
struct AbstractFreeGadgetType end
struct FreeGadget end
struct ThreeMatchFreeGadget end
struct EdgeFreeGadget end
struct ThreadFreeGadget end
struct GadgetGraph end
struct GadgetVertex end
struct GadgetEdge end
add_gadget!(args...; kwargs...) = nothing
connect_gadgets!(args...; kwargs...) = nothing
gadget_degree(args...; kwargs...) = nothing
most_connected(args...; kwargs...) = nothing
sample_gadget!(args...; kwargs...) = nothing
walk_gadgets!(args...; kwargs...) = nothing
gadget_color(args...; kwargs...) = nothing
gadget_fingerprint(args...; kwargs...) = nothing
gadget_seed(args...; kwargs...) = nothing
gadget_arity(args...; kwargs...) = nothing
gadget_apply(args...; kwargs...) = nothing
struct TritDirection end
struct TRIT_BACK end
struct TRIT_STAY end
struct TRIT_FORWARD end
struct TritWalk end
struct TritWalkState end
struct TritParallelConfig end
trit_parallel_walk!(args...; kwargs...) = nothing
trit_step!(args...; kwargs...) = nothing
merge_trit_walks(args...; kwargs...) = nothing
struct ThreadGadgetCandidate end
extract_thread_gadgets(args...; kwargs...) = nothing
rank_by_connectivity(args...; kwargs...) = nothing
find_most_connected_thread(args...; kwargs...) = nothing
struct GayMCBridge end
bridge_sample!(args...; kwargs...) = nothing
bridge_walk!(args...; kwargs...) = nothing
demo_abstract_free_gadget(args...; kwargs...) = nothing
demo_trit_walk(args...; kwargs...) = nothing

end # module FreeGadgetBridge
