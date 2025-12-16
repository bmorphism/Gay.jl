# Auto-generated stub for gamut_learnable.jl
module GamutLearnable

export GamutConstraint, GaySRGBGamut, GayP3Gamut, GayRec2020Gamut, LearnableGamutMap, GamutParameters, map_to_gamut, is_in_gamut, gamut_distance, learn_gamut_map!, gamut_loss, chroma_preservation_loss, GayChain, chain_to_gamut, verify_chain_in_gamut, process_gay_chain, enzyme_gamut_gradient, enzyme_learn_gamut!

# Stub definitions
struct GamutConstraint end
struct GaySRGBGamut end
struct GayP3Gamut end
struct GayRec2020Gamut end
struct LearnableGamutMap end
struct GamutParameters end
map_to_gamut(args...; kwargs...) = nothing
is_in_gamut(args...; kwargs...) = nothing
gamut_distance(args...; kwargs...) = nothing
learn_gamut_map!(args...; kwargs...) = nothing
gamut_loss(args...; kwargs...) = nothing
chroma_preservation_loss(args...; kwargs...) = nothing
struct GayChain end
chain_to_gamut(args...; kwargs...) = nothing
verify_chain_in_gamut(args...; kwargs...) = nothing
process_gay_chain(args...; kwargs...) = nothing
enzyme_gamut_gradient(args...; kwargs...) = nothing
enzyme_learn_gamut!(args...; kwargs...) = nothing

end # module GamutLearnable
