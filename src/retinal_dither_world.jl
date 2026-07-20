# Retinal Dither Name World
#
# A compact runtime counterpart to docs/RETINAL_DITHER_NAMES.md and
# artifacts/retinal_dither_name_world.json. This keeps the naming work
# tileable as a Gay.jl world: countable, mergeable, and fingerprintable.

struct RetinalDitherInterpretation
    slug::Symbol
    display::String
    pronunciation_hint::String
    gf3_role::Int8
    role_name::Symbol
    interpretation::String
    biological_pattern::String
    information_geometry::String
    base_hex::String
    gay_index::Int
    chain_seed::UInt64
    chain_hex::Vector{String}
    associations::Vector{String}
    tile_uris::Vector{String}
end

struct RetinalDitherNameWorld
    canonical_slug::Symbol
    formal_name::String
    event_unit::String
    triad_seed::UInt64
    interpretations::Vector{RetinalDitherInterpretation}
    candidates::Vector{Pair{Symbol,String}}
    pairwise_ciede2000::Dict{Tuple{Symbol,Symbol},Float64}
    source_artifact::String
    fingerprint::UInt64
end

Base.length(w::RetinalDitherNameWorld) = length(w.interpretations)

retinal_dither_interpretations(w::RetinalDitherNameWorld) = w.interpretations
retinal_dither_fingerprint(w::RetinalDitherNameWorld)::UInt64 = w.fingerprint
fingerprint(w::RetinalDitherNameWorld)::UInt64 = w.fingerprint

function retinal_dither_color_chains(w::RetinalDitherNameWorld)
    Dict(i.slug => copy(i.chain_hex) for i in w.interpretations)
end

function _retinal_dither_world_fingerprint(
    seed::UInt64,
    interpretations::Vector{RetinalDitherInterpretation},
    candidates::Vector{Pair{Symbol,String}},
)::UInt64
    fp = stable_seed("retinal_dither_name_world"; seed=seed)

    for interp in sort(interpretations; by=x -> String(x.slug))
        fp = xor(fp, stable_seed((interp.slug, interp.base_hex, interp.chain_seed, interp.gf3_role); seed=seed))
        for (index, hex) in enumerate(interp.chain_hex)
            fp = xor(fp, stable_seed((interp.slug, index, hex); seed=interp.chain_seed))
        end
        for uri in interp.tile_uris
            fp = xor(fp, stable_seed((interp.slug, uri); seed=seed))
        end
    end

    for candidate in sort(candidates; by=x -> String(first(x)))
        fp = xor(fp, stable_seed((first(candidate), last(candidate)); seed=seed))
    end

    return splitmix64(fp)
end

function _retinal_dither_make_world(
    canonical_slug::Symbol,
    formal_name::String,
    event_unit::String,
    triad_seed::UInt64,
    interpretations::Vector{RetinalDitherInterpretation},
    candidates::Vector{Pair{Symbol,String}},
    distances::Dict{Tuple{Symbol,Symbol},Float64},
    source_artifact::String,
)::RetinalDitherNameWorld
    fp = _retinal_dither_world_fingerprint(triad_seed, interpretations, candidates)
    RetinalDitherNameWorld(
        canonical_slug,
        formal_name,
        event_unit,
        triad_seed,
        interpretations,
        candidates,
        distances,
        source_artifact,
        fp,
    )
end

function Base.merge(a::RetinalDitherNameWorld, b::RetinalDitherNameWorld)::RetinalDitherNameWorld
    by_slug = Dict{Symbol,RetinalDitherInterpretation}()
    for interp in a.interpretations
        by_slug[interp.slug] = interp
    end
    for interp in b.interpretations
        by_slug[interp.slug] = interp
    end

    interpretations = sort!(collect(values(by_slug)); by=x -> String(x.slug))
    candidates = sort!(unique(vcat(a.candidates, b.candidates)); by=x -> String(first(x)))
    distances = merge(a.pairwise_ciede2000, b.pairwise_ciede2000)
    seed = a.triad_seed == b.triad_seed ? a.triad_seed : splitmix64(xor(a.triad_seed, b.triad_seed))

    _retinal_dither_make_world(
        a.canonical_slug,
        a.formal_name,
        a.event_unit,
        seed,
        interpretations,
        candidates,
        distances,
        a.source_artifact,
    )
end

function world_retinal_dither_names()::RetinalDitherNameWorld
    interpretations = RetinalDitherInterpretation[
        RetinalDitherInterpretation(
            :retinal_dither,
            "Retinal Dither",
            "RET-in-al DITH-er",
            Int8(1),
            :play,
            "Local exploratory sampling: small gaze perturbations create fast perceptual refresh without claiming a full world reset.",
            "A retinal mosaic receives slightly shifted photon samples; downstream circuits compare changes rather than static pixels.",
            "A local chart update on a receptor manifold: high information gain with low motor amplitude.",
            "#AFFB3E",
            619,
            UInt64(4708092008076594128),
            ["#9C5E5C", "#A5E9B0", "#993FC2", "#16A66D", "#99B4FE", "#A322B5", "#8AA2F8"],
            ["sampling", "refresh", "mosaic", "play", "fast update"],
            ["world://retina/dither", "discopy://color/retinal-dither", "scip://gay/retinal_dither"],
        ),
        RetinalDitherInterpretation(
            :mosaic_refresh,
            "Mosaic Refresh",
            "mo-ZAY-ik re-FRESH",
            Int8(0),
            :witness,
            "Shared observation: the same environment is re-read through overlapping receptor neighborhoods.",
            "Cone and rod mosaics tile the field unevenly; refresh semantics come from changing overlap, not a frame clock.",
            "A witness pass over adjacent tiles: the environment is held fixed while sampling basis changes.",
            "#0C068C",
            641,
            UInt64(855570893831936446),
            ["#EF6BB7", "#6F0537", "#A176D6", "#F2CDE7", "#C26315", "#B91F5F", "#913DD2"],
            ["witnessing", "tile overlap", "basis change", "shared now"],
            ["world://retina/mosaic-refresh", "discopy://color/mosaic-refresh", "scip://gay/mosaic_refresh"],
        ),
        RetinalDitherInterpretation(
            :ganglion_sync,
            "Ganglion Sync",
            "GANG-lee-on sink",
            Int8(-1),
            :coplay,
            "Constraint and correction: retinal ganglion outputs synchronize change signals so unfit interpretations decay before resource exhaustion.",
            "Ganglion cells integrate receptive-field contrasts and send sparse change-bearing spikes down the optic nerve.",
            "A coplay regularizer: synchronize only the deltas that survive local inhibition and shared salience.",
            "#F5B6F6",
            493,
            UInt64(13499727420772396991),
            ["#02C534", "#B8D4F6", "#BE8505", "#224614", "#D8B048", "#897778", "#9EB079"],
            ["constraint", "contrast", "synchrony", "inhibition", "survival"],
            ["world://retina/ganglion-sync", "discopy://color/ganglion-sync", "scip://gay/ganglion_sync"],
        ),
    ]

    candidates = Pair{Symbol,String}[
        :retinal_dither => "Recommended canonical term: accurate enough biologically, clear aloud, and good for tileable world updates.",
        :retinal_mosaic_dither => "More formal variant when the receptor tiling matters.",
        :mosaic_refresh => "Good witness-side term for shared re-reading of an environment.",
        :ganglion_sync => "Good coplay-side term for constraint, sparse deltas, and correction.",
        :foveal_flicker => "Useful only when central vision is specifically meant; otherwise too narrow.",
        :ocular_jitter => "Mechanically evocative, but less tied to retinal information geometry.",
    ]

    distances = Dict{Tuple{Symbol,Symbol},Float64}(
        (:retinal_dither, :mosaic_refresh) => 113.63596451748482,
        (:retinal_dither, :ganglion_sync) => 72.45014477675956,
        (:mosaic_refresh, :ganglion_sync) => 67.25885431192907,
    )

    _retinal_dither_make_world(
        :retinal_dither,
        "Retinal Mosaic Dither",
        "dither pulse",
        UInt64(13418111070463605249),
        interpretations,
        candidates,
        distances,
        "artifacts/retinal_dither_name_world.json",
    )
end
