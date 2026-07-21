#!/usr/bin/env julia

using Gay

const STRICT = "--strict" in ARGS
const LIST = "--list" in ARGS
const SOURCES = "--sources" in ARGS

exported = names(Gay; all=false, imported=false)
undefined = filter(name -> !isdefined(Gay, name), exported)
defined_count = length(exported) - length(undefined)

println("exported=$(length(exported))")
println("defined=$defined_count")
println("undefined=$(length(undefined))")

if LIST
    foreach(name -> println("undefined\t", name), sort!(undefined; by=string))
end

if SOURCES
    source_dir = normpath(joinpath(@__DIR__, "..", "src"))
    stub_exports = Set{Symbol}()
    substantive_exports = Set{Symbol}()
    substantive_sources = Dict{Symbol, Vector{String}}()

    for source_file in readdir(source_dir; join=true)
        endswith(source_file, ".jl") || continue
        basename(source_file) == "Gay.jl" && continue
        source = read(source_file, String)
        target = occursin("# Auto-generated stub", source) ? stub_exports : substantive_exports

        for export_match in eachmatch(r"(?m)^\s*export\s+([^\n#]+)", source)
            for raw_name in split(export_match.captures[1], ',')
                name = strip(raw_name)
                isempty(name) && continue
                occursin(r"^[^\s]+$", name) || continue
                symbol = Symbol(name)
                push!(target, symbol)
                if target === substantive_exports
                    push!(get!(substantive_sources, symbol, String[]), basename(source_file))
                end
            end
        end
    end

    undefined_set = Set(undefined)
    substantive_backed = intersect(undefined_set, substantive_exports)
    stub_only_backed = intersect(undefined_set, setdiff(stub_exports, substantive_exports))
    source_unknown = setdiff(undefined_set, union(stub_exports, substantive_exports))

    println("undefined_stub_only_backed=$(length(stub_only_backed))")
    println("undefined_substantive_backed=$(length(substantive_backed))")
    println("undefined_source_unknown=$(length(source_unknown))")

    if LIST
        for name in sort!(collect(substantive_backed); by=string)
            files = join(sort!(unique(substantive_sources[name])), ',')
            println("substantive\t", name, '\t', files)
        end
        foreach(
            name -> println("stub_only\t", name),
            sort!(collect(stub_only_backed); by=string),
        )
        foreach(
            name -> println("source_unknown\t", name),
            sort!(collect(source_unknown); by=string),
        )
    end
end

if STRICT && !isempty(undefined)
    exit(1)
end
