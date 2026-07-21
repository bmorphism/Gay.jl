#!/usr/bin/env julia

using Gay

const STRICT = "--strict" in ARGS
const LIST = "--list" in ARGS

exported = names(Gay; all=false, imported=false)
undefined = filter(name -> !isdefined(Gay, name), exported)
defined_count = length(exported) - length(undefined)

println("exported=$(length(exported))")
println("defined=$defined_count")
println("undefined=$(length(undefined))")

if LIST
    foreach(name -> println("undefined\t", name), sort!(undefined; by=string))
end

if STRICT && !isempty(undefined)
    exit(1)
end
