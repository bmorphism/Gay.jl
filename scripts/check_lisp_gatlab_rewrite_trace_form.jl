#!/usr/bin/env julia

using Gay
using JSON3

root = dirname(@__DIR__)
input = isempty(ARGS) ? joinpath(root, "artifacts", "lisp_gatlab_rewrite_program_trace.lisp") : ARGS[1]
form = read(input, String)
validation = eval(lisp_gatlab_rewrite_trace_compile(form; target=:validation))
payload = lisp_gatlab_rewrite_trace_validation_payload(validation)

if length(ARGS) >= 2
    output = ARGS[2]
    mkpath(dirname(output))
    open(output, "w") do io
        JSON3.pretty(io, payload)
        println(io)
    end
else
    JSON3.pretty(stdout, payload)
    println()
end

validation.valid || error("Lisp/GATlab rewrite trace form failed replay validation: $input")
