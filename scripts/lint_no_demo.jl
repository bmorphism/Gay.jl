#!/usr/bin/env julia
"""
Lint script to detect forbidden demo markers in source identifiers.
Enforces composable `world_`/`verify_` naming over `demo_` and `_demo`.

Usage: julia scripts/lint_no_demo.jl
Exit code: 0 if clean, 1 if violations found
"""

function find_violations(src_dir::String)
    violations = Tuple{String, Int, String, String}[]  # (file, line, pattern, suggestion)

    forbidden_name = "(?:demo_\\w+|\\w+_demo)"
    function_pattern = Regex("function\\s+($forbidden_name)\\b")
    export_pattern = Regex("export\\s+.*\\b($forbidden_name)\\b")

    function suggestion_for(name::AbstractString)
        if startswith(name, "demo_")
            return replace(name, "demo_" => "world_")
        elseif startswith(name, "world_") && endswith(name, "_demo")
            return name[1:end-length("_demo")]
        else
            return replace(name, "_demo" => "_world")
        end
    end

    for (root, dirs, files) in walkdir(src_dir)
        for file in files
            endswith(file, ".jl") || continue
            filepath = joinpath(root, file)

            lines = try
                readlines(filepath)
            catch e
                @warn "Could not read $filepath: $e"
                continue
            end

            for (lineno, line) in enumerate(lines)
                # Check for function definitions
                m = match(function_pattern, line)
                if m !== nothing
                    name = m.captures[1]
                    suggestion = suggestion_for(name)
                    push!(violations, (filepath, lineno, "function $name", suggestion))
                end

                # Check for exports
                for m in eachmatch(export_pattern, line)
                    name = m.captures[1]
                    suggestion = suggestion_for(name)
                    push!(violations, (filepath, lineno, "export $name", suggestion))
                end
            end
        end
    end

    return violations
end

function main()
    script_dir = @__DIR__
    src_dir = normpath(joinpath(script_dir, "..", "src"))

    if !isdir(src_dir)
        println(stderr, "ERROR: src directory not found at $src_dir")
        exit(1)
    end

    violations = find_violations(src_dir)

    if isempty(violations)
        println("◆ No demo identifier violations found")
        exit(0)
    else
        println("Found $(length(violations)) violation(s):\n")
        for (file, line, pattern, suggestion) in violations
            println("VIOLATION: $file:$line: $pattern")
            println("  FIX: Rename to $suggestion\n")
        end
        exit(1)
    end
end

main()
