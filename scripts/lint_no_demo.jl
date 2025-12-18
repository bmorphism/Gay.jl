#!/usr/bin/env julia
"""
Lint script to detect forbidden `demo_` patterns in the codebase.
Enforces `world_` naming convention over `demo_`.

Usage: julia scripts/lint_no_demo.jl
Exit code: 0 if clean, 1 if violations found
"""

function find_violations(src_dir::String)
    violations = Tuple{String, Int, String, String}[]  # (file, line, pattern, suggestion)
    
    function_pattern = r"function\s+(demo_\w+)"
    export_pattern = r"export\s+.*\b(demo_\w+)"
    
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
                    suggestion = replace(name, "demo_" => "world_")
                    push!(violations, (filepath, lineno, "function $name", suggestion))
                end
                
                # Check for exports
                for m in eachmatch(export_pattern, line)
                    name = m.captures[1]
                    suggestion = replace(name, "demo_" => "world_")
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
        println("✓ No demo_ violations found")
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
