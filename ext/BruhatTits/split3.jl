# split3.jl  
module Split3

export @split3_read, @split3_write, SplitContext

const GAY_SEED = UInt64(0x6761795f636f6c6f)

sm64(z) = let z = z + 0x9E3779B97F4A7C15
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    z ⊻ (z >> 31)
end

struct SplitContext
    parent_color::UInt64
    child_colors::NTuple{3, UInt64}
    operation::Symbol
end

function split3(parent_color::UInt64, op::Symbol)
    c1 = sm64(parent_color)
    c2 = sm64(c1)
    c3 = sm64(c2)
    SplitContext(parent_color, (c1, c2, c3), op)
end

function execute_split3(ctx::SplitContext, f1, f2, f3)
    results = Vector{Any}(undef, 3)
    Threads.@threads for i in 1:3
        color = ctx.child_colors[i]
        if i == 1
            results[1] = f1(color)
        elseif i == 2
            results[2] = f2(color)
        else
            results[3] = f3(color)
        end
    end
    # Aggregate fingerprint
    fp = ctx.child_colors[1] ⊻ ctx.child_colors[2] ⊻ ctx.child_colors[3]
    (results=results, fingerprint=fp)
end

macro split3_read(parent_color, f1, f2, f3)
    quote
        ctx = split3($(esc(parent_color)), :read)
        execute_split3(ctx, $(esc(f1)), $(esc(f2)), $(esc(f3)))
    end
end

macro split3_write(parent_color, f1, f2, f3)
    quote
        ctx = split3($(esc(parent_color)), :write)
        execute_split3(ctx, $(esc(f1)), $(esc(f2)), $(esc(f3)))
    end
end

end
