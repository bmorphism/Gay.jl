# Metal.jl extension for Gay.jl - Apple Silicon GPU acceleration
module GayMetalExt

using Gay
using Metal

function __init__()
    if Metal.functional()
        @info "Gay.jl Metal extension loaded - GPU acceleration available 🚀"
        Gay.set_backend!(Metal.MetalBackend())
    end
end

end
