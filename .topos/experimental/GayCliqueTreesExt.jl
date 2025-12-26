module GayCliqueTreesExt

using Gay
using CliqueTrees

function __init__()
    # Register CliqueTrees backend with the parallel color scheduler
    Gay._register_cliquetrees_backend!(
        tree_decomposition,
        treewidth
    )
end

end # module
