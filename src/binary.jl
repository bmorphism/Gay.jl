# Gay.jl Binary Analysis: Growing the Gay Binary
#
# Enhanced binary analysis combining:
# - Radare2 MCP for disassembly, decompilation, xrefs
# - Tree-sitter for pseudocode AST parsing
# - SPI coloring for deterministic visualization
# - Abduction for pattern inference
# - Diaphora-style AST prime hashing for similarity
#
# Inspired by:
# - r2diaphora: AST-based function matching
# - Peirce's semiotic triangle: binary → sign → interpretant
# - Heisenberg exchange: XOR parity for parallel CFG analysis

export GayBinary, BinaryAnalysis, ASTHash, FunctionSignature
export analyze_binary!, parse_pseudocode, hash_ast_primes
export abduce_function_match, abduce_binary_seed
export render_cfg_graph, render_call_graph

# ═══════════════════════════════════════════════════════════════════════════
# AST Prime Hashing (à la Diaphora)
# ═══════════════════════════════════════════════════════════════════════════

# Prime numbers for AST node types (following diaphora's scheme)
const AST_PRIMES = Dict{Symbol, UInt64}(
    :function_definition => 2,
    :compound_statement => 3,
    :if_statement => 5,
    :else_clause => 7,
    :while_statement => 11,
    :for_statement => 13,
    :do_statement => 17,
    :switch_statement => 19,
    :case_statement => 23,
    :break_statement => 29,
    :continue_statement => 31,
    :return_statement => 37,
    :goto_statement => 41,
    :labeled_statement => 43,
    :expression_statement => 47,
    :declaration => 53,
    :assignment_expression => 59,
    :call_expression => 61,
    :binary_expression => 67,
    :unary_expression => 71,
    :conditional_expression => 73,
    :cast_expression => 79,
    :pointer_expression => 83,
    :subscript_expression => 89,
    :field_expression => 97,
    :sizeof_expression => 101,
    :identifier => 103,
    :number_literal => 107,
    :string_literal => 109,
    :char_literal => 113,
)

"""
    ASTHash

Hash of pseudocode AST for function matching.
Based on r2diaphora's prime product approach.
"""
struct ASTHash
    prime_product::UInt64       # Product of primes for node types
    node_count::Int             # Total AST nodes
    depth_max::Int              # Maximum nesting depth
    call_count::Int             # Number of function calls
    branch_count::Int           # if/switch/loop count
    color::RGB                  # SPI color from hash
end

"""
    hash_ast_primes(nodes::Vector{Symbol}; seed=GAY_SEED) -> ASTHash

Compute AST hash from node type sequence.
Product of primes is invariant to some code transformations.
"""
function hash_ast_primes(nodes::Vector{Symbol}; seed::Integer=GAY_SEED)
    product = UInt64(1)
    call_count = 0
    branch_count = 0
    
    for node in nodes
        prime = get(AST_PRIMES, node, UInt64(127))  # Default prime for unknown
        product = (product * prime) % (UInt64(1) << 63)  # Prevent overflow
        
        if node == :call_expression
            call_count += 1
        elseif node in [:if_statement, :switch_statement, :while_statement, :for_statement]
            branch_count += 1
        end
    end
    
    # Color from hash
    color = color_at(Int(product % 0xFFFF), SRGB(); seed=UInt64(seed))
    
    ASTHash(product, length(nodes), 0, call_count, branch_count, color)
end

# ═══════════════════════════════════════════════════════════════════════════
# Function Signature for Matching
# ═══════════════════════════════════════════════════════════════════════════

"""
    FunctionSignature

Rich function signature for similarity matching.
Combines multiple hashing strategies.
"""
struct FunctionSignature
    address::UInt64
    name::Union{String, Nothing}
    size::Int
    
    # Hashes for matching
    ast_hash::Union{ASTHash, Nothing}       # AST prime product
    bytes_hash::Union{UInt64, Nothing}      # First N bytes hash
    constants_hash::Union{UInt64, Nothing}  # Referenced constants
    strings_hash::Union{UInt64, Nothing}    # Referenced strings
    
    # Metrics
    cyclomatic::Int                         # Cyclomatic complexity
    basic_blocks::Int                       # Number of basic blocks
    
    # SPI coloring
    color::RGB
    spin::Int  # ±1 based on address parity
end

"""
    function_signature(addr, name, size; seed=R2_SEED, ast_nodes=nothing)

Create a function signature with computed hashes.
"""
function function_signature(addr::UInt64, name::Union{String,Nothing}, size::Int;
                           seed::UInt64=R2_SEED,
                           ast_nodes::Union{Vector{Symbol},Nothing}=nothing,
                           constants::Vector{UInt64}=UInt64[],
                           strings::Vector{String}=String[])
    # AST hash if nodes provided
    ast_hash = isnothing(ast_nodes) ? nothing : hash_ast_primes(ast_nodes; seed=seed)
    
    # Constants hash via FNV-1a
    constants_hash = isempty(constants) ? nothing : fnv1a_hash(constants)
    
    # Strings hash
    strings_hash = isempty(strings) ? nothing : fnv1a_hash(strings)
    
    # Color and spin
    color = r2_color_at(addr; seed=seed)
    spin = (addr & 1) == 0 ? 1 : -1
    
    FunctionSignature(
        addr, name, size,
        ast_hash, nothing, constants_hash, strings_hash,
        0, 0,  # Cyclomatic and blocks computed separately
        color, spin
    )
end

"""
    fnv1a_hash(data) -> UInt64

FNV-1a hash for deterministic hashing.
"""
function fnv1a_hash(data::Vector{UInt64})
    h = UInt64(0xcbf29ce484222325)  # FNV offset basis
    for x in data
        for i in 0:7
            byte = (x >> (8*i)) & 0xFF
            h = h ⊻ byte
            h = h * UInt64(0x100000001b3)  # FNV prime
        end
    end
    return h
end

function fnv1a_hash(strings::Vector{String})
    h = UInt64(0xcbf29ce484222325)
    for s in strings
        for c in codeunits(s)
            h = h ⊻ UInt64(c)
            h = h * UInt64(0x100000001b3)
        end
    end
    return h
end

# ═══════════════════════════════════════════════════════════════════════════
# Binary Analysis State
# ═══════════════════════════════════════════════════════════════════════════

"""
    GayBinary

Complete binary analysis with SPI coloring throughout.
The "gay binary" - a colored, signed, abducible representation.
"""
mutable struct GayBinary
    path::String
    seed::UInt64
    
    # Core analysis
    functions::Dict{UInt64, FunctionSignature}
    xrefs::Vector{ColoredXref}
    strings::Vector{Tuple{UInt64, String, RGB}}
    
    # Derived data
    call_graph::Dict{UInt64, Vector{UInt64}}  # caller → callees
    cfg_blocks::Dict{UInt64, Vector{ColoredBlock}}  # func → blocks
    
    # Abduction state
    abducer::GayAbducer
    inferred_patterns::Vector{Symbol}
    
    # Analysis level
    level::Int  # 0=headers, 1=symbols, 2=functions, 3=xrefs, 4=deep
end

"""
    GayBinary(path; seed=R2_SEED)

Create empty binary analysis container.
"""
function GayBinary(path::String; seed::Integer=R2_SEED)
    GayBinary(
        path, UInt64(seed),
        Dict{UInt64, FunctionSignature}(),
        ColoredXref[],
        Tuple{UInt64, String, RGB}[],
        Dict{UInt64, Vector{UInt64}}(),
        Dict{UInt64, Vector{ColoredBlock}}(),
        GayAbducer(),
        Symbol[],
        0
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Tree-sitter Pseudocode Parsing
# ═══════════════════════════════════════════════════════════════════════════

"""
    parse_pseudocode(code::String) -> Vector{Symbol}

Parse C-like pseudocode and extract AST node types.
Returns sequence of node type symbols for hashing.

Note: This is a simplified regex-based parser.
Full implementation would use tree-sitter-c via TreeSitter.jl or MCP.
"""
function parse_pseudocode(code::String)
    nodes = Symbol[]
    
    # Control flow patterns
    for m in eachmatch(r"\bif\s*\(", code)
        push!(nodes, :if_statement)
    end
    for m in eachmatch(r"\belse\b", code)
        push!(nodes, :else_clause)
    end
    for m in eachmatch(r"\bwhile\s*\(", code)
        push!(nodes, :while_statement)
    end
    for m in eachmatch(r"\bfor\s*\(", code)
        push!(nodes, :for_statement)
    end
    for m in eachmatch(r"\bswitch\s*\(", code)
        push!(nodes, :switch_statement)
    end
    for m in eachmatch(r"\bcase\b", code)
        push!(nodes, :case_statement)
    end
    for m in eachmatch(r"\breturn\b", code)
        push!(nodes, :return_statement)
    end
    for m in eachmatch(r"\bbreak\b", code)
        push!(nodes, :break_statement)
    end
    for m in eachmatch(r"\bcontinue\b", code)
        push!(nodes, :continue_statement)
    end
    for m in eachmatch(r"\bgoto\b", code)
        push!(nodes, :goto_statement)
    end
    
    # Expressions
    for m in eachmatch(r"\w+\s*\([^)]*\)\s*[;{]", code)  # Function calls
        push!(nodes, :call_expression)
    end
    for m in eachmatch(r"\w+\s*[+\-*/&|^]=", code)  # Compound assignment
        push!(nodes, :assignment_expression)
    end
    for m in eachmatch(r"\w+\s*=\s*[^=]", code)  # Simple assignment
        push!(nodes, :assignment_expression)
    end
    
    # Literals
    for m in eachmatch(r"\"[^\"]*\"", code)
        push!(nodes, :string_literal)
    end
    for m in eachmatch(r"\b0x[0-9a-fA-F]+\b", code)
        push!(nodes, :number_literal)
    end
    for m in eachmatch(r"\b\d+\b", code)
        push!(nodes, :number_literal)
    end
    
    return nodes
end

"""
    parse_pseudocode_with_treesitter(code::String, mcp_query_fn) -> Vector{Symbol}

Parse using tree-sitter MCP (when available).
`mcp_query_fn` should be a function that runs tree-sitter queries.
"""
function parse_pseudocode_with_treesitter(code::String, mcp_query_fn::Function)
    # Query for control flow and expressions
    query = """
    (if_statement) @if
    (while_statement) @while
    (for_statement) @for
    (switch_statement) @switch
    (return_statement) @return
    (call_expression) @call
    (assignment_expression) @assign
    """
    
    results = mcp_query_fn(code, "c", query)
    
    # Convert results to symbols
    nodes = Symbol[]
    for (capture_name, _) in results
        push!(nodes, Symbol(replace(capture_name, "@" => "") * "_statement"))
    end
    
    return nodes
end

# ═══════════════════════════════════════════════════════════════════════════
# Abduction: Pattern Inference
# ═══════════════════════════════════════════════════════════════════════════

"""
    abduce_function_match(sig1::FunctionSignature, sig2::FunctionSignature) -> Float64

Compute similarity score between two function signatures.
Returns 0.0 to 1.0 (1.0 = identical).
"""
function abduce_function_match(sig1::FunctionSignature, sig2::FunctionSignature)
    score = 0.0
    weight_sum = 0.0
    
    # AST hash match (highest weight)
    if !isnothing(sig1.ast_hash) && !isnothing(sig2.ast_hash)
        if sig1.ast_hash.prime_product == sig2.ast_hash.prime_product
            score += 0.5  # 50% weight for AST match
        else
            # Partial match based on node counts
            node_sim = 1.0 - abs(sig1.ast_hash.node_count - sig2.ast_hash.node_count) / 
                            max(sig1.ast_hash.node_count, sig2.ast_hash.node_count, 1)
            score += 0.2 * node_sim
        end
        weight_sum += 0.5
    end
    
    # Constants hash
    if !isnothing(sig1.constants_hash) && !isnothing(sig2.constants_hash)
        if sig1.constants_hash == sig2.constants_hash
            score += 0.25
        end
        weight_sum += 0.25
    end
    
    # Strings hash
    if !isnothing(sig1.strings_hash) && !isnothing(sig2.strings_hash)
        if sig1.strings_hash == sig2.strings_hash
            score += 0.15
        end
        weight_sum += 0.15
    end
    
    # Size similarity
    size_sim = 1.0 - abs(sig1.size - sig2.size) / max(sig1.size, sig2.size, 1)
    score += 0.1 * size_sim
    weight_sum += 0.1
    
    return weight_sum > 0 ? score / weight_sum : 0.0
end

"""
    abduce_binary_seed(binary::GayBinary, reference_colors)

Infer the seed used to color a binary by matching against known function colors.
"""
function abduce_binary_seed(binary::GayBinary, reference_colors::Vector{<:Tuple{UInt64, <:Any}})
    abducer = GayAbducer()
    
    for (addr, expected_color) in reference_colors
        # Convert address to index
        idx = Int(addr % 0xFFFF)
        register_observation!(abducer, expected_color; index=idx)
    end
    
    inferred_seed = infer_seed(abducer)
    binary.abducer = abducer
    
    return (inferred_seed, abducer.confidence)
end

# ═══════════════════════════════════════════════════════════════════════════
# Graph Rendering
# ═══════════════════════════════════════════════════════════════════════════

"""
    render_call_graph(binary::GayBinary; max_nodes=50) -> String

Render call graph as Mermaid diagram with SPI colors.
"""
function render_call_graph(binary::GayBinary; max_nodes::Int=50)
    lines = ["graph TD"]
    
    node_count = 0
    for (caller, callees) in binary.call_graph
        node_count >= max_nodes && break
        
        caller_sig = get(binary.functions, caller, nothing)
        caller_name = isnothing(caller_sig) ? "sub_$(string(caller, base=16))" : 
                      something(caller_sig.name, "sub_$(string(caller, base=16))")
        
        for callee in callees
            callee_sig = get(binary.functions, callee, nothing)
            callee_name = isnothing(callee_sig) ? "sub_$(string(callee, base=16))" : 
                          something(callee_sig.name, "sub_$(string(callee, base=16))")
            
            push!(lines, "    $(caller_name) --> $(callee_name)")
            node_count += 1
        end
    end
    
    return join(lines, "\n")
end

"""
    render_cfg_graph(blocks::Vector{ColoredBlock}; func_name="function") -> String

Render CFG as Mermaid diagram with checkerboard coloring.
"""
function render_cfg_graph(blocks::Vector{ColoredBlock}; func_name::String="function")
    lines = ["graph TD", "    subgraph $(func_name)"]
    
    for block in blocks
        addr_str = string(block.address, base=16)
        parity_class = block.parity == 0 ? "even" : "odd"
        push!(lines, "    bb_$(addr_str)[\"0x$(addr_str)\"]:::$(parity_class)")
        
        for succ in block.successors
            succ_str = string(succ, base=16)
            push!(lines, "    bb_$(addr_str) --> bb_$(succ_str)")
        end
    end
    
    push!(lines, "    end")
    push!(lines, "    classDef even fill:#4a9,stroke:#2a7,color:#fff")
    push!(lines, "    classDef odd fill:#a49,stroke:#72a,color:#fff")
    
    return join(lines, "\n")
end

# ═══════════════════════════════════════════════════════════════════════════
# Integration Demo
# ═══════════════════════════════════════════════════════════════════════════

"""
    world_binary_analysis(; seed=0xDEADBEEF)

Build composable binary analysis state with SPI coloring.
"""
function world_binary_analysis(; seed::UInt64=UInt64(0xDEADBEEF))
    pseudocode1 = """
    int getPortz() {
        if (access("/usr/bin/python", 0) == -1) {
            if (access("/usr/bin/perl", 0) == -1) {
                return 0;
            }
        }
        return 1;
    }
    """

    pseudocode2 = """
    undefined4 getPortz() {
        if (sym_access(ptr1) == -1) {
            if (sym_access(ptr2) == -1) {
                return 0;
            }
        }
        return 1;
    }
    """

    nodes1 = parse_pseudocode(pseudocode1)
    nodes2 = parse_pseudocode(pseudocode2)

    hash1 = hash_ast_primes(nodes1; seed=seed)
    hash2 = hash_ast_primes(nodes2; seed=seed)

    sig1 = function_signature(UInt64(0x401000), "getPortz", 128;
                             seed=seed, ast_nodes=nodes1,
                             strings=["python", "perl"])
    sig2 = function_signature(UInt64(0x501000), "sym.getPortz", 132;
                             seed=seed, ast_nodes=nodes2,
                             strings=["python", "perl"])

    similarity = abduce_function_match(sig1, sig2)

    binary = GayBinary("/example/binary"; seed=seed)
    reference = [
        (UInt64(0x401000), r2_color_at(UInt64(0x401000); seed=seed)),
        (UInt64(0x401100), r2_color_at(UInt64(0x401100); seed=seed)),
        (UInt64(0x401200), r2_color_at(UInt64(0x401200); seed=seed)),
    ]
    inferred, confidence = abduce_binary_seed(binary, reference)

    blocks = color_blocks([
        (0x401000, 16, 4, [0x401010, 0x401020], UInt64[]),
        (0x401010, 8, 2, [0x401020], [0x401000]),
        (0x401020, 12, 3, UInt64[], [0x401000, 0x401010]),
    ], UInt64(0x401000); seed=seed)

    (
        ast_hashing = (
            nodes1 = nodes1,
            nodes2 = nodes2,
            hash1 = hash1,
            hash2 = hash2,
            match = hash1.prime_product == hash2.prime_product,
        ),
        signatures = (
            sig1 = sig1,
            sig2 = sig2,
            similarity = similarity,
        ),
        seed_abduction = (
            binary = binary,
            inferred = inferred,
            actual = seed,
            confidence = confidence,
            correct = inferred == seed,
        ),
        cfg_blocks = blocks,
        seed = seed,
    )
end

export world_binary_analysis
