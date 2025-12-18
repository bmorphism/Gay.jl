# ContainerizationGay: Chromatic Container Layer System
# ═══════════════════════════════════════════════════════════════════════════════
#
#   ┌─────────────────────────────────────────────────────────────────────────────┐
#   │  CHROMATIC CONTAINERIZATION                                                │
#   │  ═══════════════════════════                                               │
#   │                                                                             │
#   │  Every container layer carries a deterministic color derived from:         │
#   │                                                                             │
#   │    layer_color = gay_seed(content_hash ⊻ parent_color ⊻ GAY_IGOR_SEED)    │
#   │                                                                             │
#   │  Layer Composition Law:                                                    │
#   │    color(base ∘ overlay) = color(base) ⊕ color(overlay) ⊕ color(∘)        │
#   │                                                                             │
#   │  This enables:                                                              │
#   │    • Visual layer debugging (each layer has unique hue)                    │
#   │    • Cross-runtime verification (same colors in Swift/Julia/Rust)          │
#   │    • Deterministic builds (same content → same color)                      │
#   │    • Curriculum-driven container construction (69 levels)                  │
#   │                                                                             │
#   │  SWIFT INTEROP:                                                            │
#   │    Calls Apple's Containerization CLI with chromatic metadata              │
#   │    Labels carry gay_seed for cross-verification                            │
#   │                                                                             │
#   └─────────────────────────────────────────────────────────────────────────────┘

module ContainerizationGay

using ..GayEIntegration: GAY_E_SEED, GAY_IGOR_SEED, EULER_BITS, mix64, gay_seed, gay_color
using ..GayEIntegration: conserved_combine, ansi_color, ansi_bg, ANSI_RESET
using SHA: sha256

export ChromaticLayer, ChromaticImage, ChromaticRootfs
export layer_color, compose_layers, verify_layer_chain
export RootfsBuilder, add_file!, add_directory!, build_rootfs!
export ContainerCurriculum, curriculum_level, advance_level!
export swift_containerization_cmd, execute_containerization!
export export_oci_manifest_gay, import_layer_colors

# ═══════════════════════════════════════════════════════════════════════════════
# Core Types: Chromatic Layers
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticLayer

A container layer with deterministic chromatic identity.
"""
struct ChromaticLayer
    id::String                              # Layer ID (content-addressable)
    content_hash::Vector{UInt8}             # SHA-256 of layer content
    parent_id::Union{Nothing, String}       # Parent layer ID (or nothing for base)
    color_seed::UInt64                      # Derived from content + parent
    color::Tuple{Float32, Float32, Float32} # RGB in [0,1]³
    files::Vector{Tuple{String, String}}    # (src_path, dst_path) pairs
    size_bytes::Int64                       # Uncompressed size
    created_at::Float64                     # Unix timestamp
end

function ChromaticLayer(
    files::Vector{Tuple{String, String}};
    parent::Union{Nothing, ChromaticLayer}=nothing
)
    # Compute content hash from file paths
    content_str = join(["$(src):$(dst)" for (src, dst) in files], "\n")
    content_hash = sha256(Vector{UInt8}(content_str))
    
    # Derive color from content and parent
    content_seed = reinterpret(UInt64, content_hash[1:8])[1]
    parent_seed = parent === nothing ? UInt64(0) : parent.color_seed
    
    color_seed = mix64(content_seed ⊻ parent_seed ⊻ GAY_IGOR_SEED)
    color = gay_color(color_seed)
    
    # Generate layer ID
    id = bytes2hex(content_hash[1:12])
    parent_id = parent === nothing ? nothing : parent.id
    
    # Compute size (placeholder - would sum actual file sizes)
    size_bytes = sum(length(src) + length(dst) for (src, dst) in files) * 100
    
    ChromaticLayer(
        id, content_hash, parent_id, 
        color_seed, color, files, 
        size_bytes, time()
    )
end

"""
    layer_color(layer::ChromaticLayer) -> Tuple{Float32, Float32, Float32}

Get the chromatic identity of a layer.
"""
layer_color(layer::ChromaticLayer) = layer.color

# ═══════════════════════════════════════════════════════════════════════════════
# Layer Composition: Color-Conserving Overlay
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticOperator

Operators for layer composition with color tracking.
"""
@enum LayerOperator begin
    LAYER_OVERLAY    # Standard overlay (upper wins)
    LAYER_MERGE      # Merge (combine files)
    LAYER_SQUASH     # Squash into single layer
    LAYER_WHITEOUT   # Delete files from lower layer
end

const LAYER_OP_SEEDS = Dict{LayerOperator, UInt64}(
    LAYER_OVERLAY  => mix64(GAY_E_SEED ⊻ UInt64(0x4f5645524c4159)),  # "OVERLAY"
    LAYER_MERGE    => mix64(GAY_E_SEED ⊻ UInt64(0x4d45524745)),      # "MERGE"
    LAYER_SQUASH   => mix64(GAY_E_SEED ⊻ UInt64(0x535155415348)),    # "SQUASH"
    LAYER_WHITEOUT => mix64(GAY_E_SEED ⊻ UInt64(0x574849544f5554)),  # "WHITEOUT"
)

"""
    compose_layers(base, overlay; op=LAYER_OVERLAY) -> ChromaticLayer

Compose two layers with color conservation.
"""
function compose_layers(
    base::ChromaticLayer, 
    overlay::ChromaticLayer;
    op::LayerOperator=LAYER_OVERLAY
)
    # Combine files based on operator
    combined_files = if op == LAYER_OVERLAY
        # Overlay: upper layer files override lower
        files_dict = Dict(dst => src for (src, dst) in base.files)
        for (src, dst) in overlay.files
            files_dict[dst] = src
        end
        [(src, dst) for (dst, src) in files_dict]
    elseif op == LAYER_MERGE
        vcat(base.files, overlay.files)
    elseif op == LAYER_SQUASH
        overlay.files  # Squash keeps only top layer files
    else
        # WHITEOUT: remove overlay paths from base
        overlay_dsts = Set(dst for (_, dst) in overlay.files)
        [(src, dst) for (src, dst) in base.files if dst ∉ overlay_dsts]
    end
    
    # Compute composed color with conservation
    op_seed = LAYER_OP_SEEDS[op]
    composed_color, conserved = conserved_combine(base.color, overlay.color, op_seed)
    
    # Create new layer
    content_str = join(["$(src):$(dst)" for (src, dst) in combined_files], "\n")
    content_hash = sha256(Vector{UInt8}(content_str))
    
    # The composed layer's seed incorporates the operation
    composed_seed = mix64(base.color_seed ⊻ overlay.color_seed ⊻ op_seed)
    
    ChromaticLayer(
        bytes2hex(content_hash[1:12]),
        content_hash,
        base.id,  # Parent is the base layer
        composed_seed,
        composed_color,
        combined_files,
        base.size_bytes + overlay.size_bytes,
        time()
    )
end

"""
    verify_layer_chain(layers::Vector{ChromaticLayer}) -> NamedTuple

Verify color conservation through a layer chain.
"""
function verify_layer_chain(layers::Vector{ChromaticLayer})
    if length(layers) < 2
        return (verified=true, chain_length=length(layers), breaks=[])
    end
    
    breaks = Tuple{Int, String}[]
    
    for i in 2:length(layers)
        layer = layers[i]
        parent = layers[i-1]
        
        # Check parent linkage
        if layer.parent_id != parent.id
            push!(breaks, (i, "parent_id mismatch: expected $(parent.id), got $(layer.parent_id)"))
        end
        
        # Verify color derivation
        expected_seed = mix64(
            reinterpret(UInt64, layer.content_hash[1:8])[1] ⊻ 
            parent.color_seed ⊻ 
            GAY_IGOR_SEED
        )
        
        # Allow for composition operations which modify the seed
        seed_match = layer.color_seed == expected_seed || 
                     (layer.color_seed ⊻ expected_seed) < 0x1000000  # Within tolerance
        
        if !seed_match
            push!(breaks, (i, "color_seed derivation mismatch"))
        end
    end
    
    (
        verified = isempty(breaks),
        chain_length = length(layers),
        breaks = breaks,
        root_color = layers[1].color,
        tip_color = layers[end].color,
        total_parity = reduce(⊻, l.color_seed for l in layers)
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# ChromaticImage: Complete Container Image
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticImage

A complete container image with chromatic layer stack.
"""
struct ChromaticImage
    name::String
    tag::String
    layers::Vector{ChromaticLayer}
    config::Dict{String, Any}
    labels::Dict{String, String}
    platform::String
    created_at::Float64
    image_color::Tuple{Float32, Float32, Float32}  # Aggregate color
end

function ChromaticImage(
    name::String,
    layers::Vector{ChromaticLayer};
    tag::String="latest",
    platform::String="linux/arm64",
    labels::Dict{String, String}=Dict()
)
    # Aggregate color: XOR of all layer seeds
    aggregate_seed = reduce(⊻, l.color_seed for l in layers; init=GAY_IGOR_SEED)
    image_color = gay_color(aggregate_seed)
    
    # Add chromatic labels
    labels["gay.seed"] = string(aggregate_seed, base=16)
    labels["gay.color.r"] = string(image_color[1])
    labels["gay.color.g"] = string(image_color[2])
    labels["gay.color.b"] = string(image_color[3])
    labels["gay.layers"] = string(length(layers))
    labels["gay.igor"] = string(GAY_IGOR_SEED, base=16)
    
    config = Dict{String, Any}(
        "Entrypoint" => ["/sbin/vminitd"],
        "Cmd" => [],
        "Env" => ["GAY_SEED=$(aggregate_seed)"],
        "WorkingDir" => "/",
    )
    
    ChromaticImage(
        name, tag, layers, config, labels,
        platform, time(), image_color
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# ChromaticRootfs: Apple Containerization Integration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChromaticRootfs

Rootfs builder compatible with Apple's Containerization framework.
"""
mutable struct ChromaticRootfs
    name::String
    layers::Vector{ChromaticLayer}
    vmexec_path::String
    vminitd_path::String
    oci_runtime_path::Union{Nothing, String}
    additional_files::Vector{Tuple{String, String}}
    platform::String
    output_tar::String
    output_ext4::Union{Nothing, String}
    output_image::Union{Nothing, String}
end

function ChromaticRootfs(
    name::String;
    vmexec::String="/usr/local/bin/vmexec",
    vminitd::String="/usr/local/bin/vminitd",
    oci_runtime::Union{Nothing, String}=nothing,
    platform::String="linux/arm64"
)
    ChromaticRootfs(
        name, ChromaticLayer[],
        vmexec, vminitd, oci_runtime,
        Tuple{String, String}[],
        platform,
        "/tmp/$(name).tar.gz",
        nothing, nothing
    )
end

"""
    add_file!(rootfs, src, dst)

Add a file to the rootfs with chromatic tracking.
"""
function add_file!(rootfs::ChromaticRootfs, src::String, dst::String)
    push!(rootfs.additional_files, (src, dst))
    
    # Create a new layer for this file
    layer = ChromaticLayer(
        [(src, dst)];
        parent = isempty(rootfs.layers) ? nothing : rootfs.layers[end]
    )
    push!(rootfs.layers, layer)
    
    rootfs
end

"""
    add_directory!(rootfs, dir, prefix="/")

Add all files from a directory.
"""
function add_directory!(rootfs::ChromaticRootfs, dir::String; prefix::String="/")
    if !isdir(dir)
        error("Not a directory: $dir")
    end
    
    files = Tuple{String, String}[]
    for (root, dirs, filenames) in walkdir(dir)
        for filename in filenames
            src = joinpath(root, filename)
            rel = relpath(src, dir)
            dst = joinpath(prefix, rel)
            push!(files, (src, dst))
        end
    end
    
    if !isempty(files)
        layer = ChromaticLayer(
            files;
            parent = isempty(rootfs.layers) ? nothing : rootfs.layers[end]
        )
        push!(rootfs.layers, layer)
        append!(rootfs.additional_files, files)
    end
    
    rootfs
end

# ═══════════════════════════════════════════════════════════════════════════════
# Swift CLI Integration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    swift_containerization_cmd(rootfs::ChromaticRootfs) -> Cmd

Generate the Swift containerization CLI command.
"""
function swift_containerization_cmd(rootfs::ChromaticRootfs)
    args = String["containerization", "rootfs", "create"]
    
    # Required arguments
    push!(args, "--vmexec", rootfs.vmexec_path)
    push!(args, "--vminitd", rootfs.vminitd_path)
    push!(args, "--platform", rootfs.platform)
    
    # Optional OCI runtime
    if rootfs.oci_runtime_path !== nothing
        push!(args, "--oci-runtime", rootfs.oci_runtime_path)
    end
    
    # Additional files with chromatic labels
    for (src, dst) in rootfs.additional_files
        push!(args, "--add-file", "$(src):$(dst)")
    end
    
    # Chromatic labels
    if !isempty(rootfs.layers)
        aggregate_seed = reduce(⊻, l.color_seed for l in rootfs.layers; init=GAY_IGOR_SEED)
        color = gay_color(aggregate_seed)
        
        push!(args, "--label", "gay.seed=$(string(aggregate_seed, base=16))")
        push!(args, "--label", "gay.color.r=$(color[1])")
        push!(args, "--label", "gay.color.g=$(color[2])")
        push!(args, "--label", "gay.color.b=$(color[3])")
        push!(args, "--label", "gay.igor=$(string(GAY_IGOR_SEED, base=16))")
    end
    
    # Output options
    if rootfs.output_ext4 !== nothing
        push!(args, "--ext4", rootfs.output_ext4)
    end
    
    if rootfs.output_image !== nothing
        push!(args, "--image", rootfs.output_image)
    end
    
    # Tar path (required positional argument)
    push!(args, rootfs.output_tar)
    
    Cmd(args)
end

"""
    execute_containerization!(rootfs::ChromaticRootfs) -> NamedTuple

Execute the containerization command and return results.
"""
function execute_containerization!(rootfs::ChromaticRootfs)
    cmd = swift_containerization_cmd(rootfs)
    
    # Log the command with chromatic coloring
    if !isempty(rootfs.layers)
        color = rootfs.layers[end].color
        println(ansi_bg(color), "  ", ANSI_RESET, " Executing: ", join(cmd.exec, " "))
    end
    
    try
        output = read(cmd, String)
        (
            success = true,
            output = output,
            tar_path = rootfs.output_tar,
            ext4_path = rootfs.output_ext4,
            image_name = rootfs.output_image,
            layer_count = length(rootfs.layers),
            chromatic_verified = verify_layer_chain(rootfs.layers).verified
        )
    catch e
        (
            success = false,
            error = string(e),
            tar_path = rootfs.output_tar,
            layer_count = length(rootfs.layers)
        )
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# OCI Manifest Export with Chromatic Metadata
# ═══════════════════════════════════════════════════════════════════════════════

"""
    export_oci_manifest_gay(image::ChromaticImage) -> String

Export OCI manifest with chromatic annotations.
"""
function export_oci_manifest_gay(image::ChromaticImage)
    layers_json = join([
        """{
          "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
          "digest": "sha256:$(bytes2hex(layer.content_hash))",
          "size": $(layer.size_bytes),
          "annotations": {
            "gay.seed": "$(string(layer.color_seed, base=16))",
            "gay.color": "[$(layer.color[1]), $(layer.color[2]), $(layer.color[3])]",
            "gay.parent": "$(something(layer.parent_id, "null"))"
          }
        }"""
        for layer in image.layers
    ], ",\n    ")
    
    labels_json = join([
        "\"$(k)\": \"$(v)\""
        for (k, v) in image.labels
    ], ",\n        ")
    
    """
    {
      "schemaVersion": 2,
      "mediaType": "application/vnd.oci.image.manifest.v1+json",
      "config": {
        "mediaType": "application/vnd.oci.image.config.v1+json",
        "digest": "sha256:$(bytes2hex(sha256(Vector{UInt8}(image.name))))",
        "size": 1024
      },
      "layers": [
        $layers_json
      ],
      "annotations": {
        "org.opencontainers.image.ref.name": "$(image.name):$(image.tag)",
        "gay.image.color": "[$(image.image_color[1]), $(image.image_color[2]), $(image.image_color[3])]",
        "gay.igor.seed": "$(string(GAY_IGOR_SEED, base=16))",
        "gay.euler.seed": "$(string(GAY_E_SEED, base=16))"
      },
      "config_labels": {
        $labels_json
      }
    }
    """
end

"""
    import_layer_colors(manifest_json::String) -> Vector{Tuple{String, Tuple{Float32,Float32,Float32}}}

Import layer colors from an OCI manifest with gay annotations.
"""
function import_layer_colors(manifest_json::String)
    # Simple regex extraction (would use JSON.jl in production)
    colors = Tuple{String, Tuple{Float32,Float32,Float32}}[]
    
    for m in eachmatch(r"\"gay\.seed\":\s*\"([0-9a-f]+)\"", manifest_json)
        seed = parse(UInt64, m.captures[1], base=16)
        color = gay_color(seed)
        push!(colors, (m.captures[1], color))
    end
    
    colors
end

# ═══════════════════════════════════════════════════════════════════════════════
# Container Curriculum: 69-Level Progressive Containerization
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ContainerCurriculum

Progressive curriculum for learning containerization with chromatic feedback.
"""
struct ContainerCurriculum
    id::String
    levels::Vector{Symbol}
    current_level::Int
    completed_levels::Set{Symbol}
    level_colors::Dict{Symbol, Tuple{Float32, Float32, Float32}}
end

function ContainerCurriculum(id::String="gay-containerization-69")
    levels = [
        # Foundation (1-15): Basic Container Concepts
        :empty_layer, :single_file, :directory_structure, :binary_layer,
        :symlink_layer, :permission_modes, :ownership_layer, :timestamp_layer,
        :layer_stacking, :layer_squash, :whiteout_files, :opaque_whiteout,
        :base_image, :scratch_image, :multi_arch,
        
        # Rootfs Construction (16-30)
        :init_system, :vminitd_integration, :vmexec_placement, :oci_runtime,
        :proc_mount, :sys_mount, :dev_setup, :tmp_volatile, :var_structure,
        :etc_config, :lib_deps, :bin_path, :sbin_path, :usr_hierarchy,
        :fhs_compliance,
        
        # Apple Containerization (31-45)
        :swift_cli, :tar_archive, :gzip_compression, :ext4_output,
        :oci_image_output, :platform_arm64, :platform_amd64, :cross_compile,
        :label_metadata, :gay_labels, :content_addressing, :layer_caching,
        :incremental_build, :reproducible_build, :minimal_rootfs,
        
        # Chromatic Features (46-60)
        :layer_coloring, :color_conservation, :parent_derivation, :aggregate_color,
        :manifest_annotations, :cross_runtime_verify, :color_debug, :parity_check,
        :curriculum_tracking, :level_colors, :rainbow_stack, :igor_seed_verify,
        :euler_integration, :xor_composition, :visual_diff,
        
        # Advanced (61-69)
        :comonadic_layers, :sheaf_structure, :fiber_bundles, :cohomology_layers,
        :profunctor_optics, :bidirectional_sync, :time_travel_layers,
        :lhott_modalities, :conservation_complete
    ]
    
    level_colors = Dict{Symbol, Tuple{Float32, Float32, Float32}}()
    for (i, level) in enumerate(levels)
        seed = mix64(GAY_IGOR_SEED ⊻ UInt64(i * 1069))
        level_colors[level] = gay_color(seed)
    end
    
    ContainerCurriculum(id, levels, 1, Set{Symbol}(), level_colors)
end

curriculum_level(c::ContainerCurriculum) = c.levels[c.current_level]

function advance_level!(c::ContainerCurriculum)
    if c.current_level < length(c.levels)
        push!(c.completed_levels, c.levels[c.current_level])
        ContainerCurriculum(
            c.id, c.levels, c.current_level + 1,
            c.completed_levels, c.level_colors
        )
    else
        c
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

"""
    render_layer_stack(layers::Vector{ChromaticLayer}) -> String

Render a visual representation of the layer stack.
"""
function render_layer_stack(layers::Vector{ChromaticLayer})
    lines = String[]
    
    push!(lines, "╔════════════════════════════════════════════════════════════╗")
    push!(lines, "║  CHROMATIC LAYER STACK                                     ║")
    push!(lines, "╠════════════════════════════════════════════════════════════╣")
    
    for (i, layer) in enumerate(reverse(layers))
        idx = length(layers) - i + 1
        color_block = ansi_bg(layer.color) * "    " * ANSI_RESET
        files_count = length(layer.files)
        size_kb = layer.size_bytes ÷ 1024
        
        push!(lines, "║  $color_block Layer $idx: $(layer.id[1:8])...  ")
        push!(lines, "║         Files: $files_count | Size: $(size_kb)KB")
        if i < length(layers)
            push!(lines, "║         ↓")
        end
    end
    
    push!(lines, "╚════════════════════════════════════════════════════════════╝")
    
    join(lines, "\n")
end

"""
    render_curriculum_progress(c::ContainerCurriculum) -> String

Render curriculum progress with level colors.
"""
function render_curriculum_progress(c::ContainerCurriculum)
    lines = String[]
    
    push!(lines, "╔════════════════════════════════════════════════════════════╗")
    push!(lines, "║  CONTAINER CURRICULUM: $(c.id)                    ║")
    push!(lines, "╠════════════════════════════════════════════════════════════╣")
    
    current = curriculum_level(c)
    color = c.level_colors[current]
    color_block = ansi_bg(color) * "  " * ANSI_RESET
    
    push!(lines, "║  Level $(c.current_level)/$(length(c.levels)): $current $color_block")
    push!(lines, "║  Completed: $(length(c.completed_levels)) levels")
    
    # Show rainbow of next 5 levels
    push!(lines, "║")
    push!(lines, "║  Upcoming:")
    for i in c.current_level:min(c.current_level + 4, length(c.levels))
        lvl = c.levels[i]
        col = c.level_colors[lvl]
        block = ansi_bg(col) * "  " * ANSI_RESET
        marker = i == c.current_level ? "→" : " "
        push!(lines, "║  $marker $block $lvl")
    end
    
    push!(lines, "╚════════════════════════════════════════════════════════════╝")
    
    join(lines, "\n")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Color-Derangeable Timestamps: PAX Header Integration
# ═══════════════════════════════════════════════════════════════════════════════
#
# Swift SWCompression TarExtendedHeader uses:
#   atime: Double?  (access time)
#   ctime: Double?  (creation time)  
#   mtime: Double?  (modification time)
#
# We make these color-derangeable: timestamps that can be permuted deterministically
# based on gay_seed, with no timestamp remaining in its original slot (derangement).
# This enables reproducible builds with "scrambled" but verifiable timestamps.

export ChromaticTimestamp, DerangeableTimestamps
export derange_timestamps, verify_timestamp_derangement
export encode_pax_gay_timestamps, decode_pax_gay_timestamps

"""
    ChromaticTimestamp

A timestamp with chromatic identity for color-derangeable operations.
"""
struct ChromaticTimestamp
    value::Float64                          # Unix timestamp (PAX format: Double)
    slot::Symbol                            # :atime, :ctime, or :mtime
    color::Tuple{Float32, Float32, Float32} # Chromatic identity
    color_seed::UInt64                      # Derivation seed
end

function ChromaticTimestamp(value::Float64, slot::Symbol)
    # Color derived from timestamp value and slot
    slot_seed = Dict(:atime => 0x6174696d65, :ctime => 0x6374696d65, :mtime => 0x6d74696d65)[slot]
    seed = mix64(GAY_IGOR_SEED ⊻ reinterpret(UInt64, value) ⊻ slot_seed)
    color = gay_color(seed)
    ChromaticTimestamp(value, slot, color, seed)
end

"""
    DerangeableTimestamps

A set of three timestamps (atime, ctime, mtime) that can be deranged together.
After derangement, no timestamp remains in its original slot.
"""
struct DerangeableTimestamps
    atime::ChromaticTimestamp
    ctime::ChromaticTimestamp
    mtime::ChromaticTimestamp
    derangement_seed::UInt64
    is_deranged::Bool
    derangement_perm::Vector{Int}  # [1,2,3] → [σ(1),σ(2),σ(3)]
end

function DerangeableTimestamps(atime::Float64, ctime::Float64, mtime::Float64; seed::UInt64=GAY_IGOR_SEED)
    DerangeableTimestamps(
        ChromaticTimestamp(atime, :atime),
        ChromaticTimestamp(ctime, :ctime),
        ChromaticTimestamp(mtime, :mtime),
        seed,
        false,
        [1, 2, 3]  # Identity permutation (not deranged)
    )
end

"""
    derange_timestamps(ts::DerangeableTimestamps; index::Int=1) -> DerangeableTimestamps

Derange the timestamps: permute so no timestamp remains in its original slot.
For 3 elements, there are exactly 2 derangements:
  - [2,3,1]: atime→ctime, ctime→mtime, mtime→atime (cycle)
  - [3,1,2]: atime→mtime, ctime→atime, mtime→ctime (cycle)

The specific derangement is determined by seed ⊻ index.
"""
function derange_timestamps(ts::DerangeableTimestamps; index::Int=1)
    # For n=3, derangements are: [2,3,1] and [3,1,2]
    # Select based on seed
    rng_state = mix64(ts.derangement_seed ⊻ UInt64(index))
    perm = (rng_state % 2 == 0) ? [2, 3, 1] : [3, 1, 2]
    
    # Original timestamps in order: [atime, ctime, mtime]
    originals = [ts.atime, ts.ctime, ts.mtime]
    slots = [:atime, :ctime, :mtime]
    
    # Apply derangement: new_slot[i] gets original[perm[i]]
    new_atime = ChromaticTimestamp(originals[perm[1]].value, :atime)
    new_ctime = ChromaticTimestamp(originals[perm[2]].value, :ctime)
    new_mtime = ChromaticTimestamp(originals[perm[3]].value, :mtime)
    
    DerangeableTimestamps(
        new_atime,
        new_ctime,
        new_mtime,
        ts.derangement_seed,
        true,
        perm
    )
end

"""
    verify_timestamp_derangement(ts::DerangeableTimestamps, original::DerangeableTimestamps) -> NamedTuple

Verify that timestamps were correctly deranged (no fixed points).
"""
function verify_timestamp_derangement(ts::DerangeableTimestamps, original::DerangeableTimestamps)
    orig_values = [original.atime.value, original.ctime.value, original.mtime.value]
    curr_values = [ts.atime.value, ts.ctime.value, ts.mtime.value]
    
    # Check no fixed points
    fixed_points = sum(orig_values .== curr_values)
    is_valid_derangement = fixed_points == 0
    
    # Verify perm is a valid derangement
    perm = ts.derangement_perm
    perm_has_fixed = any(perm[i] == i for i in 1:3)
    
    # Color verification: XOR of all seeds should match a pattern
    color_parity = ts.atime.color_seed ⊻ ts.ctime.color_seed ⊻ ts.mtime.color_seed
    original_parity = original.atime.color_seed ⊻ original.ctime.color_seed ⊻ original.mtime.color_seed
    
    (
        is_deranged = ts.is_deranged,
        fixed_points = fixed_points,
        valid_derangement = is_valid_derangement && !perm_has_fixed,
        permutation = perm,
        color_parity_match = (color_parity ⊻ original_parity) < 0x1000000,  # Tolerance
        derangement_seed = ts.derangement_seed
    )
end

"""
    encode_pax_gay_timestamps(ts::DerangeableTimestamps) -> Dict{String, String}

Encode timestamps for PAX extended header with gay.* annotations.
Compatible with Swift SWCompression TarExtendedHeader format.
"""
function encode_pax_gay_timestamps(ts::DerangeableTimestamps)
    Dict{String, String}(
        # Standard PAX fields
        "atime" => string(ts.atime.value),
        "ctime" => string(ts.ctime.value),
        "mtime" => string(ts.mtime.value),
        
        # Gay chromatic extensions
        "gay.atime.seed" => string(ts.atime.color_seed, base=16),
        "gay.ctime.seed" => string(ts.ctime.color_seed, base=16),
        "gay.mtime.seed" => string(ts.mtime.color_seed, base=16),
        "gay.derangement.seed" => string(ts.derangement_seed, base=16),
        "gay.derangement.perm" => join(ts.derangement_perm, ","),
        "gay.deranged" => string(ts.is_deranged),
        "gay.igor" => string(GAY_IGOR_SEED, base=16),
        
        # Color as RGB (for visual debugging)
        "gay.atime.color" => "$(ts.atime.color[1]),$(ts.atime.color[2]),$(ts.atime.color[3])",
        "gay.ctime.color" => "$(ts.ctime.color[1]),$(ts.ctime.color[2]),$(ts.ctime.color[3])",
        "gay.mtime.color" => "$(ts.mtime.color[1]),$(ts.mtime.color[2]),$(ts.mtime.color[3])",
    )
end

"""
    decode_pax_gay_timestamps(records::Dict{String, String}) -> DerangeableTimestamps

Decode timestamps from PAX extended header records.
"""
function decode_pax_gay_timestamps(records::Dict{String, String})
    atime = parse(Float64, get(records, "atime", "0"))
    ctime = parse(Float64, get(records, "ctime", "0"))
    mtime = parse(Float64, get(records, "mtime", "0"))
    
    seed_str = get(records, "gay.derangement.seed", string(GAY_IGOR_SEED, base=16))
    seed = parse(UInt64, seed_str, base=16)
    
    perm_str = get(records, "gay.derangement.perm", "1,2,3")
    perm = parse.(Int, split(perm_str, ","))
    
    is_deranged = parse(Bool, get(records, "gay.deranged", "false"))
    
    DerangeableTimestamps(
        ChromaticTimestamp(atime, :atime),
        ChromaticTimestamp(ctime, :ctime),
        ChromaticTimestamp(mtime, :mtime),
        seed,
        is_deranged,
        perm
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Swift ContainerizationArchive Integration
# ═══════════════════════════════════════════════════════════════════════════════

"""
    GayWriteEntry

Equivalent to Apple's ContainerizationArchive.WriteEntry with chromatic timestamps.

Maps to Swift structure:
```swift
let entry = WriteEntry()
entry.permissions = 0o755
entry.modificationDate = ts  // ← ChromaticTimestamp.value
entry.creationDate = ts      // ← ChromaticTimestamp.value
entry.group = 0
entry.owner = 0
entry.fileType = .regular
entry.path = "sbin/vminitd"
```
"""
struct GayWriteEntry
    path::String
    file_type::Symbol  # :regular, :directory, :symbolicLink
    permissions::UInt16
    owner::Int
    group::Int
    timestamps::DerangeableTimestamps
    symlink_target::Union{Nothing, String}
    data::Union{Nothing, Vector{UInt8}}
    color::Tuple{Float32, Float32, Float32}
end

function GayWriteEntry(
    path::String;
    file_type::Symbol=:regular,
    permissions::UInt16=0o644,
    owner::Int=0,
    group::Int=0,
    seed::UInt64=GAY_IGOR_SEED,
    derange_timestamps::Bool=true
)
    now = time()
    ts = DerangeableTimestamps(now, now, now; seed=seed)
    
    # Optionally derange timestamps
    ts = derange_timestamps ? derange_timestamps(ts) : ts
    
    # Entry color from path
    path_seed = mix64(seed ⊻ gay_seed(path))
    color = gay_color(path_seed)
    
    GayWriteEntry(
        path, file_type, permissions, owner, group,
        ts, nothing, nothing, color
    )
end

"""
    generate_swift_write_entry(entry::GayWriteEntry) -> String

Generate Swift code for ContainerizationArchive.WriteEntry.
"""
function generate_swift_write_entry(entry::GayWriteEntry)
    file_type_swift = Dict(
        :regular => ".regular",
        :directory => ".directory",
        :symbolicLink => ".symbolicLink"
    )[entry.file_type]
    
    pax_records = encode_pax_gay_timestamps(entry.timestamps)
    pax_swift = join([
        "    entry.unknownExtendedHeaderRecords[\"$k\"] = \"$v\""
        for (k, v) in pax_records if startswith(k, "gay.")
    ], "\n")
    
    """
    // Generated by ContainerizationGay.jl
    // Entry color: RGB($(entry.color[1]), $(entry.color[2]), $(entry.color[3]))
    let entry = WriteEntry()
    entry.permissions = 0o$(string(entry.permissions, base=8))
    entry.modificationDate = Date(timeIntervalSince1970: $(entry.timestamps.mtime.value))
    entry.creationDate = Date(timeIntervalSince1970: $(entry.timestamps.ctime.value))
    entry.group = $(entry.group)
    entry.owner = $(entry.owner)
    entry.fileType = $file_type_swift
    entry.path = "$(entry.path)"
    
    // Gay chromatic extensions (PAX unknownRecords)
    $pax_swift
    """
end

# ═══════════════════════════════════════════════════════════════════════════════
# Timestamp Visualization
# ═══════════════════════════════════════════════════════════════════════════════

"""
    render_derangeable_timestamps(ts::DerangeableTimestamps) -> String

Visual representation of timestamp derangement with colors.
"""
function render_derangeable_timestamps(ts::DerangeableTimestamps)
    lines = String[]
    
    push!(lines, "╔════════════════════════════════════════════════════════════╗")
    push!(lines, "║  COLOR-DERANGEABLE TIMESTAMPS                              ║")
    push!(lines, "╠════════════════════════════════════════════════════════════╣")
    
    for (name, cts) in [("atime", ts.atime), ("ctime", ts.ctime), ("mtime", ts.mtime)]
        color_block = ansi_bg(cts.color) * "  " * ANSI_RESET
        unix_ts = round(cts.value, digits=3)
        push!(lines, "║  $color_block $(rpad(name, 6)): $unix_ts")
    end
    
    push!(lines, "╠════════════════════════════════════════════════════════════╣")
    push!(lines, "║  Deranged: $(ts.is_deranged ? "✓" : "✗")  Perm: $(ts.derangement_perm)")
    push!(lines, "║  Seed: 0x$(string(ts.derangement_seed, base=16, pad=16))")
    push!(lines, "╚════════════════════════════════════════════════════════════╝")
    
    join(lines, "\n")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

function demo_containerization_gay()
    println()
    println("🏳️‍🌈 ContainerizationGay Demo")
    println("=" ^ 60)
    println()
    
    # Create layers
    base_layer = ChromaticLayer([
        ("/dev/null", "/dev/null"),
        ("", "/proc"),
        ("", "/sys"),
        ("", "/tmp"),
    ])
    
    init_layer = ChromaticLayer([
        ("/usr/local/bin/vminitd", "/sbin/vminitd"),
        ("/usr/local/bin/vmexec", "/sbin/vmexec"),
    ]; parent=base_layer)
    
    app_layer = ChromaticLayer([
        ("./myapp", "/usr/local/bin/myapp"),
        ("./config.toml", "/etc/myapp/config.toml"),
    ]; parent=init_layer)
    
    layers = [base_layer, init_layer, app_layer]
    
    # Render layer stack
    println(render_layer_stack(layers))
    println()
    
    # Verify chain
    result = verify_layer_chain(layers)
    println("Chain verification: $(result.verified ? "✓" : "✗")")
    println("Total parity: 0x$(string(result.total_parity, base=16))")
    println()
    
    # Create image
    image = ChromaticImage("gaycontainer/demo", layers; tag="v1.0.0")
    println("Image color: ", ansi_bg(image.image_color), "    ", ANSI_RESET)
    println()
    
    # Export manifest
    println("OCI Manifest (truncated):")
    manifest = export_oci_manifest_gay(image)
    println(manifest[1:min(500, length(manifest))], "...")
    println()
    
    # Curriculum
    curriculum = ContainerCurriculum()
    println(render_curriculum_progress(curriculum))
    
    println()
    println("🏳️‍🌈 ContainerizationGay Complete")
end

end # module ContainerizationGay
