#!/usr/bin/env julia

using Pkg
using Libdl

function valid_qhull_root(path)
    isdir(joinpath(path, "include", "libqhull_r")) &&
        isfile(joinpath(path, "lib", "libqhull_r.$(Libdl.dlext)"))
end

function nix_qhull_roots()
    store = "/nix/store"
    isdir(store) || return String[]
    roots = String[]
    for name in readdir(store)
        occursin(r"-qhull-[0-9]", name) || continue
        root = joinpath(store, name)
        valid_qhull_root(root) && push!(roots, root)
    end
    sort!(roots)
end

function miniqhull_package_root()
    for depot in DEPOT_PATH
        root = joinpath(depot, "packages", "MiniQhull")
        isdir(root) || continue
        versions = filter(isdir, readdir(root; join=true))
        isempty(versions) && continue
        legacy = filter(p -> isfile(joinpath(p, "deps", "build.jl")), versions)
        pool = isempty(legacy) ? versions : legacy
        return first(sort(pool; by=p -> stat(p).mtime, rev=true))
    end
    error("MiniQhull is not installed in DEPOT_PATH; run Gay.jl tests once to instantiate test dependencies.")
end

qhull_root = get(ENV, "QHULL_ROOT_DIR", "")
if isempty(qhull_root)
    roots = nix_qhull_roots()
    isempty(roots) && error("No Nix Qhull root found. Set QHULL_ROOT_DIR=/path/to/qhull and rerun.")
    qhull_root = last(roots)
end

valid_qhull_root(qhull_root) || error("QHULL_ROOT_DIR does not look like a Qhull root: $qhull_root")
ENV["QHULL_ROOT_DIR"] = qhull_root

pkgroot = miniqhull_package_root()
println("QHULL_ROOT_DIR=", qhull_root)
println("MiniQhull package=", pkgroot)

Pkg.activate(pkgroot)
Pkg.instantiate()

deps_jl = joinpath(pkgroot, "deps", "deps.jl")
if isfile(joinpath(pkgroot, "deps", "build.jl"))
    Pkg.build()
    isfile(deps_jl) || error("MiniQhull build finished without creating $deps_jl")
    println("MiniQhull deps.jl=", deps_jl)
else
    println("MiniQhull uses a JLL-backed package layout; no deps.jl build artifact expected.")
end

@eval using MiniQhull
cells = MiniQhull.delaunay([0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0])
cells == Int32[4 4; 2 3; 1 1] || error("MiniQhull smoke test returned unexpected cells: $cells")
println("MiniQhull smoke test passed.")
