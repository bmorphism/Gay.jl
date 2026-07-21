#!/usr/bin/env julia

root = normpath(joinpath(@__DIR__, ".."))
strict = "--strict" in ARGS
full = "--full" in ARGS

function check(label, command)
    println("\n==> ", label)
    run(Cmd(command; dir=root))
end

check("claim ledger", Cmd(["bb", "scripts/verify_iacr_claims.bb"]))
check("artifact manifest", Cmd(["bb", "scripts/verify_iacr_artifact.bb"]))

readiness = ["julia", "--project=.", "scripts/audit_iacr_paper.jl"]
strict && push!(readiness, "--strict")
check("paper readiness", Cmd(readiness))
check("repository terminology",
      Cmd(["julia", "--project=.", "scripts/lint_no_demo.jl"]))

if full
    check("full package tests",
          Cmd(["julia", "--project=.", "-e", "using Pkg; Pkg.test()"]))
end

println("\nIACR paper checks passed", strict ? " in strict mode" : "")
