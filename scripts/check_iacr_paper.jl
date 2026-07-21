#!/usr/bin/env julia

root = normpath(joinpath(@__DIR__, ".."))
strict = "--strict" in ARGS
full = "--full" in ARGS
pdf = full || "--pdf" in ARGS

function check(label, command)
    println("\n==> ", label)
    run(Cmd(command; dir=root))
end

check("claim ledger", Cmd(["bb", "scripts/verify_iacr_claims.bb"]))
check("standards ledger", Cmd(["bb", "scripts/verify_iacr_standards.bb"]))
check("artifact manifest", Cmd(["bb", "scripts/verify_iacr_artifact.bb"]))
check("referent boundary",
      Cmd(["bb", "scripts/verify_referent_boundary.bb", "--self-test"]))
check("interaction boundary",
      Cmd(["bb", "scripts/verify_higher_order_interactions.bb", "--self-test"]))

readiness = ["julia", "--project=.", "scripts/audit_iacr_paper.jl"]
strict && push!(readiness, "--strict")
check("paper readiness", Cmd(readiness))
check("repository terminology",
      Cmd(["julia", "--project=.", "scripts/lint_no_demo.jl"]))
pdf && check("rendered PDF", Cmd(["bb", "scripts/check_iacr_pdf.bb"]))

if full
    check("full package tests",
          Cmd(["julia", "--project=.", "-e", "using Pkg; Pkg.test()"]))
end

println("\nIACR paper checks passed", strict ? " in strict mode" : "")
