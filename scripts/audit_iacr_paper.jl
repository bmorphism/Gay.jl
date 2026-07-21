#!/usr/bin/env julia

root = normpath(joinpath(@__DIR__, ".."))
paper_dir = joinpath(root, "papers", "iacr-entropy-as-color")
tex_path = joinpath(paper_dir, "main.tex")
bib_path = joinpath(paper_dir, "refs.bib")
readme_path = joinpath(paper_dir, "README.md")
readiness_path = joinpath(paper_dir, "READINESS.md")

tex = read(tex_path, String)
bib = read(bib_path, String)
artifact_readme = read(readme_path, String)
readiness = read(readiness_path, String)

struct AuditItem
    id::Symbol
    passed::Bool
    evidence::String
end

items = AuditItem[]
check(id, passed, evidence) = push!(items, AuditItem(id, passed, evidence))

for (term, id) in [
    ("security parameter", :security_parameter),
    ("ideal functionality", :ideal_functionality),
    ("simulator", :simulator),
    ("adversary", :adversary_model),
    ("environment", :uc_environment),
]
    check(id, occursin(term, lowercase(tex)), "manuscript contains `$term`")
end

check(:no_proof_sketch,
      !occursin("[Proof sketch]", tex),
      "theorem bodies contain no proof-sketch placeholder")
check(:no_unqualified_uc_claim,
      !occursin("achieves universal composability", lowercase(tex)),
      "no UC conclusion before a complete UC definition and proof")
check(:no_unqualified_amplification,
      !occursin("amplify effective entropy", lowercase(tex)),
      "no entropy-amplification claim without extractor parameters and bounds")
check(:no_false_leftover_hash_attribution,
      !occursin("leftover hash lemma applied to independent sources", lowercase(tex)),
      "leftover-hash lemma is not used to justify bare concatenation")

lean_sources = filter(path -> endswith(path, ".lean"),
                      readdir(joinpath(root, "lean4"); join=true))
lean_text = join(read.(lean_sources, String), "\n")
for theorem_name in ["char_three", "moebius_preserves_balance"]
    check(Symbol("lean_" * theorem_name),
          occursin(Regex("theorem\\s+" * theorem_name * "\\b"), lean_text),
          "Lean source declares `$theorem_name`")
end
positive_security_claim = occursin("achieves universal composability", lowercase(tex))
check(:formal_security_crosswalk,
      !positive_security_claim ||
          occursin("universal_composability", lean_text) ||
          occursin("idealFunctionality", lean_text),
      positive_security_claim ?
          "positive security claim has a formal definition crosswalk" :
          "not applicable: manuscript makes no positive composable-security claim")

cites = Set(m.captures[1] for m in eachmatch(r"\\cite\{([^}]+)\}", tex))
cites = Set(strip(key) for group in cites for key in split(group, ','))
bibkeys = Set(m.captures[1] for m in eachmatch(r"@[A-Za-z]+\{([^,]+),", bib))
missing_cites = sort!(collect(setdiff(cites, bibkeys)))
check(:citations_resolve, isempty(missing_cites),
      isempty(missing_cites) ? "all citation keys resolve" : "missing: $(join(missing_cites, ", "))")
check(:no_placeholder_authors,
      !occursin("and others", lowercase(bib)),
      "bibliography contains complete author lists")

for (term, id) in [
    ("dependencies", :artifact_dependencies),
    ("platform", :artifact_platform),
    ("runtime", :artifact_runtime),
    ("expected output", :artifact_expected_output),
    ("license", :artifact_license),
]
    check(id, occursin(term, lowercase(artifact_readme)),
          "artifact README documents $term")
end
check(:result_crosswalk,
      occursin("claim", lowercase(artifact_readme)) && occursin("command", lowercase(artifact_readme)),
      "artifact README maps paper claims to commands and outputs")
check(:selected_venue_contract,
      occursin("## Selected target venue", artifact_readme),
      "one target venue, dated call, format, anonymity, and deadline contract is frozen")
open_rubric = collect(eachmatch(r"\*\*(?:Missing|Incomplete|Contradicted|Failing|Not yet attestable)", readiness))
check(:no_open_rubric_blockers,
      isempty(open_rubric),
      isempty(open_rubric) ?
          "readiness rubric has no unresolved blocking status" :
          "readiness rubric still contains $(length(open_rubric)) unresolved blocking statuses")

passed = count(item -> item.passed, items)
failed = length(items) - passed
println("IACR paper readiness: $passed/$(length(items)) gates pass")
for item in items
    marker = item.passed ? "PASS" : "FAIL"
    println(rpad(marker, 5), " ", rpad(string(item.id), 34), " ", item.evidence)
end
println((ready=failed == 0, passed=passed, failed=failed))

if "--strict" in ARGS && failed > 0
    exit(1)
end
