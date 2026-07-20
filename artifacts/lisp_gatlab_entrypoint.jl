# Generated executable LispSyntax entrypoint artifact.
# It requires Gay.jl and uses the gat"..." and gat_rewrite"..." string macros.
using Gay

const LISP_GATLAB_BRIDGE_WORLD = gat"""
(gat
  (ob TestWitness)
  (ob ClosureAspect)
  (ob CounterfactualAssignment)
  (ob CatColabDecl)
  (ob LispForm)
  (ob SharedArena)
  (attrtype Color)
  (attrtype Trit)
  (attrtype Cost)
  (attrtype CounterfactualEffect)
  (attrtype ScipAddress)
  (hom has-aspect TestWitness ClosureAspect)
  (hom has-counterfactual TestWitness CounterfactualAssignment)
  (hom from-aspect CounterfactualAssignment ClosureAspect)
  (hom to-aspect CounterfactualAssignment ClosureAspect)
  (hom declares-object ClosureAspect CatColabDecl)
  (hom as-declared-object CounterfactualAssignment CatColabDecl)
  (hom observed-as LispForm TestWitness)
  (hom language-assigns-aspect LispForm ClosureAspect)
  (hom shared-in CounterfactualAssignment SharedArena)
  (hom witness-arena TestWitness SharedArena)
  (attr witness-color TestWitness Color)
  (attr aspect-trit ClosureAspect Trit)
  (attr counterfactual-cost CounterfactualAssignment Cost)
  (attr counterfactual-effect CounterfactualAssignment CounterfactualEffect)
  (attr scip-uri CatColabDecl ScipAddress)
  (eq (compose has-counterfactual from-aspect)
      (compose has-aspect))
  (eq (compose has-counterfactual to-aspect declares-object)
      (compose has-counterfactual as-declared-object))
  (eq (compose observed-as has-aspect)
      (compose language-assigns-aspect))
  (eq (compose has-counterfactual shared-in)
      (compose witness-arena)))
"""

const LISP_GATLAB_REWRITE_TRACE_FORM = raw"""
(rewrite-trace
  (fingerprint "0x8daeed567f6917d4")
  (program-fingerprint "0x36a6ddc85c3c8142")
  (program-execution-fingerprint "0x51b507c10de3c235")
  (request-count 3)
  (execution-count 3)
  (step-count 3)
  (coverage-complete true)
  (all-selected-all-materialized false)
  (all-selected-all-targets false)
  (selected-ordinals 1 2)
  (repeated-ordinals 1)
  (backends projection projection projection)
  (program
    (rewrite-program
      (rewrite-execution
        (query witness 1)
        (max-samples 2)
        (backend projection))
      (rewrite-execution
        (query effect positive-shift)
        (max-samples 1)
        (backend projection))
      (rewrite-execution
        (query ordinal 1)
        (max-samples 1)
        (backend projection))
    )
  )
  (steps
    (step 1
      (backend projection)
      (selected-ordinals 1 2)
      (introduced-ordinals 1 2)
      (selected-all-materialized false)
      (selected-all-targets false)
      (request-fingerprint "0x54a75fc4905b7b6b")
      (plan-fingerprint "0x0dc9db51db198335")
      (execution-fingerprint "0x1b6c2b6a0b54f012")
      (fingerprint "0x3ece7cd9477d51f1")
      (request
        (rewrite-execution
          (query witness 1)
          (max-samples 2)
          (backend projection))
      ))
    (step 2
      (backend projection)
      (selected-ordinals 1)
      (introduced-ordinals)
      (selected-all-materialized false)
      (selected-all-targets false)
      (request-fingerprint "0x50bd432c25252bb4")
      (plan-fingerprint "0x7b13993ec64e049b")
      (execution-fingerprint "0x77f3c1cb010e5e78")
      (fingerprint "0xbfcd0501580ddc68")
      (request
        (rewrite-execution
          (query effect positive-shift)
          (max-samples 1)
          (backend projection))
      ))
    (step 3
      (backend projection)
      (selected-ordinals 1)
      (introduced-ordinals)
      (selected-all-materialized false)
      (selected-all-targets false)
      (request-fingerprint "0xb531c4b50e55c56f")
      (plan-fingerprint "0xaa6c8febfa84171c")
      (execution-fingerprint "0xf6ac6a502bb3d1ba")
      (fingerprint "0x21553f34263e70f9")
      (request
        (rewrite-execution
          (query ordinal 1)
          (max-samples 1)
          (backend projection))
      ))
  )
)
"""

lisp_gatlab_bridge_summary(LISP_GATLAB_BRIDGE_WORLD)
parse_lisp_gatlab_rewrite_form(default_lisp_gatlab_rewrite_form())
gat_rewrite"""(rewrite-execution (query ordinal 1) (max-samples 1) (backend projection))"""
parse_lisp_gatlab_rewrite_program(default_lisp_gatlab_rewrite_program_form())
default_lisp_gatlab_rewrite_trace_form()
gat_rewrite_program"""(rewrite-program (rewrite-execution (query ordinal 1) (max-samples 1) (backend projection)))"""
lisp_gatlab_query(:ordinal, 1)
lisp_gatlab_rewrite_plan(:ordinal, 1)
lisp_gatlab_rewrite_execution(lisp_gatlab_rewrite_plan(:ordinal, 1), :projection)
lisp_gatlab_rewrite_plan(parse_lisp_gatlab_rewrite_form("(rewrite-execution (query ordinal 1) (max-samples 1) (backend projection))"))
eval(lisp_gatlab_rewrite_compile("(rewrite-execution (query ordinal 1) (max-samples 1) (backend projection))"; target=:plan))
eval(lisp_gatlab_rewrite_program_compile("(rewrite-program (rewrite-execution (query ordinal 1) (max-samples 1) (backend projection)))"; target=:execution))
eval(lisp_gatlab_rewrite_program_compile("(rewrite-program (rewrite-execution (query ordinal 1) (max-samples 1) (backend projection)))"; target=:trace))
eval(lisp_gatlab_rewrite_program_compile("(rewrite-program (rewrite-execution (query ordinal 1) (max-samples 1) (backend projection)))"; target=:trace_form))
eval(lisp_gatlab_rewrite_trace_compile(LISP_GATLAB_REWRITE_TRACE_FORM; target=:validation))
eval(lisp_gatlab_rewrite_trace_compile(LISP_GATLAB_REWRITE_TRACE_FORM; target=:validation_json))
sexp_eval("(lisp-gatlab-query 'ordinal 1)", Gay)
sexp_eval("(lisp-gatlab-rewrite-plan 'ordinal 1)", Gay)
sexp_eval("(lisp-gatlab-rewrite-execution (lisp-gatlab-rewrite-plan 'ordinal 1) 'projection)", Gay)
sexp_eval("(lisp-gatlab-rewrite-trace-validation-payload (default-lisp-gatlab-rewrite-trace-form))", Gay)
