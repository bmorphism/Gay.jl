# Generated projection artifact; requires Catlab/GATlab to execute.
# fingerprint: 0x0a59ea3e6ed448d8
@present SchGayCounterfactualClosure(FreeSchema) begin
    TestWitness::Ob
    ClosureAspect::Ob
    CounterfactualAssignment::Ob
    CatColabDecl::Ob
    LispForm::Ob
    SharedArena::Ob
    Color::AttrType
    Trit::AttrType
    Cost::AttrType
    CounterfactualEffect::AttrType
    ScipAddress::AttrType

    has_aspect::Hom(TestWitness, ClosureAspect)
    has_counterfactual::Hom(TestWitness, CounterfactualAssignment)
    from_aspect::Hom(CounterfactualAssignment, ClosureAspect)
    to_aspect::Hom(CounterfactualAssignment, ClosureAspect)
    declares_object::Hom(ClosureAspect, CatColabDecl)
    as_declared_object::Hom(CounterfactualAssignment, CatColabDecl)
    observed_as::Hom(LispForm, TestWitness)
    language_assigns_aspect::Hom(LispForm, ClosureAspect)
    shared_in::Hom(CounterfactualAssignment, SharedArena)
    witness_arena::Hom(TestWitness, SharedArena)
    witness_color::Attr(TestWitness, Color)
    aspect_trit::Attr(ClosureAspect, Trit)
    counterfactual_cost::Attr(CounterfactualAssignment, Cost)
    counterfactual_effect::Attr(CounterfactualAssignment, CounterfactualEffect)
    scip_uri::Attr(CatColabDecl, ScipAddress)

    # equation 1: has_counterfactual ; from_aspect == has_aspect
    # equation 2: has_counterfactual ; to_aspect ; declares_object == has_counterfactual ; as_declared_object
    # equation 3: observed_as ; has_aspect == language_assigns_aspect
    # equation 4: has_counterfactual ; shared_in == witness_arena
end
