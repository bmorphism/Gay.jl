# Generated projection artifact; requires GATlab to execute.
# fingerprint: 0x0a59ea3e6ed448d8
@theory ThGayCounterfactualClosure begin
    TestWitness::TYPE
    ClosureAspect::TYPE
    CounterfactualAssignment::TYPE
    CatColabDecl::TYPE
    LispForm::TYPE
    SharedArena::TYPE
    Color::TYPE
    Trit::TYPE
    Cost::TYPE
    CounterfactualEffect::TYPE
    ScipAddress::TYPE

    has_aspect(x1::TestWitness)::ClosureAspect
    has_counterfactual(x2::TestWitness)::CounterfactualAssignment
    from_aspect(x3::CounterfactualAssignment)::ClosureAspect
    to_aspect(x4::CounterfactualAssignment)::ClosureAspect
    declares_object(x5::ClosureAspect)::CatColabDecl
    as_declared_object(x6::CounterfactualAssignment)::CatColabDecl
    observed_as(x7::LispForm)::TestWitness
    language_assigns_aspect(x8::LispForm)::ClosureAspect
    shared_in(x9::CounterfactualAssignment)::SharedArena
    witness_arena(x10::TestWitness)::SharedArena
    witness_color(x11::TestWitness)::Color
    aspect_trit(x12::ClosureAspect)::Trit
    counterfactual_cost(x13::CounterfactualAssignment)::Cost
    counterfactual_effect(x14::CounterfactualAssignment)::CounterfactualEffect
    scip_uri(x15::CatColabDecl)::ScipAddress

    # law 1 under x1::TestWitness
    # from_aspect(has_counterfactual(x1)) == has_aspect(x1)
    # law 2 under x2::TestWitness
    # declares_object(to_aspect(has_counterfactual(x2))) == as_declared_object(has_counterfactual(x2))
    # law 3 under x3::LispForm
    # has_aspect(observed_as(x3)) == language_assigns_aspect(x3)
    # law 4 under x4::TestWitness
    # shared_in(has_counterfactual(x4)) == witness_arena(x4)
end
