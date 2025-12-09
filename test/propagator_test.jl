# Tests for SDF-style Propagator system
# Based on Software Design for Flexibility (Sussman & Hanson)

using Test
using Gay
using Gay.Propagator
using Gay.PropagatorLisp

@testset "Propagator Core" begin
    
    @testset "Special Values" begin
        @test isnothing_prop(TheNothing)
        @test !isnothing_prop(42)
        @test is_contradiction(TheContradiction())
        @test is_contradiction(TheContradiction("info"))
        @test !is_contradiction(42)
    end
    
    @testset "Premises and Support Sets" begin
        p1 = Premise(:p1)
        p2 = Premise(:p2)
        
        @test is_premise_in(p1)
        mark_premise_out!(p1)
        @test !is_premise_in(p1)
        mark_premise_in!(p1)
        @test is_premise_in(p1)
        
        s1 = support_set(p1, p2)
        @test all_premises_in(s1)
        
        mark_premise_out!(p2)
        @test !all_premises_in(s1)
    end
    
    @testset "Hypothetical Premises" begin
        h = hypothetical(:choice1)
        @test h.is_hypothetical
        @test is_premise_in(h)  # Initially believed
    end
    
    @testset "Merge Values" begin
        @test merge_values(TheNothing, 42) == 42
        @test merge_values(42, TheNothing) == 42
        @test merge_values(42, 42) == 42
        @test is_contradiction(merge_values(42, 43))
        @test merge_values(TheContradiction(), 42) isa TheContradictionType
    end
    
    @testset "Merge Intervals" begin
        i1 = (1.0, 10.0)
        i2 = (5.0, 15.0)
        merged = merge_values(i1, i2)
        @test merged == (5.0, 10.0)
        
        # Non-overlapping → contradiction
        i3 = (20.0, 30.0)
        @test is_contradiction(merge_values(i1, i3))
    end
    
    @testset "Cells" begin
        initialize_scheduler!()
        
        c = make_cell(:test_cell)
        @test cell_strongest(c) === TheNothing
        
        add_content!(c, 42)
        @test cell_strongest(c) == 42
        
        # Adding same value is fine
        add_content!(c, 42)
        @test cell_strongest(c) == 42
        
        # Adding conflicting value → contradiction
        add_content!(c, 43)
        @test is_contradiction(cell_strongest(c))
    end
    
    @testset "Cell Chromatic Identity" begin
        c1 = make_cell(:alpha)
        c2 = make_cell(:beta)
        c3 = make_cell(:alpha)  # Same name
        
        @test c1.color == c3.color  # Same name → same color
        @test c1.color != c2.color  # Different names → different colors (with high probability)
    end
    
end

@testset "Primitive Propagators" begin
    initialize_scheduler!()
    
    @testset "Addition" begin
        a = make_cell(:a)
        b = make_cell(:b)
        sum = make_cell(:sum)
        
        p_add(a, b, sum)
        
        add_content!(a, 3)
        add_content!(b, 4)
        run!()
        
        @test cell_strongest(sum) == 7
    end
    
    @testset "Multiplication" begin
        initialize_scheduler!()
        
        x = make_cell(:x)
        y = make_cell(:y)
        product = make_cell(:product)
        
        p_mul(x, y, product)
        
        add_content!(x, 5)
        add_content!(y, 6)
        run!()
        
        @test cell_strongest(product) == 30
    end
end

@testset "Constraint Propagators" begin
    
    @testset "c:+ Bidirectional Addition" begin
        initialize_scheduler!()
        
        a = make_cell(:a)
        b = make_cell(:b)
        sum = make_cell(:sum)
        
        c_add(a, b, sum)
        
        # Forward: a + b → sum
        add_content!(a, 3)
        add_content!(b, 4)
        run!()
        @test cell_strongest(sum) == 7
        
        # Now test backward propagation
        initialize_scheduler!()
        a2 = make_cell(:a2)
        b2 = make_cell(:b2)
        sum2 = make_cell(:sum2)
        
        c_add(a2, b2, sum2)
        
        add_content!(sum2, 10)
        add_content!(a2, 3)
        run!()
        
        # sum - a should give b
        @test cell_strongest(b2) == 7
    end
    
    @testset "c:* Bidirectional Multiplication" begin
        initialize_scheduler!()
        
        x = make_cell(:x)
        y = make_cell(:y)
        product = make_cell(:product)
        
        c_mul(x, y, product)
        
        add_content!(product, 30)
        add_content!(x, 5)
        run!()
        
        @test cell_strongest(y) == 6.0
    end
end

@testset "PropagatorLisp API" begin
    using Gay.PropagatorLisp
    
    @testset "define-cell" begin
        reset_env!()
        
        c = define_cell(:my_cell)
        @test c isa Cell
        @test c.name == :my_cell
        
        c2 = define_cell(:initialized_cell, 42)
        @test cell_strongest(c2) == 42
    end
    
    @testset "tell and content" begin
        reset_env!()
        
        define_cell(:x)
        tell_cell(:x, 100)
        
        val = cell_value(:x)
        @test val == 100
    end
    
    @testset "Propagator API" begin
        reset_env!()
        
        define_cell(:a, 3)
        define_cell(:b, 4)
        define_cell(:sum)
        prop_add(:a, :b, :sum)
        run!()
        
        @test cell_value(:sum) == 7
    end
end

@testset "Chromatic Network Visualization" begin
    using Gay.Propagator: network_palette
    using Colors: RGB
    
    initialize_scheduler!()
    
    a = make_cell(:a)
    b = make_cell(:b)
    c = make_cell(:c)
    
    p = c_add(a, b, c)
    
    palette = network_palette([a, b, c], [p])
    
    @test haskey(palette.cells, :a)
    @test haskey(palette.cells, :b)
    @test haskey(palette.cells, :c)
    @test haskey(palette.propagators, :c_add)
    
    # Colors should be RGB values
    @test palette.cells[:a] isa RGB
end

@testset "Stellar Distance Example (SDF §7.1)" begin
    # Just test that we can set up the network (propagation with intervals needs more work)
    reset_env!()
    
    define_cell(:vega_parallax)
    define_cell(:vega_distance)
    define_cell(:t)
    define_cell(:AU, 4.848136811095e-6)
    
    constraint_mul(:t, :vega_distance, :AU)
    
    env = get_env()
    
    @test haskey(env.cells, :vega_parallax)
    @test haskey(env.cells, :vega_distance)
    @test haskey(env.cells, :AU)
    
    # AU should be set
    @test cell_strongest(env.cells[:AU]) ≈ 4.848136811095e-6
end
