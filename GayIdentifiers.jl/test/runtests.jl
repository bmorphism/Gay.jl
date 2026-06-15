using Test
using GayIdentifiers

@testset "GayIdentifiers — parse + Gay layer" begin
    d = parse(GayID, "https://doi.org/10.1371/journal.pone.0068810")
    @test idkind(d) == :doi
    @test shortcode(d) == "10.1371/journal.pone.0068810"
    @test startswith(purl(d), "https://doi.org/")
    @test occursin(r"^#[0-9A-F]{6}$", gaycolor(d))
    @test gaytrit(d) in Int8(-1):Int8(1)

    o = parse(GayID, "0000-0002-1825-0097")
    @test idkind(o) == :orcid
    @test shortcode(o) == "0000-0002-1825-0097"

    a = parse(GayID, "arXiv:2507.08892")
    @test idkind(a) == :arxiv
    @test startswith(purl(a), "https://arxiv.org/abs/")

    w = parse(GayID, "world://securities")
    @test idkind(w) == :world && shortcode(w) == "securities"
    v = parse(GayID, "vm://sierpinski/012")
    @test idkind(v) == :vm

    # canonicalisation ⇒ same colour from equivalent inputs
    @test gaycolor(d) == gaycolor(parse(GayID, "doi:10.1371/journal.pone.0068810"))
    # distinct identifiers ⇒ distinct colours
    @test gaycolor(w) != gaycolor(v)
end

@testset "GF(3) identity audit (world://groupoid Σ)" begin
    a = gay_audit(["world://securities", "world://groupoid", "world://morphism"])
    @test a.n == 3
    @test a.sigma in 0:2
    @test length(a.colors) == 3
    @test all(c -> occursin(r"^#[0-9A-F]{6}$", c), values(a.colors))
end
