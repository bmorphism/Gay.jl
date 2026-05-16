using Test

module XFKernelFixture
include(joinpath(@__DIR__, "..", "..", "src", "xf", "kernels.jl"))
end

@testset "XF SplitMixRGB" begin
    seed = UInt64(0x78656e6f66656d21)
    stream = XFKernelFixture.SplitMixRGB(seed)

    @test XFKernelFixture.XF_SEED == seed
    @test stream(1) == XFKernelFixture.splitmix_rgb(1; seed=seed)
    @test stream(42) == XFKernelFixture.splitmix_rgb(seed, 42)
    @test XFKernelFixture.splitmix_rgb(42; seed=seed) ==
          XFKernelFixture.hash_color_rgb(seed, UInt64(42))

    colors = [stream(i) for i in 1:128]
    threaded = Vector{NTuple{3, Float32}}(undef, 128)
    Threads.@threads for i in 1:128
        threaded[i] = stream(i)
    end

    @test colors == threaded
    @test all(c -> all(x -> 0.0f0 <= x <= 1.0f0, c), colors)
    @test XFKernelFixture.with_splitmix_rgb(seed) do continued
        (continued.seed, continued(7))
    end == (seed, stream(7))

    mat = XFKernelFixture.xf_ka_colors(128, seed; backend=XFKernelFixture.CPU())
    @test all(i -> Tuple(mat[i, :]) == stream(i), 1:128)
end
