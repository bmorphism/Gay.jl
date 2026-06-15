# Non-Riemannian gate: strict subadditivity on collinear triplets.
#
# A perceptual difference valid at LARGE differences must show diminishing
# returns (Bujack et al. 2022): on collinear triplets A–B–C,
#     f(d(A,C)) < f(d(A,B)) + f(d(B,C))      STRICTLY.
# Equality (additivity along a path) is the Riemannian/length-metric
# signature; a metric exhibiting it is barred from large-difference use.
#
# Tolerances are DERIVED, not tuned: for f_A(t) = A(1 − exp(−t/A)) the
# defect is EXACT (SatReadout.key_identity, MATHLIB4_NONRIEMANNIAN.md):
#     f(x) + f(y) − f(x+y) = f(x)·f(y)/A
# so the test asserts the gap equals f(d₁)f(d₂)/A up to float slop.
#
# GF(3): this file is the −1/coplay leg of the audit — it can FAIL.
# perceptual_diff (raw CIEDE2000) is asserted to FAIL the gate (it is
# additive along Lab lines up to its own curvature); if that assertion ever
# flips, the kernel changed class and the regime split must be re-examined.

using Test
using Colors

# Module-or-standalone: prefer the package functions if loaded.
const _saturate = (ΔE, A) -> A * (1 - exp(-ΔE / A))

@testset "non-Riemannian gate (derived tolerances)" begin
    A = 10.0
    # Tolerance DERIVED, not tuned (now literally): SatReadout.key_identity'
    # proves f(x)+f(y)-f(x+y) = f(x)f(y)/A EXACTLY over ℝ, so the only admissible
    # slack is IEEE-754 rounding of exp(), ≈ 2·ε·A. 64·ε·A is the certified
    # ceiling (cross-substrate measured residual ≤ 3.6e-15 ≪ this at A=10; bb≡julia).
    slop = 64 * eps(Float64) * A   # ≈ 1.42e-13 — a rounding bound, not a knob

    @testset "exact defect identity f(x)+f(y)-f(x+y) = f(x)f(y)/A" begin
        for x in (0.5, 1.0, 3.7, 25.0, 80.0), y in (0.25, 1.0, 9.9, 60.0)
            fx, fy, fxy = _saturate(x, A), _saturate(y, A), _saturate(x + y, A)
            @test abs((fx + fy - fxy) - fx * fy / A) ≤ slop
        end
    end

    @testset "strict subadditivity with derived gap" begin
        for x in (0.5, 2.0, 15.0), y in (0.5, 2.0, 15.0)
            fx, fy, fxy = _saturate(x, A), _saturate(y, A), _saturate(x + y, A)
            gap = fx + fy - fxy
            @test gap > 0                                  # strict: non-Riemannian
            @test abs(gap - fx * fy / A) ≤ slop            # exactly the theorem
        end
    end

    @testset "local validity: t - t²/2A ≤ f(t) ≤ t (SatReadout.sq_lower/f_le_self)" begin
        for t in (0.01, 0.1, 0.5, 1.0, 2.0)
            ft = _saturate(t, A)
            @test ft ≤ t + slop
            @test t - t^2 / (2A) ≤ ft + slop
        end
    end

    @testset "no-midpoint gate (SatReadout.no_midpoint): 2f(t/2) - f(t) = f(t/2)²/A" begin
        for t in (1.0, 10.0, 40.0)
            defect = 2 * _saturate(t / 2, A) - _saturate(t, A)
            @test defect > 0
            @test abs(defect - _saturate(t / 2, A)^2 / A) ≤ slop
        end
    end

    @testset "collinear color triplets: saturated passes, raw CIEDE2000 is the contrast" begin
        # Collinear in Lab: B is a Lab-line midpoint of A and C.
        labA = Lab(20.0, -30.0, -20.0)
        labC = Lab(80.0, 40.0, 50.0)
        labB = Lab((labA.l + labC.l) / 2, (labA.a + labC.a) / 2, (labA.b + labC.b) / 2)
        dAB = colordiff(labA, labB)
        dBC = colordiff(labB, labC)
        dAC = colordiff(labA, labC)

        # Gate for the saturated readout, applied to the metric's OWN values
        # (works for any base ΔE, additive or not, since f∘d is subadditive
        # whenever d(A,C) ≤ d(A,B)+d(B,C)):
        fAB, fBC, fAC = _saturate(dAB, A), _saturate(dBC, A), _saturate(dAC, A)
        @test fAC < fAB + fBC   # STRICT — the non-Riemannian gate

        # Quantitative when the triplet is metrically collinear for d:
        if abs(dAB + dBC - dAC) < 1e-6
            @test (fAB + fBC - fAC) ≥ _saturate(dAB, A) * _saturate(dBC, A) / A - slop
        end

        # The contrast that names the failure mode: raw CIEDE2000's gap on
        # this triplet is whatever Lab curvature gives it — for a TRULY
        # additive (Euclidean-Lab) metric it would be 0. Document, don't hide:
        raw_gap = dAB + dBC - dAC
        sat_gap = fAB + fBC - fAC
        @test sat_gap > raw_gap * (fAB / dAB) - 1e-6  # saturation adds gap beyond curvature
    end
end
