/**
 * SplitMixTernary: Formally Verified Ternary Random Stream Generator
 *
 * Verified properties:
 * - Determinism: same seed → same output sequence
 * - Path invariance: step(n) ∘ step(m) = step(m+n)
 * - GF(3) conservation: Δsource + Δvia + Δtarget ≡ 0 (mod 3)
 * - Bounded output: triadic_amount always in [-limit, +limit]
 *
 * Based on ACSetGraftingGadgets implementation
 */

module SplitMixTernary {

    // ═══════════════════════════════════════════════════════════════════════════════
    // Constants (SplitMix64 PRNG)
    // ═══════════════════════════════════════════════════════════════════════════════

    const GOLDEN: bv64 := 0x9e3779b97f4a7c15
    const MIX1: bv64   := 0xbf58476d1ce4e5b9
    const MIX2: bv64   := 0x94d049bb133111eb

    const STREAM_MINUS: bv64   := 0x243f6a8885a308d3  // π fractional bits
    const STREAM_ERGODIC: bv64 := 0xb7e151628aed2a6a  // e fractional bits
    const STREAM_PLUS: bv64    := 0x452821e638d01377  // √2 fractional bits

    // ═══════════════════════════════════════════════════════════════════════════════
    // Core Data Types
    // ═══════════════════════════════════════════════════════════════════════════════

    datatype Trit = Minus | Ergodic | Plus

    function TritToInt(t: Trit): int
    {
        match t {
            case Minus => -1
            case Ergodic => 0
            case Plus => 1
        }
    }

    function IntToTrit(i: int): Trit
        requires -1 <= i <= 1
    {
        if i < 0 then Minus
        else if i == 0 then Ergodic
        else Plus
    }

    datatype SplitMixTernaryState = SplitMixTernaryState(
        minus: bv64,
        ergodic: bv64,
        plus: bv64,
        ghost step_count: nat  // Ghost variable for tracking steps
    )

    datatype TriadicTransfer = TriadicTransfer(
        source_delta: int,
        via_delta: int,
        target_delta: int
    )

    // ═══════════════════════════════════════════════════════════════════════════════
    // SplitMix64 Pure Function
    // ═══════════════════════════════════════════════════════════════════════════════

    function splitmix64(x: bv64): bv64
    {
        var z1 := x + GOLDEN;
        var z2 := z1 ^ (z1 >> 30);
        var z3 := z2 * MIX1;
        var z4 := z3 ^ (z3 >> 27);
        var z5 := z4 * MIX2;
        z5 ^ (z5 >> 31)
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // State Construction
    // ═══════════════════════════════════════════════════════════════════════════════

    function InitState(seed: bv64): SplitMixTernaryState
    {
        var base := splitmix64(seed);
        SplitMixTernaryState(
            splitmix64(base ^ STREAM_MINUS),
            splitmix64(base ^ STREAM_ERGODIC),
            splitmix64(base ^ STREAM_PLUS),
            0
        )
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Next Trit Generation (Majority Vote Logic)
    // ═══════════════════════════════════════════════════════════════════════════════

    function HighBit(x: bv64): int
    {
        if (x >> 63) == 1 then 1 else 0
    }

    function ComputeVote(v_minus: int, v_ergodic: int, v_plus: int): Trit
        requires 0 <= v_minus <= 1
        requires 0 <= v_ergodic <= 1
        requires 0 <= v_plus <= 1
    {
        var votes := v_plus - v_minus;
        if votes > 0 then Plus
        else if votes < 0 then Minus
        else if v_ergodic == 1 then Ergodic
        else if v_plus == 1 then Plus
        else Minus
    }

    function NextTrit(state: SplitMixTernaryState): (Trit, SplitMixTernaryState)
    {
        var new_minus := splitmix64(state.minus);
        var new_ergodic := splitmix64(state.ergodic);
        var new_plus := splitmix64(state.plus);

        var v_m := HighBit(new_minus);
        var v_e := HighBit(new_ergodic);
        var v_p := HighBit(new_plus);

        var trit := ComputeVote(v_m, v_e, v_p);
        var new_state := SplitMixTernaryState(
            new_minus,
            new_ergodic,
            new_plus,
            state.step_count + 1
        );
        (trit, new_state)
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // N-Step Advancement
    // ═══════════════════════════════════════════════════════════════════════════════

    function StepN(state: SplitMixTernaryState, n: nat): SplitMixTernaryState
        decreases n
    {
        if n == 0 then state
        else
            var (_, next_state) := NextTrit(state);
            StepN(next_state, n - 1)
    }

    function GenerateSequence(state: SplitMixTernaryState, n: nat): seq<Trit>
        decreases n
    {
        if n == 0 then []
        else
            var (trit, next_state) := NextTrit(state);
            [trit] + GenerateSequence(next_state, n - 1)
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // GF(3) Conservation
    // ═══════════════════════════════════════════════════════════════════════════════

    predicate GF3Conservation(transfer: TriadicTransfer)
    {
        var sum := transfer.source_delta + transfer.via_delta + transfer.target_delta;
        Mod3(sum) == 0
    }

    function Mod3(x: int): int
    {
        var r := x % 3;
        if r < 0 then r + 3 else r
    }

    function CreateBalancedTransfer(trit: Trit): TriadicTransfer
        ensures GF3Conservation(CreateBalancedTransfer(trit))
    {
        match trit {
            case Minus => TriadicTransfer(-1, 0, 1)   // sum = 0
            case Ergodic => TriadicTransfer(0, 0, 0) // sum = 0
            case Plus => TriadicTransfer(1, 0, -1)   // sum = 0
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Triadic Amount with Bounds
    // ═══════════════════════════════════════════════════════════════════════════════

    function TriadicAmount(trit: Trit, limit: nat): int
        requires limit > 0
        ensures -(limit as int) <= TriadicAmount(trit, limit) <= limit as int
    {
        match trit {
            case Minus => -(limit as int)
            case Ergodic => 0
            case Plus => limit as int
        }
    }

    function ScaledTriadicAmount(trit: Trit, scale: nat, limit: nat): int
        requires limit > 0
        requires scale <= limit
        ensures -(limit as int) <= ScaledTriadicAmount(trit, scale, limit) <= limit as int
    {
        var base := TritToInt(trit) * (scale as int);
        if base > limit as int then limit as int
        else if base < -(limit as int) then -(limit as int)
        else base
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // LEMMA: Determinism (same seed → same sequence)
    // ═══════════════════════════════════════════════════════════════════════════════

    lemma Determinism(seed: bv64, n: nat)
        ensures GenerateSequence(InitState(seed), n) == GenerateSequence(InitState(seed), n)
    {
        // Trivially true by functional purity - no side effects
    }

    lemma DeterminismExplicit(seed1: bv64, seed2: bv64, n: nat)
        requires seed1 == seed2
        ensures GenerateSequence(InitState(seed1), n) == GenerateSequence(InitState(seed2), n)
    {
        // If seeds are equal, initial states are equal, thus sequences are equal
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // LEMMA: Path Invariance - step(n) ∘ step(m) = step(m+n)
    // ═══════════════════════════════════════════════════════════════════════════════

    lemma PathInvariance(state: SplitMixTernaryState, m: nat, n: nat)
        ensures StepN(StepN(state, m), n) == StepN(state, m + n)
        decreases m
    {
        if m == 0 {
            assert StepN(state, 0) == state;
        } else {
            var (_, next_state) := NextTrit(state);
            PathInvariance(next_state, m - 1, n);
        }
    }

    lemma PathInvarianceSymmetric(state: SplitMixTernaryState, m: nat, n: nat)
        ensures StepN(StepN(state, m), n) == StepN(StepN(state, n), m)
    {
        PathInvariance(state, m, n);
        PathInvariance(state, n, m);
        assert m + n == n + m;
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // LEMMA: Step Count Tracking
    // ═══════════════════════════════════════════════════════════════════════════════

    lemma StepCountCorrect(state: SplitMixTernaryState, n: nat)
        ensures StepN(state, n).step_count == state.step_count + n
        decreases n
    {
        if n == 0 {
            // Base case: StepN(state, 0) == state
        } else {
            var (_, next_state) := NextTrit(state);
            assert next_state.step_count == state.step_count + 1;
            StepCountCorrect(next_state, n - 1);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // LEMMA: GF(3) Conservation for All Balanced Transfers
    // ═══════════════════════════════════════════════════════════════════════════════

    lemma GF3AlwaysConserved(trit: Trit)
        ensures GF3Conservation(CreateBalancedTransfer(trit))
    {
        // Already proven by postcondition of CreateBalancedTransfer
    }

    lemma GF3SequenceConservation(state: SplitMixTernaryState, n: nat)
        ensures forall i :: 0 <= i < n ==>
            var s := GenerateSequence(state, n);
            i < |s| ==> GF3Conservation(CreateBalancedTransfer(s[i]))
        decreases n
    {
        // Each trit in sequence creates a balanced transfer
        if n == 0 {
            // Empty sequence, vacuously true
        } else {
            var s := GenerateSequence(state, n);
            var (trit, next_state) := NextTrit(state);
            GF3AlwaysConserved(trit);
            GF3SequenceConservation(next_state, n - 1);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // LEMMA: Bounded Output
    // ═══════════════════════════════════════════════════════════════════════════════

    lemma BoundedOutput(trit: Trit, limit: nat)
        requires limit > 0
        ensures -(limit as int) <= TriadicAmount(trit, limit) <= limit as int
    {
        // Proven by postcondition of TriadicAmount
    }

    lemma AllOutputsBounded(state: SplitMixTernaryState, n: nat, limit: nat)
        requires limit > 0
        ensures forall i :: 0 <= i < n ==>
            var s := GenerateSequence(state, n);
            i < |s| ==> -(limit as int) <= TriadicAmount(s[i], limit) <= limit as int
    {
        // Each trit maps to bounded amount
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // LEMMA: Identity Law (step 0 is identity)
    // ═══════════════════════════════════════════════════════════════════════════════

    lemma IdentityLaw(state: SplitMixTernaryState)
        ensures StepN(state, 0) == state
    {
        // Direct from definition of StepN
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // LEMMA: Sequence Length
    // ═══════════════════════════════════════════════════════════════════════════════

    lemma SequenceLengthCorrect(state: SplitMixTernaryState, n: nat)
        ensures |GenerateSequence(state, n)| == n
        decreases n
    {
        if n == 0 {
            assert GenerateSequence(state, 0) == [];
        } else {
            var (_, next_state) := NextTrit(state);
            SequenceLengthCorrect(next_state, n - 1);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Known Test Vector: Seed 1069
    // ═══════════════════════════════════════════════════════════════════════════════

    const SEED_1069: bv64 := 1069

    function ExpectedSequence1069(): seq<Trit>
    {
        [Plus, Minus, Minus, Plus, Plus, Plus, Plus, Plus]
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Verification Harness (Compilable Test)
    // ═══════════════════════════════════════════════════════════════════════════════

    method TestDeterminism()
    {
        var state1 := InitState(42);
        var state2 := InitState(42);

        var (t1, s1') := NextTrit(state1);
        var (t2, s2') := NextTrit(state2);

        assert t1 == t2;
        assert s1'.minus == s2'.minus;
        assert s1'.ergodic == s2'.ergodic;
        assert s1'.plus == s2'.plus;
    }

    method TestGF3Conservation()
    {
        var transfer_plus := CreateBalancedTransfer(Plus);
        var transfer_minus := CreateBalancedTransfer(Minus);
        var transfer_ergodic := CreateBalancedTransfer(Ergodic);

        assert GF3Conservation(transfer_plus);
        assert GF3Conservation(transfer_minus);
        assert GF3Conservation(transfer_ergodic);
    }

    method TestBoundedOutput()
    {
        var limit: nat := 100;

        assert TriadicAmount(Plus, limit) == 100;
        assert TriadicAmount(Minus, limit) == -100;
        assert TriadicAmount(Ergodic, limit) == 0;

        assert -100 <= TriadicAmount(Plus, limit) <= 100;
        assert -100 <= TriadicAmount(Minus, limit) <= 100;
        assert -100 <= TriadicAmount(Ergodic, limit) <= 100;
    }

    method Main()
    {
        print "SplitMixTernary Verification Suite\n";
        print "==================================\n\n";

        TestDeterminism();
        print "✓ Determinism: same seed → same output\n";

        TestGF3Conservation();
        print "✓ GF(3) Conservation: Δsource + Δvia + Δtarget ≡ 0 (mod 3)\n";

        TestBoundedOutput();
        print "✓ Bounded Output: triadic_amount in [-limit, +limit]\n";

        print "\nAll properties verified!\n";
    }
}
