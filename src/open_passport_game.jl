# Open Passport Game — HumanAOS-shaped open games where players ARE did:gay:* holders
#
# Three scales of the same structure:
#   1. Syntax:  Gay.jl S-expression coloring (Ising spins on terms)
#   2. Formal:  Ghani-Hedges open games (play/coplay duality)
#   3. Economy: HumanAOS bounty board (human agents as spin sites)
#
# The passport is the identity layer that pins agents across all three.
# Magnetization ≈ 0 at equilibrium means supply ≈ demand.
#
# Stellogen notation:
#   Bounty(task, reward) : Play(did:gay:agent) ⊗ CoPlay(did:gay:verifier) → Outcome
#   ⟨M⟩ = Σ commitment_spin / N_agents ≈ 0 at Nash equilibrium

module OpenPassportGame

using SplittableRandoms: SplittableRandom, split
using Printf: @sprintf

# Import from sibling modules via parent
import ..GayPassport: Passport, GayDID, did_gay, passport_fingerprint,
    trust_score, verification_level, verify_passport, is_human,
    VerificationLevel, UNVERIFIED, DEVICE_VERIFIED, WITNESS_VERIFIED, MULTI_WITNESS

export Bounty, BountyBoard, PassportPlayer, Commitment
export OpenGame, Play, CoPlay, Outcome
export post_bounty!, commit!, verify_and_pay!, withdraw!
export board_magnetization, is_equilibrium, board_fingerprint
export open_game_compose, open_game_tensor
export world_open_passport_game, OpenPassportGameWorld

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

const GOLDEN = UInt64(0x9e3779b97f4a7c15)

function splitmix64(x::UInt64)
    x = (x ⊻ (x >> 30)) * UInt64(0xbf58476d1ce4e5b9)
    x = (x ⊻ (x >> 27)) * UInt64(0x94d049bb133111eb)
    x ⊻ (x >> 31)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Core Types — Ghani-Hedges Open Game on Passports
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Bounty

A task on the bounty board. In open game terms:
- `reward` is the utility
- `tags` define the constraint graph edges
- `required_level` gates participation by passport hardening mode
"""
struct Bounty
    id::UInt64
    description::String
    reward::Float64
    tags::Vector{Symbol}
    location::String
    required_level::VerificationLevel
    posted_by::Passport
    seed::UInt64  # deterministic color for this bounty
end

function Bounty(desc::String, reward::Float64, posted_by::Passport;
                tags::Vector{Symbol}=Symbol[],
                location::String="anywhere",
                required_level::VerificationLevel=DEVICE_VERIFIED)
    seed = splitmix64(hash(desc) ⊻ posted_by.did.seed)
    Bounty(seed, desc, reward, tags, location, required_level, posted_by, seed)
end

"""
    Commitment

An agent's commitment to a bounty — the "spin" in the Ising model.
- spin = +1: committed (supply)
- spin = -1: requesting (demand)
- spin = 0: withdrawn/neutral

Magnetization ⟨M⟩ = Σspin/N ≈ 0 at equilibrium (supply meets demand).
"""
struct Commitment
    bounty_id::UInt64
    agent::Passport
    spin::Int8        # +1 committed, -1 requesting, 0 withdrawn
    timestamp::Float64
    stake::Float64    # trust_score at time of commitment
end

"""
    PassportPlayer

An agent in the open game, identified by their did:gay:* passport.
The trust score determines their influence on equilibrium.
"""
struct PassportPlayer
    passport::Passport
    commitments::Vector{Commitment}
    total_earned::Float64
    total_staked::Float64
end

PassportPlayer(passport::Passport) = PassportPlayer(passport, Commitment[], 0.0, 0.0)

# ═══════════════════════════════════════════════════════════════════════════════
# Play / CoPlay — Ghani-Hedges Duality
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Play

Forward pass: agent commits to performing a bounty task.
In Ghani-Hedges: Play : Σ → X (strategy → observation)
In HumanAOS: agent picks up a task from the board.
In Ising: spin site flips to +1 (committed).
"""
struct Play
    player::PassportPlayer
    bounty::Bounty
    commitment::Commitment
end

"""
    CoPlay

Backward pass: verifier witnesses task completion and releases reward.
In Ghani-Hedges: CoPlay : Σ × R → () (strategy × utility → continuation)
In HumanAOS: verifier confirms work, payment flows.
In Ising: interaction energy J between neighboring spins.
"""
struct CoPlay
    verifier::PassportPlayer
    play::Play
    verified::Bool
    reward_released::Float64
end

"""
    Outcome

Result of Play ⊗ CoPlay composition.
"""
struct Outcome
    play::Play
    coplay::CoPlay
    net_utility::Float64  # reward - stake cost
    fingerprint::UInt64
end

"""
    OpenGame

A composable open game over passport holders.
Notation: G : S → (X, R) where
  S = strategy profiles (set of commitments)
  X = observations (bounty board state)
  R = utility (reward distribution)
"""
struct OpenGame
    players::Vector{PassportPlayer}
    bounties::Vector{Bounty}
    plays::Vector{Play}
    coplays::Vector{CoPlay}
    outcomes::Vector{Outcome}
end

OpenGame() = OpenGame(PassportPlayer[], Bounty[], Play[], CoPlay[], Outcome[])

# ═══════════════════════════════════════════════════════════════════════════════
# Bounty Board — The Marketplace as Constraint Graph
# ═══════════════════════════════════════════════════════════════════════════════

"""
    BountyBoard

The full marketplace. Each bounty is a node in the constraint graph,
each commitment is a spin, and magnetization measures equilibrium.
"""
mutable struct BountyBoard
    bounties::Dict{UInt64, Bounty}
    commitments::Vector{Commitment}
    players::Dict{String, PassportPlayer}  # DID string → player
    completed::Vector{Outcome}
end

BountyBoard() = BountyBoard(Dict{UInt64, Bounty}(), Commitment[],
                             Dict{String, PassportPlayer}(), Outcome[])

"""
    post_bounty!(board, bounty) -> BountyBoard

Post a new bounty to the board. The poster's passport must be verified.
"""
function post_bounty!(board::BountyBoard, bounty::Bounty)
    verification_level(bounty.posted_by) >= bounty.required_level ||
        error("Poster does not meet required verification level")
    board.bounties[bounty.id] = bounty
    # Register poster as player if not already
    did = did_gay(bounty.posted_by.did)
    if !haskey(board.players, did)
        board.players[did] = PassportPlayer(bounty.posted_by)
    end
    board
end

"""
    commit!(board, passport, bounty_id; spin=+1) -> Play

Commit to a bounty (spin=+1) or request fulfillment (spin=-1).
Returns the Play (forward pass of the open game).

Gate: passport verification level must meet bounty requirements.
"""
function commit!(board::BountyBoard, passport::Passport, bounty_id::UInt64; spin::Int8=Int8(1))
    bounty = get(board.bounties, bounty_id, nothing)
    bounty !== nothing || error("Bounty not found: $bounty_id")
    verification_level(passport) >= bounty.required_level ||
        error("Passport does not meet required verification level")

    stake = trust_score(passport)
    c = Commitment(bounty_id, passport, spin, time(), stake)
    push!(board.commitments, c)

    did = did_gay(passport.did)
    if !haskey(board.players, did)
        board.players[did] = PassportPlayer(passport)
    end
    player = board.players[did]
    push!(player.commitments, c)

    Play(player, bounty, c)
end

"""
    verify_and_pay!(board, verifier_passport, play) -> Outcome

CoPlay: verifier witnesses completion and releases reward.
The verifier must be a different passport holder with sufficient trust.
This is the IRL witnessing model applied to work verification.
"""
function verify_and_pay!(board::BountyBoard, verifier_passport::Passport, play::Play)
    # Verifier must be different from player
    did_gay(verifier_passport.did) != did_gay(play.player.passport.did) ||
        error("Cannot verify own work")

    is_human(verifier_passport) ||
        error("Verifier must be a blessed passport holder (is_human)")

    did_v = did_gay(verifier_passport.did)
    if !haskey(board.players, did_v)
        board.players[did_v] = PassportPlayer(verifier_passport)
    end
    verifier = board.players[did_v]

    reward = play.bounty.reward
    cp = CoPlay(verifier, play, true, reward)

    # Compute outcome
    net = reward - play.commitment.stake * 0.1  # small stake cost
    fp = splitmix64(passport_fingerprint(play.player.passport) ⊻
                    passport_fingerprint(verifier_passport) ⊻
                    play.bounty.seed)
    outcome = Outcome(play, cp, net, fp)
    push!(board.completed, outcome)

    outcome
end

"""
    withdraw!(board, passport, bounty_id) -> Nothing

Withdraw a commitment (spin → 0).
"""
function withdraw!(board::BountyBoard, passport::Passport, bounty_id::UInt64)
    did = did_gay(passport.did)
    filter!(c -> !(c.bounty_id == bounty_id && did_gay(c.agent.did) == did),
            board.commitments)
    nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# Magnetization — Ising Order Parameter on the Bounty Board
# ═══════════════════════════════════════════════════════════════════════════════

"""
    board_magnetization(board::BountyBoard) -> Float64

Compute magnetization ⟨M⟩ = Σ spin / N over all active commitments.

⟨M⟩ ≈ 0: supply ≈ demand (equilibrium)
⟨M⟩ > 0: more supply than demand (workers looking for work)
⟨M⟩ < 0: more demand than supply (tasks unfilled)

This is the same quantity as Gay.jl's lattice_magnetization and
gay_sexpr_magnetization — the Ising order parameter, but over an
economic system instead of syntax or a spin lattice.
"""
function board_magnetization(board::BountyBoard)
    isempty(board.commitments) && return 0.0
    total = sum(Float64(c.spin) for c in board.commitments)
    total / length(board.commitments)
end

"""
    is_equilibrium(board::BountyBoard; tol=0.1) -> Bool

Nash equilibrium ≈ magnetization near zero.
No agent can improve their utility by unilaterally changing commitment.
"""
function is_equilibrium(board::BountyBoard; tol::Float64=0.1)
    abs(board_magnetization(board)) < tol
end

"""
    board_fingerprint(board::BountyBoard) -> UInt64

SPI-compliant XOR fingerprint of the entire board state.
"""
function board_fingerprint(board::BountyBoard)
    fp = UInt64(0)
    for (_, bounty) in board.bounties
        fp ⊻= bounty.seed
    end
    for c in board.commitments
        fp ⊻= splitmix64(passport_fingerprint(c.agent) ⊻ UInt64(c.spin + 2))
    end
    for o in board.completed
        fp ⊻= o.fingerprint
    end
    fp
end

# ═══════════════════════════════════════════════════════════════════════════════
# Composition — Sequential and Parallel Game Composition
# ═══════════════════════════════════════════════════════════════════════════════

"""
    open_game_compose(g1::OpenGame, g2::OpenGame) -> OpenGame

Sequential composition: g1 ; g2 (output of g1 feeds into g2).
In Stellogen: G₁ ∘ G₂ : S₁ × S₂ → (X₂, R₁ + R₂)
"""
function open_game_compose(g1::OpenGame, g2::OpenGame)
    OpenGame(
        vcat(g1.players, g2.players),
        vcat(g1.bounties, g2.bounties),
        vcat(g1.plays, g2.plays),
        vcat(g1.coplays, g2.coplays),
        vcat(g1.outcomes, g2.outcomes)
    )
end

"""
    open_game_tensor(g1::OpenGame, g2::OpenGame) -> OpenGame

Parallel (tensor) composition: g1 ⊗ g2 (independent games running simultaneously).
In Stellogen: G₁ ⊗ G₂ : S₁ × S₂ → (X₁ × X₂, R₁ × R₂)
"""
function open_game_tensor(g1::OpenGame, g2::OpenGame)
    # Same as compose structurally, but semantically independent
    open_game_compose(g1, g2)
end

# ═══════════════════════════════════════════════════════════════════════════════
# World Builder (AGENTS.md compliant)
# ═══════════════════════════════════════════════════════════════════════════════

struct OpenPassportGameWorld
    board::BountyBoard
    magnetization::Float64
    equilibrium::Bool
    n_bounties::Int
    n_players::Int
    n_completed::Int
    fingerprint_val::UInt64
end

function Base.length(w::OpenPassportGameWorld)
    w.n_bounties
end

function Base.merge(w1::OpenPassportGameWorld, w2::OpenPassportGameWorld)
    board = BountyBoard()
    for (k, v) in w1.board.bounties
        board.bounties[k] = v
    end
    for (k, v) in w2.board.bounties
        board.bounties[k] = v
    end
    append!(board.commitments, w1.board.commitments)
    append!(board.commitments, w2.board.commitments)
    for (k, v) in w1.board.players
        board.players[k] = v
    end
    for (k, v) in w2.board.players
        board.players[k] = v
    end
    append!(board.completed, w1.board.completed)
    append!(board.completed, w2.board.completed)

    mag = board_magnetization(board)
    OpenPassportGameWorld(
        board, mag, abs(mag) < 0.1,
        length(board.bounties), length(board.players),
        length(board.completed), board_fingerprint(board)
    )
end

"""
    world_open_passport_game(; passports, bounties) -> OpenPassportGameWorld

Build an open passport game world.

# Example: Vivarium bounty board
```julia
# Issue passports
alice = issue_passport(UInt64(1069))
witness_passport(alice, UInt64(42), "vivarium")
bob = issue_passport(UInt64(420))
witness_passport(bob, UInt64(1069), "vivarium")

# Build game world
w = world_open_passport_game(
    passports = [alice, bob],
    bounties = [
        ("Build BCI integration", 100.0, alice, [:bci, :minecraft]),
        ("Test color pipeline", 50.0, bob, [:testing, :colors]),
    ]
)
```
"""
function world_open_passport_game(;
    passports::Vector{Passport}=Passport[],
    bounties::Vector{<:Tuple}=Tuple[]
)
    board = BountyBoard()

    for (desc, reward, poster, tags) in bounties
        b = Bounty(desc, reward, poster; tags=tags)
        post_bounty!(board, b)
    end

    # Register all passport holders
    for p in passports
        did = did_gay(p.did)
        if !haskey(board.players, did)
            board.players[did] = PassportPlayer(p)
        end
    end

    mag = board_magnetization(board)
    fp = board_fingerprint(board)
    OpenPassportGameWorld(
        board, mag, abs(mag) < 0.1,
        length(board.bounties), length(board.players),
        length(board.completed), fp
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════

function Base.show(io::IO, b::Bounty)
    print(io, "Bounty(\"", b.description, "\" reward=", b.reward,
          " level=", b.required_level, " tags=", b.tags, ")")
end

function Base.show(io::IO, board::BountyBoard)
    mag = board_magnetization(board)
    eq = abs(mag) < 0.1 ? "EQUILIBRIUM" : "disequilibrium"
    print(io, "BountyBoard(bounties=", length(board.bounties),
          " players=", length(board.players),
          " commitments=", length(board.commitments),
          " completed=", length(board.completed),
          " ⟨M⟩=", @sprintf("%.3f", mag),
          " [", eq, "])")
end

function Base.show(io::IO, w::OpenPassportGameWorld)
    print(io, "OpenPassportGameWorld(bounties=", w.n_bounties,
          " players=", w.n_players,
          " completed=", w.n_completed,
          " ⟨M⟩=", @sprintf("%.3f", w.magnetization),
          " eq=", w.equilibrium,
          " fp=0x", string(w.fingerprint_val, base=16, pad=16), ")")
end

end # module OpenPassportGame
