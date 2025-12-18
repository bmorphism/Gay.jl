# Topos Institute Talks Reference
## Relevant to Kripke Semantics, Possible Worlds, Sheaves, and Modal Logic

### Core Modal/Kripke Talks

1. **Alex Kavvos: "Two-dimensional Kripke Semantics"** (Nov 2025)
   - [YouTube](https://www.youtube.com/watch?v=X7CxkndABno)
   - Key insight: Kripke semantics and categorical semantics correspond via 2-dimensional structure
   - Worlds are objects, accessibility is morphisms

2. **Benjamin Brast McKie: "The Construction of Possible Worlds"** (Nov 2025)
   - [YouTube](https://www.youtube.com/watch?v=p-z4bhj7p-g)
   - [Paper](https://www.benbrastmckie.com/wp-content/uploads/2025/11/possible_worlds.pdf)
   - Possible worlds as functions from times to world states
   - Task relation encodes possible transitions
   - Validates perpetuity principles: □p → always p

3. **Benjamin Brast-McKie: "Programmatic Semantics"** (2024)
   - [YouTube](https://www.youtube.com/watch?v=ZqTpdJKHT_4)

### Sheaves and Topoi

4. **Alex Simpson: "Three toposes for probability and randomness"** (June 2024)
   - [YouTube](https://www.youtube.com/watch?v=Y1RkPhwJ0Mo)
   - [Slides](https://topos.institute/events/topos-colloquium/slides/2024-06-06.pdf)
   - **Topos 1: Probability Sheaves** - RV functor on standard Borel probability spaces
   - **Topos 2: Random Topos** - randomness-preserving functions, all sets measurable
   - **Topos 3: Random Probability Sheaves** - synthetic probability theory
   - Key: Random variables form a sheaf

5. **Dario Stein: "Random Variables, Independence Structures and Dagger Categories"** (May 2025)
   - [YouTube](https://www.youtube.com/watch?v=tEq5OV3eTSc)
   - [Slides](https://topos.institute/events/topos-colloquium/slides/2025-05-22.pdf)
   - Sheaves over sample spaces for probability
   - Independence structures and couplings

6. **Steve Vickers: "The Fundamental Theorem of Calculus: point-free"** (May 2024)
   - [YouTube](https://www.youtube.com/watch?v=L6LPEFteLts)
   - Point-free topology via geometric logic
   - Classifying toposes

7. **Jonathan Sterling: "Synthetic Domains in the 21st Century"** (Sept 2023)
   - [YouTube](https://www.youtube.com/watch?v=lLvweTSmR40)
   - Synthetic domain theory

### Categorical Logic and Type Theory

8. **Corinthia Aberlé: "Synthetic Mathematics, Logical Frameworks, Categorical Algebra"** (2024)
   - [YouTube](https://www.youtube.com/watch?v=MjkWT6GkISI)
   - Synthetic definitions as interfaces

9. **André Joyal: "Free bicompletion of categories revisited"** (Feb 2024)
   - [YouTube](https://www.youtube.com/watch?v=uWwD0Y-DOJ0)
   - Applications to Linear Logic semantics

10. **Filippo Bonchi: "Diagrammatic Algebra of First Order Logic"** (May 2024)
    - [YouTube](https://www.youtube.com/watch?v=G6eFMEjT74w)
    - String diagrammatic extension of binary relations
    - Same expressivity as first order logic

### Modal Logic and Comonads

11. **Awodey, Kishida, Kotzsch: "Topos Semantics for Higher-Order Modal Logic"**
    - [arXiv:1403.0020](https://arxiv.org/abs/1403.0020)
    - Modal operator as comonad from geometric morphism f : F → E
    - H = f_* Ω_F as complete Heyting algebra
    - Subsumes Kripke, neighborhood, and sheaf semantics

12. **Leo Esakia (via Berkeley Seminar): "My 70 Years with Heyting Algebras"**
    - [Slides](https://topos.institute/events/berkeley-seminar/slides/2023-10-16.pdf)
    - Gödel translations: classical → intuitionistic, intuitionistic → S4-modal

### Compositional World Modeling

13. **Topos Blog: "Towards a Research Program on Compositional World-Modeling"**
    - [Blog Post](https://topos.institute/blog/2023-06-15-compositional-world-modeling/)
    - Category theory for understanding systems at global scope

### QS and Modal Logic

14. **DJ Myers: "QS: Quantum Programming via Linear Homotopy Types"** (Aug 2023)
    - [Slides](https://topos.institute/events/topos-colloquium/slides/2023-08-24.pdf)
    - S5 Kripke semantics as co-monadic descent
    - Unifies quantum logic (linear types) with epistemic modal logic (possible worlds)

---

## Key Concepts for SPI Integration

### From Kavvos (Two-dimensional Kripke)
- Kripke frames → categories (worlds = objects, accessibility = morphisms)
- □ as comonad, ◇ as monad
- 2-cells give natural transformations between interpretations

### From Simpson (Three Toposes)
- **RV functor**: faithful, preserves countable limits
- **Distribution functor D**: D : SB → Set (taut)
- **Law of random variable**: natural transformation P : RV → D
- Invariance principle, independence principle, countable dependent choice

### From Awodey-Kishida-Kotzsch
- Complete Heyting algebra H replaces Ω
- Counit ε : □p → p corresponds to reflexivity (T axiom)
- Comultiplication δ : □p → □□p corresponds to transitivity (4 axiom)

### From Brast McKie
- World states = maximal possible ways for things to be at an instant
- Task relation = possible transitions between world states
- Possible worlds = functions from times to world states
- Eliminates unnecessary degrees of freedom

---

## Layer Correspondence

| SPI Layer | Topos Concept | Talk Reference |
|-----------|---------------|----------------|
| 0-5 | Monoid/Trace structure | Base category theory |
| 6 | Kripke Frames | Kavvos, Brast McKie |
| 7 | Modal Logic □◇ | Awodey-Kishida-Kotzsch |
| 8 | Sheaf Semantics | Simpson, Vickers |
| 9+ | Probability Sheaves | Simpson's Three Toposes |
| 10+ | Synthetic Probability | Random Topos |
