#!/bin/bash
# Setup GitHub tracking for gaymc forks
# Usage: ./setup_github.sh [repo]
# Default repo: bmorphism/gaymc

REPO="${1:-bmorphism/gaymc}"

echo "🏳️‍🌈 Setting up gaymc tracking for $REPO"

# Labels
labels=(
  "spi:SPI verification:0e8a16"
  "chromatic:Chromatic identity:5319e7"
  "algorithm:Core algorithm:1d76db"
  "parallel:Parallel execution:d93f0b"
  "sheaf:Sheaf theory:c5def5"
  "narrative:Temporal narrative:fbca04"
  "energy:Energy grid:b60205"
  "bumpus:Bumpus paper:006b75"
)

for label in "${labels[@]}"; do
  IFS=':' read -r name desc color <<< "$label"
  gh label create "$name" --description "$desc" --color "$color" --repo "$REPO" 2>/dev/null || \
    echo "  Label '$name' exists"
done

# Milestones
gh api repos/$REPO/milestones -f title="v0.1.0-core" -f description="Core gaimc algorithms ported" 2>/dev/null
gh api repos/$REPO/milestones -f title="v0.2.0-algorithms" -f description="Fork-specific algorithms" 2>/dev/null
gh api repos/$REPO/milestones -f title="v0.3.0-spi-verified" -f description="Full SPI verification suite" 2>/dev/null

# Issues for v0.1.0
issues=(
  "Implement gay_bfs! with level coloring:algorithm,spi"
  "Implement gay_dfs! with discovery time coloring:algorithm,spi"
  "Implement gay_dijkstra! with distance class coloring:algorithm,spi"
  "Implement gay_mst_prim! with tree edge coloring:algorithm,spi"
  "Implement gay_scomponents! with component coloring:algorithm,spi"
  "Implement gay_corenums! with k-core coloring:algorithm,spi"
  "Add XOR fingerprint verification:spi,parallel"
  "Integrate Bumpus spined category theory:bumpus,sheaf"
)

for issue in "${issues[@]}"; do
  IFS=':' read -r title labels <<< "$issue"
  gh issue create --title "$title" --label "$labels" --milestone "v0.1.0-core" --repo "$REPO" 2>/dev/null || \
    echo "  Issue exists: $title"
done

# Fork-specific issues
gh issue create --title "Plurigrid: gay_power_flow! implementation" --label "energy,spi" --repo "$REPO" 2>/dev/null
gh issue create --title "TeglonLabs: gay_cech_cohomology! implementation" --label "sheaf,bumpus" --repo "$REPO" 2>/dev/null
gh issue create --title "Tritwies: gay_narrative_bfs! implementation" --label "narrative,bumpus" --repo "$REPO" 2>/dev/null
gh issue create --title "bmorphism: gay_tree_width! implementation" --label "algorithm,bumpus" --repo "$REPO" 2>/dev/null

echo "✅ GitHub tracking setup complete"
echo "   View: gh issue list --repo $REPO"
echo "   Board: gh project list --owner bmorphism"
