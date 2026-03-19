#!/usr/bin/env bb
;; Setup GitHub tracking for gaymc forks
;; Usage: bb setup_github.bb [repo]

(require '[babashka.process :refer [shell sh]]
         '[clojure.string :as str])

(def repo (or (first *command-line-args*) "bmorphism/gaymc"))

(println "🏳️‍🌈 Setting up gaymc tracking for" repo)

;; Labels: [name description color]
(def labels
  [["spi" "SPI verification" "0e8a16"]
   ["chromatic" "Chromatic identity" "5319e7"]
   ["algorithm" "Core algorithm" "1d76db"]
   ["parallel" "Parallel execution" "d93f0b"]
   ["sheaf" "Sheaf theory" "c5def5"]
   ["narrative" "Temporal narrative" "fbca04"]
   ["energy" "Energy grid" "b60205"]
   ["bumpus" "Bumpus paper" "006b75"]
   ["plurigrid" "Plurigrid fork" "ff6b6b"]
   ["teglonlabs" "TeglonLabs fork" "4ecdc4"]
   ["tritwies" "Tritwies fork" "ffe66d"]
   ["bmorphism" "bmorphism fork" "95e1d3"]])

(defn gh [& args]
  (try
    (:out (apply sh "gh" args))
    (catch Exception _ nil)))

(defn create-label [[name desc color]]
  (if (gh "label" "create" name 
          "--description" desc 
          "--color" color 
          "--repo" repo)
    (println "  ✓ Label:" name)
    (println "  · Label exists:" name)))

(defn create-milestone [title desc]
  (if (gh "api" (str "repos/" repo "/milestones")
          "-f" (str "title=" title)
          "-f" (str "description=" desc))
    (println "  ✓ Milestone:" title)
    (println "  · Milestone exists:" title)))

(defn create-issue [title labels-str milestone]
  (if (gh "issue" "create"
          "--title" title
          "--label" labels-str
          "--milestone" milestone
          "--repo" repo)
    (println "  ✓ Issue:" title)
    (println "  · Issue exists:" title)))

;; Create labels
(println "\n📌 Labels")
(doseq [label labels] (create-label label))

;; Create milestones
(println "\n🎯 Milestones")
(create-milestone "v0.1.0-core" "Core gaimc algorithms ported with SPI")
(create-milestone "v0.2.0-forks" "Fork-specific algorithms (energy/sheaf/narrative/spined)")
(create-milestone "v0.3.0-unified" "Cross-fork composition and full SPI verification")

;; Core algorithm issues
(println "\n📋 Issues: Core Algorithms (v0.1.0)")
(def core-issues
  [["Implement gay_bfs! with level coloring" "algorithm,spi"]
   ["Implement gay_dfs! with discovery time coloring" "algorithm,spi"]
   ["Implement gay_dijkstra! with distance class coloring" "algorithm,spi"]
   ["Implement gay_mst_prim! with tree edge coloring" "algorithm,spi"]
   ["Implement gay_scomponents! with component coloring" "algorithm,spi"]
   ["Implement gay_corenums! with k-core coloring" "algorithm,spi"]
   ["Add XOR fingerprint verification to all algorithms" "spi,parallel"]])

(doseq [[title labels] core-issues]
  (create-issue title labels "v0.1.0-core"))

;; Fork-specific issues
(println "\n📋 Issues: Fork Specialization (v0.2.0)")
(def fork-issues
  [["Plurigrid: gay_power_flow! DC power flow" "energy,plurigrid,spi"]
   ["Plurigrid: gay_grid_partition! parallel decomposition" "energy,plurigrid,parallel"]
   ["Plurigrid: gay_optimal_power_flow! with chromatic constraints" "energy,plurigrid"]
   ["TeglonLabs: gay_cech_cohomology! H⁰ H¹ computation" "sheaf,teglonlabs,bumpus"]
   ["TeglonLabs: gay_local_to_global! certification" "sheaf,teglonlabs"]
   ["TeglonLabs: gay_sheaf_decidability! bounded width" "sheaf,teglonlabs,bumpus"]
   ["Tritwies: gay_narrative_bfs! spatiotemporal search" "narrative,tritwies,bumpus"]
   ["Tritwies: gay_interval_sheaf! time interval sheaves" "narrative,tritwies,sheaf"]
   ["Tritwies: gay_snapshot_compose! narrative composition" "narrative,tritwies"]
   ["bmorphism: gay_tree_width! minimum-degree elimination" "algorithm,bmorphism,bumpus"]
   ["bmorphism: gay_triangulate! chordal completion" "algorithm,bmorphism"]
   ["bmorphism: gay_spined_functor! triangulation functor" "algorithm,bmorphism,bumpus"]])

(doseq [[title labels] fork-issues]
  (create-issue title labels "v0.2.0-forks"))

;; Bumpus paper issues
(println "\n📋 Issues: Bumpus Papers")
(def bumpus-issues
  [["Implement Spined Categories (EJC 2023)" "bumpus,bmorphism"]
   ["Implement Compositional Algorithms on Compositional Data (2023)" "bumpus,teglonlabs,sheaf"]
   ["Implement Towards Unified Theory of Time-varying Data (2024)" "bumpus,tritwies,narrative"]
   ["Implement Additive Invariants of Open Petri Nets (2024)" "bumpus,plurigrid"]])

(doseq [[title labels] bumpus-issues]
  (create-issue title labels "v0.3.0-unified"))

(println "\n✅ GitHub tracking setup complete")
(println "   View issues: gh issue list --repo" repo)
(println "   View milestones: gh api repos/" repo "/milestones")
