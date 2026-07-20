#!/usr/bin/env bb

(ns freesurfer-uncertainty-propagation
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]))

(def default-path ".topos/estimates/freesurfer_humans_BA2645.edn")
(def default-samples 4096)
(def bases [2 3 5 7 11 13 17 19])

(defn fail! [message]
  (binding [*out* *err*]
    (println (str "FAIL: " message)))
  (System/exit 1))

(defn parse-int [s]
  (Integer/parseInt s))

(defn round-to [x quantum]
  (* quantum (Math/round (/ x (double quantum)))))

(defn round1 [x]
  (/ (Math/round (* 10.0 (double x))) 10.0))

(defn round3 [x]
  (/ (Math/round (* 1000.0 (double x))) 1000.0))

(defn round6 [x]
  (/ (Math/round (* 1000000.0 (double x))) 1000000.0))

(defn mean [xs]
  (if (seq xs)
    (/ (reduce + xs) (double (count xs)))
    0.0))

(defn sample-variance [xs]
  (let [n (count xs)
        mu (mean xs)]
    (if (< n 2)
      0.0
      (/ (reduce + (map #(let [d (- % mu)] (* d d)) xs))
         (double (dec n))))))

(defn weighted-log-merge [branches]
  (Math/exp
   (reduce
    +
    (for [branch branches]
      (* (:branch/weight branch)
         (Math/log (:branch/point branch)))))))

(defn vdc [n base]
  (loop [n n
         denom (double base)
         result 0.0]
    (if (zero? n)
      result
      (recur (quot n base)
             (* denom base)
             (+ result (/ (mod n base) denom))))))

(defn lhs-u [i n base]
  (/ (+ i (vdc (inc i) base)) (double n)))

(defn log-triangular-quantile [u [lo hi] point]
  (let [a (Math/log lo)
        b (Math/log hi)
        c (Math/log point)
        width (- b a)
        left (- c a)
        right (- b c)
        fc (if (pos? width) (/ left width) 0.5)
        x (cond
            (<= width 0.0) c
            (<= left 0.0) (- b (Math/sqrt (* (- 1.0 u) width right)))
            (<= right 0.0) (+ a (Math/sqrt (* u width left)))
            (< u fc) (+ a (Math/sqrt (* u width left)))
            :else (- b (Math/sqrt (* (- 1.0 u) width right))))]
    (Math/exp x)))

(defn quantile [sorted-xs p]
  (let [n (count sorted-xs)
        pos (* p (dec n))
        lo (long (Math/floor pos))
        hi (long (Math/ceil pos))
        frac (- pos lo)
        a (nth sorted-xs lo)
        b (nth sorted-xs hi)]
    (+ a (* frac (- b a)))))

(defn quantiles [xs]
  (let [sorted-xs (vec (sort xs))]
    {:p05 (long (Math/round (quantile sorted-xs 0.05)))
     :p25 (long (Math/round (quantile sorted-xs 0.25)))
     :p50 (long (Math/round (quantile sorted-xs 0.50)))
     :p75 (long (Math/round (quantile sorted-xs 0.75)))
     :p95 (long (Math/round (quantile sorted-xs 0.95)))}))

(defn branch-sample [branch i n dimension]
  (assoc branch
         :branch/point
         (log-triangular-quantile
          (lhs-u i n (nth bases (mod dimension (count bases))))
          (:branch/range branch)
          (:branch/point branch))))

(defn simulated-merges [branches n]
  (mapv
   (fn [i]
     (weighted-log-merge
      (map-indexed #(branch-sample %2 i n %1) branches)))
   (range n)))

(defn branch-log-samples [branch n dimension]
  (mapv
   (fn [i]
     (Math/log (:branch/point (branch-sample branch i n dimension))))
   (range n)))

(defn log-variance-contribution [branches n]
  (let [rows (mapv
              (fn [dimension branch]
                (let [log-var (sample-variance (branch-log-samples branch n dimension))
                      weighted (* (:branch/weight branch) (:branch/weight branch) log-var)]
                  {:branch/id (:branch/id branch)
                   :branch/weight (:branch/weight branch)
                   :log-variance log-var
                   :weighted-log-variance weighted}))
              (range)
              branches)
        total (reduce + (map :weighted-log-variance rows))
        enriched (mapv
                  (fn [row]
                    (assoc row :share (if (pos? total)
                                        (/ (:weighted-log-variance row) total)
                                        0.0)))
                  rows)
        ranked (vec (sort-by (juxt (comp - :share)
                                   (comp str :branch/id))
                             enriched))
        rank-map (into {} (map-indexed (fn [i row] [(:branch/id row) (inc i)]) ranked))]
    {:method :independent-logspace-variance-share
     :samples n
     :total-weighted-log-variance (round6 total)
     :priority-order (mapv :branch/id ranked)
     :branches
     (into
      (sorted-map)
      (for [row enriched
            :let [branch-id (:branch/id row)]]
        [branch-id {:rank (rank-map branch-id)
                    :branch-weight (:branch/weight row)
                    :log-variance (round6 (:log-variance row))
                    :weighted-log-variance (round6 (:weighted-log-variance row))
                    :share (round3 (:share row))}]))}))

(defn replace-branch-point [branches branch-id point]
  (mapv
   (fn [branch]
     (if (= branch-id (:branch/id branch))
       (assoc branch :branch/point point)
       branch))
   branches))

(defn branch-sensitivity [branches]
  (into
   (sorted-map)
   (for [branch branches
         :let [branch-id (:branch/id branch)
               [lo hi] (:branch/range branch)
               lo-merge (weighted-log-merge
                         (replace-branch-point branches branch-id lo))
               hi-merge (weighted-log-merge
                         (replace-branch-point branches branch-id hi))]]
     [branch-id {:low-branch-point lo
                 :low-merge-rounded (long (round-to lo-merge 10000))
                 :high-branch-point hi
                 :high-merge-rounded (long (round-to hi-merge 10000))
                 :span-rounded (- (long (round-to hi-merge 10000))
                                  (long (round-to lo-merge 10000)))}])))

(defn anchor-by-id [estimate id]
  (some #(when (= id (:anchor/id %)) %) (:source/anchors estimate)))

(defn branch-by-id [estimate id]
  (some #(when (= id (:branch/id %)) %) (:estimate/branches estimate)))

(defn nested [m ks]
  (get-in m ks))

(defn scenario [branches publication-branch calibration-ratio full-core-role scenario-id ratio]
  (let [implied-multiplier (/ (:branch/point publication-branch)
                              (* full-core-role calibration-ratio))
        branch-point (* full-core-role ratio implied-multiplier)
        merged (weighted-log-merge
                (replace-branch-point branches
                                      (:branch/id publication-branch)
                                      branch-point))]
    [scenario-id {:method-to-role-ratio ratio
                  :publication-branch-point (long (Math/round branch-point))
                  :merge-rounded (long (round-to merged 10000))
                  :implied-helper-multiplier (round3 implied-multiplier)}]))

(defn publication-uplift-sensitivity [estimate]
  (let [branches (:estimate/branches estimate)
        publication-branch (branch-by-id estimate :publication-authorship)
        core-value (:anchor/value (anchor-by-id estimate :openalex/freesurfer-core-citing-union))
        oa-value (:anchor/value (anchor-by-id estimate :openalex/freesurfer-oa-methods-probe))
        abstract-value (:anchor/value (anchor-by-id estimate :openalex/freesurfer-abstract-methods-calibration))
        benchmark (:pdf-first-stratified-benchmark oa-value)
        uplift (:fulltext-uplift-estimator benchmark)
        full-core-role (:role-weighted-author-equivalents core-value)
        calibration-ratio (:method-to-role-ratio oa-value)
        scenarios {:abstract-title-lower (:method-to-role-ratio abstract-value)
                   :oa-default (:method-to-role-ratio oa-value)
                   :stratified-lower-bound (:method-to-role-ratio benchmark)
                   :stratified-shrunk-uplift (:poststratified-method-to-role-ratio uplift)}]
    {:full-core-role-weighted-author-equivalents full-core-role
     :calibration-ratio calibration-ratio
     :calibration-source :openalex/freesurfer-oa-methods-probe
     :scenario-source :pdf-first-stratified-benchmark
     :scenarios (into (sorted-map)
                      (for [[scenario-id ratio] scenarios]
                        (scenario branches publication-branch calibration-ratio full-core-role scenario-id ratio)))}))

(defn summarize [estimate samples path]
  (let [branches (:estimate/branches estimate)
        merges (simulated-merges branches samples)
        baseline-raw (weighted-log-merge branches)]
    {:refinement/id :freesurfer-uncertainty-propagation
     :estimate/id (:estimate/id estimate)
     :source/path path
     :simulation {:method :deterministic-log-triangular-latin-hypercube
                  :samples samples
                  :merge-method :weighted-logspace-zigzag
                  :rounding-quantum 10000}
     :baseline {:raw (long (Math/round baseline-raw))
                :rounded (long (round-to baseline-raw 10000))
                :edn-rounded (get-in estimate [:estimate/merge :point])}
     :merge-quantiles (quantiles merges)
     :branch-sensitivity (branch-sensitivity branches)
     :log-variance-contribution (log-variance-contribution branches samples)
     :publication-uplift-sensitivity (publication-uplift-sensitivity estimate)
     :interpretation
     "Deterministic uncertainty propagation: sample each branch over its stated range with a log-triangular distribution centered on the branch point, merge in log-space, and report publication-branch scenarios induced by observed method-to-role calibration."}))

(defn parse-args [args]
  (loop [args args
         opts {:path default-path
               :samples default-samples}]
    (if-let [arg (first args)]
      (case arg
        "--path" (recur (nnext args) (assoc opts :path (second args)))
        "--samples" (recur (nnext args) (assoc opts :samples (parse-int (second args))))
        (fail! (str "unknown argument: " arg)))
      opts)))

(defn -main [& args]
  (let [{:keys [path samples]} (parse-args args)
        file (io/file path)]
    (when-not (.exists file)
      (fail! (str "estimate file not found: " path)))
    (println (pr-str (summarize (edn/read-string (slurp file)) samples path)))))

(apply -main *command-line-args*)
