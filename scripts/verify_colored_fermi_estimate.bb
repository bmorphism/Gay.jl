#!/usr/bin/env bb

(ns verify-colored-fermi-estimate
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]))

(def default-path ".topos/estimates/freesurfer_humans_BA2645.edn")

(defn fail! [message]
  (binding [*out* *err*]
    (println (str "FAIL: " message)))
  (System/exit 1))

(defn require* [ok? message]
  (when-not ok?
    (fail! message)))

(defn approx= [a b eps]
  (<= (Math/abs (- (double a) (double b))) eps))

(defn within? [x [lo hi]]
  (and (<= lo x) (<= x hi)))

(defn sum-vals [m]
  (reduce + 0 (vals (or m {}))))

(defn monotone-nondecreasing? [xs]
  (every? (fn [[a b]] (<= a b)) (partition 2 1 xs)))

(defn weighted-log-merge [branches]
  (Math/exp
   (reduce
    +
    (for [branch branches]
      (* (:branch/weight branch)
         (Math/log (:branch/point branch)))))))

(defn replace-branch-point [branches branch-id point]
  (mapv
   (fn [branch]
     (if (= branch-id (:branch/id branch))
       (assoc branch :branch/point point)
       branch))
   branches))

(defn round-to [x quantum]
  (* quantum (Math/round (/ x (double quantum)))))

(defn anchor-ids [anchors]
  (set (map :anchor/id anchors)))

(defn branch-ids [branches]
  (set (map :branch/id branches)))

(defn unreconciled-color-reuses [colors]
  (->> colors
       (group-by :tick/hex)
       vals
       (mapcat rest)
       (remove :event/reuse?)))

(defn verify [estimate]
  (let [colors (:gay/color-chain estimate)
        anchors (:source/anchors estimate)
        branches (:estimate/branches estimate)
        merge (:estimate/merge estimate)
        uncertainty (:estimate/uncertainty-propagation estimate)
        downstream-calibration (:estimate/downstream-multiplier-calibration estimate)
        hexes (map :tick/hex colors)
        trits (map :tick/trit colors)
        weights (map :branch/weight branches)
        weight-sum (reduce + weights)
        raw (weighted-log-merge branches)
        rounded (round-to raw 10000)
        ids (anchor-ids anchors)
        bids (branch-ids branches)]
    (require* (= :freesurfer/humans-ever-interacted (:estimate/id estimate))
              "unexpected :estimate/id")
    (require* (seq colors) "missing :gay/color-chain")
    (require* (= "#BA2645" (:tick/hex (first colors)))
              "color chain must start from current next_color seed #BA2645")
    (require* (empty? (unreconciled-color-reuses colors))
              "color chain contains overlapping/reused colors without :event/reuse?")
    (require* (zero? (reduce + trits))
              "GF(3) trits must balance to zero")
    (require* (seq anchors) "missing :source/anchors")
    (doseq [anchor anchors]
      (require* (:source/url anchor)
                (str "anchor lacks URL: " (:anchor/id anchor)))
      (require* (:source/strength anchor)
                (str "anchor lacks source strength: " (:anchor/id anchor)))
      (let [value (:anchor/value anchor)
            fetched (:citation-slots-fetched value)
            filter-count (:citation-slots-filter-count value)
            unique-authors (or (:unique-authors-fetched value)
                               (:unique-authors-sampled value))
            unique-works (:unique-citing-works value)
            role-weighted (:role-weighted-author-equivalents value)
            fetch-summary (:fetch-summary value)
            pmc-summary (:pmc-summary value)
            pdf-summary (:pdf-summary value)
            pdf-first-benchmark (:pdf-first-benchmark value)
            pdf-first-stratified-benchmark (:pdf-first-stratified-benchmark value)
            abstract-classes (:abstract-class-counts value)
            final-classes (:final-class-counts value)]
        (when (:works-full-pagination value)
          (require* (= fetched filter-count)
                    (str "full-pagination anchor fetched/filter mismatch: "
                         (:anchor/id anchor))))
        (when role-weighted
          (require* (and unique-authors
                         (pos? role-weighted)
                         (<= role-weighted unique-authors))
                    (str "role-weighted author equivalent is invalid: "
                         (:anchor/id anchor))))
        (when fetch-summary
          (let [attempted (:attempted fetch-summary)
                text-fetched (:text-fetched fetch-summary)
                no-fetchable (:no-fetchable-text fetch-summary)
                not-attempted (:not-attempted fetch-summary)
                pdf-attempted (:pdf-attempted fetch-summary)
                pdf-attempt-count (:pdf-attempt-count fetch-summary)]
            (require* (and unique-works
                           (every? number? [attempted text-fetched no-fetchable not-attempted]))
                      (str "fetch-summary lacks numeric coverage fields: "
                           (:anchor/id anchor)))
            (require* (= unique-works (+ attempted not-attempted))
                      (str "fetch-summary attempted/not-attempted does not match unique works: "
                           (:anchor/id anchor)))
            (require* (= attempted (+ text-fetched no-fetchable))
                      (str "fetch-summary text/no-fetchable does not match attempted: "
                           (:anchor/id anchor)))
            (when (or pdf-attempted pdf-attempt-count)
              (require* (and (number? pdf-attempted)
                             (number? pdf-attempt-count)
                             (<= 0 pdf-attempted pdf-attempt-count))
                        (str "fetch-summary PDF attempt coverage is invalid: "
                             (:anchor/id anchor))))))
        (when pmc-summary
          (let [pmids (:works-with-pmid pmc-summary)
                pmcids (:works-with-pmcid pmc-summary)]
            (require* (and unique-works
                           (number? pmids)
                           (number? pmcids)
                           (<= 0 pmcids pmids unique-works))
                      (str "pmc-summary coverage is invalid: "
                           (:anchor/id anchor)))))
        (when pdf-summary
          (let [available (set (:available-extractors pdf-summary))
                selected (:selected-extractor pdf-summary)
                pdf-works (:works-with-pdf-candidate pdf-summary)
                attempted-works (:attempted-works pdf-summary)
                attempt-count (:attempt-count pdf-summary)
                text-fetched (:text-fetched pdf-summary)]
            (require* (and unique-works
                           (number? pdf-works)
                           (number? attempted-works)
                           (number? attempt-count)
                           (number? text-fetched)
                           (<= 0 text-fetched attempted-works pdf-works unique-works)
                           (<= attempted-works attempt-count))
                      (str "pdf-summary coverage is invalid: "
                           (:anchor/id anchor)))
            (when selected
              (require* (contains? available selected)
                        (str "selected PDF extractor is not listed as available: "
                             (:anchor/id anchor))))
            (when fetch-summary
              (require* (= attempted-works (:pdf-attempted fetch-summary))
                        (str "pdf-summary attempted works disagrees with fetch-summary: "
                             (:anchor/id anchor)))
              (require* (= attempt-count (:pdf-attempt-count fetch-summary))
                        (str "pdf-summary attempt count disagrees with fetch-summary: "
                             (:anchor/id anchor))))))
        (doseq [[benchmark-name benchmark] [[:pdf-first-benchmark pdf-first-benchmark]
                                            [:pdf-first-stratified-benchmark
                                             pdf-first-stratified-benchmark]]
                :when benchmark]
          (let [unique-bench (:unique-citing-works benchmark)
                pdf-works (:works-with-pdf-candidate benchmark)
                attempted-works (:attempted-works benchmark)
                attempt-count (:attempt-count benchmark)
                pdf-text-fetched (:text-fetched benchmark)
                text-fetched-total (:text-fetched-total benchmark)
                source-counts (:source-counts benchmark)
                selection (:selection benchmark)
                uplift (:fulltext-uplift-estimator benchmark)]
            (require* (and unique-works
                           (number? unique-bench)
                           (number? pdf-works)
                           (number? attempted-works)
                           (number? attempt-count)
                           (number? pdf-text-fetched)
                           (number? text-fetched-total)
                           (<= 0 pdf-text-fetched attempted-works attempt-count)
                           (<= attempted-works pdf-works unique-bench unique-works)
                           (<= pdf-text-fetched text-fetched-total unique-bench))
                      (str benchmark-name " coverage is invalid: "
                           (:anchor/id anchor)))
            (require* (= attempt-count (sum-vals (:attempt-status-counts benchmark)))
                      (str benchmark-name " attempt statuses do not sum: "
                           (:anchor/id anchor)))
            (require* (= text-fetched-total (sum-vals source-counts))
                      (str benchmark-name " source counts do not sum to text total: "
                           (:anchor/id anchor)))
            (when selection
              (let [candidate-count (:candidate-count selection)
                    selected-count (:selected-count selection)
                    candidate-pdfs (:candidate-pdf-candidate-count selection)
                    selected-pdfs (:selected-pdf-candidate-count selection)
                    fetch-attempted (:fetch-attempted benchmark)]
                (require* (and (number? candidate-count)
                               (number? selected-count)
                               (number? candidate-pdfs)
                               (number? selected-pdfs)
                               (<= 0 selected-count candidate-count unique-bench)
                               (<= 0 selected-pdfs selected-count)
                               (<= 0 candidate-pdfs candidate-count))
                          (str benchmark-name " selection coverage is invalid: "
                               (:anchor/id anchor)))
                (require* (= candidate-count
                             (sum-vals (:candidate-stratum-counts selection)))
                          (str benchmark-name " candidate strata do not sum: "
                               (:anchor/id anchor)))
                (require* (= selected-count
                             (sum-vals (:selected-stratum-counts selection)))
                          (str benchmark-name " selected strata do not sum: "
                               (:anchor/id anchor)))
                (require* (<= candidate-count
                              (sum-vals (:candidate-source-work-counts selection)))
                          (str benchmark-name " candidate source counts undercount candidates: "
                               (:anchor/id anchor)))
                (require* (<= selected-count
                              (sum-vals (:selected-source-work-counts selection)))
                          (str benchmark-name " selected source counts undercount selected works: "
                               (:anchor/id anchor)))
                (when fetch-attempted
                  (require* (= fetch-attempted selected-count)
                            (str benchmark-name " fetch-attempted disagrees with selection: "
                                 (:anchor/id anchor))))))
            (when uplift
              (let [population (:population-count uplift)
                    text-count (:text-fetched-count uplift)
                    prior-strength (:prior-strength uplift)
                    observed-strata (:observed-strata uplift)
                    stratum-count (:stratum-count uplift)
                    abstract-rate (:abstract-rate uplift)
                    lower-bound-rate (:lower-bound-final-rate uplift)
                    post-rate (:poststratified-rate uplift)
                    post-method (:poststratified-method-weighted-author-equivalents uplift)
                    post-ratio (:poststratified-method-to-role-ratio uplift)
                    strata (:strata uplift)]
                (require* (and (number? population)
                               (number? text-count)
                               (number? prior-strength)
                               (number? observed-strata)
                               (number? stratum-count)
                               (number? abstract-rate)
                               (number? lower-bound-rate)
                               (number? post-rate)
                               (number? post-method)
                               (number? post-ratio)
                               (<= 0 text-count population)
                               (<= 0 observed-strata stratum-count)
                               (<= 0.0 abstract-rate 1.0)
                               (<= 0.0 lower-bound-rate 1.0)
                               (<= 0.0 post-rate 1.0)
                               (not (neg? prior-strength))
                               (not (neg? post-method))
                               (not (neg? post-ratio)))
                          (str benchmark-name " fulltext uplift summary is invalid: "
                               (:anchor/id anchor)))
                (require* (= population unique-bench)
                          (str benchmark-name " fulltext uplift population disagrees with benchmark: "
                               (:anchor/id anchor)))
                (require* (= text-count text-fetched-total)
                          (str benchmark-name " fulltext uplift text count disagrees with fetched text: "
                               (:anchor/id anchor)))
                (when (:active-equivalent-work-rate benchmark)
                  (require* (approx= lower-bound-rate
                                     (:active-equivalent-work-rate benchmark)
                                     1.0e-9)
                            (str benchmark-name " fulltext uplift lower-bound rate disagrees: "
                                 (:anchor/id anchor))))
                (require* (= population (sum-vals (into {}
                                                     (map (fn [[k row]]
                                                            [k (:population-count row)])
                                                          strata))))
                          (str benchmark-name " fulltext uplift strata populations do not sum: "
                               (:anchor/id anchor)))
                (require* (= text-count (sum-vals (into {}
                                                    (map (fn [[k row]]
                                                           [k (:text-fetched-count row)])
                                                         strata))))
                          (str benchmark-name " fulltext uplift strata text counts do not sum: "
                               (:anchor/id anchor)))
                (doseq [[stratum row] strata]
                  (let [row-pop (:population-count row)
                        row-text (:text-fetched-count row)
                        row-abstract (:abstract-rate row)
                        row-lower (:lower-bound-final-rate row)
                        row-mean (:mean-fulltext-uplift row)
                        row-shrinkage (:uplift-shrinkage row)
                        row-shrunk (:shrunk-fulltext-uplift row)
                        row-post (:poststratified-rate row)]
                    (require* (and (number? row-pop)
                                   (number? row-text)
                                   (number? row-abstract)
                                   (number? row-lower)
                                   (number? row-mean)
                                   (number? row-shrinkage)
                                   (number? row-shrunk)
                                   (number? row-post)
                                   (<= 0 row-text row-pop)
                                   (<= 0.0 row-abstract 1.0)
                                   (<= 0.0 row-lower 1.0)
                                   (<= 0.0 row-shrinkage 1.0)
                                   (<= 0.0 row-post 1.0)
                                   (<= (Math/abs (double row-shrunk))
                                       (+ (Math/abs (double row-mean)) 1.0e-9)))
                              (str benchmark-name " fulltext uplift row is invalid: "
                                   (:anchor/id anchor) " / " stratum))))))))
        (when abstract-classes
          (require* (= unique-works (sum-vals abstract-classes))
                    (str "abstract class counts do not sum to unique works: "
                         (:anchor/id anchor))))
        (when final-classes
          (require* (= unique-works (sum-vals final-classes))
                    (str "final class counts do not sum to unique works: "
                         (:anchor/id anchor))))
        (when (= :openalex/pipeline-dataset-contact-probe (:anchor/id anchor))
          (let [proxy-works (:proxy-works value)
                proxy-count (:proxy-count value)
                work-ids (:proxy-work-ids value)
                overlap (:overlap value)
                sum-authors (:sum-proxy-unique-authors overlap)
                union-authors (:union-unique-authors overlap)
                retention (:overlap-retention-ratio overlap)
                naive (:naive-contact-weighted-author-equivalents value)
                adjusted (:overlap-adjusted-contact-author-equivalents value)
                multiplier (:human-contact-multiplier value)
                human (:pipeline-contact-human-equivalents value)
                [range-lo range-hi] (:branch-range value)
                branch-point (:branch-current-point value)
                range-low-mult (:implied-multiplier-for-range-low value)
                current-mult (:implied-multiplier-for-current-point value)]
            (require* (and (= proxy-count (count proxy-works) (count work-ids))
                           (pos? proxy-count)
                           (pos? (:sample-size-per-proxy-seed value))
                           (seq (:seeds value)))
                      "pipeline contact proxy query shape is invalid")
            (require* (and (number? sum-authors)
                           (number? union-authors)
                           (number? retention)
                           (<= 0 union-authors sum-authors)
                           (<= 0.0 retention 1.0)
                           (approx= retention
                                    (if (pos? sum-authors)
                                      (/ union-authors (double sum-authors))
                                      1.0)
                                    1.0e-3))
                      "pipeline contact overlap summary is invalid")
            (require* (and (pos? naive)
                           (pos? adjusted)
                           (pos? multiplier)
                           (pos? human)
                           (<= adjusted naive)
                           (approx= adjusted (* naive retention) 1.0)
                           (approx= human (* adjusted multiplier) 1.0))
                      "pipeline contact estimate arithmetic is invalid")
            (require* (and (number? range-lo)
                           (number? range-hi)
                           (number? branch-point)
                           (<= range-lo branch-point range-hi)
                           (approx= range-low-mult (/ range-lo adjusted) 1.0e-3)
                           (approx= current-mult (/ branch-point adjusted) 1.0e-3))
                      "pipeline contact implied branch multipliers are invalid")
            (doseq [proxy proxy-works]
              (let [role (:role-weighted-author-equivalents proxy)
                    contact-weight (:contact-weight proxy)
                    contact (:contact-weighted-author-equivalents proxy)]
                (require* (and (:proxy-id proxy)
                               (:work-id proxy)
                               (:doi proxy)
                               (pos? (:citing-filter-count proxy))
                               (pos? (:sample-union-unique-authors proxy))
                               (number? (:median-role-weighted-density proxy))
                               (pos? role)
                               (<= 0.0 contact-weight 1.0)
                               (pos? contact)
                               (approx= contact (* role contact-weight) 0.2))
                          (str "pipeline contact proxy row is invalid: "
                               (:proxy-id proxy)))))))))
    (require* (seq branches) "missing :estimate/branches")
    (require* (approx= weight-sum 1.0 1.0e-9)
              (str "branch weights must sum to 1.0, got " weight-sum))
    (doseq [branch branches]
      (require* (:branch/color branch)
                (str "branch lacks color: " (:branch/id branch)))
      (require* (within? (:branch/point branch) (:branch/range branch))
                (str "branch point outside range: " (:branch/id branch)))
      (require* (seq (:branch/forward branch))
                (str "branch lacks forward reasons: " (:branch/id branch)))
      (require* (seq (:branch/backtrack branch))
                (str "branch lacks backtracking reasons: " (:branch/id branch)))
      (doseq [source (:branch/sources branch)]
        (require* (contains? ids source)
                  (str "branch source has no matching anchor: "
                       (:branch/id branch) " -> " source))))
    (doseq [refinement (:estimate/refinements estimate)]
      (let [script (:refinement/script refinement)
            impacted-branch (get-in refinement [:refinement/branch-impact :branch/id])]
        (require* (:refinement/color refinement)
                  (str "refinement lacks color: " (:refinement/id refinement)))
        (require* (:refinement/technique refinement)
                  (str "refinement lacks technique: " (:refinement/id refinement)))
        (require* (and script (.exists (io/file script)))
                  (str "refinement script missing: " script))
        (require* (contains? bids impacted-branch)
                  (str "refinement impacts unknown branch: " impacted-branch))))
    (require* (= (:point merge) rounded)
              (str "merge point should be rounded weighted log merge "
                   rounded ", got " (:point merge)))
    (require* (= (:logspace-point-raw merge)
                 (long (Math/round raw)))
              (str "raw log merge mismatch: expected "
                   (long (Math/round raw))
                   ", got " (:logspace-point-raw merge)))
    (require* (within? (:point merge) (:range merge))
              "merged point outside central range")
    (let [[central-lo central-hi] (:range merge)
          [direct-lo direct-hi] (:direct-ran-or-installed-range merge)
          [broad-lo broad-hi] (:broad-derived-output-contact-range merge)]
      (require* (<= direct-lo direct-hi)
                "direct range is malformed")
      (require* (<= broad-lo broad-hi)
                "broad derived-output range is malformed")
      (require* (<= direct-lo central-lo)
                "direct lower bound should not exceed central lower bound")
      (require* (>= broad-hi central-hi)
                "broad upper bound should dominate central upper bound"))
    (when uncertainty
      (let [script (:refinement/script uncertainty)
            simulation (:simulation uncertainty)
            baseline (:baseline uncertainty)
            quantiles (:merge-quantiles uncertainty)
            quantile-values (map quantiles [:p05 :p25 :p50 :p75 :p95])
            sensitivity (:branch-sensitivity uncertainty)
            variance (:log-variance-contribution uncertainty)
            publication (:publication-uplift-sensitivity uncertainty)
            scenarios (:scenarios publication)
            publication-branch (some #(when (= :publication-authorship (:branch/id %)) %) branches)]
        (require* (and script (.exists (io/file script)))
                  (str "uncertainty propagation script missing: " script))
        (require* (and (= :deterministic-log-triangular-latin-hypercube
                          (:method simulation))
                       (= :weighted-logspace-zigzag (:merge-method simulation))
                       (pos? (:samples simulation))
                       (pos? (:rounding-quantum simulation)))
                  "uncertainty propagation simulation metadata is invalid")
        (require* (= (:raw baseline) (long (Math/round raw)))
                  "uncertainty baseline raw does not match current merge")
        (require* (= (:rounded baseline) rounded)
                  "uncertainty baseline rounded does not match current merge")
        (require* (= (:edn-rounded baseline) (:point merge))
                  "uncertainty baseline EDN rounded does not match merge point")
        (require* (and (every? number? quantile-values)
                       (monotone-nondecreasing? quantile-values)
                       (<= (:p05 quantiles) (:point merge) (:p95 quantiles)))
                  "uncertainty quantiles are invalid")
        (require* (= bids (set (keys sensitivity)))
                  "uncertainty branch sensitivity does not cover every branch")
        (doseq [branch branches]
          (let [branch-id (:branch/id branch)
                row (get sensitivity branch-id)
                [lo hi] (:branch/range branch)
                low-merge (:low-merge-rounded row)
                high-merge (:high-merge-rounded row)]
            (require* (= lo (:low-branch-point row))
                      (str "uncertainty low branch point mismatch: " branch-id))
            (require* (= hi (:high-branch-point row))
                      (str "uncertainty high branch point mismatch: " branch-id))
            (require* (and (number? low-merge)
                           (number? high-merge)
                           (= (:span-rounded row) (- high-merge low-merge)))
                      (str "uncertainty branch sensitivity span mismatch: " branch-id))))
        (when variance
          (let [variance-branches (:branches variance)
                priority (:priority-order variance)
                shares (map :share (vals variance-branches))]
            (require* (and (= :independent-logspace-variance-share (:method variance))
                           (= (:samples simulation) (:samples variance))
                           (pos? (:total-weighted-log-variance variance))
                           (= bids (set (keys variance-branches)))
                           (= bids (set priority))
                           (= (count bids) (count priority))
                           (approx= (reduce + shares) 1.0 0.01))
                      "uncertainty log variance contribution metadata is invalid")
            (doseq [[idx branch-id] (map-indexed vector priority)
                    :let [row (get variance-branches branch-id)
                          branch (some #(when (= branch-id (:branch/id %)) %) branches)]]
              (require* (and row branch
                             (= (:rank row) (inc idx))
                             (approx= (:branch-weight row)
                                      (:branch/weight branch)
                                      1.0e-9)
                             (not (neg? (:log-variance row)))
                             (not (neg? (:weighted-log-variance row)))
                             (<= 0.0 (:share row) 1.0))
                        (str "uncertainty log variance row is invalid: " branch-id)))))
        (when publication
          (require* (contains? ids (:calibration-source publication))
                    "publication uplift sensitivity calibration source is unknown")
          (require* (and (pos? (:full-core-role-weighted-author-equivalents publication))
                         (pos? (:calibration-ratio publication))
                         (map? scenarios)
                         (contains? scenarios :oa-default))
                    "publication uplift sensitivity metadata is invalid")
          (require* (= (:publication-branch-point (:oa-default scenarios))
                       (:branch/point publication-branch))
                    "publication uplift default scenario does not match branch point")
          (doseq [[scenario-id scenario] scenarios]
            (require* (and (number? (:method-to-role-ratio scenario))
                           (number? (:publication-branch-point scenario))
                           (number? (:merge-rounded scenario))
                           (number? (:implied-helper-multiplier scenario))
                           (pos? (:method-to-role-ratio scenario))
                           (pos? (:publication-branch-point scenario))
                           (pos? (:merge-rounded scenario))
                           (pos? (:implied-helper-multiplier scenario)))
                      (str "publication uplift scenario is invalid: " scenario-id))))))
    (when downstream-calibration
      (let [script (:refinement/script downstream-calibration)
            inputs (:inputs downstream-calibration)
            overlap-adjusted (:overlap-adjusted-contact-author-equivalents inputs)
            branch-id (:pipeline-branch inputs)
            branch (some #(when (= branch-id (:branch/id %)) %) branches)
            [range-lo range-hi] (:branch/range branch)
            scenarios (:scenarios downstream-calibration)
            required-scenarios #{:observed-proxy-lower-bound
                                 :branch-range-low
                                 :publication-helper-calibrated
                                 :current-branch-point
                                 :branch-range-high}]
        (require* (and script (.exists (io/file script)))
                  (str "downstream multiplier calibration script missing: " script))
        (require* (and (contains? ids (:pipeline-anchor inputs))
                       (contains? bids branch-id)
                       (pos? overlap-adjusted)
                       (pos? (:publication-helper-multiplier inputs))
                       (= (:branch-current-point inputs) (:branch/point branch))
                       (= (:branch-range inputs) (:branch/range branch))
                       (= required-scenarios (set (keys scenarios))))
                  "downstream multiplier calibration inputs are invalid")
        (doseq [[scenario-id scenario] scenarios]
          (let [multiplier (:multiplier scenario)
                branch-point (:pipeline-branch-point scenario)
                expected-branch-point (* overlap-adjusted multiplier)
                scenario-merge (weighted-log-merge
                                (replace-branch-point branches branch-id branch-point))
                expected-rounded (long (round-to scenario-merge 10000))
                inside? (<= range-lo branch-point range-hi)
                relative (/ branch-point (double (:branch/point branch)))]
            (require* (and (pos? multiplier)
                           (pos? branch-point)
                           (pos? (:merge-raw scenario))
                           (pos? (:merge-rounded scenario))
                           (number? (:relative-to-current-branch-point scenario))
                           (approx= branch-point expected-branch-point 10.0)
                           (approx= (:merge-raw scenario)
                                    (long (Math/round scenario-merge))
                                    10.0)
                           (= (:merge-rounded scenario) expected-rounded)
                           (= (:inside-current-branch-range? scenario) inside?)
                           (approx= (:relative-to-current-branch-point scenario)
                                    relative
                                    1.0e-3))
                      (str "downstream multiplier scenario is invalid: "
                           scenario-id))))
        (require* (= (:pipeline-branch-point (:current-branch-point scenarios))
                     (:branch/point branch))
                  "downstream current scenario does not match branch point")
        (require* (= (:merge-rounded (:current-branch-point scenarios))
                     (:point merge))
                  "downstream current scenario does not match merge point")))
    {:status :ok
     :estimate/id (:estimate/id estimate)
     :color-count (count colors)
     :trit-sum (reduce + trits)
     :branch-count (count branches)
     :raw-logspace-point (long (Math/round raw))
     :rounded-point rounded
     :central-range (:range merge)}))

(defn -main [& args]
  (let [path (or (first args) default-path)
        file (io/file path)]
    (require* (.exists file) (str "estimate file not found: " path))
    (let [estimate (edn/read-string (slurp file))
          result (verify estimate)]
      (println (pr-str result)))))

(apply -main *command-line-args*)
