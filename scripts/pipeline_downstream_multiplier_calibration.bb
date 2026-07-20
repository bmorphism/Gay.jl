#!/usr/bin/env bb

(ns pipeline-downstream-multiplier-calibration
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]))

(def default-path ".topos/estimates/freesurfer_humans_BA2645.edn")

(defn fail! [message]
  (binding [*out* *err*]
    (println (str "FAIL: " message)))
  (System/exit 1))

(defn round-to [x quantum]
  (* quantum (Math/round (/ x (double quantum)))))

(defn round1 [x]
  (/ (Math/round (* 10.0 (double x))) 10.0))

(defn round3 [x]
  (/ (Math/round (* 1000.0 (double x))) 1000.0))

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

(defn anchor-value [estimate id]
  (:anchor/value
   (some #(when (= id (:anchor/id %)) %) (:source/anchors estimate))))

(defn branch-by-id [estimate id]
  (some #(when (= id (:branch/id %)) %) (:estimate/branches estimate)))

(defn publication-helper-multiplier [estimate]
  (get-in estimate
          [:estimate/uncertainty-propagation
           :publication-uplift-sensitivity
           :scenarios
           :oa-default
           :implied-helper-multiplier]))

(defn scenario [estimate overlap-adjusted branch scenario-id multiplier]
  (let [branches (:estimate/branches estimate)
        branch-point (* overlap-adjusted multiplier)
        rounded-branch-point (long (Math/round branch-point))
        merged (weighted-log-merge
                (replace-branch-point branches
                                      (:branch/id branch)
                                      branch-point))
        [lo hi] (:branch/range branch)]
    [scenario-id
     {:multiplier (round3 multiplier)
      :pipeline-branch-point rounded-branch-point
      :merge-raw (long (Math/round merged))
      :merge-rounded (long (round-to merged 10000))
      :inside-current-branch-range? (<= lo branch-point hi)
      :relative-to-current-branch-point
      (round3 (/ branch-point (:branch/point branch)))}]))

(defn summarize [estimate path]
  (let [pipeline-anchor (anchor-value estimate :openalex/pipeline-dataset-contact-probe)
        branch (branch-by-id estimate :pipeline-dataset-contact)
        overlap-adjusted (:overlap-adjusted-contact-author-equivalents pipeline-anchor)
        observed-mult (:human-contact-multiplier pipeline-anchor)
        [range-lo range-hi] (:branch/range branch)
        publication-mult (publication-helper-multiplier estimate)
        current-mult (/ (:branch/point branch) overlap-adjusted)
        high-mult (/ range-hi overlap-adjusted)
        low-mult (/ range-lo overlap-adjusted)
        scenarios (into
                   (sorted-map)
                   [(scenario estimate overlap-adjusted branch
                              :observed-proxy-lower-bound observed-mult)
                    (scenario estimate overlap-adjusted branch
                              :branch-range-low low-mult)
                    (scenario estimate overlap-adjusted branch
                              :publication-helper-calibrated publication-mult)
                    (scenario estimate overlap-adjusted branch
                              :current-branch-point current-mult)
                    (scenario estimate overlap-adjusted branch
                              :branch-range-high high-mult)])]
    {:refinement/id :pipeline-downstream-multiplier-calibration
     :source/path path
     :inputs {:pipeline-anchor :openalex/pipeline-dataset-contact-probe
              :pipeline-branch :pipeline-dataset-contact
              :overlap-adjusted-contact-author-equivalents overlap-adjusted
              :publication-helper-multiplier publication-mult
              :branch-current-point (:branch/point branch)
              :branch-range (:branch/range branch)}
     :scenarios scenarios
     :interpretation
     "Downstream multiplier calibration: reuse the sampled pipeline/dataset proxy lower bound and compare observed, range-boundary, publication-helper-calibrated, current-branch, and high-range multipliers through the same weighted log-space merge."}))

(defn parse-args [args]
  (loop [args args
         opts {:path default-path}]
    (if-let [arg (first args)]
      (case arg
        "--path" (recur (nnext args) (assoc opts :path (second args)))
        (fail! (str "unknown argument: " arg)))
      opts)))

(defn -main [& args]
  (let [{:keys [path]} (parse-args args)
        file (io/file path)]
    (when-not (.exists file)
      (fail! (str "estimate file not found: " path)))
    (println (pr-str (summarize (edn/read-string (slurp file)) path)))))

(apply -main *command-line-args*)
