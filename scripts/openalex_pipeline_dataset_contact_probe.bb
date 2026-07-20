#!/usr/bin/env bb

(ns openalex-pipeline-dataset-contact-probe
  (:require [cheshire.core :as json]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.set :as set]
            [clojure.string :as str]))

(def default-path ".topos/estimates/freesurfer_humans_BA2645.edn")
(def default-sample-size 100)
(def default-seeds [1069 1070])
(def default-human-contact-multiplier 1.8)

(def role-weight-scheme
  {:first 0.75
   :last 0.45
   :middle 0.18
   :unknown 0.15
   :corresponding-bonus 0.20
   :cap 1.0})

(def default-proxy-works
  [{:proxy/id :hcp-neuroimaging
    :work/id "W2507387536"
    :proxy/contact-weight 0.55
    :proxy/kind :dataset-pipeline-paper
    :proxy/reason "HCP neuroimaging workflows expose FreeSurfer-derived surfaces, parcellations, and structural outputs."}
   {:proxy/id :uk-biobank-image-processing
    :work/id "W2607804943"
    :proxy/contact-weight 0.50
    :proxy/kind :dataset-processing-paper
    :proxy/reason "UK Biobank imaging processing papers proxy downstream contact with released structural imaging derivatives."}
   {:proxy/id :abcd-image-processing
    :work/id "W2966883685"
    :proxy/contact-weight 0.50
    :proxy/kind :dataset-processing-paper
    :proxy/reason "ABCD image processing methods proxy downstream contact with curated neuroimaging derivatives."}
   {:proxy/id :fmriprep
    :work/id "W2951583631"
    :proxy/contact-weight 0.65
    :proxy/kind :pipeline-software-paper
    :proxy/reason "fMRIPrep citations proxy users of pipelines that frequently interoperate with FreeSurfer reconstruction outputs."}
   {:proxy/id :mne-python
    :work/id "W2169918686"
    :proxy/contact-weight 0.25
    :proxy/kind :analysis-software-paper
    :proxy/reason "MNE-Python citations weakly proxy contact with FreeSurfer-derived source spaces and cortical surfaces."}
   {:proxy/id :adni-progress
    :work/id "W2001477615"
    :proxy/contact-weight 0.25
    :proxy/kind :dataset-cohort-paper
    :proxy/reason "ADNI citations weakly proxy use of structural MRI derivative ecosystems, only some of which involve FreeSurfer."}])

(defn fail! [message]
  (binding [*out* *err*]
    (println (str "FAIL: " message)))
  (System/exit 1))

(defn parse-int [s]
  (Integer/parseInt s))

(defn parse-double [s]
  (Double/parseDouble s))

(defn round1 [x]
  (/ (Math/round (* 10.0 (double x))) 10.0))

(defn round3 [x]
  (/ (Math/round (* 1000.0 (double x))) 1000.0))

(defn median [xs]
  (let [v (vec (sort xs))
        n (count v)]
    (cond
      (zero? n) 0.0
      (odd? n) (double (v (quot n 2)))
      :else (/ (+ (v (dec (quot n 2))) (v (quot n 2))) 2.0))))

(defn curl-json [url]
  (let [{:keys [out err exit]} (sh "curl" "-L" "-sS" "--fail"
                                   "--connect-timeout" "10"
                                   "--max-time" "30"
                                   url)]
    (when-not (zero? exit)
      (fail! (str "curl failed for " url ": " err)))
    (json/parse-string out true)))

(defn work-api-url [work-id]
  (str "https://api.openalex.org/works/" work-id))

(defn citing-count-url [work-id]
  (str "https://api.openalex.org/works?filter=cites:" work-id
       "&per-page=1&select=id"))

(defn sample-url [work-id sample-size seed]
  (format "https://api.openalex.org/works?filter=cites:%s&sample=%d&seed=%d&per-page=%d&select=id,publication_year,authorships"
          work-id sample-size seed sample-size))

(defn author-id [authorship]
  (get-in authorship [:author :id]))

(defn authorship-weight [authorship]
  (let [position (keyword (or (:author_position authorship) "unknown"))
        base (get role-weight-scheme position (:unknown role-weight-scheme))
        bonus (if (:is_corresponding authorship)
                (:corresponding-bonus role-weight-scheme)
                0.0)]
    (min (:cap role-weight-scheme) (+ base bonus))))

(defn author-weight-map [authorships]
  (reduce
   (fn [m authorship]
     (if-let [id (author-id authorship)]
       (update m id (fnil max 0.0) (authorship-weight authorship))
       m))
   {}
   authorships))

(defn role-weighted-total [author-weights]
  (reduce + 0.0 (vals author-weights)))

(defn fetch-work-meta [proxy]
  (let [work (curl-json (work-api-url (:work/id proxy)))
        filter-count (get-in (curl-json (citing-count-url (:work/id proxy)))
                             [:meta :count])]
    (merge proxy
           {:work/title (:title work)
            :work/doi (:doi work)
            :work/year (:publication_year work)
            :work/cited-by-count (:cited_by_count work)
            :query/citing-filter-count filter-count})))

(defn fetch-seed-sample [proxy sample-size seed]
  (let [data (curl-json (sample-url (:work/id proxy) sample-size seed))
        works (:results data)
        authorships (mapcat :authorships works)
        author-weights (author-weight-map authorships)
        authors (set (keys author-weights))
        role-weighted (role-weighted-total author-weights)]
    {:seed seed
     :sampled-works (count works)
     :unique-authors (count authors)
     :role-weighted-author-equivalents (round1 role-weighted)
     :role-weighted-density (if (seq works)
                              (/ role-weighted (double (count works)))
                              0.0)
     :authors authors}))

(defn summarize-proxy [proxy sample-size seeds]
  (let [proxy (fetch-work-meta proxy)
        samples (mapv #(fetch-seed-sample proxy sample-size %) seeds)
        density (median (map :role-weighted-density samples))
        filter-count (:query/citing-filter-count proxy)
        estimated-role (* density filter-count)
        contact-weight (:proxy/contact-weight proxy)
        contact-weighted (* estimated-role contact-weight)
        sample-authors (apply set/union (map :authors samples))]
    (merge
     (select-keys proxy
                  [:proxy/id :proxy/kind :proxy/contact-weight :proxy/reason
                   :work/id :work/title :work/doi :work/year :work/cited-by-count
                   :query/citing-filter-count])
     {:sample/observations (mapv #(select-keys %
                                               [:seed :sampled-works :unique-authors
                                                :role-weighted-author-equivalents])
                                  samples)
      :sample/union-unique-authors (count sample-authors)
      :sample/median-role-weighted-density (round3 density)
      :estimate/role-weighted-author-equivalents (round1 estimated-role)
      :estimate/contact-weighted-author-equivalents (round1 contact-weighted)
      :_authors sample-authors})))

(defn strip-private [proxy-summary]
  (dissoc proxy-summary :_authors))

(defn branch-by-id [estimate id]
  (some #(when (= id (:branch/id %)) %) (:estimate/branches estimate)))

(defn overlap-summary [proxy-summaries]
  (let [sets (map :_authors proxy-summaries)
        sum-unique (reduce + (map count sets))
        union-unique (count (apply set/union sets))
        retention (if (pos? sum-unique)
                    (/ union-unique (double sum-unique))
                    1.0)]
    {:sample/sum-proxy-unique-authors sum-unique
     :sample/union-unique-authors union-unique
     :sample/overlap-retention-ratio (round3 retention)
     :sample/overlap-discount-ratio (round3 (- 1.0 retention))}))

(defn summarize [estimate sample-size seeds human-contact-multiplier]
  (let [proxy-summaries (mapv #(summarize-proxy % sample-size seeds)
                              default-proxy-works)
        overlap (overlap-summary proxy-summaries)
        naive-contact (reduce + (map :estimate/contact-weighted-author-equivalents
                                     proxy-summaries))
        retention (:sample/overlap-retention-ratio overlap)
        overlap-adjusted (* naive-contact retention)
        human-contact (* overlap-adjusted human-contact-multiplier)
        branch (branch-by-id estimate :pipeline-dataset-contact)
        [range-low range-high] (:branch/range branch)]
    {:refinement/id :openalex-pipeline-dataset-contact-probe
     :query/sample-size-per-proxy-seed sample-size
     :query/seeds seeds
     :query/proxy-count (count default-proxy-works)
     :query/proxy-work-ids (mapv :work/id default-proxy-works)
     :proxy/works (mapv strip-private proxy-summaries)
     :sample/overlap overlap
     :estimate/naive-contact-weighted-author-equivalents (round1 naive-contact)
     :estimate/overlap-adjusted-contact-author-equivalents (round1 overlap-adjusted)
     :estimate/human-contact-multiplier human-contact-multiplier
     :estimate/pipeline-contact-human-equivalents (long (Math/round human-contact))
     :estimate/lower-bound-proxy? true
     :branch/current-point (:branch/point branch)
     :branch/range (:branch/range branch)
     :branch/inside-range? (and branch
                                (<= range-low
                                    human-contact
                                    range-high))
     :branch/implied-multiplier-for-range-low (when (pos? overlap-adjusted)
                                                (round3 (/ range-low overlap-adjusted)))
     :branch/implied-multiplier-for-current-point (when (pos? overlap-adjusted)
                                                   (round3 (/ (:branch/point branch)
                                                             overlap-adjusted)))
     :interpretation
     "Dataset/pipeline-contact probe: sample OpenAlex authors citing proxy papers for HCP, UK Biobank imaging processing, ABCD image processing, fMRIPrep, MNE-Python, and ADNI; apply conservative FreeSurfer-contact weights, discount sampled cross-proxy author overlap, then multiply by a small non-author/user-contact factor."}))

(defn parse-args [args]
  (loop [args args
         opts {:path default-path
               :sample-size default-sample-size
               :seeds default-seeds
               :human-contact-multiplier default-human-contact-multiplier}]
    (if-let [arg (first args)]
      (case arg
        "--path" (recur (nnext args) (assoc opts :path (second args)))
        "--sample-size" (recur (nnext args) (assoc opts :sample-size (parse-int (second args))))
        "--seeds" (recur (nnext args)
                         (assoc opts :seeds
                                (mapv parse-int (str/split (second args) #","))))
        "--human-contact-multiplier" (recur (nnext args)
                                            (assoc opts :human-contact-multiplier
                                                   (parse-double (second args))))
        (fail! (str "unknown argument: " arg)))
      opts)))

(defn -main [& args]
  (let [{:keys [path sample-size seeds human-contact-multiplier]} (parse-args args)
        file (io/file path)]
    (when-not (.exists file)
      (fail! (str "estimate file not found: " path)))
    (println
     (pr-str
      (summarize (edn/read-string (slurp file))
                 sample-size
                 seeds
                 human-contact-multiplier)))))

(apply -main *command-line-args*)
