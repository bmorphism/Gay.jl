#!/usr/bin/env bb

(ns openalex-freesurfer-methods-calibration
  (:require [cheshire.core :as json]
            [clojure.java.shell :refer [sh]]
            [clojure.set :as set]
            [clojure.string :as str]))

(def default-sample-size 100)
(def default-seeds [1069 1070])

(def default-core-works
  [{:work/id "W2151721316" :work/label :surface-analysis-i}
   {:work/id "W2113319997" :work/label :surface-analysis-ii}
   {:work/id "W2004293194" :work/label :whole-brain-segmentation}
   {:work/id "W2151130155" :work/label :cortical-parcellation}
   {:work/id "W2157270343" :work/label :skull-stripping}
   {:work/id "W4241074797" :work/label :freesurfer-overview}])

(def role-weight-scheme
  {:first 0.75
   :last 0.45
   :middle 0.18
   :unknown 0.15
   :corresponding-bonus 0.20
   :cap 1.0})

(def class-weight
  {:active-explicit 1.0
   :mention-only 0.45
   :pipeline-inherited 0.25
   :no-abstract-evidence 0.10})

(def active-patterns
  [#"\brecon-?all\b"
   #"\bfreeview\b"
   #"\bmri_[a-z0-9_]+\b"
   #"\bfsaverage\b"
   #"\baseg\b"
   #"\baparc\b"
   #"\bfreesurfer.{0,100}\b(processed|processing|reconstructed|reconstruction|segmented|segmentation|parcellated|parcellation|quality control|qc|manual|edited|cortical thickness|surface area)\b"
   #"\b(processed|reconstructed|segmented|parcellated).{0,100}\bfreesurfer\b"])

(def pipeline-patterns
  [#"\bfmriprep\b"
   #"\bqsiprep\b"
   #"\bhuman connectome project\b"
   #"\bhcp\b"
   #"\buk biobank\b"
   #"\babcd\b"
   #"\badni\b"
   #"\bpreprocessed\b"
   #"\bpipeline\b"])

(defn fail! [message]
  (binding [*out* *err*]
    (println (str "FAIL: " message)))
  (System/exit 1))

(defn parse-int [s]
  (Integer/parseInt s))

(defn curl-json [url]
  (let [{:keys [out err exit]} (sh "curl" "-sS" "--fail"
                                   "--connect-timeout" "10"
                                   "--max-time" "30"
                                   url)]
    (when-not (zero? exit)
      (fail! (str "curl failed for " url ": " err)))
    (json/parse-string out true)))

(defn sample-url [work-id sample-size seed]
  (format "https://api.openalex.org/works?filter=cites:%s&sample=%d&seed=%d&per-page=%d&select=id,title,publication_year,abstract_inverted_index,authorships"
          work-id sample-size seed sample-size))

(defn citing-count-url [work-id]
  (str "https://api.openalex.org/works?filter=cites:" work-id
       "&per-page=1&select=id"))

(defn author-id [authorship]
  (get-in authorship [:author :id]))

(defn authorship-weight [authorship]
  (let [position (keyword (or (:author_position authorship) "unknown"))
        base (get role-weight-scheme position (:unknown role-weight-scheme))
        bonus (if (:is_corresponding authorship)
                (:corresponding-bonus role-weight-scheme)
                0.0)]
    (min (:cap role-weight-scheme) (+ base bonus))))

(defn round3 [x]
  (/ (Math/round (* 1000.0 (double x))) 1000.0))

(defn round1 [x]
  (/ (Math/round (* 10.0 (double x))) 10.0))

(defn abstract-text [inverted]
  (when (seq inverted)
    (->> inverted
         (mapcat (fn [[word positions]]
                   (map (fn [pos] [pos word]) positions)))
         (sort-by first)
         (map second)
         (str/join " "))))

(defn matches-any? [patterns text]
  (boolean (some #(re-find % text) patterns)))

(defn classify-text [title abstract]
  (let [text (str/lower-case (str title "\n" abstract))
        has-free (boolean (re-find #"\bfreesurfer\b" text))
        active? (matches-any? active-patterns text)
        pipeline? (matches-any? pipeline-patterns text)]
    (cond
      active? :active-explicit
      has-free :mention-only
      pipeline? :pipeline-inherited
      :else :no-abstract-evidence)))

(defn work-evidence [work]
  (let [abstract (abstract-text (:abstract_inverted_index work))
        class (classify-text (:title work) abstract)]
    {:work/id (:id work)
     :work/title (:title work)
     :work/year (:publication_year work)
     :evidence/class class
     :evidence/weight (class-weight class)
     :evidence/has-abstract? (boolean (seq abstract))
     :authorships (:authorships work)}))

(defn fetch-sample [work sample-size seed]
  (let [data (curl-json (sample-url (:work/id work) sample-size seed))]
    {:work/id (:work/id work)
     :work/label (:work/label work)
     :seed seed
     :works (mapv work-evidence (:results data))}))

(defn dedupe-works [samples]
  (vals
   (reduce
    (fn [m work]
      (if (contains? m (:work/id work))
        m
        (assoc m (:work/id work) work)))
    {}
    (mapcat :works samples))))

(defn class-counts [works]
  (frequencies (map :evidence/class works)))

(defn weighted-work-rate [works]
  (if (seq works)
    (round3 (/ (reduce + (map :evidence/weight works)) (count works)))
    0.0))

(defn author-weight-map [works]
  (reduce
   (fn [m work]
     (let [w (:evidence/weight work)]
       (reduce
        (fn [m2 authorship]
          (if-let [id (author-id authorship)]
            (update m2 id (fnil max 0.0) (* w (authorship-weight authorship)))
            m2))
        m
        (:authorships work))))
   {}
   works))

(defn role-only-author-weight-map [works]
  (reduce
   (fn [m work]
     (reduce
      (fn [m2 authorship]
        (if-let [id (author-id authorship)]
          (update m2 id (fnil max 0.0) (authorship-weight authorship))
          m2))
      m
      (:authorships work)))
   {}
   works))

(defn summarize [sample-size seeds]
  (let [counts-by-work (into {}
                             (for [work default-core-works]
                               [(:work/id work)
                                (get-in (curl-json (citing-count-url (:work/id work)))
                                        [:meta :count])]))
        samples (vec
                 (for [work default-core-works
                       seed seeds]
                   (fetch-sample work sample-size seed)))
        works (vec (dedupe-works samples))
        class-counts (class-counts works)
        role-weighted (role-only-author-weight-map works)
        method-weighted (author-weight-map works)
        n (count works)
        examples (->> works
                      (filter #(not= :no-abstract-evidence (:evidence/class %)))
                      (take 12)
                      (mapv #(select-keys % [:work/id :work/title :work/year :evidence/class :evidence/weight])))]
    {:refinement/id :openalex-abstract-methods-calibration
     :query/core-work-ids (mapv :work/id default-core-works)
     :query/citing-filter-counts counts-by-work
     :sample/size-per-work-seed sample-size
     :sample/seeds seeds
     :sample/raw-slots (* sample-size (count seeds) (count default-core-works))
     :sample/unique-citing-works n
     :sample/class-counts class-counts
     :sample/class-weights class-weight
     :sample/active-equivalent-work-rate (weighted-work-rate works)
     :sample/unique-authors (count role-weighted)
     :sample/role-weighted-author-equivalents (round1 (reduce + (vals role-weighted)))
     :sample/method-weighted-author-equivalents (round1 (reduce + (vals method-weighted)))
     :sample/method-to-role-ratio (if (pos? (reduce + (vals role-weighted)))
                                    (round3 (/ (reduce + (vals method-weighted))
                                               (reduce + (vals role-weighted))))
                                    0.0)
     :sample/examples examples
     :interpretation
     "Abstract/title language calibration: active FreeSurfer verbs are a lower-bound signal because detailed processing methods often live outside the abstract."}))

(defn parse-args [args]
  (loop [args args
         opts {:sample-size default-sample-size
               :seeds default-seeds}]
    (if-let [arg (first args)]
      (case arg
        "--sample-size" (recur (nnext args) (assoc opts :sample-size (parse-int (second args))))
        "--seeds" (recur (nnext args)
                         (assoc opts :seeds
                                (mapv parse-int (str/split (second args) #","))))
        (fail! (str "unknown argument: " arg)))
      opts)))

(defn -main [& args]
  (let [{:keys [sample-size seeds]} (parse-args args)]
    (println (pr-str (summarize sample-size seeds)))))

(apply -main *command-line-args*)
