#!/usr/bin/env bb

(ns sample-openalex-freesurfer-authors
  (:require [cheshire.core :as json]
            [clojure.java.shell :refer [sh]]
            [clojure.set :as set]
            [clojure.string :as str]))

(def default-work-id "W4241074797")
(def default-sample-size 200)
(def default-seeds [1069 1070 1071 1072 1073])

(defn fail! [message]
  (binding [*out* *err*]
    (println (str "FAIL: " message)))
  (System/exit 1))

(defn parse-int [s]
  (Integer/parseInt s))

(defn curl-json [url]
  (let [{:keys [out err exit]} (sh "curl" "-s" url)]
    (when-not (zero? exit)
      (fail! (str "curl failed for " url ": " err)))
    (json/parse-string out true)))

(defn sample-url [work-id sample-size seed]
  (format "https://api.openalex.org/works?filter=cites:%s&sample=%d&seed=%d&per-page=%d&select=id,publication_year,authorships"
          work-id sample-size seed sample-size))

(defn citing-count-url [work-id]
  (format "https://api.openalex.org/works?filter=cites:%s&per-page=1&select=id" work-id))

(defn work-url [work-id]
  (str "https://api.openalex.org/works/" work-id))

(defn author-id [authorship]
  (get-in authorship [:author :id]))

(defn fetch-sample [work-id sample-size seed]
  (let [data (curl-json (sample-url work-id sample-size seed))
        works (:results data)
        author-events (mapcat :authorships works)
        authors (set (keep author-id author-events))]
    {:seed seed
     :works (count works)
     :author-events (count author-events)
     :unique-authors (count authors)
     :authors authors}))

(defn pairwise-lp [samples]
  (for [i (range (count samples))
        j (range (inc i) (count samples))
        :let [a (:authors (samples i))
              b (:authors (samples j))
              overlap (count (set/intersection a b))
              estimate (when (pos? overlap)
                         (long (/ (* (count a) (count b)) overlap)))]]
    {:seed-a (:seed (samples i))
     :seed-b (:seed (samples j))
     :overlap overlap
     :lincoln-petersen estimate}))

(defn median [xs]
  (let [v (vec (sort xs))
        n (count v)]
    (cond
      (zero? n) nil
      (odd? n) (v (quot n 2))
      :else (/ (+ (v (dec (quot n 2))) (v (quot n 2))) 2.0))))

(defn capture-recapture-summary [lp-values]
  (if (seq lp-values)
    {:min (apply min lp-values)
     :median (median lp-values)
     :max (apply max lp-values)}
    {:min nil
     :median nil
     :max nil
     :note "No pairwise sample overlap; increase --sample-size or --seeds."}))

(defn summarize [work-id sample-size seeds]
  (let [work (curl-json (work-url work-id))
        citing-count (get work :cited_by_count)
        citing-filter-count (get-in (curl-json (citing-count-url work-id)) [:meta :count])
        samples (mapv #(fetch-sample work-id sample-size %) seeds)
        sample-public (mapv #(select-keys % [:seed :works :author-events :unique-authors])
                            samples)
        union-authors (count (apply set/union (map :authors samples)))
        lp (vec (pairwise-lp samples))
        lp-values (keep :lincoln-petersen lp)]
    {:refinement/id :openalex-author-capture-recapture
     :work/id work-id
     :work/openalex-url (str "https://openalex.org/" work-id)
     :work/title (:title work)
     :work/cited-by-count citing-count
     :query/citing-filter (str "cites:" work-id)
     :query/citing-filter-count citing-filter-count
     :sample/size sample-size
     :sample/seeds seeds
     :sample/observations sample-public
     :sample/union-unique-authors union-authors
     :capture-recapture/pairs lp
     :capture-recapture/summary (capture-recapture-summary lp-values)
     :interpretation
     "Seeded samples estimate unique OpenAlex authors in the Fischl 2012 citing population; this is a publication-authorship proxy, not a direct FreeSurfer user census."}))

(defn parse-args [args]
  (loop [args args
         opts {:work-id default-work-id
               :sample-size default-sample-size
               :seeds default-seeds}]
    (if-let [arg (first args)]
      (case arg
        "--work-id" (recur (nnext args) (assoc opts :work-id (second args)))
        "--sample-size" (recur (nnext args) (assoc opts :sample-size (parse-int (second args))))
        "--seeds" (recur (nnext args)
                         (assoc opts :seeds
                                (mapv parse-int (str/split (second args) #","))))
        (fail! (str "unknown argument: " arg)))
      opts)))

(defn -main [& args]
  (let [{:keys [work-id sample-size seeds]} (parse-args args)]
    (println (pr-str (summarize work-id sample-size seeds)))))

(apply -main *command-line-args*)
