#!/usr/bin/env bb

(ns openalex-freesurfer-core-union
  (:require [cheshire.core :as json]
            [clojure.java.shell :refer [sh]]
            [clojure.set :as set]
            [clojure.string :as str])
  (:import [java.net URLEncoder]))

(def default-per-page 200)

(def role-weight-scheme
  {:first 0.75
   :last 0.45
   :middle 0.18
   :unknown 0.15
   :corresponding-bonus 0.20
   :cap 1.0})

(def default-core-works
  [{:work/id "W2151721316"
    :work/doi "10.1006/nimg.1998.0395"
    :work/label :surface-analysis-i}
   {:work/id "W2113319997"
    :work/doi "10.1006/nimg.1998.0396"
    :work/label :surface-analysis-ii}
   {:work/id "W2004293194"
    :work/doi "10.1016/S0896-6273(02)00569-X"
    :work/label :whole-brain-segmentation}
   {:work/id "W2151130155"
    :work/doi "10.1093/cercor/bhg087"
    :work/label :cortical-parcellation}
   {:work/id "W2157270343"
    :work/doi "10.1016/j.neuroimage.2004.03.032"
    :work/label :skull-stripping}
   {:work/id "W4241074797"
    :work/doi "10.1016/j.neuroimage.2012.01.021"
    :work/label :freesurfer-overview}])

(defn fail! [message]
  (binding [*out* *err*]
    (println (str "FAIL: " message)))
  (System/exit 1))

(defn parse-int [s]
  (Integer/parseInt s))

(defn url-encode [s]
  (URLEncoder/encode s "UTF-8"))

(defn curl-json [url]
  (let [{:keys [out err exit]} (sh "curl" "-sS" "--fail"
                                   "--connect-timeout" "10"
                                   "--max-time" "30"
                                   url)]
    (when-not (zero? exit)
      (fail! (str "curl failed for " url ": " err)))
    (json/parse-string out true)))

(defn work-api-url [work-id]
  (str "https://api.openalex.org/works/" work-id))

(defn citing-page-url [work-id per-page cursor]
  (str "https://api.openalex.org/works?filter=cites:" work-id
       "&per-page=" per-page
       "&cursor=" (url-encode cursor)
       "&select=id,authorships"))

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

(defn round1 [x]
  (/ (Math/round (* 10.0 (double x))) 10.0))

(defn author-weight-map [authorships]
  (reduce
   (fn [m authorship]
     (if-let [id (author-id authorship)]
       (update m id (fnil max 0.0) (authorship-weight authorship))
       m))
   {}
   authorships))

(defn role-counts [authorships]
  (reduce
   (fn [m authorship]
     (let [position (keyword (or (:author_position authorship) "unknown"))]
       (cond-> (update m position (fnil inc 0))
         (:is_corresponding authorship) (update :corresponding (fnil inc 0)))))
   {}
   authorships))

(defn role-weighted-total [author-weights]
  (round1 (reduce + 0.0 (vals author-weights))))

(defn fetch-work-meta [work]
  (let [data (curl-json (work-api-url (:work/id work)))
        filter-count (get-in (curl-json (citing-count-url (:work/id work)))
                             [:meta :count])]
    (assoc work
           :work/title (:title data)
           :work/year (:publication_year data)
           :work/cited-by-count (:cited_by_count data)
           :query/citing-filter-count filter-count)))

(defn fetch-citing-authors
  [{:keys [work/id] :as work} {:keys [per-page max-pages]}]
  (loop [cursor "*"
         page 0
         seen-cursors #{}
         citing-works #{}
         authors #{}
         author-weights {}
         role-events {}
         author-events 0]
    (if (and max-pages (>= page max-pages))
      {:pages page
       :truncated? true
       :citing-works citing-works
       :authors authors
       :author-weights author-weights
       :role-events role-events
       :author-events author-events}
      (do
        (when (contains? seen-cursors cursor)
          (fail! (str "OpenAlex cursor cycle for " id " at cursor " cursor)))
        (let [data (curl-json (citing-page-url id per-page cursor))
            results (:results data)
            next-cursor (get-in data [:meta :next_cursor])]
        (if (empty? results)
          {:pages page
           :truncated? false
           :citing-works citing-works
           :authors authors
           :author-weights author-weights
           :role-events role-events
           :author-events author-events}
          (let [page-work-ids (set (keep :id results))
                page-authorships (mapcat :authorships results)
                page-authors (set (keep author-id page-authorships))
                page-author-weights (author-weight-map page-authorships)
                page-role-events (role-counts page-authorships)]
            (if next-cursor
              (recur next-cursor
                     (inc page)
                     (conj seen-cursors cursor)
                     (set/union citing-works page-work-ids)
                     (set/union authors page-authors)
                     (merge-with max author-weights page-author-weights)
                     (merge-with + role-events page-role-events)
                     (+ author-events (count page-authorships)))
              {:pages (inc page)
               :truncated? false
               :citing-works (set/union citing-works page-work-ids)
               :authors (set/union authors page-authors)
               :author-weights (merge-with max author-weights page-author-weights)
               :role-events (merge-with + role-events page-role-events)
               :author-events (+ author-events (count page-authorships))}))))))))

(defn summarize-work [work opts]
  (let [meta-work (fetch-work-meta work)
        fetched (fetch-citing-authors meta-work opts)
        filter-count (:query/citing-filter-count meta-work)
        fetched-count (count (:citing-works fetched))]
    (merge
     (select-keys meta-work
                  [:work/id :work/doi :work/label :work/title :work/year
                   :work/cited-by-count :query/citing-filter-count])
     {:pagination/pages (:pages fetched)
      :pagination/truncated? (:truncated? fetched)
      :pagination/fetched-citing-works fetched-count
      :pagination/coverage (if (pos? filter-count)
                             (/ fetched-count (double filter-count))
                             0.0)
      :authors/unique (count (:authors fetched))
      :authors/events (:author-events fetched)
      :authors/role-weighted-equivalents (role-weighted-total (:author-weights fetched))
      :authors/role-events (:role-events fetched)
      :_sets {:citing-works (:citing-works fetched)
              :authors (:authors fetched)
              :author-weights (:author-weights fetched)}})))

(defn strip-sets [m]
  (dissoc m :_sets))

(defn summarize-union [work-summaries]
  (let [author-sets (map #(get-in % [:_sets :authors]) work-summaries)
        citing-work-sets (map #(get-in % [:_sets :citing-works]) work-summaries)
        union-authors (apply set/union author-sets)
        union-citing-works (apply set/union citing-work-sets)
        union-author-weights (apply merge-with max
                                    (map #(get-in % [:_sets :author-weights])
                                         work-summaries))
        citation-slots (reduce + (map :query/citing-filter-count work-summaries))
        fetched-slots (reduce + (map :pagination/fetched-citing-works work-summaries))
        all-full? (every? (comp not :pagination/truncated?) work-summaries)]
    {:works/count (count work-summaries)
     :works/full-pagination? all-full?
     :citation-slots/filter-count citation-slots
     :citation-slots/fetched fetched-slots
     :citation-slots/coverage (if (pos? citation-slots)
                                (/ fetched-slots (double citation-slots))
                                0.0)
     :citing-works/unique-fetched (count union-citing-works)
     :authors/unique-fetched (count union-authors)
     :authors/events-fetched (reduce + (map :authors/events work-summaries))
     :authors/role-weighted-equivalents (role-weighted-total union-author-weights)
     :authors/role-weight-scheme role-weight-scheme
     :authors/role-events-fetched (apply merge-with +
                                         (map :authors/role-events work-summaries))}))

(defn parse-args [args]
  (loop [args args
         opts {:per-page default-per-page
               :max-pages nil}]
    (if-let [arg (first args)]
      (case arg
        "--per-page" (recur (nnext args) (assoc opts :per-page (parse-int (second args))))
        "--max-pages" (recur (nnext args) (assoc opts :max-pages (parse-int (second args))))
        "--work-ids" (recur (nnext args)
                            (assoc opts :work-ids
                                   (mapv str/trim (str/split (second args) #","))))
        (fail! (str "unknown argument: " arg)))
      opts)))

(defn select-works [opts]
  (if-let [ids (:work-ids opts)]
    (mapv (fn [id] {:work/id id :work/label (keyword id)}) ids)
    default-core-works))

(defn -main [& args]
  (let [opts (parse-args args)
        works (select-works opts)
        work-summaries (mapv #(summarize-work % opts) works)
        public-work-summaries (mapv strip-sets work-summaries)
        union-summary (summarize-union work-summaries)]
    (println
     (pr-str
      {:refinement/id :openalex-core-paper-author-union
       :query/per-page (:per-page opts)
       :query/max-pages (:max-pages opts)
       :query/work-ids (mapv :work/id works)
       :work/summaries public-work-summaries
       :union/summary union-summary
       :interpretation
       "Paginated author-id union over works citing canonical FreeSurfer papers. This is a publication-author proxy; it still overcounts passive coauthors and undercounts non-publishing users."}))))

(apply -main *command-line-args*)
