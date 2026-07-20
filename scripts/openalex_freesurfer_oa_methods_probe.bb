#!/usr/bin/env bb

(ns openalex-freesurfer-oa-methods-probe
  (:require [cheshire.core :as json]
            [clojure.java.shell :refer [sh]]
            [clojure.set :as set]
            [clojure.string :as str])
  (:import [java.io File]))

(def default-sample-size 25)
(def default-seeds [1069])
(def default-fetch-limit 12)
(def default-pdf-extractor :auto)
(def default-fetch-order :text-first)
(def default-stratify-by :none)
(def default-uplift-prior-strength 8.0)

(def default-core-works
  [{:work/id "W2151721316" :work/label :surface-analysis-i}
   {:work/id "W2113319997" :work/label :surface-analysis-ii}
   {:work/id "W2004293194" :work/label :whole-brain-segmentation}
   {:work/id "W2151130155" :work/label :cortical-parcellation}
   {:work/id "W2157270343" :work/label :skull-stripping}
   {:work/id "W4241074797" :work/label :freesurfer-overview}])

(def source-label-order
  (zipmap (map :work/label default-core-works) (range)))

(def stratify-modes
  #{:none :core-work :oa-status :core-work+oa-status :pdf-availability})

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
   :no-text-evidence 0.10})

(def active-patterns
  [#"\brecon-?all\b"
   #"\bfreeview\b"
   #"\bmri_[a-z0-9_]+\b"
   #"\bfsaverage\b"
   #"\baseg\b"
   #"\baparc\b"
   #"\bfreesurfer.{0,140}\b(processed|processing|reconstructed|reconstruction|segmented|segmentation|parcellated|parcellation|quality control|qc|manual|edited|cortical thickness|surface area|thickness|pial|white surface)\b"
   #"\b(processed|reconstructed|segmented|parcellated|estimated|computed|derived|measured).{0,140}\bfreesurfer\b"])

(def pipeline-patterns
  [#"\bfmriprep\b"
   #"\bqsiprep\b"
   #"\bhuman connectome project\b"
   #"\bhcp\b"
   #"\buk biobank\b"
   #"\babcd\b"
   #"\badni\b"
   #"\bbids\b"
   #"\bpreprocessed\b"
   #"\bpipeline\b"])

(defn fail! [message]
  (binding [*out* *err*]
    (println (str "FAIL: " message)))
  (System/exit 1))

(defn parse-int [s]
  (Integer/parseInt s))

(defn parse-double [s]
  (Double/parseDouble s))

(defn parse-stratify-by [s]
  (let [mode (keyword s)]
    (if (contains? stratify-modes mode)
      mode
      (fail! (str "unknown stratify mode: " s
                  " (expected one of " (str/join ", " (sort (map name stratify-modes))) ")")))))

(defn round3 [x]
  (/ (Math/round (* 1000.0 (double x))) 1000.0))

(defn round1 [x]
  (/ (Math/round (* 10.0 (double x))) 10.0))

(defn curl-json [url]
  (let [{:keys [out err exit]} (sh "curl" "-L" "-sS" "--fail"
                                   "--connect-timeout" "10"
                                   "--max-time" "30"
                                   url)]
    (when-not (zero? exit)
      (fail! (str "curl failed for " url ": " err)))
    (json/parse-string out true)))

(defn curl-text [url]
  (let [{:keys [out err exit]} (sh "curl" "-L" "-sS" "--fail"
                                   "--compressed"
                                   "--connect-timeout" "10"
                                   "--max-time" "20"
                                   "--max-filesize" "5000000"
                                   "-A" "Gay.jl estimating probe (https://github.com/bmorphism/Gay.jl)"
                                   url)]
    (if (zero? exit)
      {:status :ok :text out}
      {:status :failed :error (str/trim err)})))

(defn curl-file [url path]
  (let [{:keys [err exit]} (sh "curl" "-L" "-sS" "--fail"
                               "--compressed"
                               "--connect-timeout" "10"
                               "--max-time" "30"
                               "--max-filesize" "12000000"
                               "-A" "Gay.jl estimating probe (https://github.com/bmorphism/Gay.jl)"
                               "-o" path
                               url)]
    (if (zero? exit)
      {:status :ok}
      {:status :failed :error (str/trim err)})))

(defn sample-url [work-id sample-size seed]
  (format "https://api.openalex.org/works?filter=cites:%s&sample=%d&seed=%d&per-page=%d&select=id,title,doi,ids,publication_year,abstract_inverted_index,authorships,open_access,best_oa_location,primary_location"
          work-id sample-size seed sample-size))

(defn idconv-url [pmids]
  (str "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids="
       (str/join "," pmids)
       "&format=json&tool=Gay.jl-estimating-probe"))

(defn pmc-xml-url [pmcid]
  (str "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id="
       pmcid
       "&retmode=xml&tool=Gay.jl-estimating-probe"))

(defn command-exists? [cmd]
  (zero? (:exit (sh "sh" "-c" (str "command -v " cmd " >/dev/null 2>&1")))))

(defn python-module-exists? [module]
  (zero? (:exit (sh "python3" "-c" (str "import " module)))))

(defn available-pdf-extractors []
  (cond-> []
    (command-exists? "pdftotext") (conj :pdftotext)
    (command-exists? "mutool") (conj :mutool)
    (python-module-exists? "pypdf") (conj :python-pypdf)
    (python-module-exists? "PyPDF2") (conj :python-pypdf2)
    (python-module-exists? "pdfminer.high_level") (conj :python-pdfminer)
    (command-exists? "clojure") (conj :clojure-pdfbox)
    (command-exists? "strings") (conj :strings)))

(defn choose-pdf-extractor [requested]
  (let [available (set (available-pdf-extractors))]
    (cond
      (= :none requested) nil
      (= :auto requested) (first (available-pdf-extractors))
      (contains? available requested) requested
      :else nil)))

(defn extract-pdf-file [pdf-path extractor]
  (let [cmd-result
        (case extractor
          :pdftotext (sh "pdftotext" "-layout" "-nopgbrk" pdf-path "-")
          :mutool (sh "mutool" "draw" "-F" "txt" "-o" "-" pdf-path)
          :clojure-pdfbox
          (sh "clojure"
              "-Sdeps" "{:deps {org.apache.pdfbox/pdfbox {:mvn/version \"3.0.3\"}}}"
              (str "-J-Dgay.pdf.path=" pdf-path)
              "-M"
              "-e"
              "(do
                 (import '(java.io File)
                         '(org.apache.pdfbox Loader)
                         '(org.apache.pdfbox.text PDFTextStripper))
                 (let [path (System/getProperty \"gay.pdf.path\")
                       doc (Loader/loadPDF (File. path))
                       stripper (doto (PDFTextStripper.) (.setEndPage 40))]
                   (try
                     (print (.getText stripper doc))
                     (finally (.close doc)))))")
          :python-pypdf
          (sh "python3" "-c"
              "import sys\nfrom pypdf import PdfReader\nreader = PdfReader(sys.argv[1])\nprint('\\n'.join((page.extract_text() or '') for page in reader.pages[:40]))"
              pdf-path)
          :python-pypdf2
          (sh "python3" "-c"
              "import sys\nfrom PyPDF2 import PdfReader\nreader = PdfReader(sys.argv[1])\nprint('\\n'.join((page.extract_text() or '') for page in reader.pages[:40]))"
              pdf-path)
          :python-pdfminer
          (sh "python3" "-c"
              "import sys\nfrom pdfminer.high_level import extract_text\nprint(extract_text(sys.argv[1], maxpages=40))"
              pdf-path)
          :strings (sh "strings" "-n" "5" pdf-path)
          {:exit 1 :err (str "unsupported PDF extractor: " extractor) :out ""})]
    (if (zero? (:exit cmd-result))
      {:status :ok
       :extractor extractor
       :text (:out cmd-result)}
      {:status :failed
       :extractor extractor
       :error (str/trim (:err cmd-result))})))

(defn fetch-pdf-text [url extractor]
  (if-not extractor
    {:status :skipped
     :extractor nil
     :error "No PDF text extractor available."}
    (let [tmp (File/createTempFile "gay-freesurfer-oa-" ".pdf")
          path (.getAbsolutePath tmp)]
      (try
        (let [downloaded (curl-file url path)]
          (if-not (= :ok (:status downloaded))
            {:status :failed
             :extractor extractor
             :error (:error downloaded)}
            (extract-pdf-file path extractor)))
        (finally
          (.delete tmp))))))

(defn author-id [authorship]
  (get-in authorship [:author :id]))

(defn authorship-weight [authorship]
  (let [position (keyword (or (:author_position authorship) "unknown"))
        base (get role-weight-scheme position (:unknown role-weight-scheme))
        bonus (if (:is_corresponding authorship)
                (:corresponding-bonus role-weight-scheme)
                0.0)]
    (min (:cap role-weight-scheme) (+ base bonus))))

(defn abstract-text [inverted]
  (when (seq inverted)
    (->> inverted
         (mapcat (fn [[word positions]]
                   (map (fn [pos] [pos word]) positions)))
         (sort-by first)
         (map second)
         (str/join " "))))

(defn html->text [s]
  (-> s
      (str/replace #"(?is)<script.*?</script>" " ")
      (str/replace #"(?is)<style.*?</style>" " ")
      (str/replace #"(?is)<[^>]+>" " ")
      (str/replace #"&nbsp;" " ")
      (str/replace #"&amp;" "&")
      (str/replace #"&lt;" "<")
      (str/replace #"&gt;" ">")
      (str/replace #"&#\d+;" " ")
      (str/replace #"\s+" " ")
      str/trim))

(defn pmid-from-work [work]
  (some-> (get-in work [:work/ids :pmid])
          (str/replace #"https?://pubmed\.ncbi\.nlm\.nih\.gov/" "")
          (str/replace #"/$" "")))

(defn text-like? [s]
  (and (string? s)
       (not (str/starts-with? s "%PDF"))
       (not (str/includes? (subs s 0 (min 200 (count s))) "\u0000"))
       (> (count (re-seq #"[A-Za-z]" (subs s 0 (min 5000 (count s))))) 200)))

(defn normalize-text [s]
  (let [s2 (if (re-find #"(?is)<html|<!doctype|<body|<article" s)
             (html->text s)
             (str/replace s #"\s+" " "))]
    (subs s2 0 (min 250000 (count s2)))))

(defn matches-any? [patterns text]
  (boolean (some #(re-find % text) patterns)))

(defn classify-text [text]
  (let [text (str/lower-case (or text ""))
        has-free (boolean (re-find #"\bfreesurfer\b" text))
        active? (matches-any? active-patterns text)
        pipeline? (matches-any? pipeline-patterns text)]
    (cond
      active? :active-explicit
      has-free :mention-only
      pipeline? :pipeline-inherited
      :else :no-text-evidence)))

(defn candidate-urls [work]
  (if-let [urls (:oa/candidate-urls work)]
    urls
    (let [open-access (:open_access work)
          best (:best_oa_location work)
          primary (:primary_location work)
          urls [(:oa_url open-access)
                (:landing_page_url best)
                (:pdf_url best)
                (:landing_page_url primary)
                (:pdf_url primary)]]
      (->> urls
           (remove str/blank?)
           distinct
           vec))))

(defn pdf-url? [url]
  (boolean (re-find #"(?i)(/pdf\b|\.pdf(\?|$)|/pdf/|format=pdf)" url)))

(defn oa-profile [work]
  (let [urls (candidate-urls work)
        open-access (:open_access work)]
    {:oa/is-oa? (boolean (:is_oa open-access))
     :oa/status (keyword (or (:oa_status open-access) "unknown"))
     :oa/any-repository-has-fulltext? (boolean (:any_repository_has_fulltext open-access))
     :oa/has-candidate-url? (boolean (seq urls))
     :oa/has-pdf-candidate? (boolean (some pdf-url? urls))
     :oa/has-nonpdf-candidate? (boolean (some (complement pdf-url?) urls))
     :oa/candidate-urls urls}))

(defn attach-pmcids [works]
  (let [pmids (->> works
                   (keep pmid-from-work)
                   distinct
                   vec)]
    (if (seq pmids)
      (let [records (:records (curl-json (idconv-url pmids)))
            pmid->pmcid (into {}
                              (for [record records
                                    :let [pmid (some-> (:pmid record) str)
                                          pmcid (:pmcid record)]
                                    :when (and pmid pmcid)]
                                [pmid pmcid]))]
        (mapv
         (fn [work]
           (let [pmid (pmid-from-work work)]
             (cond-> (assoc work :work/pmid pmid)
               (get pmid->pmcid pmid) (assoc :work/pmcid (get pmid->pmcid pmid)))))
         works))
      works)))

(defn fetch-work-text [work pdf-extractor fetch-order]
  (let [urls (candidate-urls work)
        pmc-url (some-> (:work/pmcid work) pmc-xml-url)
        text-candidates (concat
                         (when pmc-url [{:url pmc-url :source :pmc-xml}])
                         (map #(hash-map :url % :source :oa-or-landing)
                              (remove pdf-url? urls)))
        pdf-candidates (map #(hash-map :url % :source :pdf)
                            (filter pdf-url? urls))
        candidates (if (= :pdf-first fetch-order)
                     (concat pdf-candidates text-candidates)
                     (concat text-candidates pdf-candidates))]
    (loop [[candidate & more] candidates
           attempts []]
      (cond
        (nil? candidate)
        {:fetch/status :no-fetchable-text
         :fetch/attempts attempts}

        (= :pdf (:source candidate))
        (let [fetched (fetch-pdf-text (:url candidate) pdf-extractor)
              pdf-source (keyword (str "pdf-" (name (or (:extractor fetched) :none))))
              attempt (cond-> {:url (:url candidate)
                               :source pdf-source
                               :status (:status fetched)
                               :extractor (:extractor fetched)}
                        (:error fetched) (assoc :error (:error fetched)))]
          (if (and (= :ok (:status fetched))
                   (text-like? (:text fetched)))
            {:fetch/status :text-fetched
             :fetch/source pdf-source
             :fetch/url (:url candidate)
             :fetch/pdf-extractor (:extractor fetched)
             :fetch/attempts (conj attempts attempt)
             :fetch/text (normalize-text (:text fetched))}
            (recur more (conj attempts attempt))))

        :else
        (let [fetched (curl-text (:url candidate))
              attempt (cond-> {:url (:url candidate)
                               :source (:source candidate)
                               :status (:status fetched)}
                        (:error fetched) (assoc :error (:error fetched)))]
          (if (and (= :ok (:status fetched))
                   (text-like? (:text fetched)))
            {:fetch/status :text-fetched
             :fetch/source (:source candidate)
             :fetch/url (:url candidate)
             :fetch/attempts (conj attempts attempt)
             :fetch/text (normalize-text (:text fetched))}
            (recur more (conj attempts attempt))))))))

(defn work-evidence [source-work seed work]
  (let [abstract (abstract-text (:abstract_inverted_index work))
        abstract-class (classify-text (str (:title work) "\n" abstract))
        oa (oa-profile work)]
    (merge
     {:work/id (:id work)
      :work/title (:title work)
      :work/year (:publication_year work)
      :work/doi (:doi work)
      :work/ids (:ids work)
      :evidence/abstract-class abstract-class
      :evidence/abstract-has-text? (boolean (seq abstract))
      :sample/source-work-ids [(:work/id source-work)]
      :sample/source-work-labels [(:work/label source-work)]
      :sample/source-seeds [seed]
      :authorships (:authorships work)}
     oa)))

(defn fetch-sample [work sample-size seed]
  (let [data (curl-json (sample-url (:work/id work) sample-size seed))]
    {:work/id (:work/id work)
     :work/label (:work/label work)
     :seed seed
     :works (mapv #(work-evidence work seed %) (:results data))}))

(defn ordered-source-labels [labels]
  (->> labels
       distinct
       (sort-by #(get source-label-order % 999))
       vec))

(defn ordered-source-ids [ids]
  (let [id-order (zipmap (map :work/id default-core-works) (range))]
    (->> ids
         distinct
         (sort-by #(get id-order % 999))
         vec)))

(defn ordered-seeds [seeds]
  (->> seeds distinct sort vec))

(defn merge-work-evidence [a b]
  (-> a
      (assoc :sample/source-work-ids
             (ordered-source-ids (concat (:sample/source-work-ids a)
                                         (:sample/source-work-ids b)))
             :sample/source-work-labels
             (ordered-source-labels (concat (:sample/source-work-labels a)
                                            (:sample/source-work-labels b)))
             :sample/source-seeds
             (ordered-seeds (concat (:sample/source-seeds a)
                                    (:sample/source-seeds b))))))

(defn dedupe-works [samples]
  (->> (reduce
        (fn [m work]
          (update m (:work/id work)
                  #(if %
                     (merge-work-evidence % work)
                     work)))
        {}
        (mapcat :works samples))
       vals
       (sort-by :work/id)
       vec))

(defn primary-source-label [work]
  (or (first (:sample/source-work-labels work))
      :unknown))

(defn stratum-key [stratify-by work]
  (case stratify-by
    :none :all
    :core-work (primary-source-label work)
    :oa-status (:oa/status work)
    :core-work+oa-status [(primary-source-label work) (:oa/status work)]
    :pdf-availability (if (:oa/has-pdf-candidate? work) :pdf :no-pdf)
    :all))

(defn work-selection-key [fetch-order work]
  [(if (and (= :pdf-first fetch-order)
            (:oa/has-pdf-candidate? work))
     0
     1)
   (:work/id work)])

(defn pr-str-compare [a b]
  (compare (pr-str a) (pr-str b)))

(defn sorted-frequencies [xs]
  (into (sorted-map-by pr-str-compare) (frequencies xs)))

(defn round-robin-take [limit groups]
  (loop [queues (mapv (fn [[k works]] [k (vec works)]) groups)
         picked []]
    (cond
      (or (not (pos? limit)) (>= (count picked) limit) (empty? queues))
      picked

      :else
      (let [step (reduce
                  (fn [state [k works]]
                    (cond
                      (>= (count (:picked state)) limit)
                      state

                      (seq works)
                      (let [more (subvec works 1)
                            state' (-> state
                                       (update :picked conj (first works))
                                       (assoc :advanced? true))]
                        (cond-> state'
                          (seq more) (update :queues conj [k more])))

                      :else state))
                  {:queues [] :picked picked :advanced? false}
                  queues)]
        (if (:advanced? step)
          (recur (:queues step) (:picked step))
          (:picked step))))))

(defn fetch-candidates [works]
  (filter :oa/has-candidate-url? works))

(defn grouped-fetch-candidates [works fetch-order stratify-by]
  (->> (fetch-candidates works)
       (group-by #(stratum-key stratify-by %))
       (map (fn [[k grouped]]
              [k (sort-by #(work-selection-key fetch-order %) grouped)]))
       (sort-by #(pr-str (first %)))))

(defn select-fetch-work-ids [works fetch-limit fetch-order stratify-by]
  (->> (grouped-fetch-candidates works fetch-order stratify-by)
       (round-robin-take fetch-limit)
       (map :work/id)
       set))

(defn source-work-counts [works]
  (sorted-frequencies (mapcat :sample/source-work-labels works)))

(defn stratum-counts [works stratify-by]
  (sorted-frequencies (map #(stratum-key stratify-by %) works)))

(defn fetch-selection-summary [works selected-work-ids stratify-by]
  (let [candidates (vec (fetch-candidates works))
        selected (filter #(contains? selected-work-ids (:work/id %)) works)]
    {:selection/stratify-by stratify-by
     :selection/candidate-count (count candidates)
     :selection/selected-count (count selected)
     :selection/candidate-pdf-candidate-count
     (count (filter :oa/has-pdf-candidate? candidates))
     :selection/selected-pdf-candidate-count
     (count (filter :oa/has-pdf-candidate? selected))
     :selection/candidate-source-work-counts (source-work-counts candidates)
     :selection/selected-source-work-counts (source-work-counts selected)
     :selection/candidate-stratum-counts (stratum-counts candidates stratify-by)
     :selection/selected-stratum-counts (stratum-counts selected stratify-by)}))

(defn enrich-fetches [works selected-work-ids pdf-extractor fetch-order]
  (let [eligible selected-work-ids]
    (mapv
     (fn [work]
       (if (contains? eligible (:work/id work))
         (let [fetched (fetch-work-text work pdf-extractor fetch-order)
               attempts (:fetch/attempts fetched)
               pdf-attempts (filter #(some-> (:source %) name (str/starts-with? "pdf-"))
                                    attempts)
               full-class (when (= :text-fetched (:fetch/status fetched))
                            (classify-text (:fetch/text fetched)))
               final-class (or full-class (:evidence/abstract-class work))]
           (-> work
               (assoc :fetch/status (:fetch/status fetched)
                      :fetch/source (:fetch/source fetched)
                      :fetch/url (:fetch/url fetched)
                      :fetch/pdf-extractor (:fetch/pdf-extractor fetched)
                      :fetch/attempt-count (count attempts)
                      :fetch/attempt-sources (vec (map :source attempts))
                      :fetch/pdf-attempt-count (count pdf-attempts)
                      :fetch/pdf-attempt-status-counts (frequencies (map :status pdf-attempts))
                      :evidence/fulltext-class full-class
                      :evidence/final-class final-class
                      :evidence/final-weight (class-weight final-class))
               (dissoc :fetch/text)))
         (assoc work
                :fetch/status :not-attempted
                :fetch/attempt-count 0
                :fetch/attempt-sources []
                :fetch/pdf-attempt-count 0
                :fetch/pdf-attempt-status-counts {}
                :evidence/fulltext-class nil
                :evidence/final-class (:evidence/abstract-class work)
                :evidence/final-weight (class-weight (:evidence/abstract-class work)))))
     works)))

(defn author-weight-map-with [works work-weight]
  (reduce
   (fn [m work]
     (let [w (work-weight work)]
       (reduce
        (fn [m2 authorship]
          (if-let [id (author-id authorship)]
            (update m2 id (fnil max 0.0) (* w (authorship-weight authorship)))
            m2))
        m
        (:authorships work))))
   {}
   works))

(defn author-weight-map [works]
  (author-weight-map-with works :evidence/final-weight))

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

(defn class-counts [works class-key]
  (frequencies (map class-key works)))

(defn clamp [lo hi x]
  (max lo (min hi x)))

(defn mean [xs]
  (if (seq xs)
    (/ (reduce + xs) (double (count xs)))
    0.0))

(defn abstract-weight [work]
  (class-weight (:evidence/abstract-class work)))

(defn final-weight [work]
  (:evidence/final-weight work))

(defn text-fetched? [work]
  (= :text-fetched (:fetch/status work)))

(defn availability-summary [works]
  {:works/is-oa (count (filter :oa/is-oa? works))
   :works/with-candidate-url (count (filter :oa/has-candidate-url? works))
   :works/with-nonpdf-candidate (count (filter :oa/has-nonpdf-candidate? works))
   :works/with-pdf-candidate (count (filter :oa/has-pdf-candidate? works))
   :works/repository-fulltext-flagged (count (filter :oa/any-repository-has-fulltext? works))
   :oa/status-counts (frequencies (map :oa/status works))})

(defn pmc-summary [works]
  {:works/with-pmid (count (filter :work/pmid works))
   :works/with-pmcid (count (filter :work/pmcid works))
   :pmc/coverage (if (seq works)
                   (round3 (/ (count (filter :work/pmcid works))
                              (double (count works))))
                   0.0)})

(defn fetch-summary [works]
  (let [attempted (remove #(= :not-attempted (:fetch/status %)) works)]
    {:fetch/attempted (count attempted)
     :fetch/text-fetched (count (filter #(= :text-fetched (:fetch/status %)) attempted))
     :fetch/no-fetchable-text (count (filter #(= :no-fetchable-text (:fetch/status %)) attempted))
     :fetch/not-attempted (count (filter #(= :not-attempted (:fetch/status %)) works))
     :fetch/status-counts (frequencies (map :fetch/status works))
     :fetch/source-counts (frequencies (keep :fetch/source works))
     :fetch/pdf-attempted (count (filter #(pos? (:fetch/pdf-attempt-count % 0)) works))
     :fetch/pdf-attempt-count (reduce + (map #(:fetch/pdf-attempt-count % 0) works))}))

(defn pdf-summary [works requested-pdf-extractor selected-pdf-extractor]
  (let [pdf-status-counts (apply merge-with +
                                 (map :fetch/pdf-attempt-status-counts works))]
    {:pdf/requested-extractor requested-pdf-extractor
     :pdf/selected-extractor selected-pdf-extractor
     :pdf/available-extractors (available-pdf-extractors)
     :pdf/works-with-pdf-candidate (count (filter :oa/has-pdf-candidate? works))
     :pdf/attempted-works (count (filter #(pos? (:fetch/pdf-attempt-count % 0)) works))
     :pdf/attempt-count (reduce + (map #(:fetch/pdf-attempt-count % 0) works))
     :pdf/attempt-status-counts pdf-status-counts
     :pdf/text-fetched (count (filter #(and (= :text-fetched (:fetch/status %))
                                            (some-> (:fetch/source %) name (str/starts-with? "pdf-")))
                                      works))}))

(defn weighted-work-rate-with [works work-weight]
  (if (seq works)
    (round3 (/ (reduce + (map work-weight works)) (count works)))
    0.0))

(defn weighted-work-rate [works]
  (weighted-work-rate-with works final-weight))

(defn uplift-row [prior-strength k grouped]
  (let [observed (filter text-fetched? grouped)
        mean-uplift (mean (map #(- (final-weight %) (abstract-weight %)) observed))
        shrinkage (if (pos? (count observed))
                    (/ (count observed) (+ (double (count observed)) prior-strength))
                    0.0)
        shrunk-uplift (* mean-uplift shrinkage)
        corrected (fn [work]
                    (clamp 0.0 1.0 (+ (abstract-weight work) shrunk-uplift)))]
    [k {:population-count (count grouped)
        :text-fetched-count (count observed)
        :abstract-rate (weighted-work-rate-with grouped abstract-weight)
        :lower-bound-final-rate (weighted-work-rate-with grouped final-weight)
        :mean-fulltext-uplift (round3 mean-uplift)
        :uplift-shrinkage (round3 shrinkage)
        :shrunk-fulltext-uplift (round3 shrunk-uplift)
        :poststratified-rate (weighted-work-rate-with grouped corrected)
        :observed? (pos? (count observed))}]))

(defn stratified-uplift-summary [works stratify-by role-weighted-total prior-strength]
  (let [groups (->> works
                    (group-by #(stratum-key stratify-by %))
                    (sort-by #(pr-str (first %))))
        rows (into (sorted-map-by pr-str-compare)
                   (map (fn [[k grouped]]
                          (uplift-row prior-strength k grouped))
                        groups))
        stratum-uplift (fn [work]
                         (:shrunk-fulltext-uplift
                          (get rows (stratum-key stratify-by work)
                               {:shrunk-fulltext-uplift 0.0})))
        corrected-weight (fn [work]
                           (clamp 0.0 1.0
                                  (+ (abstract-weight work)
                                     (stratum-uplift work))))
        post-method-weighted (author-weight-map-with works corrected-weight)
        post-method-total (reduce + (vals post-method-weighted))
        observed-strata (count (filter :observed? (vals rows)))]
    {:uplift/stratify-by stratify-by
     :uplift/prior-strength prior-strength
     :uplift/population-count (count works)
     :uplift/text-fetched-count (count (filter text-fetched? works))
     :uplift/observed-strata observed-strata
     :uplift/stratum-count (count rows)
     :uplift/abstract-rate (weighted-work-rate-with works abstract-weight)
     :uplift/lower-bound-final-rate (weighted-work-rate works)
     :uplift/poststratified-rate (weighted-work-rate-with works corrected-weight)
     :uplift/poststratified-method-weighted-author-equivalents (round1 post-method-total)
     :uplift/poststratified-method-to-role-ratio (if (pos? role-weighted-total)
                                                   (round3 (/ post-method-total role-weighted-total))
                                                   0.0)
     :uplift/strata rows}))

(defn summarize [sample-size seeds fetch-limit requested-pdf-extractor fetch-order stratify-by uplift-prior-strength]
  (let [samples (vec
                 (for [work default-core-works
                       seed seeds]
                   (fetch-sample work sample-size seed)))
        works (attach-pmcids (vec (dedupe-works samples)))
        selected-pdf-extractor (choose-pdf-extractor requested-pdf-extractor)
        selected-work-ids (select-fetch-work-ids works fetch-limit fetch-order stratify-by)
        enriched (enrich-fetches works selected-work-ids selected-pdf-extractor fetch-order)
        role-weighted (role-only-author-weight-map enriched)
        method-weighted (author-weight-map enriched)
        role-weighted-total (reduce + (vals role-weighted))
        method-weighted-total (reduce + (vals method-weighted))
        uplift (stratified-uplift-summary enriched stratify-by role-weighted-total uplift-prior-strength)
        examples (->> enriched
                      (filter #(or (= :text-fetched (:fetch/status %))
                                   (not= :no-text-evidence (:evidence/final-class %))))
                      (take 12)
                      (mapv #(select-keys %
                                           [:work/id :work/title :work/year :work/doi
                                            :work/pmid :work/pmcid
                                            :sample/source-work-labels
                                            :oa/status :fetch/status :fetch/source :fetch/url
                                            :fetch/pdf-extractor
                                            :fetch/pdf-attempt-count
                                            :fetch/pdf-attempt-status-counts
                                            :evidence/abstract-class
                                            :evidence/fulltext-class
                                            :evidence/final-class
                                            :evidence/final-weight])))]
    {:refinement/id :openalex-oa-methods-probe
     :query/core-work-ids (mapv :work/id default-core-works)
     :query/fetch-order fetch-order
     :query/stratify-by stratify-by
     :sample/size-per-work-seed sample-size
     :sample/seeds seeds
     :sample/raw-slots (* sample-size (count seeds) (count default-core-works))
     :sample/unique-citing-works (count enriched)
     :sample/source-work-counts (source-work-counts enriched)
     :open-access/summary (availability-summary enriched)
     :pmc/summary (pmc-summary enriched)
     :fetch/selection (fetch-selection-summary enriched selected-work-ids stratify-by)
     :fetch/summary (fetch-summary enriched)
     :pdf/summary (pdf-summary enriched requested-pdf-extractor selected-pdf-extractor)
     :sample/abstract-class-counts (class-counts enriched :evidence/abstract-class)
     :sample/final-class-counts (class-counts enriched :evidence/final-class)
     :sample/class-weights class-weight
     :sample/active-equivalent-work-rate (weighted-work-rate enriched)
     :sample/fulltext-uplift-estimator uplift
     :sample/unique-authors (count role-weighted)
     :sample/role-weighted-author-equivalents (round1 role-weighted-total)
     :sample/method-weighted-author-equivalents (round1 method-weighted-total)
     :sample/method-to-role-ratio (if (pos? role-weighted-total)
                                    (round3 (/ method-weighted-total
                                               role-weighted-total))
                                    0.0)
     :sample/examples examples
     :interpretation
     "Open-access methods probe: classifies title/abstract evidence for every sampled citing work, maps PMIDs to PMCIDs, enriches fetchable PMC XML or OA/HTML candidates, and attempts PDF text extraction when an extractor is available."}))

(defn parse-args [args]
  (loop [args args
         opts {:sample-size default-sample-size
               :seeds default-seeds
               :fetch-limit default-fetch-limit
               :pdf-extractor default-pdf-extractor
               :fetch-order default-fetch-order
               :stratify-by default-stratify-by
               :uplift-prior-strength default-uplift-prior-strength}]
    (if-let [arg (first args)]
      (case arg
        "--sample-size" (recur (nnext args) (assoc opts :sample-size (parse-int (second args))))
        "--seeds" (recur (nnext args)
                         (assoc opts :seeds
                                (mapv parse-int (str/split (second args) #","))))
        "--fetch-limit" (recur (nnext args) (assoc opts :fetch-limit (parse-int (second args))))
        "--pdf-extractor" (recur (nnext args) (assoc opts :pdf-extractor (keyword (second args))))
        "--fetch-order" (recur (nnext args) (assoc opts :fetch-order (keyword (second args))))
        "--stratify-by" (recur (nnext args) (assoc opts :stratify-by (parse-stratify-by (second args))))
        "--uplift-prior-strength" (recur (nnext args) (assoc opts :uplift-prior-strength (parse-double (second args))))
        "--pdf-first" (recur (next args) (assoc opts :fetch-order :pdf-first))
        "--no-pdf" (recur (next args) (assoc opts :pdf-extractor :none))
        "--no-fetch" (recur (next args) (assoc opts :fetch-limit 0))
        (fail! (str "unknown argument: " arg)))
      opts)))

(defn -main [& args]
  (let [{:keys [sample-size seeds fetch-limit pdf-extractor fetch-order stratify-by uplift-prior-strength]} (parse-args args)]
    (println (pr-str (summarize sample-size seeds fetch-limit pdf-extractor fetch-order stratify-by uplift-prior-strength)))))

(apply -main *command-line-args*)
