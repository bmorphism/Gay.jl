#!/usr/bin/env bb

(require '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.string :as str]
         '[babashka.process :refer [shell]])

(def root (.getCanonicalFile (io/file (or (System/getenv "GAY_ROOT") "."))))
(def ledger-file (io/file root "papers/iacr-entropy-as-color/standards.edn"))

(defn require* [condition message]
  (when-not condition (throw (ex-info message {}))))

(defn tracked-files []
  (-> (shell {:dir root :out :string} "git" "ls-files") :out str/split-lines set))

(defn verify [ledger]
  (let [sources (:standards/sources ledger)
        requirements (:standards/requirements ledger)
        source-ids (set (map :source/id sources))
        requirement-ids (map :requirement/id requirements)
        tracked (tracked-files)
        allowed-statuses #{:satisfied :incomplete :blocked-external :not-applicable}]
    (require* (= 1 (:standards/version ledger)) "unsupported standards ledger version")
    (require* (re-matches #"\d{4}-\d{2}-\d{2}" (:standards/retrieved ledger))
              "retrieval date must use YYYY-MM-DD")
    (require* (= :iacr-eprint (:standards/target ledger)) "target must be explicit")
    (require* (= (count sources) (count source-ids)) "duplicate standards source id")
    (require* (= (count requirement-ids) (count (set requirement-ids)))
              "duplicate requirement id")
    (doseq [{:source/keys [id url authority]} sources]
      (require* (= :iacr authority) (str "non-IACR authority: " id))
      (require* (and (string? url) (str/starts-with? url "https://"))
                (str "invalid source URL: " id)))
    (doseq [{:requirement/keys [id source status evidence reason]} requirements]
      (require* (contains? source-ids source) (str "unknown source for " id))
      (require* (contains? allowed-statuses status) (str "invalid status for " id))
      (if (#{:satisfied :not-applicable} status)
        (do
          (require* (seq evidence) (str "missing evidence for " id))
          (doseq [path evidence]
            (require* (contains? tracked path) (str "untracked evidence for " id ": " path))))
        (require* (and (string? reason) (not (str/blank? reason)))
                  (str "unexplained blocker for " id))))
    {:valid true
     :sources (count sources)
     :requirements (count requirements)
     :status-counts (frequencies (map :requirement/status requirements))}))

(try
  (println (verify (edn/read-string (slurp ledger-file))))
  (catch Exception error
    (binding [*out* *err*]
      (println "IACR standards ledger invalid:" (ex-message error)))
    (System/exit 1)))
