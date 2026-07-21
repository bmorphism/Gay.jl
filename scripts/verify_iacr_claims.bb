#!/usr/bin/env bb

(require '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.string :as str])

(def root (.getCanonicalFile (io/file (or (System/getenv "GAY_ROOT") "."))))
(def ledger-path (io/file root "papers/iacr-entropy-as-color/claims.edn"))

(defn require* [condition message]
  (when-not condition (throw (ex-info message {}))))

(defn source-text [relative-path]
  (let [file (io/file root relative-path)]
    (require* (.isFile file) (str "missing evidence file: " relative-path))
    (slurp file)))

(defn symbol-present? [{:evidence/keys [type path symbol]}]
  (let [text (source-text path)]
    (case type
      :lean-symbol
      (boolean (re-find (re-pattern (str "(?m)^theorem\\s+"
                                         (java.util.regex.Pattern/quote symbol)
                                         "\\b")) text))
      :julia-symbol
      (boolean (re-find (re-pattern (str "(?m)^(?:function|struct|mutable struct)\\s+"
                                         (java.util.regex.Pattern/quote symbol)
                                         "\\b")) text))
      (:test-text :manuscript-counterexample :counterexample)
      (str/includes? text symbol)
      false)))

(defn verify-ledger [ledger]
  (let [statuses (:statuses ledger)
        claims (:claims ledger)
        ids (map :claim/id claims)]
    (require* (= 1 (:ledger/version ledger)) "unsupported ledger version")
    (require* (= (count ids) (count (set ids))) "duplicate claim IDs")
    (doseq [claim claims]
      (let [status (:claim/status claim)
            evidence (:evidence claim)]
        (require* (contains? statuses status)
                  (str "unknown status for " (:claim/id claim)))
        (require* (boolean? (:claim/asserted? claim))
                  (str "claim lacks asserted flag: " (:claim/id claim)))
        (require* (not (str/blank? (:claim/text claim)))
                  (str "claim lacks text: " (:claim/id claim)))
        (require* (not (and (:claim/asserted? claim)
                            (#{:unverified :contradicted} status)))
                  (str "manuscript asserts unsupported claim: " (:claim/id claim)))
        (when (#{:verified :scoped :contradicted} status)
          (require* (seq evidence)
                    (str "claim lacks evidence: " (:claim/id claim)))
          (doseq [item evidence]
            (require* (symbol-present? item)
                      (str "unresolved evidence for " (:claim/id claim) ": " item))))))
    {:valid true
     :claims (count claims)
     :asserted (count (filter :claim/asserted? claims))
     :status-counts (frequencies (map :claim/status claims))}))

(try
  (println (verify-ledger (edn/read-string (slurp ledger-path))))
  (catch Exception error
    (binding [*out* *err*]
      (println "IACR claim ledger invalid:" (ex-message error)))
    (System/exit 1)))
