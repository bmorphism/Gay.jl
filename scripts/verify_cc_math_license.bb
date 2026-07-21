#!/usr/bin/env bb

(require '[clojure.java.io :as io]
         '[clojure.string :as str])

(def root (.getCanonicalFile (io/file (or (System/getenv "GAY_ROOT") "."))))
(def license-file (io/file root "LICENSE-CC-MATH-DRAFT.md"))

(defn require* [condition message]
  (when-not condition (throw (ex-info message {}))))

(def colors
  [{:trit "+1" :name "PLUS" :tile "🟥" :rgb "#ff6464"}
   {:trit "0" :name "ERGODIC" :tile "🟩" :rgb "#64c864"}
   {:trit "−1" :name "MINUS" :tile "🟦" :rgb "#6464ff"}])

(defn verify [text]
  (require* (str/includes? text "Not presently operative")
            "draft must remain explicitly non-operative")
  (require* (str/includes? text "not identify a right")
            "color/identity boundary is missing")
  (require* (str/includes? text "RGB does not")
            "RGB representation disclaimer is missing")
  (doseq [{:keys [trit name tile rgb]} colors]
    (require* (str/includes? text (str tile " " trit " " name))
              (str "missing semantic tile: " name))
    (require* (str/includes? text rgb)
              (str "missing presentation color: " name)))
  (let [colored-headings (re-seq #"(?m)^## [🟥🟩🟦] (?:\+1|0|−1) (?:PLUS|ERGODIC|MINUS) — " text)]
    (require* (<= 8 (count colored-headings))
              "each normative section must carry an audit color")
    {:valid true
     :status :counterfactual
     :colored-sections (count colored-headings)
     :trits (mapv :trit colors)
     :identity-bearing :typed-referent
     :color-role :contextual-presentation}))

(try
  (println (verify (slurp license-file)))
  (catch Exception error
    (binding [*out* *err*]
      (println "CC-MATH color audit failed:" (ex-message error)))
    (System/exit 1)))
