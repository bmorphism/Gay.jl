#!/usr/bin/env bb

(require '[babashka.process :refer [shell]]
         '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.string :as str])
(import '[java.math BigInteger]
        '[java.security MessageDigest])

(def root (.getCanonicalFile (io/file (or (System/getenv "GAY_ROOT") "."))))
(def registry-file (io/file root "licenses/cc-math/versions.edn"))

(defn require* [condition message]
  (when-not condition (throw (ex-info message {}))))

(defn sha256 [file]
  (let [digest (MessageDigest/getInstance "SHA-256")]
    (with-open [input (io/input-stream file)]
      (let [buffer (byte-array 8192)]
        (loop []
          (let [size (.read input buffer)]
            (when (pos? size)
              (.update digest buffer 0 size)
              (recur))))))
    (format "%064x" (BigInteger. 1 (.digest digest)))))

(def colors
  [{:trit "+1" :name "PLUS" :tile "🟥" :rgb "#ff6464"}
   {:trit "0" :name "ERGODIC" :tile "🟩" :rgb "#64c864"}
   {:trit "−1" :name "MINUS" :tile "🟦" :rgb "#6464ff"}])

(defn verify-node [versions [version-id version]]
  (require* (= version-id (:version/id version)) "version key/id mismatch")
  (require* (uuid? version-id) "version identity must be a typed UUID")
  (require* (pos-int? (:version/ordinal version)) "ordinal must be positive")
  (require* (set? (:version/parents version)) "parents must be a set")
  (doseq [parent (:version/parents version)]
    (require* (contains? versions parent) (str "unknown parent: " parent)))
  (let [content-file (io/file root (:content/path version))]
    (require* (.isFile content-file) (str "missing version content: " content-file))
    (require* (= (:content/sha256 version) (sha256 content-file))
              (str "content digest mismatch: " (:content/path version)))))

(defn verify [world]
  (let [versions (:license/versions world)
        artifacts (:license/artifacts world)
        license-head (get-in artifacts [:cc-math/license :artifact/head])
        license-text (slurp (io/file root (:content/path (get versions license-head))))]
    (require* (= :cc-math/version-graph (:cc-math/schema world))
              "unsupported CC-MATH version graph")
    (require* (= 1 (:cc-math/schema-revision world))
              "unsupported schema revision")
    (require* (= :cc-math (get-in world [:license/referent :referent/id]))
              "license referent is missing")
    (doseq [[_ {:artifact/keys [head]}] artifacts]
      (require* (contains? versions head) (str "artifact head is unknown: " head)))
    (doseq [node versions] (verify-node versions node))
    (require* (str/includes? license-text "Not presently operative")
              "draft must remain explicitly non-operative")
    (require* (not (re-find #"CC-MATH\s+[0-9]+(?:\.[0-9]+)*" license-text))
              "prose must not carry the authoritative version")
    (require* (str/includes? license-text "RGB does not")
              "RGB representation disclaimer is missing")
    (doseq [{:keys [trit name tile rgb]} colors]
      (require* (str/includes? license-text (str tile " " trit " " name))
                (str "missing semantic tile: " name))
      (require* (str/includes? license-text rgb)
                (str "missing presentation color: " name)))
    (let [result (shell {:dir root :out :string :err :string}
                        "jank" "--module-path" "jank"
                        "run-main" "cc-math-license")]
      (require* (zero? (:exit result)) (str "native jank check failed: " (:err result)))
      (require* (str/includes? (:out result) ":optimization-target :jank-ir")
                "native jank processor did not report its target")
      (require* (str/includes? (:out result) ":input-preserved true")
                "native jank processor mutated or replaced its rejected input")
      (require* (str/includes? (:out result) ":invalid-version :rejected")
                "native jank processor accepted an untyped version"))
    {:valid true
     :status :counterfactual
     :version-nodes (count versions)
     :artifact-heads (count artifacts)
     :identity-bearing :typed-referent
     :version-identity :typed-uuid
     :persistence :immutable
     :native-runtime :jank
     :optimization-target :jank-ir}))

(try
  (println (verify (edn/read-string (slurp registry-file))))
  (catch Exception error
    (binding [*out* *err*]
      (println "CC-MATH version audit failed:" (ex-message error)))
    (System/exit 1)))
