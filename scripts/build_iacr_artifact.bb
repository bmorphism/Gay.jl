#!/usr/bin/env bb

(require '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.string :as str]
         '[babashka.process :refer [shell]])

(def root (.getCanonicalFile (io/file (or (System/getenv "GAY_ROOT") "."))))
(def manifest-file (io/file root "papers/iacr-entropy-as-color/artifact.edn"))

(defn fail [message]
  (binding [*out* *err*] (println "IACR artifact build refused:" message))
  (System/exit 1))

(when-not (= 1 (count *command-line-args*))
  (fail "usage: bb scripts/build_iacr_artifact.bb OUTPUT_DIRECTORY"))

(let [output-dir (.getCanonicalFile (io/file (first *command-line-args*)))
      manifest (edn/read-string (slurp manifest-file))
      paths (vec (distinct (concat (:artifact/source-files manifest)
                                   (:artifact/license-files manifest))))
      dirty (-> (apply shell {:dir root :out :string :continue true}
                         "git" "status" "--porcelain" "--" paths)
                :out str/trim)
      commit (-> (shell {:dir root :out :string} "git" "rev-parse" "HEAD") :out str/trim)
      short-commit (subs commit 0 12)
      archive-name (str "gay-iacr-entropy-as-color-" short-commit ".tar.gz")
      archive (io/file output-dir archive-name)]
  (when-not (str/blank? dirty)
    (fail (str "manifest paths differ from HEAD:\n" dirty)))
  (.mkdirs output-dir)
  (apply shell {:dir root}
         "git" "archive" "--format=tar.gz"
         (str "--prefix=gay-iacr-entropy-as-color-" short-commit "/")
         (str "--output=" (.getPath archive)) "HEAD" "--" paths)
  (let [digest (-> (shell {:out :string} "shasum" "-a" "256" (.getPath archive))
                   :out (str/split #"\s+") first)]
    (println {:valid true
              :commit commit
              :archive (.getPath archive)
              :sha256 digest
              :files (count paths)})))
