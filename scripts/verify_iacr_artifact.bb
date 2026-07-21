#!/usr/bin/env bb

(require '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.string :as str]
         '[babashka.process :refer [shell]])

(def root (.getCanonicalFile (io/file (or (System/getenv "GAY_ROOT") "."))))
(def manifest-file (io/file root "papers/iacr-entropy-as-color/artifact.edn"))

(defn require* [condition message]
  (when-not condition (throw (ex-info message {}))))

(defn tracked-files []
  (-> (shell {:dir root :out :string} "git" "ls-files")
      :out
      str/split-lines
      set))

(defn verify [manifest]
  (let [tracked (tracked-files)
        sources (:artifact/source-files manifest)
        generated (:artifact/generated-files manifest)
        licenses (:artifact/license-files manifest)
        commands (:artifact/commands manifest)]
    (require* (= 1 (:artifact/version manifest)) "unsupported artifact version")
    (require* (= (count sources) (count (set sources))) "duplicate source path")
    (doseq [path sources]
      (require* (.isFile (io/file root path)) (str "missing source file: " path))
      (require* (contains? tracked path) (str "source file is not tracked: " path)))
    (doseq [path licenses]
      (require* (.isFile (io/file root path)) (str "missing license file: " path)))
    (doseq [path generated]
      (require* (not (contains? tracked path))
                (str "generated paper output is tracked: " path)))
    (doseq [{:command/keys [id run expect]} commands]
      (require* (keyword? id) (str "invalid command id: " id))
      (require* (and (string? run) (not (str/blank? run)))
                (str "empty command: " id))
      (require* (and (string? expect) (not (str/blank? expect)))
                (str "missing expected output: " id)))
    {:valid true
     :sources (count sources)
     :generated-untracked (count generated)
     :commands (count commands)
     :licenses (count licenses)}))

(try
  (println (verify (edn/read-string (slurp manifest-file))))
  (catch Exception error
    (binding [*out* *err*]
      (println "IACR artifact manifest invalid:" (ex-message error)))
    (System/exit 1)))
