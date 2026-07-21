#!/usr/bin/env bb

(require '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[clojure.string :as str]
         '[babashka.process :refer [shell]])
(import '[java.math BigInteger]
        '[java.security MessageDigest])

(def root (.getCanonicalFile (io/file (or (System/getenv "GAY_ROOT") "."))))
(def paper-dir (io/file root "papers/iacr-entropy-as-color"))
(def toolchain-file (io/file paper-dir "toolchain.edn"))

(defn require* [condition message]
  (when-not condition (throw (ex-info message {}))))

(defn sha256 [file]
  (let [digest (MessageDigest/getInstance "SHA-256")]
    (with-open [input (io/input-stream file)]
      (let [buffer (byte-array 8192)]
        (loop []
          (let [n (.read input buffer)]
            (when (pos? n)
              (.update digest buffer 0 n)
              (recur))))))
    (format "%064x" (BigInteger. 1 (.digest digest)))))

(defn verify []
  (let [toolchain (edn/read-string (slurp toolchain-file))
        expected-version (:tectonic/version toolchain)
        actual-version (-> (shell {:out :string} "tectonic" "--version")
                           :out str/trim (str/replace #"^Tectonic\s+" ""))
        result (shell {:dir paper-dir :out :string :err :string :continue true
                       :extra-env {"SOURCE_DATE_EPOCH" (:pdf/source-date-epoch toolchain)}}
                      "tectonic" "--web-bundle" (:tectonic/bundle-url toolchain)
                      "--keep-logs" "--keep-intermediates" "main.tex")
        console (str (:out result) "\n" (:err result))
        source (slurp (io/file paper-dir "main.tex"))
        log-file (io/file paper-dir "main.log")
        bbl-file (io/file paper-dir "main.bbl")
        pdf-file (io/file paper-dir "main.pdf")]
    (require* (= expected-version actual-version)
              (str "Tectonic version mismatch: expected " expected-version
                   ", found " actual-version))
    (require* (zero? (:exit result)) (str "Tectonic failed:\n" console))
    (let [allowed (:tectonic/allowed-console-warning toolchain)
          warning-lines (filter #(str/starts-with? % "warning:")
                                (str/split-lines console))]
      (doseq [line warning-lines]
        (require* (or (str/includes? line allowed)
                      (= line "warning: warnings were issued by the TeX engine; use --print and/or --keep-logs for details."))
                  (str "unexpected Tectonic warning: " line))))
    (let [keywords-index (.indexOf source "\\keywords{")
          abstract-index (.indexOf source "\\begin{abstract}")]
      (require* (and (not (neg? keywords-index))
                     (not (neg? abstract-index))
                     (< keywords-index abstract-index))
                "iacrtrans requires keywords to be set before the abstract"))
    (doseq [[file label] [[log-file "log"] [bbl-file "bibliography"] [pdf-file "PDF"]]]
      (require* (.isFile file) (str "missing generated " label)))
    (let [log (slurp log-file)
          forbidden ["Overfull \\hbox" "Underfull \\hbox" "LaTeX Error"
                     "Undefined control sequence" "There were undefined references"
                     "Citation `" "multiply defined"]]
      (doseq [pattern forbidden]
        (require* (not (str/includes? log pattern))
                  (str "TeX log contains forbidden diagnostic: " pattern))))
    (require* (str/includes? (slurp bbl-file) "\\bibitem")
              "BibTeX output contains no bibliography items")
    (let [bytes (.length pdf-file)
          header (with-open [input (io/input-stream pdf-file)]
                   (let [buffer (byte-array 5)]
                     (.read input buffer)
                     (String. buffer "US-ASCII")))]
      (require* (= "%PDF-" header) "output does not have a PDF header")
      (require* (>= bytes (:pdf/minimum-bytes toolchain)) "PDF is unexpectedly small")
      {:valid true
       :tectonic actual-version
       :bundle (:tectonic/bundle-digest toolchain)
       :allowed-upstream-warning (:tectonic/allowed-console-warning toolchain)
       :pdf-bytes bytes
       :pdf-sha256 (sha256 pdf-file)})))

(try
  (println (verify))
  (catch Exception error
    (binding [*out* *err*]
      (println "IACR PDF check failed:" (ex-message error)))
    (System/exit 1)))
