#!/usr/bin/env bb

(ns verify-higher-order-interactions
  (:require [clojure.edn :as edn]
            [clojure.string :as str]))

(def default-path ".topos/higher-order-interactions.edn")

(def required-laws
  #{:interactions-are-append-only-observations
    :higher-order-targets-exist
    :causality-is-explicit-and-acyclic
    :replay-creates-a-new-interaction
    :capability-names-persist-but-grants-do-not
    :artifact-hashes-do-not-identify-referents})

(def forbidden-authority-keys
  #{:capability/grant
    :capability/token
    :authorization/token
    :credential
    :credentials
    :secret})

(defn fail! [message]
  (throw (ex-info message {:type ::validation-failure})))

(defn require* [ok? message]
  (when-not ok?
    (fail! message)))

(defn qualified-keyword? [value]
  (and (keyword? value) (some? (namespace value))))

(defn typed-ref [referent]
  [(:referent/type referent) (:referent/key referent)])

(defn nested-map-entries [value]
  (tree-seq coll? seq value))

(defn read-edn [path]
  (edn/read-string (slurp path)))

(defn boundary-path [document document-path]
  (let [source (:boundary/source document)
        direct (java.io.File. source)
        source-file (java.io.File. source)
        adjacent (java.io.File. (.getParentFile (java.io.File. document-path))
                                (.getName source-file))]
    (cond
      (.exists direct) (.getPath direct)
      (.exists adjacent) (.getPath adjacent)
      :else (fail! (str "missing referent boundary: " source)))))

(defn cyclic? [edges]
  (letfn [(visit [node visiting visited]
            (cond
              (contains? visiting node) true
              (contains? visited node) false
              :else
              (some true?
                    (map #(visit % (conj visiting node) (conj visited node))
                         (get edges node #{})))))]
    (some true? (map #(visit % #{} #{}) (keys edges)))))

(defn verify [document boundary]
  (let [referent-refs (set (map typed-ref (:referents boundary)))
        interface-routes (set (map :interface/route (:interfaces boundary)))
        artifacts (:artifacts document)
        artifact-keys (set (map :artifact/key artifacts))
        interactions (:interactions document)
        interaction-keys (set (map :interaction/key interactions))
        causal-edges (into {} (map (juxt :interaction/key
                                         #(or (:causality/after %) #{}))
                                   interactions))
        used-authority-keys (->> (nested-map-entries document)
                                 (filter map-entry?)
                                 (map key)
                                 (filter forbidden-authority-keys)
                                 set)
        laws (into {} (map (juxt :law/name :law/status) (:laws document)))]
    (require* (= 1 (:persistence/version document))
              "unsupported persistence version")
    (require* (= (count artifacts) (count artifact-keys))
              "duplicate artifact key")
    (require* (= (count interactions) (count interaction-keys))
              "duplicate interaction key")
    (require* (empty? used-authority-keys)
              (str "persisted authority or credential: " used-authority-keys))
    (doseq [artifact artifacts]
      (require* (and (string? (:artifact/key artifact))
                     (not (str/blank? (:artifact/key artifact))))
                (str "invalid artifact key: " artifact))
      (require* (and (string? (:artifact/hash artifact))
                     (str/starts-with? (:artifact/hash artifact) "sha256:"))
                (str "artifact lacks representation hash: " artifact)))
    (doseq [interaction interactions]
      (let [key (:interaction/key interaction)
            targets (or (:interaction/targets interaction) #{})
            after (or (:causality/after interaction) #{})
            replay-of (:replay/of interaction)]
        (require* (and (string? key)
                       (not (str/blank? key))
                       (not (str/includes? key "://")))
                  (str "interaction key must be local and non-URI: " interaction))
        (require* (qualified-keyword? (:interaction/type interaction))
                  (str "interaction type must be qualified: " interaction))
        (require* (contains? interface-routes (:interaction/interface interaction))
                  (str "unknown interface route: " (:interaction/interface interaction)))
        (require* (every? referent-refs (:interaction/participants interaction))
                  (str "unknown typed participant: " key))
        (require* (every? artifact-keys
                          (concat (:interaction/input-artifacts interaction)
                                  (:interaction/output-artifacts interaction)))
                  (str "unknown artifact edge: " key))
        (require* (and (set? (:interaction/requires interaction))
                       (every? qualified-keyword? (:interaction/requires interaction)))
                  (str "capability requirements must be typed names: " key))
        (require* (every? interaction-keys (concat targets after))
                  (str "unknown interaction edge: " key))
        (require* (not (contains? (set (concat targets after)) key))
                  (str "self-referential interaction: " key))
        (when replay-of
          (require* (contains? interaction-keys replay-of)
                    (str "replay target does not exist: " replay-of))
          (require* (not= key replay-of)
                    "replay must create a new interaction")
          (require* (contains? after replay-of)
                    "replay must be causally after its source"))))
    (require* (not (cyclic? causal-edges)) "causal graph contains a cycle")
    (require* (= required-laws (set (keys laws))) "required law set differs")
    (require* (every? #(= :required %) (vals laws))
              "all persistence laws must be required")
    {:valid true
     :artifacts (count artifacts)
     :interactions (count interactions)
     :higher-order (count (filter :interaction/targets interactions))
     :replays (count (filter :replay/of interactions))
     :laws (count laws)}))

(defn rejected? [document boundary]
  (try
    (verify document boundary)
    false
    (catch clojure.lang.ExceptionInfo error
      (if (= ::validation-failure (:type (ex-data error)))
        true
        (throw error)))))

(defn self-test [document boundary]
  (let [mutations
        {:duplicate-interaction
         (update document :interactions conj (first (:interactions document)))

         :dangling-higher-order-target
         (assoc-in document [:interactions 2 :interaction/targets] #{"missing"})

         :causal-cycle
         (assoc-in document [:interactions 0 :causality/after] #{"audit-1"})

         :replay-reuses-source-key
         (assoc-in document [:interactions 1 :interaction/key] "query-1")

         :persisted-capability-grant
         (assoc-in document [:interactions 0 :capability/grant] "ambient")

         :artifact-used-as-participant
         (assoc-in document [:interactions 0 :interaction/participants]
                   [[ :artifact.type/edn "query-input-1"]])}
        outcomes (into {} (map (fn [[name mutation]]
                                 [name (rejected? mutation boundary)])
                               mutations))]
    (require* (every? true? (vals outcomes))
              (str "negative persistence case escaped: " outcomes))
    {:valid true
     :positive (verify document boundary)
     :negative-cases outcomes}))

(defn run! [args]
  (let [self-test? (= "--self-test" (first args))
        path (if self-test?
               (or (second args) default-path)
               (or (first args) default-path))
        document (read-edn path)
        boundary (read-edn (boundary-path document path))]
    (if self-test?
      (self-test document boundary)
      (verify document boundary))))

(try
  (println (pr-str (run! *command-line-args*)))
  (catch clojure.lang.ExceptionInfo error
    (binding [*out* *err*]
      (println (str "FAIL: " (.getMessage error))))
    (System/exit 1)))
