#!/usr/bin/env bb

(ns verify-higher-order-interactions
  (:require [clojure.edn :as edn]
            [clojure.set :as set]
            [clojure.string :as str]))

(def default-path ".topos/higher-order-interactions.edn")

(def required-laws
  #{:interactions-are-append-only-observations
    :higher-order-targets-exist
    :causality-is-explicit-and-acyclic
    :replay-creates-a-new-interaction
    :capability-names-persist-but-grants-do-not
    :artifact-hashes-do-not-identify-referents
    :colors-annotate-tiles-not-identities
    :adhesion-colors-follow-actual-spans})

(def forbidden-authority-keys
  #{:capability/grant
    :capability/token
    :authorization/token
    :credential
    :credentials
    :secret})

(def genesis-hex
  {1 "#55B0E6"
   2 "#C8A0C2"
   3 "#FFA6C2"
   4 "#789A20"
   5 "#54C1ED"
   6 "#285DD0"
   7 "#6233EF"
   8 "#D4BE57"
   9 "#389BC3"
   10 "#7278C0"
   11 "#5FA42B"
   12 "#C3F7FA"
   13 "#DE1FBE"})

(defn fail! [message]
  (throw (ex-info message {:type ::validation-failure})))

(defn require* [ok? message]
  (when-not ok?
    (fail! message)))

(defn qualified-keyword? [value]
  (and (keyword? value) (some? (namespace value))))

(defn valid-color-tile? [tile]
  (and (= 1069 (:color/seed tile))
       (pos-int? (:color/index tile))
       (string? (:color/hex tile))
       (boolean (re-matches #"#[0-9A-F]{6}" (:color/hex tile)))
       (= (get genesis-hex (:color/index tile)) (:color/hex tile))
       (not (contains? tile :color/identity))))

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
        decomposition (:decomposition document)
        bags (:decomposition/bags decomposition)
        bag-keys (set (map :bag/key bags))
        adhesions (:decomposition/adhesions decomposition)
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
                (str "artifact lacks representation hash: " artifact))
      (require* (valid-color-tile? (:color/tile artifact))
                (str "artifact lacks a valid color tile: " artifact)))
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
        (require* (valid-color-tile? (:color/tile interaction))
                  (str "interaction lacks a valid color tile: " key))
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
    (require* (= :structured/tree (:decomposition/type decomposition))
              "persistence cover must be a structured tree decomposition")
    (require* (= (count bags) (count bag-keys)) "duplicate bag key")
    (require* (= (dec (count bags)) (count adhesions))
              "tree decomposition must have bags - 1 adhesions")
    (doseq [bag bags]
      (require* (every? interaction-keys (:bag/members bag))
                (str "bag contains unknown interaction: " (:bag/key bag)))
      (require* (valid-color-tile? (:color/tile bag))
                (str "bag lacks a valid color tile: " (:bag/key bag))))
    (require* (= interaction-keys (set (mapcat :bag/members bags)))
              "decomposition bags must cover every interaction")
    (doseq [adhesion adhesions]
      (let [left (:adhesion/left adhesion)
            right (:adhesion/right adhesion)
            left-bag (first (filter #(= left (:bag/key %)) bags))
            right-bag (first (filter #(= right (:bag/key %)) bags))]
        (require* (and left-bag right-bag)
                  (str "adhesion endpoint is not a bag: " adhesion))
        (let [actual-span (set/intersection (:bag/members left-bag)
                                            (:bag/members right-bag))]
          (require* (= actual-span (:adhesion/apex adhesion))
                    (str "adhesion apex differs from actual bag overlap: " adhesion))
          (require* (seq actual-span)
                    (str "adhesion must have a nonempty span: " adhesion)))
        (require* (valid-color-tile? (:color/tile adhesion))
                  (str "adhesion lacks a valid color tile: " adhesion))))
    (require* (not (cyclic? causal-edges)) "causal graph contains a cycle")
    (require* (= required-laws (set (keys laws))) "required law set differs")
    (require* (every? #(= :required %) (vals laws))
              "all persistence laws must be required")
    {:valid true
     :artifacts (count artifacts)
     :interactions (count interactions)
     :higher-order (count (filter :interaction/targets interactions))
     :replays (count (filter :replay/of interactions))
     :bags (count bags)
     :adhesions (count adhesions)
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
                   [[:artifact.type/edn "query-input-1"]])

         :color-claims-identity
         (assoc-in document [:interactions 0 :color/tile :color/identity] "query-1")

         :color-does-not-match-seed-index
         (assoc-in document [:interactions 0 :color/tile :color/hex] "#000000")

         :adhesion-uses-wrong-span
         (assoc-in document [:decomposition :decomposition/adhesions 0 :adhesion/apex]
                   #{"query-1"})}
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
