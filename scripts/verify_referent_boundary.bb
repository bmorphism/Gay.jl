#!/usr/bin/env bb

(ns verify-referent-boundary
  (:require [clojure.edn :as edn]
            [clojure.string :as str]))

(def default-path ".topos/referent-boundary.edn")

(def forbidden-identity-keys
  #{:identity
    :uri/identity
    :tile/identity
    :artifact/identity
    :interface/identity
    :adapter/identity})

(def required-laws
  #{:identity-only-on-typed-referents
    :representations-reference-existing-referents
    :artifact-hash-is-not-referent-identity
    :interface-route-is-not-referent-identity
    :web-is-only-a-dns-https-adapter})

(defn fail! [message]
  (binding [*out* *err*]
    (println (str "FAIL: " message)))
  (System/exit 1))

(defn require* [ok? message]
  (when-not ok?
    (fail! message)))

(defn qualified-keyword? [value]
  (and (keyword? value) (some? (namespace value))))

(defn uri [value]
  (try
    (java.net.URI. value)
    (catch Exception _
      (fail! (str "invalid route: " value)))))

(defn referent-ref [referent]
  [(:referent/type referent) (:referent/key referent)])

(defn nested-map-entries [value]
  (tree-seq coll? seq value))

(defn verify-referents [referents]
  (require* (seq referents) "missing typed referents")
  (doseq [referent referents]
    (require* (qualified-keyword? (:referent/type referent))
              (str "referent type must be a qualified keyword: " referent))
    (require* (and (string? (:referent/key referent))
                   (not (str/blank? (:referent/key referent)))
                   (not (str/includes? (:referent/key referent) "://")))
              (str "referent key must be a non-URI local key: " referent)))
  (let [refs (map referent-ref referents)]
    (require* (= (count refs) (count (set refs)))
              "duplicate typed referent")))

(defn verify-representations [document referent-refs]
  (doseq [tile (:tiles document)]
    (require* (contains? referent-refs (:tile/represents tile))
              (str "tile references unknown referent: " tile)))
  (doseq [artifact (:artifacts document)
          reference (:artifact/represents artifact)]
    (require* (contains? referent-refs reference)
              (str "artifact references unknown referent: " reference))))

(defn verify-interfaces [interfaces]
  (doseq [interface interfaces]
    (let [route (uri (:interface/route interface))]
      (require* (= "clojure" (.getScheme route))
                (str "interface must use clojure:// routing: " interface))
      (require* (qualified-keyword? (:interface/operation interface))
                (str "interface operation must be typed: " interface)))))

(defn verify-web-adapters [adapters]
  (doseq [adapter adapters]
    (let [route (uri (:adapter/route adapter))
          inherited (:adapter/inherited adapter)
          https-route (uri (:https/url inherited))]
      (require* (= :web (:adapter/type adapter))
                (str "unsupported adapter type: " adapter))
      (require* (= "web" (.getScheme route))
                (str "web adapter route must use web://: " adapter))
      (require* (= "https" (.getScheme https-route))
                (str "web adapter must inherit HTTPS: " adapter))
      (require* (= (.getHost route) (:dns/name inherited) (.getHost https-route))
                (str "web adapter DNS/HTTPS authorities disagree: " adapter)))))

(defn verify [document]
  (let [referents (:referents document)
        referent-refs (set (map referent-ref referents))
        law-statuses (into {} (map (juxt :law/name :law/status) (:laws document)))
        used-forbidden-keys (->> (nested-map-entries document)
                                 (filter map-entry?)
                                 (map key)
                                 (filter forbidden-identity-keys)
                                 set)]
    (require* (= 1 (:ontology/version document)) "unsupported ontology version")
    (require* (empty? used-forbidden-keys)
              (str "representation layer claims identity: " used-forbidden-keys))
    (verify-referents referents)
    (verify-representations document referent-refs)
    (verify-interfaces (:interfaces document))
    (verify-web-adapters (:adapters document))
    (require* (= required-laws (set (keys law-statuses)))
              "required law set differs")
    (require* (every? #(= :required %) (vals law-statuses))
              "all boundary laws must be required")
    {:valid true
     :referents (count referents)
     :tiles (count (:tiles document))
     :artifacts (count (:artifacts document))
     :interfaces (count (:interfaces document))
     :adapters (count (:adapters document))
     :laws (count law-statuses)}))

(let [path (or (first *command-line-args*) default-path)
      result (verify (edn/read-string (slurp path)))]
  (println (pr-str result)))
