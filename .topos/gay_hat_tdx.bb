#!/usr/bin/env bb
;; gay hat transducer - FST for aperiodic tiling adjacency

(def GOLDEN (unchecked-long 0x9e3779b97f4a7c15))
(def MASK64 (unchecked-long 0xffffffffffffffff))

(defn u64 [n] (bit-and (unchecked-long n) MASK64))
(defn sm64 [s]
  (let [s (u64 (unchecked-add s GOLDEN))
        z (u64 (unchecked-multiply (bit-xor s (unsigned-bit-shift-right s 30)) (unchecked-long 0xbf58476d1ce4e5b9)))
        z (u64 (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 27)) (unchecked-long 0x94d049bb133111eb)))]
    (u64 (bit-xor z (unsigned-bit-shift-right z 31)))))

(defn rgb [s] [(bit-and (unsigned-bit-shift-right s 16) 0xFF) (bit-and (unsigned-bit-shift-right s 8) 0xFF) (bit-and s 0xFF)])
(defn show [[r g b]] (format "\u001b[48;2;%d;%d;%dm  \u001b[0m" r g b))

(defn color-at [typ pos]
  (rgb (sm64 (u64 (unchecked-add (unchecked-long typ) (unchecked-multiply (unchecked-long pos) GOLDEN))))))

(defn tdx-step [typ pos edge]
  (let [next-typ (mod (+ typ edge 1) 4)
        next-pos (mod (+ pos edge) 4)]
    {:type next-typ :pos next-pos :color (color-at next-typ next-pos)}))

(defn run-tdx [seed n]
  (loop [typ (mod seed 4) pos 0 i 0 trace []]
    (if (>= i n)
      trace
      (let [edge (mod (sm64 (u64 (unchecked-add seed (unchecked-multiply (unchecked-long i) GOLDEN)))) 3)
            state {:type typ :pos pos :edge edge :color (color-at typ pos)}
            next (tdx-step typ pos edge)]
        (recur (:type next) (:pos next) (inc i) (conj trace state))))))

(let [seed (or (some-> (first *command-line-args*) parse-long) 69)
      n (or (some-> (second *command-line-args*) parse-long) 8)
      trace (run-tdx seed n)
      fp (reduce bit-xor 0 (map #(sm64 (u64 (+ (:type %) (:pos %)))) trace))]
  (println)
  (println "gay hat tdx - FST adjacency")
  (println (format "seed=%d steps=%d" seed n))
  (println)
  (doseq [[i s] (map-indexed vector trace)]
    (print (format "  %d: %s t=%d p=%d e=%d" i (show (:color s)) (:type s) (:pos s) (:edge s)))
    (println))
  (println)
  (print "  ")
  (doseq [s trace] (print (show (:color s))))
  (println " gay")
  (println (format "\nfp=0x%x" fp)))
