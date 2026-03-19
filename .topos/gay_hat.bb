#!/usr/bin/env bb
;; just gay hat - aperiodic monotile with SPI
;; After Tatham's combinatorial coordinates

(def GOLDEN (unchecked-long 0x9e3779b97f4a7c15))
(def MASK64 (unchecked-long 0xffffffffffffffff))

(defn u64 [n] (bit-and (unchecked-long n) MASK64))

(defn sm64 [s]
  (let [s (u64 (unchecked-add s GOLDEN))
        z (u64 (unchecked-multiply (bit-xor s (unsigned-bit-shift-right s 30)) (unchecked-long 0xbf58476d1ce4e5b9)))
        z (u64 (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 27)) (unchecked-long 0x94d049bb133111eb)))]
    (u64 (bit-xor z (unsigned-bit-shift-right z 31)))))

(defn rgb [seed]
  [(bit-and (unsigned-bit-shift-right seed 16) 0xFF)
   (bit-and (unsigned-bit-shift-right seed 8) 0xFF)
   (bit-and seed 0xFF)])

(defn show [[r g b]] (format "\u001b[48;2;%d;%d;%dm  \u001b[0m" r g b))

;; 4 metatile types: H T P F (hat paper naming)
(def METATILES [:H :T :P :F])

;; Substitution children per metatile (simplified)
(def SUBS {:H [:H :T :P :F]  :T [:H :H :T]  :P [:H :P :F]  :F [:H :T :F :F]})

;; Combinatorial coordinate: [metatile-type child-path kite-idx]
(defn hat-coord [seed depth]
  (loop [d 0 s seed path [] typ :H]
    (if (>= d depth)
      {:type typ :path path :seed s}
      (let [children (get SUBS typ)
            idx (mod (sm64 s) (count children))
            child (nth children idx)]
        (recur (inc d) (sm64 (+ s idx)) (conj path idx) child)))))

;; Generate n hats at depth d
(defn gen-hats [seed n depth]
  (for [i (range n)]
    (let [s (sm64 (u64 (unchecked-add seed (unchecked-multiply (unchecked-long i) GOLDEN))))
          coord (hat-coord s depth)]
      (assoc coord :color (rgb (sm64 (:seed coord)))))))

(let [seed (or (some-> (first *command-line-args*) parse-long) 69)
      n (or (some-> (second *command-line-args*) parse-long) 12)
      depth 4
      hats (gen-hats seed n depth)
      fp (reduce bit-xor 0 (map #(sm64 (:seed %)) hats))]
  (println)
  (println "gay hat - aperiodic monotile")
  (println (format "seed=%d n=%d depth=%d" seed n depth))
  (println)
  (doseq [h hats]
    (print (format "  %s " (show (:color h))))
    (println (format "%-2s path=%s" (name (:type h)) (:path h))))
  (println)
  (print "  ")
  (doseq [h hats] (print (show (:color h))))
  (println " gay")
  (println (format "\nfp=0x%x" fp)))
