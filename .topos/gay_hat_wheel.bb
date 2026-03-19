#!/usr/bin/env bb
;; gay hat wheel - find repeating cycles in tiling transducer

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
(defn color-at [typ pos] (rgb (sm64 (u64 (unchecked-add (unchecked-long typ) (unchecked-multiply (unchecked-long pos) GOLDEN))))))

(def TYPES ["H" "T" "P" "F"])

(defn find-wheel [seed max-n]
  (loop [typ (mod seed 4) pos 0 i 0 seen {} trace []]
    (let [state [typ pos]
          color (color-at typ pos)]
      (cond
        (>= i max-n) {:wheel nil :trace trace}
        (contains? seen state) 
          {:wheel state :period (- i (get seen state)) :start (get seen state) :trace trace}
        :else 
          (let [edge (mod (sm64 (u64 (unchecked-add seed (unchecked-multiply (unchecked-long i) GOLDEN)))) 3)
                next-typ (mod (+ typ edge 1) 4)
                next-pos (mod (+ pos edge) 4)]
            (recur next-typ next-pos (inc i) 
                   (assoc seen state i)
                   (conj trace {:typ typ :pos pos :color color :edge edge})))))))

(let [seed (or (some-> (first *command-line-args*) parse-long) 69)
      {:keys [wheel period start trace]} (find-wheel seed 100)]
  (println)
  (println "gay hat wheel - repeating cycle finder")
  (println (format "seed=%d" seed))
  (println)
  (if wheel
    (do
      (println (format "🔄 WHEEL found at step %d" start))
      (println (format "   state=[%s,%d] period=%d" (nth TYPES (first wheel)) (second wheel) period))
      (println)
      (println "trace to wheel:")
      (doseq [[i s] (map-indexed vector (take (+ start period 1) trace))]
        (let [marker (cond (= i start) "→" (> i start) "○" :else " ")]
          (print (format "  %s%2d: %s %s p=%d e=%d" marker i (show (:color s)) (nth TYPES (:typ s)) (:pos s) (:edge s)))
          (println)))
      (println)
      (println "wheel cycle:")
      (print "  ")
      (doseq [s (subvec trace start (+ start period))]
        (print (show (:color s))))
      (println (format " (period %d)" period)))
    (println "no wheel found"))
  (println)
  (print "full: ")
  (doseq [s (take 16 trace)] (print (show (:color s))))
  (println " gay"))
