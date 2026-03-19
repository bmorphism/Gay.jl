#!/usr/bin/env bb
;; gaymc quick verification - babashka edition

(def GOLDEN (unchecked-long 0x9e3779b97f4a7c15))
(def MASK64 (unchecked-long 0xffffffffffffffff))

(defn u64 [n] (bit-and (unchecked-long n) MASK64))

(defn splitmix64 [state]
  (let [s (u64 (unchecked-add state GOLDEN))
        z (u64 (unchecked-multiply (bit-xor s (unsigned-bit-shift-right s 30)) (unchecked-long 0xbf58476d1ce4e5b9)))
        z (u64 (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 27)) (unchecked-long 0x94d049bb133111eb)))]
    [(u64 (bit-xor z (unsigned-bit-shift-right z 31))) s]))

(defn hash-color [seed idx]
  (let [[h _] (splitmix64 (u64 (unchecked-add seed (u64 (unchecked-multiply (unchecked-long idx) GOLDEN)))))]
    [(bit-and (unsigned-bit-shift-right h 16) 0xFF) 
     (bit-and (unsigned-bit-shift-right h 8) 0xFF) 
     (bit-and h 0xFF)]))

(defn show [[r g b]] (format "\u001b[48;2;%d;%d;%dm  \u001b[0m" r g b))

(defn fp [colors]
  (reduce bit-xor 0 (map #(+ (* (% 0) 65536) (* (% 1) 256) (% 2)) colors)))

(let [seed (or (some-> (first *command-line-args*) parse-long) 69)
      n 6
      colors (mapv #(hash-color seed %) (range n))]
  (println)
  (println "gaymc chromatic identity")
  (println (format "seed=%d (0x%x)" seed seed))
  (println)
  (print "  ")
  (doseq [c colors] (print (show c)))
  (println " gay")
  (println)
  (println (format "fingerprint=0x%x" (fp colors)))
  (println)
  (println "forks: Plurigrid | TeglonLabs | Tritwies | bmorphism")
  (doseq [[i fork] (map-indexed vector ["energy" "sheaf" "temporal" "spined"])]
    (let [fork-seed (bit-xor seed (unchecked-long (* (inc i) 0x1111111111111111)))
          fc (mapv #(hash-color fork-seed %) (range 3))]
      (print (format "  %-8s " fork))
      (doseq [c fc] (print (show c)))
      (println (format " fp=0x%x" (fp fc))))))
