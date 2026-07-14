#!/usr/bin/env bb
;; Polyglot legitimacy kernel — Babashka/Clojure implementation. See SPEC.md.
;; Signed-long equivalents of the unsigned SplitMix64 constants (wrap mod 2^64).
(def ^:const G  -7046029254386353131)   ;; 0x9e3779b97f4a7c15
(def ^:const M1 -4658895280553007687)   ;; 0xbf58476d1ce4e5b9
(def ^:const M2 -7723592293110705685)   ;; 0x94d049bb133111eb

(defn fin ^long [^long z]
  (let [z (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 30)) M1)
        z (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 27)) M2)]
    (bit-xor z (unsigned-bit-shift-right z 31))))

(defn sm ^long [^long seed ^long k]
  (fin (unchecked-add seed (unchecked-multiply G k))))

(def SEED 1069)
(def N 8)
(def T 16)

(defn orders [rnd agents]
  (let [m    (mod rnd 16)
        amp  (if (<= m 8) (* m 64) (* (- 16 m) 64))
        grid (+ 1000 amp)
        rows (for [i agents
                   :let [h    (sm SEED (unchecked-add (bit-shift-left (long i) 32) (long rnd)))
                         u1   (bit-and (unsigned-bit-shift-right h 16) 0xFF)
                         u2   (bit-and (unsigned-bit-shift-right h 8) 0xFF)
                         u3   (bit-and h 0xFF)
                         load (+ 100 (bit-and (sm SEED (+ 0xA000 i)) 0xFF))
                         rad  (unsigned-bit-shift-right (* amp (+ 200 (* 10 i))) 8)
                         q    (+ (- rad load) (- u1 128))]]
               {:i i :q q
                :ask (unsigned-bit-shift-right (* grid (+ 115 (bit-shift-right u2 1))) 8)
                :bid (unsigned-bit-shift-right (* grid (+ 179 (bit-shift-right u3 1))) 8)})
        sellers (->> rows (filter #(pos? (:q %)))
                     (map (fn [{:keys [ask i q]}] [ask i q]))
                     (sort-by (fn [[a i _]] [a i]))
                     vec)
        buyers  (->> rows (filter #(neg? (:q %)))
                     (map (fn [{:keys [bid i q]}] [bid i (- q)]))
                     (sort-by (fn [[b i _]] [(- b) i]))
                     vec)]
    [sellers buyers]))

(defn clear [sellers buyers]
  (loop [si 0 bi 0
         srem (if (seq sellers) (get-in sellers [0 2]) 0)
         brem (if (seq buyers) (get-in buyers [0 2]) 0)
         fills [] surplus 0]
    (if (and (< si (count sellers)) (< bi (count buyers))
             (>= (get-in buyers [bi 0]) (get-in sellers [si 0])))
      (let [ask   (get-in sellers [si 0])
            bid   (get-in buyers [bi 0])
            take' (min brem srem)
            price (bit-shift-right (+ bid ask) 1)
            fills (conj fills [price take' (get-in buyers [bi 1]) (get-in sellers [si 1])])
            surplus (+ surplus (* (- bid ask) take'))
            brem (- brem take')
            srem (- srem take')
            [bi brem] (if (zero? brem)
                        [(inc bi) (if (< (inc bi) (count buyers)) (get-in buyers [(inc bi) 2]) 0)]
                        [bi brem])
            [si srem] (if (zero? srem)
                        [(inc si) (if (< (inc si) (count sellers)) (get-in sellers [(inc si) 2]) 0)]
                        [si srem])]
        (recur si bi srem brem fills surplus))
      (let [rs (if (< si (count sellers))
                 (into [[(get-in sellers [si 0]) (get-in sellers [si 1]) srem]] (subvec sellers (inc si)))
                 [])
            rb (if (< bi (count buyers))
                 (into [[(get-in buyers [bi 0]) (get-in buyers [bi 1]) brem]] (subvec buyers (inc bi)))
                 [])]
        [fills surplus rs rb]))))

(defn crossing? [rb rs]
  (and (seq rb) (seq rs) (>= (get-in rb [0 0]) (get-in rs [0 0]))))

(loop [r 0 total 0 legit-n 0 wevsum 0]
  (if (< r T)
    (let [[su-fills su _ _] (apply clear (orders r (range 0 N)))
          [_ sa rsa rba] (apply clear (orders r (range 0 4)))
          [_ sb rsb rbb] (apply clear (orders r (range 4 8)))
          wev  (- su (+ sa sb))
          play (if (seq su-fills) 1 0)
          wit  0
          cop  (if (or (crossing? rba rsb) (crossing? rbb rsa)) 0 -1)
          s3   (mod (+ play wit cop 3) 3)
          legit (if (and (= play 1) (= cop -1)) 1 0)
          fp   (reduce (fn [fp [p q b s]]
                         (bit-xor fp (fin (bit-xor (bit-shift-left (long p) 40)
                                                   (bit-shift-left (long q) 20)
                                                   (bit-shift-left (long b) 8)
                                                   (long s)))))
                       0 su-fills)
          fp   (bit-xor fp (fin (bit-xor (bit-shift-left (long wev) 8) (long s3) (bit-shift-left (long legit) 4))))
          total (bit-xor total (fin (bit-xor fp (long r))))
          clearstr (if (seq su-fills) (str (first (peek su-fills))) "-")]
      (println (format "r=%d clear=%s fills=%d legs=%d,%d,%d sum3=%d wev=%d fp=%016x"
                       r clearstr (count su-fills) play wit cop s3 wev fp))
      (recur (inc r) total (+ legit-n legit) (+ wevsum wev)))
    (println (format "TOTAL fp=%016x legit=%d/16 wev=%d" total legit-n wevsum))))
