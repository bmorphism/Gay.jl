#!/usr/bin/env bb
;; QUIC Terminal Bridge - Maximum buffer PTY over QUIC
;; Uses cloudflare/quiche or quinn for 0-RTT connection

(require '[babashka.process :refer [shell process]]
         '[clojure.java.io :as io])

(def config
  {:max-data       (* 64 1024 1024)      ; 64MB connection buffer
   :max-stream     (* 16 1024 1024)      ; 16MB per stream
   :idle-timeout   (* 30 60 1000)        ; 30 min idle
   :initial-rtt    100                    ; 100ms initial RTT estimate
   :port           4433
   :cert           "~/.config/quic/cert.pem"
   :key            "~/.config/quic/key.pem"})

(defn server-cmd [{:keys [port max-data max-stream]}]
  (str "quiche-server"
       " --listen 0.0.0.0:" port
       " --max-data " max-data
       " --max-stream-data " max-stream
       " --cert " (:cert config)
       " --key " (:key config)
       " -- /bin/zsh -i"))

(defn client-cmd [host {:keys [port max-data max-stream]}]
  (str "quiche-client"
       " --max-data " max-data
       " --max-stream-data " max-stream
       " --no-verify"  ; dev only
       " https://" host ":" port))

;; Alternative: Use Rust quinn for better performance
(def quinn-server
  "quinn-server --listen 0.0.0.0:4433 --max-concurrent-bidi 100")

(def quinn-client
  "quinn-client --max-data 67108864")

;; Fastest: netcat over WireGuard (UDP, kernel-level)
(defn wireguard-terminal [peer-ip]
  (println "WireGuard + socat (kernel UDP, maximum speed):")
  (println (str "  Server: socat TCP-LISTEN:5555 EXEC:/bin/zsh,pty,stderr"))
  (println (str "  Client: socat - TCP:" peer-ip ":5555")))

;; QUIC with SPI coloring from Gay.jl
(defn gay-quic-bridge [seed]
  (println "Gay.jl QUIC bridge with chromatic path probes:")
  (println "  julia -e 'using Gay; demo_quic_pathfinding(seed=" seed ")'"))

(defn usage []
  (println "QUIC Terminal Bridge")
  (println "")
  (println "  Server: bb quic_terminal.bb serve")
  (println "  Client: bb quic_terminal.bb connect HOST")
  (println "")
  (println "Fastest options (ranked):")
  (println "  1. WireGuard + socat (kernel UDP)")
  (println "  2. quinn (Rust QUIC, 0-RTT)")
  (println "  3. quiche (Cloudflare QUIC)")
  (println "  4. mosh + tmux (battle-tested)"))

(defn main [args]
  (case (first args)
    "serve"   (do (println "Starting QUIC server...")
                  (println (server-cmd config))
                  (shell (server-cmd config)))
    "connect" (let [host (second args)]
                (println "Connecting to" host)
                (println (client-cmd host config))
                (shell (client-cmd host config)))
    "wg"      (wireguard-terminal (second args))
    "gay"     (gay-quic-bridge (or (second args) "0x42"))
    (usage)))

(main *command-line-args*)
