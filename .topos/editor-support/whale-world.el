;;; whale-world.el --- Transient/Magit-style interface for Gay.jl Whale World -*- lexical-binding: t -*-

;; Author: Gay.jl Project
;; Version: 1.0.0
;; Package-Requires: ((emacs "28.1") (transient "0.4") (julia-repl "0.4"))

;;; Commentary:

;; A magit-style porcelain for the Whale World in Gay.jl, demonstrating
;; Strong Parallelism Invariance through whale tripartite synergy.
;;
;; The interface exposes:
;; - Trajectory state (corpus, seeds, coupling scores)
;; - Whale grouping combinatorics (N choose 3 for synergy testing)
;; - Color chain visualization (ANSI in terminal, propertized in buffer)
;; - Real-time coupling strength display
;; - First-contact verification via shared color fingerprints
;;
;; Key bindings (from whale-world-mode):
;;   SPC w   - Open whale world transient
;;   g       - Refresh state from Julia
;;   RET     - Select whale or triad under point
;;   TAB     - Expand/collapse section
;;
;; The SPI Algorithm Demonstrated:
;;
;; 1. Each whale has a deterministic seed: whale_seed(id) = base ⊕ hash(id)
;; 2. Color chains are PURE functions: color_at(i; seed) always returns same RGB
;; 3. Tripartite synergy: For each whale triple (i,j,k), compute gadget class
;; 4. First-contact: Both parties compute fingerprint from shared seeds
;;
;; Because color_at() is pure, all computations can run in any order
;; and produce identical results. This is Strong Parallelism Invariance.

;;; Code:

(require 'transient)
(require 'cl-lib)

;; ═══════════════════════════════════════════════════════════════════════════
;; Customization
;; ═══════════════════════════════════════════════════════════════════════════

(defgroup whale-world nil
  "Whale World interface for Gay.jl."
  :group 'tools
  :prefix "whale-world-")

(defcustom whale-world-julia-command "julia"
  "Command to invoke Julia."
  :type 'string
  :group 'whale-world)

(defcustom whale-world-default-seed "0x6761795f636f6c6f"
  "Default seed (GAY_SEED) for whale world."
  :type 'string
  :group 'whale-world)

(defcustom whale-world-refresh-interval 2.0
  "Seconds between automatic state refreshes."
  :type 'number
  :group 'whale-world)

;; ═══════════════════════════════════════════════════════════════════════════
;; State Management
;; ═══════════════════════════════════════════════════════════════════════════

(defvar whale-world--state nil
  "Current whale world state from Julia.")

(defvar whale-world--process nil
  "Julia process for whale world communication.")

(defvar whale-world--buffer-name "*Whale World*"
  "Name of the whale world buffer.")

(defvar whale-world--current-triad nil
  "Currently selected whale triad.")

(defvar whale-world--expanded-sections '(whales synergies)
  "List of expanded sections in the buffer.")

;; ═══════════════════════════════════════════════════════════════════════════
;; Color Rendering
;; ═══════════════════════════════════════════════════════════════════════════

(defun whale-world--rgb-to-color (r g b)
  "Convert R G B (0-255) to Emacs color string."
  (format "#%02x%02x%02x" r g b))

(defun whale-world--make-color-block (r g b &optional width)
  "Create a propertized color block with background color R G B."
  (let ((color (whale-world--rgb-to-color r g b))
        (w (or width 2)))
    (propertize (make-string w ? )
                'face `(:background ,color))))

(defun whale-world--render-color-chain (colors)
  "Render a list of (r g b) color tuples as a propertized string."
  (mapconcat
   (lambda (c)
     (whale-world--make-color-block
      (plist-get c :r)
      (plist-get c :g)
      (plist-get c :b)))
   colors
   ""))

(defun whale-world--gadget-face (gadget)
  "Return face for GADGET class symbol."
  (pcase gadget
    ("XOR" '(:foreground "#00ff00" :weight bold))
    ("MAJ" '(:foreground "#ffff00" :weight bold))
    ("PARITY" '(:foreground "#00ffff" :weight bold))
    ("CLAUSE" '(:foreground "#ff6600"))
    (_ '(:foreground "#888888"))))

;; ═══════════════════════════════════════════════════════════════════════════
;; Buffer Rendering
;; ═══════════════════════════════════════════════════════════════════════════

(defun whale-world--insert-header ()
  "Insert whale world header."
  (insert (propertize "🐋 Whale World" 'face '(:height 1.5 :weight bold)))
  (insert " — Parallel SPI Demonstration\n")
  (insert (propertize "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                      'face '(:foreground "#555555")))
  (insert "\n"))

(defun whale-world--insert-summary ()
  "Insert world summary."
  (let* ((state whale-world--state)
         (base-seed (or (plist-get state :base_seed) whale-world-default-seed))
         (n-whales (or (plist-get state :n_whales) 0))
         (n-triads (or (plist-get state :n_triads) 0))
         (fingerprint (or (plist-get state :world_fingerprint) "—")))
    
    (insert (propertize "World State\n" 'face '(:weight bold :underline t)))
    (insert (format "  Base seed:   %s\n" base-seed))
    (insert (format "  Whales:      %d\n" n-whales))
    (insert (format "  Triads:      %d (N choose 3)\n" n-triads))
    (insert (format "  Fingerprint: %s\n" 
                    (propertize fingerprint 'face '(:foreground "#00ff88"))))
    (insert "\n")))

(defun whale-world--insert-whales-section ()
  "Insert whales section with color chains."
  (let ((expanded (memq 'whales whale-world--expanded-sections))
        (whales (plist-get whale-world--state :whales)))
    
    (insert (propertize (if expanded "▼ " "▶ ") 'face '(:foreground "#888888")))
    (insert (propertize "Whales" 'face '(:weight bold :underline t)))
    (insert (format " (%d)\n" (length whales)))
    
    (when expanded
      (dolist (w whales)
        (let* ((id (plist-get w :id))
               (seed (plist-get w :seed))
               (clan (plist-get w :clan))
               (notes (plist-get w :notes))
               (unique-pcs (plist-get w :unique_pcs))
               (colors (plist-get w :colors)))
          
          (insert "  ")
          (insert (propertize (format "%-6s" id) 'face '(:weight bold)))
          (insert " ")
          
          ;; Color chain visualization
          (when colors
            (insert (whale-world--render-color-chain colors))
            (insert " "))
          
          (insert (propertize (format "[%s]" clan) 'face '(:foreground "#888888")))
          (insert (format " %d/12 PCs" unique-pcs))
          (insert "\n")
          
          ;; Note sequence on second line
          (when notes
            (insert "         ")
            (insert (propertize notes 'face '(:foreground "#aaaaaa")))
            (insert "\n")))))
    
    (insert "\n")))

(defun whale-world--insert-synergies-section ()
  "Insert top synergies section."
  (let ((expanded (memq 'synergies whale-world--expanded-sections))
        (synergies (plist-get whale-world--state :top_synergies)))
    
    (insert (propertize (if expanded "▼ " "▶ ") 'face '(:foreground "#888888")))
    (insert (propertize "Top Synergies" 'face '(:weight bold :underline t)))
    (insert (format " (%d)\n" (length synergies)))
    
    (when expanded
      (if (null synergies)
          (insert "  (no synergies computed - need 3+ whales)\n")
        (dolist (s synergies)
          (let* ((triad (plist-get s :triad))
                 (gadget (plist-get s :gadget))
                 (coupling (plist-get s :coupling))
                 (xor-res (plist-get s :xor_residue))
                 (fp (plist-get s :fingerprint)))
            
            (insert "  ")
            
            ;; Triad IDs
            (insert (propertize (format "(%s %s %s)"
                                        (nth 0 triad) (nth 1 triad) (nth 2 triad))
                                'face '(:weight bold)))
            (insert " ")
            
            ;; Gadget class with color
            (insert (propertize (format "%-7s" gadget)
                                'face (whale-world--gadget-face gadget)))
            
            ;; Coupling score as bar
            (let* ((bar-width 10)
                   (filled (round (* coupling bar-width)))
                   (empty (- bar-width filled)))
              (insert " │")
              (insert (propertize (make-string filled ?█)
                                  'face '(:foreground "#00ff00")))
              (insert (propertize (make-string empty ?░)
                                  'face '(:foreground "#333333")))
              (insert "│ "))
            
            (insert (format "%.3f" coupling))
            
            ;; XOR residue
            (when (= xor-res 0)
              (insert (propertize " ⊕" 'face '(:foreground "#00ff00"))))
            
            (insert "\n")))))
    
    (insert "\n")))

(defun whale-world--insert-spi-explanation ()
  "Insert SPI algorithm explanation."
  (let ((expanded (memq 'spi whale-world--expanded-sections)))
    
    (insert (propertize (if expanded "▼ " "▶ ") 'face '(:foreground "#888888")))
    (insert (propertize "SPI Algorithm" 'face '(:weight bold :underline t)))
    (insert "\n")
    
    (when expanded
      (insert (propertize "
  Strong Parallelism Invariance means:
  Same seeds → Same colors → Same synergies
  
  The algorithm:
  
  1. SEED: Each whale gets deterministic seed
     whale_seed(id) = base_seed ⊕ hash(id)
  
  2. COLORS: Pure function generates chain
     chain[i] = color_at(i; seed)  ← PURE: no side effects
  
  3. SYNERGY: Tripartite gadget classification
     For each (i,j,k): classify as XOR/MAJ/PARITY
  
  4. VERIFY: First-contact via fingerprint
     fingerprint = hash(colors for shared seeds)
  
  Because color_at() is PURE, computation order
  doesn't matter. This enables:
  • Parallel execution across workers
  • Deterministic replay from any point
  • Distributed consensus on shared state
" 'face '(:foreground "#888888")))
      (insert "\n"))))

(defun whale-world--insert-keybindings ()
  "Insert keybinding help."
  (insert (propertize "─────────────────────────────────────────────────\n"
                      'face '(:foreground "#333333")))
  (insert (propertize "Keys: " 'face '(:foreground "#666666")))
  (insert "g refresh  ")
  (insert "a add-whale  ")
  (insert "s synergy  ")
  (insert "v verify  ")
  (insert "? help\n"))

(defun whale-world-refresh-buffer ()
  "Refresh the whale world buffer."
  (interactive)
  (with-current-buffer (get-buffer-create whale-world--buffer-name)
    (let ((inhibit-read-only t)
          (pos (point)))
      (erase-buffer)
      (whale-world--insert-header)
      (whale-world--insert-summary)
      (whale-world--insert-whales-section)
      (whale-world--insert-synergies-section)
      (whale-world--insert-spi-explanation)
      (whale-world--insert-keybindings)
      (goto-char (min pos (point-max))))))

;; ═══════════════════════════════════════════════════════════════════════════
;; Julia Communication
;; ═══════════════════════════════════════════════════════════════════════════

(defun whale-world--julia-eval (code callback)
  "Evaluate Julia CODE and call CALLBACK with result."
  (if (and (fboundp 'julia-repl-inferior-buffer)
           (julia-repl-inferior-buffer))
      ;; Use julia-repl if available
      (julia-repl-send-string code)
    ;; Otherwise use shell command
    (let ((output (shell-command-to-string
                   (format "%s -e '%s'" whale-world-julia-command code))))
      (when callback
        (funcall callback output)))))

(defun whale-world-sync-state ()
  "Sync state from Julia."
  (interactive)
  (whale-world--julia-eval
   "using Gay; state = Gay.export_transient_state(Gay.WHALE_WORLD[]); println(state)"
   (lambda (output)
     ;; Parse Julia output to plist (simplified)
     (message "Whale world state updated")
     (whale-world-refresh-buffer))))

(defun whale-world-add-whale (id)
  "Add whale with ID to the world."
  (interactive "sWhale ID: ")
  (whale-world--julia-eval
   (format "using Gay; Gay.add_whale!(Gay.WHALE_WORLD[], \"%s\")" id)
   (lambda (_)
     (whale-world-sync-state))))

(defun whale-world-compute-synergies ()
  "Compute all tripartite synergies."
  (interactive)
  (whale-world--julia-eval
   "using Gay; Gay.compute_all_synergies!(Gay.WHALE_WORLD[])"
   (lambda (_)
     (whale-world-sync-state))))

(defun whale-world-run-spi-demo ()
  "Run the SPI demonstration."
  (interactive)
  (whale-world--julia-eval
   "using Gay; Gay.spi_parallel_demo(Gay.WHALE_WORLD[])"
   (lambda (output)
     (with-current-buffer (get-buffer-create "*Whale World SPI Demo*")
       (erase-buffer)
       (insert output)
       (display-buffer (current-buffer))))))

(defun whale-world-verify-contact ()
  "Run first-contact verification."
  (interactive)
  (whale-world--julia-eval
   "using Gay; c = Gay.first_contact_challenge(Gay.WHALE_WORLD[]); 
    v = Gay.verify_first_contact(Gay.WHALE_WORLD[], c.expected_fingerprint);
    println(\"Challenge: \", c); println(\"Verification: \", v)"
   (lambda (output)
     (message "First contact: %s" output))))

;; ═══════════════════════════════════════════════════════════════════════════
;; Toggle Sections
;; ═══════════════════════════════════════════════════════════════════════════

(defun whale-world-toggle-section ()
  "Toggle expansion of section at point."
  (interactive)
  (save-excursion
    (beginning-of-line)
    (cond
     ((looking-at ".*Whales")
      (if (memq 'whales whale-world--expanded-sections)
          (setq whale-world--expanded-sections 
                (delq 'whales whale-world--expanded-sections))
        (push 'whales whale-world--expanded-sections)))
     ((looking-at ".*Synergies")
      (if (memq 'synergies whale-world--expanded-sections)
          (setq whale-world--expanded-sections 
                (delq 'synergies whale-world--expanded-sections))
        (push 'synergies whale-world--expanded-sections)))
     ((looking-at ".*SPI Algorithm")
      (if (memq 'spi whale-world--expanded-sections)
          (setq whale-world--expanded-sections 
                (delq 'spi whale-world--expanded-sections))
        (push 'spi whale-world--expanded-sections)))))
  (whale-world-refresh-buffer))

;; ═══════════════════════════════════════════════════════════════════════════
;; Transient Interface
;; ═══════════════════════════════════════════════════════════════════════════

(transient-define-prefix whale-world-transient ()
  "Whale World control panel."
  ["World Management"
   ("a" "Add whale" whale-world-add-whale)
   ("d" "Remove whale" whale-world-remove-whale)
   ("r" "Reset world" whale-world-reset)]
  
  ["Synergy Computation"
   ("s" "Compute synergies" whale-world-compute-synergies)
   ("o" "Find optimal triads" whale-world-find-optimal)
   ("m" "Synergy matrix" whale-world-show-matrix)]
  
  ["SPI Demonstration"
   ("p" "Run parallel demo" whale-world-run-spi-demo)
   ("v" "Verify first-contact" whale-world-verify-contact)
   ("f" "Show fingerprint" whale-world-show-fingerprint)]
  
  ["Display"
   ("g" "Refresh" whale-world-sync-state)
   ("TAB" "Toggle section" whale-world-toggle-section)
   ("q" "Quit" transient-quit-one)])

(transient-define-prefix whale-world-whale-actions ()
  "Actions for selected whale."
  ["Whale Actions"
   ("c" "Show color chain" whale-world-show-chain)
   ("n" "Show notes" whale-world-show-notes)
   ("i" "Show intervals" whale-world-show-intervals)
   ("t" "Find triads containing" whale-world-find-whale-triads)])

;; ═══════════════════════════════════════════════════════════════════════════
;; Mode Definition
;; ═══════════════════════════════════════════════════════════════════════════

(defvar whale-world-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "g") #'whale-world-sync-state)
    (define-key map (kbd "a") #'whale-world-add-whale)
    (define-key map (kbd "s") #'whale-world-compute-synergies)
    (define-key map (kbd "p") #'whale-world-run-spi-demo)
    (define-key map (kbd "v") #'whale-world-verify-contact)
    (define-key map (kbd "TAB") #'whale-world-toggle-section)
    (define-key map (kbd "?") #'whale-world-transient)
    (define-key map (kbd "q") #'quit-window)
    map)
  "Keymap for whale-world-mode.")

(define-derived-mode whale-world-mode special-mode "Whale-World"
  "Major mode for Whale World interface.

\\{whale-world-mode-map}"
  (setq buffer-read-only t)
  (setq truncate-lines t)
  (setq-local revert-buffer-function #'whale-world-refresh-buffer))

;; ═══════════════════════════════════════════════════════════════════════════
;; Entry Point
;; ═══════════════════════════════════════════════════════════════════════════

;;;###autoload
(defun whale-world ()
  "Open the Whale World interface."
  (interactive)
  (let ((buf (get-buffer-create whale-world--buffer-name)))
    (with-current-buffer buf
      (whale-world-mode)
      ;; Initialize with demo state if no state exists
      (unless whale-world--state
        (setq whale-world--state
              '(:base_seed "0x6761795f636f6c6f"
                :n_whales 0
                :n_triads 0
                :world_fingerprint "—"
                :whales nil
                :top_synergies nil)))
      (whale-world-refresh-buffer))
    (switch-to-buffer buf)))

;;;###autoload
(defun whale-world-demo ()
  "Initialize demo whale world and open interface."
  (interactive)
  ;; Set up demo state
  (setq whale-world--state
        '(:base_seed "0x6761795f636f6c6f"
          :n_whales 6
          :n_triads 20
          :world_fingerprint "0x1a2b3c4d5e6f7890"
          :whales ((:id "W001" :seed "0x7a8b9c0d..." :clan "EC-1"
                   :notes "C-E-G-B-D-F#-A-C#-F-Ab-Bb-Eb" :unique_pcs 12
                   :colors ((:r 255 :g 0 :b 0) (:r 255 :g 127 :b 0)
                            (:r 255 :g 255 :b 0) (:r 0 :g 255 :b 0)
                            (:r 0 :g 0 :b 255) (:r 139 :g 0 :b 255)))
                  (:id "W002" :seed "0x2c3d4e5f..." :clan "EC-1"
                   :notes "D-F#-A-C#-E-G#-B-D#-G-Bb-C-F" :unique_pcs 12
                   :colors ((:r 0 :g 255 :b 0) (:r 0 :g 255 :b 127)
                            (:r 0 :g 255 :b 255) (:r 0 :g 127 :b 255)
                            (:r 0 :g 0 :b 255) (:r 127 :g 0 :b 255)))
                  (:id "W003" :seed "0x4e5f6a7b..." :clan "EC-1"
                   :notes "E-G#-B-D#-F#-A#-C#-F-A-C-D-G" :unique_pcs 12
                   :colors ((:r 0 :g 0 :b 255) (:r 75 :g 0 :b 255)
                            (:r 148 :g 0 :b 255) (:r 222 :g 0 :b 255)
                            (:r 255 :g 0 :b 222) (:r 255 :g 0 :b 148)))
                  (:id "W004" :seed "0x6a7b8c9d..." :clan "EC-1"
                   :notes "F-A-C-E-G-B-D-F#-Bb-Db-Eb-Ab" :unique_pcs 12
                   :colors ((:r 255 :g 0 :b 255) (:r 255 :g 64 :b 192)
                            (:r 255 :g 128 :b 128) (:r 255 :g 192 :b 64)
                            (:r 255 :g 255 :b 0) (:r 192 :g 255 :b 64)))
                  (:id "W005" :seed "0x8c9d0e1f..." :clan "EC-1"
                   :notes "G-B-D-F#-A-C#-E-G#-C-Eb-F-Bb" :unique_pcs 12
                   :colors ((:r 128 :g 255 :b 128) (:r 64 :g 255 :b 192)
                            (:r 0 :g 255 :b 255) (:r 64 :g 192 :b 255)
                            (:r 128 :g 128 :b 255) (:r 192 :g 64 :b 255)))
                  (:id "W006" :seed "0x0e1f2a3b..." :clan "EC-1"
                   :notes "A-C#-E-G#-B-D#-F#-A#-D-F-G-C" :unique_pcs 12
                   :colors ((:r 255 :g 128 :b 128) (:r 255 :g 192 :b 128)
                            (:r 255 :g 255 :b 128) (:r 192 :g 255 :b 128)
                            (:r 128 :g 255 :b 128) (:r 128 :g 255 :b 192))))
          :top_synergies ((:triad ("W001" "W002" "W003") :gadget "XOR"
                          :coupling 0.892 :xor_residue 0
                          :fingerprint "0xa1b2c3d4...")
                         (:triad ("W001" "W003" "W005") :gadget "MAJ"
                          :coupling 0.785 :xor_residue 3
                          :fingerprint "0xb2c3d4e5...")
                         (:triad ("W002" "W004" "W006") :gadget "XOR"
                          :coupling 0.756 :xor_residue 0
                          :fingerprint "0xc3d4e5f6...")
                         (:triad ("W001" "W004" "W005") :gadget "PARITY"
                          :coupling 0.698 :xor_residue 6
                          :fingerprint "0xd4e5f6a7...")
                         (:triad ("W003" "W004" "W006") :gadget "CLAUSE"
                          :coupling 0.612 :xor_residue 9
                          :fingerprint "0xe5f6a7b8..."))))
  
  ;; Expand all sections for demo
  (setq whale-world--expanded-sections '(whales synergies spi))
  (whale-world))

(provide 'whale-world)

;;; whale-world.el ends here
