(rewrite-trace
  (fingerprint "0x8daeed567f6917d4")
  (program-fingerprint "0x36a6ddc85c3c8142")
  (program-execution-fingerprint "0x51b507c10de3c235")
  (request-count 3)
  (execution-count 3)
  (step-count 3)
  (coverage-complete true)
  (all-selected-all-materialized false)
  (all-selected-all-targets false)
  (selected-ordinals 1 2)
  (repeated-ordinals 1)
  (backends projection projection projection)
  (program
    (rewrite-program
      (rewrite-execution
        (query witness 1)
        (max-samples 2)
        (backend projection))
      (rewrite-execution
        (query effect positive-shift)
        (max-samples 1)
        (backend projection))
      (rewrite-execution
        (query ordinal 1)
        (max-samples 1)
        (backend projection))
    )
  )
  (steps
    (step 1
      (backend projection)
      (selected-ordinals 1 2)
      (introduced-ordinals 1 2)
      (selected-all-materialized false)
      (selected-all-targets false)
      (request-fingerprint "0x54a75fc4905b7b6b")
      (plan-fingerprint "0x0dc9db51db198335")
      (execution-fingerprint "0x1b6c2b6a0b54f012")
      (fingerprint "0x3ece7cd9477d51f1")
      (request
        (rewrite-execution
          (query witness 1)
          (max-samples 2)
          (backend projection))
      ))
    (step 2
      (backend projection)
      (selected-ordinals 1)
      (introduced-ordinals)
      (selected-all-materialized false)
      (selected-all-targets false)
      (request-fingerprint "0x50bd432c25252bb4")
      (plan-fingerprint "0x7b13993ec64e049b")
      (execution-fingerprint "0x77f3c1cb010e5e78")
      (fingerprint "0xbfcd0501580ddc68")
      (request
        (rewrite-execution
          (query effect positive-shift)
          (max-samples 1)
          (backend projection))
      ))
    (step 3
      (backend projection)
      (selected-ordinals 1)
      (introduced-ordinals)
      (selected-all-materialized false)
      (selected-all-targets false)
      (request-fingerprint "0xb531c4b50e55c56f")
      (plan-fingerprint "0xaa6c8febfa84171c")
      (execution-fingerprint "0xf6ac6a502bb3d1ba")
      (fingerprint "0x21553f34263e70f9")
      (request
        (rewrite-execution
          (query ordinal 1)
          (max-samples 1)
          (backend projection))
      ))
  )
)
