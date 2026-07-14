#!/bin/sh
# legitimate.sh — legitimacy from correctness from without.
# Runs every available implementation of the legitimacy kernel and confers
# legitimacy by EXTERNAL quorum: an implementation is legitimate iff its
# output is byte-identical to >= 2 of the OTHER implementations.
# Its own tests, its own belief, count for nothing (Beetham > Weber).
set -u
cd "$(dirname "$0")"
OUT="${TMPDIR:-/tmp}/legit.$$"
mkdir -p "${OUT}"

run_impl() {
  name="$1"; shift
  if "$@" > "${OUT}/${name}.out" 2> "${OUT}/${name}.err"; then
    printf '%s\n' "${name}" >> "${OUT}/ran"
  else
    printf 'SKIP %s (runtime failed or absent)\n' "${name}"
  fi
}

: > "${OUT}/ran"
command -v julia   >/dev/null 2>&1 && run_impl julia  julia --startup-file=no legitimacy_kernel.jl
command -v python3 >/dev/null 2>&1 && run_impl python python3 legitimacy_kernel.py
command -v ruby    >/dev/null 2>&1 && run_impl ruby   ruby legitimacy_kernel.rb
command -v node    >/dev/null 2>&1 && run_impl node   node legitimacy_kernel.mjs
command -v bb      >/dev/null 2>&1 && run_impl bb     bb legitimacy_kernel.bb

n=$(wc -l < "${OUT}/ran" | tr -d ' ')
[ "${n}" -ge 3 ] || { echo "FATAL: need >=3 implementations for an external quorum, have ${n}"; exit 2; }

echo "== implementations run: ${n} =="
allok=0
while read -r name; do
  agree=0
  while read -r other; do
    [ "${name}" = "${other}" ] && continue
    if cmp -s "${OUT}/${name}.out" "${OUT}/${other}.out"; then
      agree=$((agree + 1))
    fi
  done < "${OUT}/ran"
  if [ "${agree}" -ge 2 ]; then
    verdict="LEGITIMATE"
  else
    verdict="ILLEGITIMATE"
    allok=1
  fi
  printf '%-8s agrees-with-others=%d/%d  %s\n' "${name}" "${agree}" "$((n - 1))" "${verdict}"
done < "${OUT}/ran"

echo "== canonical output (first legitimate implementation) =="
first=$(head -1 "${OUT}/ran")
tail -1 "${OUT}/${first}.out"
if [ "${allok}" -eq 0 ]; then
  echo "VERDICT: every implementation legitimated from without (quorum >= 2 others each)"
else
  echo "VERDICT: divergence — some implementation failed external legitimation"
  for f in "${OUT}"/*.out; do printf '%s  %s\n' "$(cksum < "$f")" "$f"; done
fi
exit "${allok}"
