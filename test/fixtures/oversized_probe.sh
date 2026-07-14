#!/bin/sh
printf 'raw-stderr-sentinel\n' >&2
i=0
while [ "$i" -lt 1100 ]; do
    printf x
    i=$((i + 1))
done
printf '\n'
