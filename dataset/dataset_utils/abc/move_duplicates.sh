#!/bin/bash
declare -A arr
shopt -s globstar

mkdir ../duplicates

for file in **; do
  [[ -f "$file" ]] || continue

  read cksm _ < <(md5sum "$file")
  if ((arr[$cksm]++)); then
    mv $file ../duplicates/
  fi
done