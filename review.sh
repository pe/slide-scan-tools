#!/bin/bash
# Open a review page for the new and rescanned racks.
#   ./review.sh crops|flags|orientation|colour|duplicates [extra options, e.g. --no-browser]
cd "$(dirname "$0")" || exit 1

RACKS=("rack-1" "rack-2" "rack-3")   # folders to review

PY=../.venv/bin/python
what=$1; shift
case $what in
    crops)       $PY review.py "${RACKS[@]}" "$@" ;;                  # port 8765
    flags)       $PY quality.py review "${RACKS[@]}" "$@" ;;          # 8767
    orientation) $PY upright.py review "${RACKS[@]}" "$@" ;;          # 8766
    colour|color) $PY color_fix.py review "${RACKS[@]}" "$@" ;;       # 8768
    duplicates)  $PY dupes.py review "${RACKS[@]}" "$@" ;;            # 8769
    *) echo "usage: $0 crops|flags|orientation|colour|duplicates [options]"; exit 1 ;;
esac
