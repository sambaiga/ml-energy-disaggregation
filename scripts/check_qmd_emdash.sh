#!/usr/bin/env bash
# Check that QMD files contain no em-dashes.
# Usage: check_qmd_emdash.sh [files...]
rc=0
for f in "$@"; do
    if grep -Pn "—" "$f"; then
        rc=1
    fi
done
if [ $rc -ne 0 ]; then
    echo "Em-dashes found in QMD files. Replace with a colon or comma."
fi
exit $rc
