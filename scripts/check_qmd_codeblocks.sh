#!/usr/bin/env bash
# Check that QMD files do not use {lang eval=false} code block syntax.
# Display-only blocks should use plain ```language fences.
# Usage: check_qmd_codeblocks.sh [files...]
rc=0
for f in "$@"; do
    if grep -Pn '\{[a-z]+ eval=false\}' "$f"; then
        rc=1
    fi
done
if [ $rc -ne 0 ]; then
    echo "Found {lang eval=false} in QMD files. Use plain \`\`\`language fences for display-only blocks."
fi
exit $rc
