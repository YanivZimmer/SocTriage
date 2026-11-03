#!/usr/bin/env bash
# Download the public GUIDE dataset (Microsoft Security Incident Prediction, Kaggle, dataset version 2) and
# verify it. About 540 MB compressed, 3.5 GB extracted. Requires a Kaggle account and API token: either the
# `kaggle` CLI configured as usual, or KAGGLE_USERNAME / KAGGLE_KEY in the environment for the curl fallback.
# The dataset is not redistributed in this repository; check the dataset's license on Kaggle before use.
#
# Usage: scripts/download_guide.sh [DEST_DIR]        (default: data/guide_raw)
set -euo pipefail
DEST="${1:-data/guide_raw}"
REF="Microsoft/microsoft-security-incident-prediction"
ARCHIVE="$DEST/archive.zip"
mkdir -p "$DEST"
if [ ! -f "$ARCHIVE" ]; then
  if command -v kaggle >/dev/null 2>&1; then
    kaggle datasets download -d "$REF" -p "$DEST" --force
    mv "$DEST/microsoft-security-incident-prediction.zip" "$ARCHIVE"
  else
    : "${KAGGLE_USERNAME:?set KAGGLE_USERNAME or install the kaggle CLI}" "${KAGGLE_KEY:?set KAGGLE_KEY}"
    curl -fL -u "$KAGGLE_USERNAME:$KAGGLE_KEY" -o "$ARCHIVE" \
      "https://www.kaggle.com/api/v1/datasets/download/$REF?datasetVersionNumber=2"
  fi
fi
python3 - "$ARCHIVE" <<'PY'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as fh:
    for b in iter(lambda: fh.read(1 << 22), b""):
        h.update(b)
want = "23ea64ee8f950c77142935064b3e515f3be1e17120bc229233d932ed4a079ed3"
print("archive sha256", h.hexdigest(), "OK" if h.hexdigest() == want else "MISMATCH (expected " + want + ")")
sys.exit(h.hexdigest() != want)
PY
unzip -n -q "$ARCHIVE" GUIDE_Train.csv GUIDE_Test.csv -d "$DEST"
echo "GUIDE files in $DEST:"; ls -l "$DEST"/GUIDE_*.csv
