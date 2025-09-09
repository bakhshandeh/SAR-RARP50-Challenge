#!/usr/bin/env bash
set -euo pipefail

# Usage: ./prepare_dataset.sh /path/to/dataset
dir="${1:?Usage: $0 /path/to/dataset}"

process_dir() {
  local split_dir="$1"

  if [[ ! -d "$split_dir" ]]; then
    echo "Skip: '$split_dir' does not exist."
    return 0
  fi

  echo "Unzipping ZIPs in: $split_dir"

  # Handle spaces safely with -print0 / read -d ''
  find "$split_dir" -maxdepth 1 -type f -iname '*.zip' -print0 |
    while IFS= read -r -d '' f; do
      echo "→ Processing: $f"
      out="${f%.*}"
      mkdir -p -- "$out"

      # Unzip into the output folder
      unzip -o -d "$out" -- "$f"

      # Run frame extraction
      echo "   Extracting frames in: $out"
      python extract_frames.py "$out"

      # Remove the video file
      echo "   Removing video file (if present)"
      rm -f -- "$out/video_left.avi"
    done

  echo "Done with: $split_dir"
}

echo "Starting..."
process_dir "$dir/test"
process_dir "$dir/train"
echo "All done."
