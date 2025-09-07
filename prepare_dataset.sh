#!/usr/bin/env bash
# Usage: ./unzip_each.sh /path/to/folder
dir="${1:?Usage: $0 /path/to/folder}"

echo "Unzip files ..."

find "$dir" -maxdepth 1 -type f -iname '*.zip' -exec sh -c '
for f do
  out="${f%.*}"          # e.g., /path/x/1.zip -> /path/x/1
  mkdir -p "$out"
  unzip -o -d "$out" "$f"

  echo "Extracting frames"
  python extract_frames.py "$out"
  echo "Removing video file"
  rm -rf $out/video_left.avi
done
' sh {} +
