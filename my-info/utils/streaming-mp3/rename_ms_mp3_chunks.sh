#!/bin/bash

# Get the directory of this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chunks_dir="$script_dir/ms_mp3_chunks"

# Check if chunks directory exists
if [ ! -d "$chunks_dir" ]; then
    echo "Error: Chunks directory $chunks_dir not found!"
    exit 1
fi

# Change to chunks directory
cd "$chunks_dir"

# Check if there are any audio_*.mp3 files
if ! ls audio_*.mp3 >/dev/null 2>&1; then
    echo "No audio_*.mp3 files found in $chunks_dir"
    exit 1
fi

echo "Found audio files in $chunks_dir"

# Create a temporary file to store filename-timestamp pairs
temp_file=$(mktemp)

# Extract timestamps and create sorted list
for file in audio_*.mp3; do
    if [[ $file =~ audio_([0-9]+)\.mp3 ]]; then
        timestamp=${BASH_REMATCH[1]}
        echo "$timestamp:$file" >> "$temp_file"
    fi
done

# Sort by timestamp and rename files
counter=0
while IFS=':' read -r timestamp filename; do
    new_name="chunk_${counter}.mp3"
    
    if [ "$filename" != "$new_name" ]; then
        mv "$filename" "$new_name"
        echo "Renamed: $filename -> $new_name (timestamp: $timestamp)"
    else
        echo "Already named: $new_name (timestamp: $timestamp)"
    fi
    
    ((counter++))
done < <(sort -n "$temp_file")

# Clean up temporary file
rm "$temp_file"

echo ""
echo "✓ Successfully renamed $counter files"
echo "✓ Files are now named: chunk_0.mp3, chunk_1.mp3, ..., chunk_$((counter-1)).mp3"
