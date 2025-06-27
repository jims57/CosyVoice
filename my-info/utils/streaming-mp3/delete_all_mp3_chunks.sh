#!/bin/bash

# Script to delete all MP3 chunk files in the mp3_chunks folder

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MP3_CHUNKS_DIR="$SCRIPT_DIR/mp3_chunks"

echo "Looking for MP3 chunks in: $MP3_CHUNKS_DIR"

# Check if the directory exists
if [ ! -d "$MP3_CHUNKS_DIR" ]; then
    echo "Error: Directory $MP3_CHUNKS_DIR does not exist!"
    exit 1
fi

# Count MP3 files before deletion
MP3_COUNT=$(find "$MP3_CHUNKS_DIR" -name "chunk_*.mp3" | wc -l)

if [ $MP3_COUNT -eq 0 ]; then
    echo "No MP3 chunk files found in $MP3_CHUNKS_DIR"
    exit 0
fi

echo "Found $MP3_COUNT MP3 chunk files to delete"

# List the files that will be deleted
echo "Files to be deleted:"
find "$MP3_CHUNKS_DIR" -name "chunk_*.mp3" -exec basename {} \;

# Delete all MP3 chunk files
find "$MP3_CHUNKS_DIR" -name "chunk_*.mp3" -delete

echo "✓ Successfully deleted $MP3_COUNT MP3 chunk files"
