#!/bin/bash

# Script to delete all MP3 chunk files in the mp3_chunks folder and combined file in mp3_combined folder

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MP3_CHUNKS_DIR="$SCRIPT_DIR/mp3_chunks"
MP3_COMBINED_DIR="$SCRIPT_DIR/mp3_combined"

echo "Looking for MP3 chunks in: $MP3_CHUNKS_DIR"
echo "Looking for combined MP3 in: $MP3_COMBINED_DIR"

# Check if the mp3_chunks directory exists
if [ ! -d "$MP3_CHUNKS_DIR" ]; then
    echo "Error: Directory $MP3_CHUNKS_DIR does not exist!"
    exit 1
fi

# Count MP3 files before deletion
MP3_COUNT=$(find "$MP3_CHUNKS_DIR" -name "chunk_*.mp3" | wc -l)

if [ $MP3_COUNT -eq 0 ]; then
    echo "No MP3 chunk files found in $MP3_CHUNKS_DIR"
else
    echo "Found $MP3_COUNT MP3 chunk files to delete"

    # List the files that will be deleted
    echo "Files to be deleted from mp3_chunks:"
    find "$MP3_CHUNKS_DIR" -name "chunk_*.mp3" -exec basename {} \;

    # Delete all MP3 chunk files
    find "$MP3_CHUNKS_DIR" -name "chunk_*.mp3" -delete

    echo "✓ Successfully deleted $MP3_COUNT MP3 chunk files"
fi

# Check and delete combined MP3 file
if [ -d "$MP3_COMBINED_DIR" ]; then
    COMBINED_FILE="$MP3_COMBINED_DIR/combined_all_chunks.mp3"
    if [ -f "$COMBINED_FILE" ]; then
        echo "Found combined MP3 file to delete: combined_all_chunks.mp3"
        rm "$COMBINED_FILE"
        echo "✓ Successfully deleted combined_all_chunks.mp3"
    else
        echo "No combined_all_chunks.mp3 found in $MP3_COMBINED_DIR"
    fi
else
    echo "Directory $MP3_COMBINED_DIR does not exist"
fi
