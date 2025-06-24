#!/bin/bash

# Convert M4A Audio Files to 16kHz Mono WAV
#
# This script converts all m4a audio files in a directory to 16kHz mono wav files,
# which are suitable for use with CosyVoice and other speech synthesis models.
#
# Usage:
#     1. Install dependencies:
#        # On Ubuntu/Debian:
#        sudo apt update && sudo apt install ffmpeg
#        
#        # On macOS:
#        brew install ffmpeg
#        
#        # On CentOS/RHEL:
#        sudo yum install ffmpeg
#     
#     2. Make script executable:
#        chmod +x convert_m4a_to_16khz_wav.sh
#     
#     3. Run the script:
#        cd /Users/mac/Documents/GitHub/CosyVoice/my-info/utils
#        ./convert_m4a_to_16khz_wav.sh
#     
#     4. The script will:
#        - Find all .m4a files in ../audios directory
#        - Convert each to 16kHz mono format using ffmpeg
#        - Save as .wav files in the same directory
#        - Keep original filenames (only change extension)
#
# Example:
#     Input:  336_1750681637.m4a
#     Output: 336_1750681637.wav (16kHz, mono)
#
# Requirements:
#     - ffmpeg: For audio conversion and processing

# Set the input directory (relative to script location)
AUDIO_DIR="../audios"

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed."
    echo "Please install ffmpeg first:"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    echo "  CentOS/RHEL: sudo yum install ffmpeg"
    exit 1
fi

# Check if audio directory exists
if [ ! -d "$AUDIO_DIR" ]; then
    echo "Error: Directory $AUDIO_DIR not found"
    exit 1
fi

# Find all m4a files
M4A_FILES=($(find "$AUDIO_DIR" -name "*.m4a" -type f))

# Check if any m4a files were found
if [ ${#M4A_FILES[@]} -eq 0 ]; then
    echo "No m4a files found in $AUDIO_DIR"
    exit 0
fi

echo "Found ${#M4A_FILES[@]} m4a files to convert"

# Convert each m4a file
for m4a_file in "${M4A_FILES[@]}"; do
    # Get filename without extension
    filename=$(basename "$m4a_file" .m4a)
    # Create output path
    output_file="${AUDIO_DIR}/${filename}.wav"
    
    echo "Converting: $(basename "$m4a_file")"
    
    # Convert using ffmpeg
    # -i: input file
    # -ar 16000: set sample rate to 16kHz
    # -ac 1: set to mono (1 channel)
    # -y: overwrite output file if it exists
    if ffmpeg -i "$m4a_file" -ar 16000 -ac 1 -y "$output_file" -loglevel quiet; then
        echo "  → Saved: ${filename}.wav"
    else
        echo "  → Error converting $(basename "$m4a_file")"
    fi
done

echo "Conversion completed!"
