#!/bin/bash

# Combine WAV Files into Single WAV
#
# This script combines all wav files in the current directory into a single wav file,
# maintaining numerical order (0.wav, 1.wav, 2.wav, etc.).
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
#        chmod +x combine_all_wav_into_a_single_wav.sh
#     
#     3. Run the script:
#        cd /Users/mac/Documents/GitHub/CosyVoice/my-info/utils/combine_wavs
#        ./combine_all_wav_into_a_single_wav.sh
#     
#     4. The script will:
#        - Find all .wav files in current directory
#        - Sort them numerically (0.wav, 1.wav, 2.wav, etc.)
#        - Normalize them to same format (16kHz, mono)
#        - Combine them into 'combined_output.wav'
#
# Example:
#     Input:  0.wav, 1.wav, 2.wav, 3.wav
#     Output: combined_output.wav (concatenated audio)
#
# Requirements:
#     - ffmpeg: For audio processing and concatenation

# Output filename
OUTPUT_FILE="combined_output.wav"

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed."
    echo "Please install ffmpeg first:"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    echo "  CentOS/RHEL: sudo yum install ffmpeg"
    exit 1
fi

# Find all wav files and sort them numerically
WAV_FILES=($(ls *.wav 2>/dev/null | grep -E '^[0-9]+\.wav$' | sort -V))

# Check if any wav files were found
if [ ${#WAV_FILES[@]} -eq 0 ]; then
    echo "No numbered wav files found in current directory"
    exit 0
fi

echo "Found ${#WAV_FILES[@]} wav files to combine:"
for file in "${WAV_FILES[@]}"; do
    echo "  - $file"
done

# Check properties of first file to get target format
echo ""
echo "Checking audio properties..."
FIRST_FILE="${WAV_FILES[0]}"

if command -v ffprobe &> /dev/null; then
    SAMPLE_RATE=$(ffprobe -v quiet -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 "$FIRST_FILE" 2>/dev/null)
    CHANNELS=$(ffprobe -v quiet -select_streams a:0 -show_entries stream=channels -of csv=p=0 "$FIRST_FILE" 2>/dev/null)
    echo "Target format: ${SAMPLE_RATE}Hz, ${CHANNELS} channel(s)"
else
    # Default to 16kHz mono if ffprobe not available
    SAMPLE_RATE="16000"
    CHANNELS="1"
    echo "Using default format: 16kHz, mono"
fi

echo ""
echo "Combining files into $OUTPUT_FILE..."

# Method 1: Try direct concatenation with format normalization
echo "Attempting concatenation with format normalization..."

# Build ffmpeg command with multiple inputs and filter
FFMPEG_CMD="ffmpeg"
FILTER_COMPLEX=""
INPUTS=""

# Add all input files
for i in "${!WAV_FILES[@]}"; do
    FFMPEG_CMD="$FFMPEG_CMD -i \"${WAV_FILES[$i]}\""
    INPUTS="$INPUTS[$i:a]"
done

# Create filter to normalize and concatenate
FILTER_COMPLEX="$INPUTS concat=n=${#WAV_FILES[@]}:v=0:a=1,aresample=$SAMPLE_RATE,pan=mono|c0=0.5*c0+0.5*c1[out]"
FFMPEG_CMD="$FFMPEG_CMD -filter_complex \"$FILTER_COMPLEX\" -map \"[out]\" -y \"$OUTPUT_FILE\""

# Execute the command
if eval $FFMPEG_CMD 2>/dev/null; then
    echo "✅ Successfully created: $OUTPUT_FILE"
else
    echo "Method 1 failed, trying Method 2..."
    
    # Method 2: Convert each file individually then concatenate
    echo "Converting files to common format first..."
    
    TEMP_DIR=$(mktemp -d)
    TEMP_LIST="$TEMP_DIR/filelist.txt"
    
    # Convert each file to normalized format
    for i in "${!WAV_FILES[@]}"; do
        temp_file="$TEMP_DIR/normalized_$i.wav"
        echo "  Converting ${WAV_FILES[$i]}..."
        
        if ffmpeg -i "${WAV_FILES[$i]}" -ar "$SAMPLE_RATE" -ac "$CHANNELS" -y "$temp_file" -loglevel quiet; then
            echo "file '$temp_file'" >> "$TEMP_LIST"
        else
            echo "❌ Error converting ${WAV_FILES[$i]}"
            rm -rf "$TEMP_DIR"
            exit 1
        fi
    done
    
    # Now concatenate the normalized files
    if ffmpeg -f concat -safe 0 -i "$TEMP_LIST" -c copy "$OUTPUT_FILE" -y -loglevel quiet; then
        echo "✅ Successfully created: $OUTPUT_FILE"
    else
        echo "❌ Error: Both methods failed to combine wav files"
        rm -rf "$TEMP_DIR"
        exit 1
    fi
    
    # Clean up temporary files
    rm -rf "$TEMP_DIR"
fi

# Get file information
if [ -f "$OUTPUT_FILE" ]; then
    # Get file size
    if command -v du &> /dev/null; then
        file_size=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "   File size: $file_size"
    fi
    
    # Get duration
    if command -v ffprobe &> /dev/null; then
        duration=$(ffprobe -i "$OUTPUT_FILE" -show_entries format=duration -v quiet -of csv="p=0" 2>/dev/null)
        if [ ! -z "$duration" ] && [ "$duration" != "" ]; then
            duration_rounded=$(echo "$duration" | cut -d'.' -f1)
            echo "   Duration: ${duration_rounded}s"
        fi
    fi
fi

echo ""
echo "Combination completed!"
echo "Output: $OUTPUT_FILE"
