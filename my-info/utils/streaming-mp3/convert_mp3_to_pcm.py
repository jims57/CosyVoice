"""
MP3 to PCM Converter

USAGE INSTRUCTIONS:
==================

1. PERMISSION SETUP:
   - Ensure you have read/write permissions for the script directory
   - If needed, run: chmod +x convert_mp3_to_pcm.py

2. PACKAGE INSTALLATION:
   - Install required Python package:
     pip install pydub
   
   - For macOS users, you may also need ffmpeg:
     brew install ffmpeg
   
   - For Linux users:
     sudo apt-get install ffmpeg
   
   - For Windows users:
     Download ffmpeg from https://ffmpeg.org/download.html and add to PATH

3. FILE PREPARATION:
   - Place your MP3 file named "combined_binary_mp3.mp3" in the same directory as this script
   - The script will automatically create a "pcm" subdirectory for output

4. BASIC USAGE:
   - Run with default settings (16kHz, mono):
     python convert_mp3_to_pcm.py
   
   - Or run directly:
     python3 convert_mp3_to_pcm.py

5. PROGRAMMATIC USAGE:
   - Import and use the function with custom parameters:
     from convert_mp3_to_pcm import convert_mp3_to_pcm
     convert_mp3_to_pcm("input.mp3", "output.pcm", sample_rate=22050, channels=2)

6. PARAMETERS:
   - input_mp3_path: Path to input MP3 file
   - output_pcm_path: Path to output PCM file
   - sample_rate: Target sample rate in Hz (default: 16000)
   - channels: Number of audio channels (1=mono, 2=stereo, default: 1)

7. OUTPUT FORMAT:
   - PCM format: 16-bit signed little-endian
   - File extension: .pcm
   - Raw audio data (no headers)

8. TROUBLESHOOTING:
   - If "ModuleNotFoundError: No module named 'pydub'": Run pip install pydub
   - If "FileNotFoundError" for ffmpeg: Install ffmpeg (see step 2)
   - If "Permission denied": Check file permissions and directory access
   - If input file not found: Ensure "combined_binary_mp3.mp3" exists in script directory

EXAMPLE DIRECTORY STRUCTURE:
============================
streaming-mp3/
├── convert_mp3_to_pcm.py (this script)
├── combined_binary_mp3.mp3 (input file)
└── pcm/
    └── combined_binary_mp3.pcm (output file)
"""

import os
from pydub import AudioSegment

def convert_mp3_to_pcm(input_mp3_path, output_pcm_path, sample_rate=16000, channels=1):
    """
    Convert MP3 file to PCM format with specified sample rate and channels
    
    Args:
        input_mp3_path: Path to input MP3 file
        output_pcm_path: Path to output PCM file
        sample_rate: Target sample rate (default: 16000 Hz)
        channels: Number of channels (default: 1 for mono)
    """
    try:
        # Load the MP3 file
        audio = AudioSegment.from_mp3(input_mp3_path)
        
        # Convert to mono and set sample rate
        audio = audio.set_channels(channels)
        audio = audio.set_frame_rate(sample_rate)
        
        # Export as raw PCM data (16-bit)
        audio.export(output_pcm_path, format="s16le")
        
        print(f"Successfully converted {input_mp3_path} to {output_pcm_path}")
        print(f"Sample rate: {sample_rate} Hz, Channels: {channels}")
        
    except Exception as e:
        print(f"Error converting file: {e}")

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input MP3 file in the same directory
    input_file = os.path.join(script_dir, "combined_binary_mp3.mp3")
    
    # Output PCM file in the pcm subdirectory
    pcm_dir = os.path.join(script_dir, "pcm")
    os.makedirs(pcm_dir, exist_ok=True)
    output_file = os.path.join(pcm_dir, "combined_binary_mp3.pcm")
    
    # Convert the file
    if os.path.exists(input_file):
        convert_mp3_to_pcm(input_file, output_file, sample_rate=16000, channels=1)
    else:
        print(f"Input file not found: {input_file}")
