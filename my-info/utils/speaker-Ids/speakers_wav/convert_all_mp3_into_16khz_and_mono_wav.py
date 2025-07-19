"""
MP3 to WAV File Converter - Convert all MP3 files to 16kHz mono WAV format

USAGE INSTRUCTIONS:
===================

1. PERMISSION SETTING:
   Make the script executable (optional):
   chmod +x convert_all_mp3_into_16khz_and_mono_wav.py

2. PACKAGE INSTALLATION:
   Install required dependencies:
   pip install pydub
   
   Note: pydub may require additional audio libraries:
   - On macOS: brew install ffmpeg
   - On Ubuntu/Debian: sudo apt-get install ffmpeg
   - On Windows: Download ffmpeg and add to PATH

3. USAGE:
   Navigate to the directory containing MP3 files and run:
   python convert_all_mp3_into_16khz_and_mono_wav.py
   
   OR run from any directory:
   cd /path/to/speakers_wav
   python convert_all_mp3_into_16khz_and_mono_wav.py

4. WHAT IT DOES:
   - Finds all .mp3 files in the current directory
   - Converts each MP3 file to 16kHz mono WAV format
   - Replaces original MP3 files with converted WAV files
   - Provides detailed logging of all operations

5. EXAMPLE OUTPUT:
   Found 2 MP3 file(s) to process:
     - speakerId-1.mp3
     - speakerId-2.mp3
   
   Processing files...
   Checking speakerId-1.mp3: 44100Hz, 2 channel(s)
     → Converting to 16kHz mono WAV...
       Converted from 2 channels to mono
       Resampled from 44100Hz to 16000Hz
     ✓ Converted and saved to speakerId-1.wav

6. SAFETY NOTES:
   - This script converts MP3 files to WAV and DELETES original MP3 files
   - Make backups if you need to preserve original MP3 files
   - Converted files will have .wav extension
"""

import os
import glob
from pydub import AudioSegment

def convert_mp3_to_16khz_mono_wav(mp3_path):
    """Convert MP3 file to 16kHz mono WAV and save with .wav extension"""
    try:
        # Load audio file
        audio = AudioSegment.from_mp3(mp3_path)
        
        # Check current properties
        current_sample_rate = audio.frame_rate
        num_channels = audio.channels
        
        print(f"Checking {os.path.basename(mp3_path)}: {current_sample_rate}Hz, {num_channels} channel(s)")
        
        print(f"  → Converting to 16kHz mono WAV...")
        
        # Convert to mono if stereo
        if num_channels > 1:
            audio = audio.set_channels(1)
            print(f"    Converted from {num_channels} channels to mono")
        
        # Resample to 16kHz if needed
        if current_sample_rate != 16000:
            audio = audio.set_frame_rate(16000)
            print(f"    Resampled from {current_sample_rate}Hz to 16000Hz")
        
        # Create WAV filename by replacing .mp3 with .wav
        wav_path = mp3_path.rsplit('.', 1)[0] + '.wav'
        
        # Save as WAV file
        audio.export(wav_path, format="wav")
        print(f"  ✓ Converted and saved to {os.path.basename(wav_path)}")
        
        # Delete original MP3 file
        os.remove(mp3_path)
        print(f"  ✓ Deleted original MP3 file: {os.path.basename(mp3_path)}")
        
    except Exception as e:
        print(f"  ✗ Error converting {os.path.basename(mp3_path)}: {e}")

def main():
    # Get current working directory (where script is being run from)
    current_dir = os.getcwd()
    
    # Find all .mp3 files in the directory
    mp3_files = glob.glob(os.path.join(current_dir, "*.mp3"))
    
    if not mp3_files:
        print("No MP3 files found in the directory")
        return
    
    print(f"Found {len(mp3_files)} MP3 file(s) to process:")
    for mp3_file in mp3_files:
        print(f"  - {os.path.basename(mp3_file)}")
    
    print("\nProcessing files...")
    
    # Process each MP3 file
    for mp3_file in mp3_files:
        convert_mp3_to_16khz_mono_wav(mp3_file)
    
    print(f"\nCompleted processing {len(mp3_files)} files")

if __name__ == "__main__":
    main()
