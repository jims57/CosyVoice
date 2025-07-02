"""
PCM to WAV Converter

This script converts PCM audio files to WAV format with configurable parameters.

USAGE INSTRUCTIONS:
==================

1. PREREQUISITES:
   - Python 3.6 or higher
   - No additional packages required (uses built-in wave module)

2. PERMISSION SETTING (Linux/Mac):
   chmod +x convert_pcm_to_wav.py

3. DIRECTORY STRUCTURE:
   Create the following structure in the same directory as this script:
   ├── convert_pcm_to_wav.py
   ├── pcm/
   │   └── your_audio_file.pcm
   └── wav/
       └── (output files will be created here)

4. USAGE METHODS:

   Method 1: Direct execution (uses default settings)
   python convert_pcm_to_wav.py

   Method 2: Import and use in another script
   from convert_pcm_to_wav import convert_pcm_to_wav
   
   convert_pcm_to_wav(
       pcm_file_path='path/to/input.pcm',
       wav_file_path='path/to/output.wav',
       sample_rate=16000,    # Hz (default: 16000)
       channels=1,           # 1=mono, 2=stereo (default: 1)
       sample_width=2        # bytes (default: 2 for 16-bit)
   )

5. PARAMETER EXPLANATION:
   - sample_rate: Audio sample rate in Hz (common: 8000, 16000, 44100, 48000)
   - channels: Number of audio channels (1=mono, 2=stereo)
   - sample_width: Bytes per sample (1=8-bit, 2=16-bit, 4=32-bit)

6. EXAMPLE USAGE:
   # Convert with custom parameters
   convert_pcm_to_wav(
       'input.pcm',
       'output.wav',
       sample_rate=44100,  # 44.1kHz
       channels=2,         # Stereo
       sample_width=2      # 16-bit
   )

7. OUTPUT:
   - Creates WAV file in specified location
   - Prints conversion information including file size and duration
   - Handles errors gracefully with informative messages

8. TROUBLESHOOTING:
   - Ensure input PCM file exists and is readable
   - Check that output directory is writable
   - Verify PCM file format matches specified parameters
"""

import wave
import os

def convert_pcm_to_wav(pcm_file_path, wav_file_path, sample_rate=16000, channels=1, sample_width=2):
    """
    Convert a PCM file to WAV format
    
    Args:
        pcm_file_path (str): Path to the input PCM file
        wav_file_path (str): Path to the output WAV file
        sample_rate (int): Sample rate in Hz (default: 16000)
        channels (int): Number of channels (default: 1 for mono)
        sample_width (int): Sample width in bytes (default: 2 for 16-bit)
    """
    try:
        # Read PCM data
        with open(pcm_file_path, 'rb') as pcm_file:
            pcm_data = pcm_file.read()
        
        # Calculate number of frames
        frames = len(pcm_data) // (sample_width * channels)
        
        print(f"PCM file info:")
        print(f"  File size: {len(pcm_data)} bytes")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {channels}")
        print(f"  Sample width: {sample_width} bytes")
        print(f"  Calculated frames: {frames}")
        
        # Create WAV file
        with wave.open(wav_file_path, 'wb') as wav_file:
            # Set WAV parameters
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.setnframes(frames)
            wav_file.setcomptype('NONE', 'not compressed')
            
            # Write PCM data to WAV file
            wav_file.writeframes(pcm_data)
        
        print(f"Successfully converted {pcm_file_path} to {wav_file_path}")
        
        # Calculate duration
        duration = frames / sample_rate
        print(f"Audio duration: {duration:.2f} seconds")
        
    except Exception as e:
        print(f"Error converting file: {e}")

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output paths
    pcm_dir = os.path.join(script_dir, 'pcm')
    wav_dir = os.path.join(script_dir, 'wav')
    
    # Create wav directory if it doesn't exist
    os.makedirs(wav_dir, exist_ok=True)
    
    # Define file paths【Jack ma】
    # pcm_file = os.path.join(pcm_dir, 'mayun_zh.pcm')
    # wav_file = os.path.join(wav_dir, 'mayun_zh_converted_from_pcm.wav')

    # Define file paths【CosyVoice】
    # pcm_file = os.path.join(pcm_dir, 'cosy_combined_wavs.pcm')
    # wav_file = os.path.join(wav_dir, 'cosyvoice_converted_from_pcm.wav')

    # Define file paths【CosyVoice pcm chunks】
    pcm_file = os.path.join(pcm_dir, 'pcm_chunks_combined.pcm')
    wav_file = os.path.join(wav_dir, 'pcm_chunks_combined.wav')
    
    # Check if input file exists
    if not os.path.exists(pcm_file):
        print(f"Error: Input file {pcm_file} not found!")
        return
    
    # Convert the file with specified parameters: 16kHz, mono, 16-bit
    convert_pcm_to_wav(pcm_file, wav_file, sample_rate=8000, channels=1, sample_width=2)

if __name__ == "__main__":
    main()
