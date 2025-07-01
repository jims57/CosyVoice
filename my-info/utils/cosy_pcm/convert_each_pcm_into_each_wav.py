import os
import wave
import numpy as np

def convert_pcm_to_wav(pcm_file_path, wav_file_path, sample_rate=8000, channels=1, sample_width=2):
    """
    Convert PCM file to WAV file
    
    Args:
        pcm_file_path: Path to input PCM file
        wav_file_path: Path to output WAV file
        sample_rate: Sample rate in Hz (default: 8000)
        channels: Number of channels (default: 1 for mono)
        sample_width: Sample width in bytes (default: 2 for 16-bit)
    """
    # Read PCM data
    with open(pcm_file_path, 'rb') as pcm_file:
        pcm_data = pcm_file.read()
    
    # Create WAV file
    with wave.open(wav_file_path, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

def main():
    # Define directories
    pcm_chunks_dir = "cosy_pcm_chunks"
    wav_chunks_dir = "cosy_wav_chunks"
    
    # Create output directory if it doesn't exist
    os.makedirs(wav_chunks_dir, exist_ok=True)
    
    # Get all PCM files from the chunks directory
    pcm_files = [f for f in os.listdir(pcm_chunks_dir) if f.endswith('.pcm')]
    
    # Convert each PCM file to WAV
    for pcm_file in pcm_files:
        pcm_file_path = os.path.join(pcm_chunks_dir, pcm_file)
        wav_file_name = pcm_file.replace('.pcm', '.wav')
        wav_file_path = os.path.join(wav_chunks_dir, wav_file_name)
        
        print(f"Converting {pcm_file} to {wav_file_name}")
        convert_pcm_to_wav(pcm_file_path, wav_file_path, sample_rate=8000, channels=1)
        print(f"Conversion complete: {wav_file_path}")

if __name__ == "__main__":
    main()
