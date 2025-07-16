import os
import wave
import glob
import re

def combine_pcm_to_wav(pcm_chunks_dir, output_wav_path, sample_rate=8000, channels=1, sample_width=2):
    """
    Combine all PCM files in order into a single WAV file
    
    Args:
        pcm_chunks_dir: Directory containing PCM chunk files
        output_wav_path: Path to output combined WAV file
        sample_rate: Sample rate in Hz (default: 8000)
        channels: Number of channels (default: 1 for mono)
        sample_width: Sample width in bytes (default: 2 for 16-bit)
    """
    # Header length constants (same as PcmHeaderExtractor.java)
    HEADER_LENGTH = 12  # 8 bytes startTimeId + 4 bytes messageId
    
    # Get all PCM files and sort them by filename to maintain order
    pcm_files = glob.glob(os.path.join(pcm_chunks_dir, "*.pcm"))
    # Sort numerically by extracting the chunk number
    pcm_files.sort(key=lambda x: int(re.search(r'chunk_(\d+)\.pcm', os.path.basename(x)).group(1)))
    
    combined_pcm_data = b''
    
    # Read and combine all PCM files in order
    for pcm_file in pcm_files:
        print(f"Reading {os.path.basename(pcm_file)}")
        with open(pcm_file, 'rb') as f:
            pcm_data_with_header = f.read()
            
            # Check if file has header (minimum 12 bytes)
            if len(pcm_data_with_header) >= HEADER_LENGTH:
                # Extract pure PCM data by skipping the first 12 bytes (header)
                pure_pcm_data = pcm_data_with_header[HEADER_LENGTH:]
                print(f"  Removed {HEADER_LENGTH} bytes header, pure PCM: {len(pure_pcm_data)} bytes")
                combined_pcm_data += pure_pcm_data
            else:
                print(f"  Warning: File too small to contain header, using as-is: {len(pcm_data_with_header)} bytes")
                combined_pcm_data += pcm_data_with_header
    
    # Create combined WAV file
    with wave.open(output_wav_path, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(combined_pcm_data)

def main():
    # Define directories
    pcm_chunks_dir = "cosy_pcm_chunks_with_header"
    combined_wav_dir = "combined_wav"
    
    # Create output directory if it doesn't exist
    os.makedirs(combined_wav_dir, exist_ok=True)
    
    # Define output file path
    output_wav_path = os.path.join(combined_wav_dir, "combined_audio.wav")
    
    print("Combining all PCM chunks with headers into a single WAV file...")
    combine_pcm_to_wav(pcm_chunks_dir, output_wav_path, sample_rate=6000, channels=1)
    print(f"Combined WAV file saved: {output_wav_path}")

if __name__ == "__main__":
    main()
