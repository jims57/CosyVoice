import os
import struct
import re

def find_data_chunk_start(wav_data):
    """
    Find the start of the 'data' chunk in WAV file to skip headers
    
    Args:
        wav_data: Raw WAV file bytes
        
    Returns:
        int: Offset where PCM data starts, or -1 if not found
    """
    # Look for 'data' chunk identifier (64617461 in hex)
    data_signature = b'data'
    
    # Start searching after the initial RIFF header
    search_start = 12  # Skip RIFF header
    
    for i in range(search_start, len(wav_data) - 8):
        if wav_data[i:i+4] == data_signature:
            # Found 'data' chunk, next 4 bytes are the data size
            # PCM data starts after these 8 bytes total
            return i + 8
    
    return -1

def extract_pcm_from_wav(wav_file_path):
    """
    Extract raw PCM data from WAV file by removing headers
    
    Args:
        wav_file_path: Path to WAV file
        
    Returns:
        bytes: Raw PCM data without WAV headers
    """
    try:
        with open(wav_file_path, 'rb') as f:
            wav_data = f.read()
        
        # Find where PCM data starts
        pcm_start = find_data_chunk_start(wav_data)
        
        if pcm_start == -1:
            print(f"  ✗ Could not find data chunk in {wav_file_path}")
            return b''
        
        # Extract only the PCM data
        pcm_data = wav_data[pcm_start:]
        
        print(f"  ✓ Extracted {len(pcm_data)} bytes PCM from {os.path.basename(wav_file_path)}")
        return pcm_data
        
    except Exception as e:
        print(f"  ✗ Error processing {wav_file_path}: {e}")
        return b''

def combine_wavs_to_pcm():
    """
    Combine all WAV chunks into a single PCM file
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output directories
    wav_chunks_dir = os.path.join(script_dir, 'wav_chunks')
    pcm_output_dir = os.path.join(script_dir, 'pcm')
    
    # Check if input directory exists
    if not os.path.exists(wav_chunks_dir):
        print(f"Error: WAV chunks directory {wav_chunks_dir} not found!")
        return
    
    # Get all WAV chunk files
    wav_files = []
    for filename in os.listdir(wav_chunks_dir):
        if filename.endswith('.wav') and filename.startswith('chunk_'):
            wav_files.append(filename)
    
    if not wav_files:
        print(f"No WAV chunk files found in {wav_chunks_dir}")
        return
    
    # Sort files by chunk number to ensure correct order
    def extract_chunk_number(filename):
        match = re.search(r'chunk_(\d+)\.wav', filename)
        return int(match.group(1)) if match else 0
    
    wav_files.sort(key=extract_chunk_number)
    
    print(f"Found {len(wav_files)} WAV chunks:")
    for i, filename in enumerate(wav_files, 1):
        file_path = os.path.join(wav_chunks_dir, filename)
        file_size = os.path.getsize(file_path)
        print(f"  {i}. {filename} ({file_size/1024:.1f} KB)")
    
    # Create output directory
    os.makedirs(pcm_output_dir, exist_ok=True)
    
    # Combine all PCM data
    combined_pcm = bytearray()
    
    print(f"\nExtracting PCM data from WAV files...")
    for i, filename in enumerate(wav_files, 1):
        file_path = os.path.join(wav_chunks_dir, filename)
        pcm_data = extract_pcm_from_wav(file_path)
        
        if pcm_data:
            combined_pcm.extend(pcm_data)
    
    # Save combined PCM
    output_file = os.path.join(pcm_output_dir, 'cosy_combined_wavs.pcm')
    
    try:
        with open(output_file, 'wb') as f:
            f.write(combined_pcm)
        
        print(f"\n✓ Successfully combined {len(wav_files)} WAV chunks")
        print(f"✓ Output file: {output_file}")
        print(f"✓ Output size: {len(combined_pcm)/1024:.1f} KB")
        
    except Exception as e:
        print(f"✗ Error saving combined PCM file: {e}")

def main():
    combine_wavs_to_pcm()

if __name__ == "__main__":
    main()
