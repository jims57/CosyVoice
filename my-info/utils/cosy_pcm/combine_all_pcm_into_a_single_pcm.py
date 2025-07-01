import os
import glob

def combine_pcm_files(pcm_chunks_dir, output_pcm_path):
    """
    Combine all PCM files in order into a single PCM file
    
    Args:
        pcm_chunks_dir: Directory containing PCM chunk files
        output_pcm_path: Path to output combined PCM file
    """
    # Get all PCM files and sort them by filename to maintain order
    pcm_files = glob.glob(os.path.join(pcm_chunks_dir, "*.pcm"))
    pcm_files.sort()  # This will sort chunk_0.pcm, chunk_1.pcm, chunk_2.pcm, etc.
    
    combined_pcm_data = b''
    
    # Read and combine all PCM files in order
    for pcm_file in pcm_files:
        print(f"Reading {os.path.basename(pcm_file)}")
        with open(pcm_file, 'rb') as f:
            pcm_data = f.read()
            combined_pcm_data += pcm_data
    
    # Write combined PCM data to output file
    with open(output_pcm_path, 'wb') as f:
        f.write(combined_pcm_data)

def main():
    # Define directories
    pcm_chunks_dir = "cosy_pcm_chunks"
    combined_pcm_dir = "combined_pcm"
    
    # Create output directory if it doesn't exist
    os.makedirs(combined_pcm_dir, exist_ok=True)
    
    # Define output file path
    output_pcm_path = os.path.join(combined_pcm_dir, "combined_audio.pcm")
    
    print("Combining all PCM chunks into a single PCM file...")
    combine_pcm_files(pcm_chunks_dir, output_pcm_path)
    print(f"Combined PCM file saved: {output_pcm_path}")

if __name__ == "__main__":
    main()
