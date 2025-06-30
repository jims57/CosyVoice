import os
import glob
import re

def combine_pcm_chunks():
    # Get the directory path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pcm_chunks_dir = os.path.join(script_dir, 'pcm_chunks')
    pcm_dir = os.path.join(script_dir, 'pcm')
    
    # Find all chunk_*.pcm files in pcm_chunks directory
    chunk_pattern = os.path.join(pcm_chunks_dir, 'chunk_*.pcm')
    chunk_files = glob.glob(chunk_pattern)
    
    # Sort files by chunk number
    def extract_chunk_number(filename):
        match = re.search(r'chunk_(\d+)\.pcm', os.path.basename(filename))
        return int(match.group(1)) if match else 0
    
    chunk_files.sort(key=extract_chunk_number)
    
    # Combine all PCM files and save to pcm folder
    output_path = os.path.join(pcm_dir, 'pcm_chunks_combined.pcm')
    
    # Create pcm directory if it doesn't exist
    os.makedirs(pcm_dir, exist_ok=True)
    
    with open(output_path, 'wb') as output_file:
        for chunk_file in chunk_files:
            print(f"Adding {os.path.basename(chunk_file)} to combined PCM...")
            with open(chunk_file, 'rb') as input_file:
                output_file.write(input_file.read())
    
    print(f"Combined {len(chunk_files)} PCM chunks into {output_path}")

if __name__ == "__main__":
    combine_pcm_chunks()
