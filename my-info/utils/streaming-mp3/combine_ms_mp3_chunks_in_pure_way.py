import os
import re

def combine_mp3_chunks_binary():
    """
    Combine MP3 chunks by directly concatenating their binary data
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input directory
    chunks_dir = os.path.join(script_dir, 'ms_mp3_chunks')
    
    # Check if chunks directory exists
    if not os.path.exists(chunks_dir):
        print(f"Error: Chunks directory {chunks_dir} not found!")
        return
    
    # Get all MP3 files in the chunks directory
    mp3_files = []
    for filename in os.listdir(chunks_dir):
        if filename.endswith('.mp3') and filename.startswith('chunk_'):
            mp3_files.append(filename)
    
    if not mp3_files:
        print(f"No MP3 chunk files found in {chunks_dir}")
        return
    
    # Sort files by chunk number to ensure correct order
    def extract_chunk_number(filename):
        match = re.search(r'chunk_(\d+)\.mp3', filename)
        return int(match.group(1)) if match else 0
    
    mp3_files.sort(key=extract_chunk_number)
    
    print(f"Found {len(mp3_files)} MP3 chunks:")
    for i, filename in enumerate(mp3_files, 1):
        file_path = os.path.join(chunks_dir, filename)
        file_size = os.path.getsize(file_path)
        print(f"  {i}. {filename} ({file_size/1024:.1f} KB)")
    
    # Create buffer to hold all binary data
    combined_buffer = bytearray()
    
    # Read each MP3 file and append to buffer
    print(f"\nCombining MP3 files...")
    for i, filename in enumerate(mp3_files, 1):
        file_path = os.path.join(chunks_dir, filename)
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
                combined_buffer.extend(file_data)
                print(f"  {i}. Added {filename} ({len(file_data)} bytes)")
        except Exception as e:
            print(f"  ✗ Error reading {filename}: {e}")
    
    # Save combined buffer to output file
    output_path = os.path.join(script_dir, 'combined_binary_mp3.mp3')
    
    try:
        with open(output_path, 'wb') as f:
            f.write(combined_buffer)
        
        print(f"\n✓ Successfully combined {len(mp3_files)} chunks")
        print(f"✓ Output file: {output_path}")
        print(f"✓ Output size: {len(combined_buffer)/1024:.1f} KB")
        
    except Exception as e:
        print(f"✗ Error saving combined file: {e}")

def main():
    combine_mp3_chunks_binary()

if __name__ == "__main__":
    main()
