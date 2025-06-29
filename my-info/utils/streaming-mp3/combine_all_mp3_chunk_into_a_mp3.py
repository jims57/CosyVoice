"""
MP3 Chunks Combiner

USAGE INSTRUCTIONS:
==================

1. PERMISSION SETUP:
   - Ensure you have read/write permissions for the script directory
   - If needed, run: chmod +x combine_all_mp3_chunk_into_a_mp3.py

2. PACKAGE INSTALLATION:
   - This script uses only Python standard library modules (os, re, argparse)
   - No additional packages required

3. FILE PREPARATION:
   - Ensure your MP3 chunk files are named in format: chunk_000.mp3, chunk_001.mp3, etc.
   - Chunks should be in a dedicated folder (default: mp3_chunks)
   - The script will automatically sort chunks by number

4. BASIC USAGE:
   - Combine chunks from default mp3_chunks folder:
     python combine_all_mp3_chunk_into_a_mp3.py
   
   - Combine chunks from custom folder:
     python combine_all_mp3_chunk_into_a_mp3.py --chunks-dir my_custom_chunks
   
   - Combine chunks and save to custom output directory:
     python combine_all_mp3_chunk_into_a_mp3.py --chunks-dir mp3_chunks --output-dir my_output

5. COMMAND LINE PARAMETERS:
   - --chunks-dir: Directory containing MP3 chunk files (default: mp3_chunks)
   - --output-dir: Directory to save combined MP3 file (default: mp3_combined)
   - --output-name: Name of the output MP3 file (default: combined_all_chunks.mp3)

6. EXAMPLES:
   # Basic usage with default settings
   python combine_all_mp3_chunk_into_a_mp3.py
   
   # Use custom chunks directory
   python combine_all_mp3_chunk_into_a_mp3.py --chunks-dir mp3_chunks
   
   # Custom input and output directories
   python combine_all_mp3_chunk_into_a_mp3.py --chunks-dir mp3_chunks --output-dir final_audio
   
   # Custom output filename
   python combine_all_mp3_chunk_into_a_mp3.py --output-name final_audio.mp3

   # Custom input and output
   python combine_all_mp3_chunk_into_a_mp3.py --chunks-dir mp3_chunks --output-dir final_audio --output-name my_audio.mp3

7. OUTPUT:
   - Creates a single MP3 file by binary concatenation
   - Preserves audio quality and format of original chunks
   - File is saved in the specified output directory

8. TROUBLESHOOTING:
   - If "No MP3 chunk files found": Check if chunks directory exists and contains chunk_*.mp3 files
   - If "Permission denied": Check file permissions and directory access
   - If chunks are out of order: Ensure files are named chunk_000.mp3, chunk_001.mp3, etc.

EXAMPLE DIRECTORY STRUCTURE:
============================
streaming-mp3/
├── combine_all_mp3_chunk_into_a_mp3.py (this script)
├── mp3_chunks/
│   ├── chunk_000.mp3
│   ├── chunk_001.mp3
│   └── chunk_002.mp3
└── mp3_combined/
    └── combined_all_chunks.mp3 (output)
"""

import os
import re
import argparse

def combine_mp3_chunks_binary(chunks_dir, output_dir, output_name):
    """
    Combine all MP3 chunks in specified folder by binary concatenation
    
    Args:
        chunks_dir: Directory containing MP3 chunk files
        output_dir: Directory to save combined MP3 file
        output_name: Name of the output MP3 file
    """
    # Check if chunks directory exists
    if not os.path.exists(chunks_dir):
        print(f"Error: Chunks directory {chunks_dir} not found!")
        return False
    
    # Get all MP3 files in the chunks directory
    mp3_files = []
    for filename in os.listdir(chunks_dir):
        if filename.endswith('.mp3') and filename.startswith('chunk_'):
            mp3_files.append(filename)
    
    if not mp3_files:
        print(f"No MP3 chunk files found in {chunks_dir}")
        return False
    
    # Sort files by chunk number to ensure correct order
    def extract_chunk_number(filename):
        match = re.search(r'chunk_(\d+)\.mp3', filename)
        return int(match.group(1)) if match else 0
    
    mp3_files.sort(key=extract_chunk_number)
    
    print(f"Found {len(mp3_files)} MP3 chunks in {chunks_dir}:")
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
            return False
    
    # Create output directory and save combined buffer
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    
    try:
        with open(output_path, 'wb') as f:
            f.write(combined_buffer)
        
        print(f"\n✓ Successfully combined {len(mp3_files)} chunks")
        print(f"✓ Output file: {output_path}")
        print(f"✓ Output size: {len(combined_buffer)/1024:.1f} KB")
        return True
        
    except Exception as e:
        print(f"✗ Error saving combined file: {e}")
        return False

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Combine MP3 chunks into a single MP3 file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python combine_all_mp3_chunk_into_a_mp3.py
  
  # Use custom chunks directory
  python combine_all_mp3_chunk_into_a_mp3.py --chunks-dir mp3_chunks
  
  # Custom input and output directories
  python combine_all_mp3_chunk_into_a_mp3.py --chunks-dir mp3_chunks --output-dir final_audio
        """
    )
    
    parser.add_argument('--chunks-dir', 
                       default='mp3_chunks',
                       help='Directory containing MP3 chunk files (default: mp3_chunks)')
    
    parser.add_argument('--output-dir', 
                       default='mp3_combined',
                       help='Output directory for combined MP3 file (default: mp3_combined)')
    
    parser.add_argument('--output-name', 
                       default='combined_all_chunks.mp3',
                       help='Name of the output MP3 file (default: combined_all_chunks.mp3)')
    
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine input chunks directory path
    if os.path.isabs(args.chunks_dir):
        chunks_dir = args.chunks_dir
    else:
        chunks_dir = os.path.join(script_dir, args.chunks_dir)
    
    # Determine output directory path
    if os.path.isabs(args.output_dir):
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(script_dir, args.output_dir)
    
    print(f"Input chunks directory: {chunks_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output filename: {args.output_name}")
    print("")
    
    # Combine the chunks
    combine_mp3_chunks_binary(chunks_dir, output_dir, args.output_name)

if __name__ == "__main__":
    main()
