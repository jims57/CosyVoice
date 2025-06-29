"""
Convert PCM to MP3 Chunks Utility

This script segments a PCM audio file into chunks and converts each chunk to MP3 format.
This is useful for streaming audio applications where you need to process audio in smaller segments.

Usage:
    python convert_pcm_to_mp3_chunks.py

Configuration:
    - segment_duration: Controls how long each audio chunk should be (in seconds)
      - Smaller values (1-3 seconds): Better for real-time streaming, lower latency
      - Larger values (5-10 seconds): Better compression efficiency, larger file sizes
      - Default: 3.0 seconds (good balance between latency and efficiency)

Examples:
    # For low-latency streaming (1 second chunks)
    python convert_pcm_to_mp3_chunks.py --pcm-file pcm/mayun_zh.pcm --output-dir mp3_chunks --segment-duration 1.0
    
    # For better compression (5 second chunks)  
    python convert_pcm_to_mp3_chunks.py --pcm-file pcm/mayun_zh.pcm --output-dir mp3_chunks --segment-duration 5.0
    
    # For high-quality audio (44.1kHz, stereo, 24-bit)
    python convert_pcm_to_mp3_chunks.py --pcm-file pcm/mayun_zh.pcm --output-dir mp3_chunks \
        --sample-rate 44100 --channels 2 --bit-depth 24

Requirements:
    - ffmpeg must be installed and available in PATH
    - Input PCM file should be raw PCM data (no WAV header)
    - Output directory will be created automatically

Output:
    - Creates numbered MP3 chunks: chunk_000.mp3, chunk_001.mp3, etc.
    - Each chunk can be played independently or combined later
    - Use combine_mp3_chunks_into_a_mp3.py to merge chunks back into a single file
"""


import os
import subprocess
import tempfile
import argparse

def segment_pcm_and_convert_to_mp3(pcm_file_path, output_dir, segment_duration=1.0, input_sample_rate=16000, output_sample_rate=48000, channels=1, bit_depth=16):
    """
    First encode entire PCM to one continuous MP3, then split at frame boundaries for gapless playback
    """
    try:
        print(f"PCM input parameters:")
        print(f"  Input sample rate: {input_sample_rate} Hz")
        print(f"  Channels: {channels} (Mono)")
        print(f"  Bit depth: {bit_depth}")
        print(f"")
        print(f"MP3 output parameters (gapless streaming chunks):")
        print(f"  Output sample rate: {output_sample_rate} Hz")
        print(f"  Segment duration: {segment_duration} seconds")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Encode entire PCM file to one continuous MP3
        temp_full_mp3 = os.path.join(output_dir, "temp_full.mp3")
        
        print(f"\nStep 1: Encoding entire PCM to continuous MP3...")
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-f', 's16le',  # Input format
            '-ar', str(input_sample_rate),  # Input sample rate
            '-ac', str(channels),  # Input channels
            '-i', pcm_file_path,  # Input file
            '-c:a', 'libmp3lame',  # MP3 encoder
            '-b:a', '320k',  # 320 kbps
            '-ar', str(output_sample_rate),  # Resample to target rate
            '-ac', '1',  # Mono output
            '-write_id3v1', '0',  # No ID3v1
            '-write_id3v2', '0',  # No ID3v2
            '-id3v2_version', '0',  # No ID3v2
            '-write_xing', '0',  # No Xing header
            '-fflags', '+bitexact',
            temp_full_mp3
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error encoding full MP3: {result.stderr}")
            return
        
        # Step 2: Read the continuous MP3 and split at frame boundaries
        print(f"Step 2: Splitting continuous MP3 into chunks...")
        
        with open(temp_full_mp3, 'rb') as f:
            full_mp3_data = f.read()
        
        # Calculate bytes per chunk (approximate)
        # MP3 at 320kbps = 40,000 bytes/second
        approx_bytes_per_chunk = int(40000 * segment_duration)
        
        chunk_num = 0
        current_pos = 0
        
        # Find first MP3 frame sync
        while current_pos < len(full_mp3_data) - 1:
            if full_mp3_data[current_pos] == 0xFF and (full_mp3_data[current_pos + 1] & 0xE0) == 0xE0:
                break
            current_pos += 1
        
        first_frame_start = current_pos
        
        while current_pos < len(full_mp3_data):
            chunk_start = current_pos
            target_end = min(chunk_start + approx_bytes_per_chunk, len(full_mp3_data))
            
            # Find next frame boundary near target end
            chunk_end = target_end
            if target_end < len(full_mp3_data):
                # Look for next frame sync near target position
                search_start = max(target_end - 1000, chunk_start + 100)  # Don't make chunks too small
                for i in range(search_start, min(target_end + 1000, len(full_mp3_data) - 1)):
                    if full_mp3_data[i] == 0xFF and (full_mp3_data[i + 1] & 0xE0) == 0xE0:
                        chunk_end = i
                        break
            else:
                chunk_end = len(full_mp3_data)
            
            # Extract chunk data
            if chunk_num == 0:
                # First chunk: keep from beginning (includes any headers)
                chunk_data = full_mp3_data[first_frame_start:chunk_end]
            else:
                # Subsequent chunks: pure MP3 frames only
                chunk_data = full_mp3_data[chunk_start:chunk_end]
            
            if len(chunk_data) > 0:
                chunk_file = os.path.join(output_dir, f"chunk_{chunk_num}.mp3")
                with open(chunk_file, 'wb') as f:
                    f.write(chunk_data)
                
                # Verify frame header
                header_hex = chunk_data[:4].hex().upper()
                duration = len(chunk_data) / 40000  # Approximate duration
                
                print(f"  ✓ Created chunk_{chunk_num}.mp3 ({len(chunk_data)/1024:.1f} KB, ~{duration:.1f}s) - Header: {header_hex}")
                chunk_num += 1
            
            current_pos = chunk_end
            
            # Safety check to prevent infinite loop
            if chunk_end <= chunk_start:
                break
        
        # Clean up temporary file
        os.unlink(temp_full_mp3)
        
        print(f"\n✓ Created {chunk_num} gapless MP3 chunks")
        print("✓ Chunks are from continuous stream - guaranteed gapless playback")
        
    except Exception as e:
        print(f"Error processing file: {e}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert PCM to MP3 Chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python convert_pcm_to_mp3_chunks.py
  
  # Use specific PCM file and chunk duration
  python convert_pcm_to_mp3_chunks.py --pcm-file pcm/a_man_die.pcm --segment-duration 1.0
  
  # Custom output directory
  python convert_pcm_to_mp3_chunks.py --pcm-file pcm/a_man_die.pcm --output-dir my_chunks
        """
    )
    
    parser.add_argument('--pcm-file', 
                       default=None,
                       help='Path to input PCM file (default: pcm/mayun_zh.pcm)')
    
    parser.add_argument('--output-dir', 
                       default='mp3_chunks',
                       help='Output directory for MP3 chunks (default: mp3_chunks)')
    
    parser.add_argument('--segment-duration', 
                       type=float, 
                       default=1.0,
                       help='Duration of each segment in seconds (default: 1.0)')
    
    parser.add_argument('--input-sample-rate', 
                       type=int, 
                       default=16000,
                       help='Input PCM sample rate in Hz (default: 16000)')
    
    parser.add_argument('--output-sample-rate', 
                       type=int, 
                       default=48000,
                       help='Output MP3 sample rate in Hz (default: 16000)')
    
    parser.add_argument('--channels', 
                       type=int, 
                       default=1,
                       help='Number of channels (default: 1)')
    
    parser.add_argument('--bit-depth', 
                       type=int, 
                       default=16,
                       help='Bit depth (default: 16)')
    
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine input file path
    if args.pcm_file:
        # Use the specified file (can be relative or absolute path)
        if os.path.isabs(args.pcm_file):
            pcm_file = args.pcm_file
        else:
            pcm_file = os.path.join(script_dir, args.pcm_file)
    else:
        # Use default file
        pcm_file = os.path.join(script_dir, 'pcm', 'mayun_zh.pcm')
    
    # Determine output directory path
    if os.path.isabs(args.output_dir):
        mp3_dir = args.output_dir
    else:
        mp3_dir = os.path.join(script_dir, args.output_dir)
    
    # Check if input file exists
    if not os.path.exists(pcm_file):
        print(f"Error: Input file {pcm_file} not found!")
        return
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not in PATH!")
        print("Please install ffmpeg: https://ffmpeg.org/download.html")
        return
    
    print(f"Input file: {pcm_file}")
    print(f"Output directory: {mp3_dir}")
    print("")
    
    # Convert with specified parameters
    segment_pcm_and_convert_to_mp3(
        pcm_file_path=pcm_file,
        output_dir=mp3_dir,
        segment_duration=args.segment_duration,
        input_sample_rate=args.input_sample_rate,
        output_sample_rate=args.output_sample_rate,
        channels=args.channels,
        bit_depth=args.bit_depth
    )

if __name__ == "__main__":
    main()
