"""
Convert PCM to MP3 Chunks Utility

This script segments a PCM audio file into frame-aligned chunks and converts each chunk to MP3 format.
This is useful for streaming audio applications where you need to process audio in smaller segments.

Usage:
    python convert_pcm_to_mp3_chunks_by_frame.py

Configuration:
    - total_frame_each_time: Controls how many audio frames to extract each time
      - Smaller values (1000-5000 frames): Better for real-time streaming, lower latency
      - Larger values (10000-50000 frames): Better compression efficiency, larger file sizes
      - Default: 8000 frames (0.5 seconds at 16kHz)

Examples:
    # For low-latency streaming (1000 frames)
    python convert_pcm_to_mp3_chunks_by_frame.py --pcm-file pcm/a_man_die.pcm --output-dir frames_aligned_mp3_chunks --total-frame-each-time 1000 --output-sample-rate 48000
    
    # For better compression (16000 frames - 1 second at 16kHz)
    python convert_pcm_to_mp3_chunks_by_frame.py --pcm-file pcm/a_man_die.pcm --output-dir frames_aligned_mp3_chunks --total-frame-each-time 16000 --output-sample-rate 48000
    
    # For high-quality audio (44.1kHz output)
    python convert_pcm_to_mp3_chunks_by_frame.py --pcm-file pcm/a_man_die.pcm --output-dir frames_aligned_mp3_chunks \
        --total-frame-each-time 8000 --output-sample-rate 44100

Requirements:
    - ffmpeg must be installed and available in PATH
    - Input PCM file should be raw PCM data (no WAV header)
    - Output directory will be created automatically

Output:
    - Creates numbered MP3 chunks: chunk_0.mp3, chunk_1.mp3, etc.
    - Each chunk can be played independently or combined later
    - Use combine_mp3_chunks_into_a_mp3.py to merge chunks back into a single file
"""


import os
import subprocess
import tempfile
import argparse
import glob

def segment_pcm_and_convert_to_mp3(pcm_file_path, output_dir, total_frame_each_time=8000, input_sample_rate=16000, output_sample_rate=48000, channels=1, bit_depth=16):
    """
    Extract exact number of audio frames from PCM and convert each chunk to MP3
    """
    try:
        print(f"PCM input parameters:")
        print(f"  Input sample rate: {input_sample_rate} Hz")
        print(f"  Channels: {channels} (Mono)")
        print(f"  Bit depth: {bit_depth}")
        print(f"")
        print(f"MP3 output parameters (frame-aligned chunks):")
        print(f"  Output sample rate: {output_sample_rate} Hz")
        print(f"  Total frames each time: {total_frame_each_time}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Delete all existing MP3 files in output directory
        mp3_pattern = os.path.join(output_dir, "*.mp3")
        existing_mp3_files = glob.glob(mp3_pattern)
        if existing_mp3_files:
            print(f"\nCleaning up {len(existing_mp3_files)} existing MP3 files...")
            for mp3_file in existing_mp3_files:
                try:
                    os.unlink(mp3_file)
                    print(f"  Deleted: {os.path.basename(mp3_file)}")
                except Exception as e:
                    print(f"  Failed to delete {os.path.basename(mp3_file)}: {e}")
            print("✓ Cleanup completed")
        
        # Calculate bytes per frame (16-bit mono = 2 bytes per frame)
        bytes_per_frame = (bit_depth // 8) * channels
        chunk_size_bytes = total_frame_each_time * bytes_per_frame
        
        print(f"  Bytes per frame: {bytes_per_frame}")
        print(f"  Chunk size: {chunk_size_bytes} bytes ({total_frame_each_time} frames)")
        
        # Read the entire PCM file
        with open(pcm_file_path, 'rb') as f:
            pcm_data = f.read()
        
        total_frames = len(pcm_data) // bytes_per_frame
        total_chunks = (total_frames + total_frame_each_time - 1) // total_frame_each_time  # Ceiling division
        
        print(f"\nInput file analysis:")
        print(f"  Total bytes: {len(pcm_data)}")
        print(f"  Total frames: {total_frames}")
        print(f"  Will create: {total_chunks} chunks")
        
        print(f"\nProcessing chunks...")
        
        chunk_num = 0
        current_pos = 0
        
        while current_pos < len(pcm_data):
            # Calculate chunk boundaries
            chunk_start = current_pos
            chunk_end = min(chunk_start + chunk_size_bytes, len(pcm_data))
            
            # Extract chunk data
            chunk_data = pcm_data[chunk_start:chunk_end]
            actual_frames = len(chunk_data) // bytes_per_frame
            
            if len(chunk_data) > 0:
                # Create temporary PCM file for this chunk
                with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as temp_pcm:
                    temp_pcm.write(chunk_data)
                    temp_pcm_path = temp_pcm.name
                
                try:
                    # Convert chunk to MP3 using ffmpeg
                    chunk_file = os.path.join(output_dir, f"chunk_{chunk_num}.mp3")
                    
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-f', 's16le',  # Input format
            '-ar', str(input_sample_rate),  # Input sample rate
            '-ac', str(channels),  # Input channels
                        '-i', temp_pcm_path,  # Input file
            '-c:a', 'libmp3lame',  # MP3 encoder
            '-b:a', '320k',  # 320 kbps
            '-ar', str(output_sample_rate),  # Resample to target rate
            '-ac', '1',  # Mono output
            '-write_id3v1', '0',  # No ID3v1
            '-write_id3v2', '0',  # No ID3v2
            '-id3v2_version', '0',  # No ID3v2
            '-write_xing', '0',  # No Xing header
            '-fflags', '+bitexact',
                        chunk_file
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
                        print(f"Error converting chunk {chunk_num}: {result.stderr}")
            else:
                        # Get output file size
                        mp3_size = os.path.getsize(chunk_file)
                        duration = actual_frames / input_sample_rate
                
                        print(f"  ✓ Created chunk_{chunk_num}.mp3 ({mp3_size/1024:.1f} KB, {actual_frames} frames, ~{duration:.2f}s)")
                chunk_num += 1
                
                finally:
                    # Clean up temporary PCM file
                    os.unlink(temp_pcm_path)
            
            current_pos = chunk_end
            
        print(f"\n✓ Created {chunk_num} frame-aligned MP3 chunks")
        print("✓ Each chunk contains exactly the specified number of frames")
        
    except Exception as e:
        print(f"Error processing file: {e}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert PCM to MP3 Chunks by Frame",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python convert_pcm_to_mp3_chunks_by_frame.py
  
  # Use specific frame count
  python convert_pcm_to_mp3_chunks_by_frame.py --pcm-file pcm/a_man_die.pcm --total-frame-each-time 16000
  
  # Custom output directory
  python convert_pcm_to_mp3_chunks_by_frame.py --pcm-file pcm/a_man_die.pcm --output-dir my_chunks
        """
    )
    
    parser.add_argument('--pcm-file', 
                       default=None,
                       help='Path to input PCM file (default: pcm/a_man_die_48khz.pcm)')
    
    parser.add_argument('--output-dir', 
                       default='frames_aligned_mp3_chunks',
                       help='Output directory for MP3 chunks (default: frames_aligned_mp3_chunks)')
    
    parser.add_argument('--total-frame-each-time', 
                       type=int, 
                       default=8000,
                       help='Number of audio frames to extract each time (default: 8000)')
    
    parser.add_argument('--input-sample-rate', 
                       type=int, 
                       default=16000,
                       help='Input PCM sample rate in Hz (default: 16000)')
    
    parser.add_argument('--output-sample-rate', 
                       type=int, 
                       default=48000,
                       help='Output MP3 sample rate in Hz (default: 48000)')
    
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
        pcm_file = os.path.join(script_dir, 'pcm', 'a_man_die_48khz.pcm')
    
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
        total_frame_each_time=args.total_frame_each_time,
        input_sample_rate=args.input_sample_rate,
        output_sample_rate=args.output_sample_rate,
        channels=args.channels,
        bit_depth=args.bit_depth
    )

if __name__ == "__main__":
    main()
