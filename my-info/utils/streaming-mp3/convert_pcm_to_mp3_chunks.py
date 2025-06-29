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
    Segment a PCM file into chunks and convert each chunk to pure raw MP3 frames
    
    Args:
        pcm_file_path (str): Path to the input PCM file
        output_dir (str): Directory to save MP3 chunks
        segment_duration (float): Duration of each segment in seconds (default: 1.0)
        input_sample_rate (int): Input PCM sample rate in Hz (default: 16000)
        output_sample_rate (int): Output MP3 sample rate in Hz (default: 48000 for FFF3E4C4)
        channels (int): Number of channels (default: 1 for mono)
        bit_depth (int): Bit depth (default: 16)
    """
    try:
        print(f"PCM input parameters:")
        print(f"  Input sample rate: {input_sample_rate} Hz (16kHz PCM)")
        print(f"  Channels: {channels} (Mono)")
        print(f"  Bit depth: {bit_depth}")
        print(f"")
        print(f"MP3 output parameters (pure raw audio frames):")
        print(f"  Output sample rate: {output_sample_rate} Hz (48kHz for MPEG-1)")
        print(f"  Target: Raw MP3 frames starting with FFF3E4C4")
        print(f"  Segment duration: {segment_duration} seconds")
        
        # Read the entire PCM file to get total info
        with open(pcm_file_path, 'rb') as pcm_file:
            pcm_data = pcm_file.read()
        
        total_bytes = len(pcm_data)
        bytes_per_second = input_sample_rate * channels * (bit_depth // 8)
        total_duration = total_bytes / bytes_per_second
        bytes_per_segment = int(bytes_per_second * segment_duration)
        
        print(f"\nInput file info:")
        print(f"  Total file size: {total_bytes} bytes")
        print(f"  Total duration: {total_duration:.2f} seconds")
        print(f"  Bytes per segment: {bytes_per_segment}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each segment separately to get pure raw frames
        chunk_num = 0
        for start_byte in range(0, total_bytes, bytes_per_segment):
            end_byte = min(start_byte + bytes_per_segment, total_bytes)
            segment_data = pcm_data[start_byte:end_byte]
            
            if len(segment_data) == 0:
                break
            
            # Create temporary PCM file for this segment
            with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as temp_pcm:
                temp_pcm.write(segment_data)
                temp_pcm_path = temp_pcm.name
            
            try:
                # Output file for this chunk
                chunk_file = os.path.join(output_dir, f"chunk_{chunk_num}.mp3")
                
                # Convert segment to raw MP3 frames using data format
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output files
                    '-f', 's16le',  # Input format
                    '-ar', str(input_sample_rate),  # Input sample rate
                    '-ac', str(channels),  # Input channels
                    '-i', temp_pcm_path,  # Input segment
                    '-f', 'data',  # Raw data output format
                    '-c:a', 'libmp3lame',  # MP3 encoder
                    '-b:a', '320k',  # 320 kbps
                    '-ar', str(output_sample_rate),  # Resample to 48kHz
                    '-ac', '1',  # Mono output
                    '-fflags', '+bitexact',  # Reproducible
                    chunk_file  # Output file
                ]
                
                # Try data format first
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    # If data format fails, try mp3 format with aggressive header removal
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output files
                        '-f', 's16le',  # Input format
                        '-ar', str(input_sample_rate),  # Input sample rate
                        '-ac', str(channels),  # Input channels
                        '-i', temp_pcm_path,  # Input segment
                        '-f', 'mp3',  # MP3 format
                        '-c:a', 'libmp3lame',  # MP3 encoder
                        '-b:a', '320k',  # 320 kbps
                        '-ar', str(output_sample_rate),  # Resample to 48kHz
                        '-ac', '1',  # Mono output
                        '-write_id3v1', '0',  # No ID3v1
                        '-write_id3v2', '0',  # No ID3v2
                        '-id3v2_version', '0',  # No ID3v2
                        '-write_xing', '0',  # No Xing header
                        '-fflags', '+bitexact',
                        chunk_file  # Output file
                    ]
                    
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists(chunk_file):
                    # Post-process to remove any remaining headers
                    with open(chunk_file, 'rb') as f:
                        data = f.read()
                    
                    # Find the first MP3 frame sync (FFF3 or FFF2)
                    mp3_start = -1
                    for i in range(len(data) - 1):
                        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
                            mp3_start = i
                            break
                    
                    if mp3_start >= 0:
                        # Extract only the MP3 frames
                        pure_mp3_data = data[mp3_start:]
                        with open(chunk_file, 'wb') as f:
                            f.write(pure_mp3_data)
                        
                        file_size = len(pure_mp3_data)
                        
                        # Verify the header
                        header_hex = pure_mp3_data[:4].hex().upper()
                        if header_hex.startswith('FFF3E4'):
                            sync_status = f"✓ {header_hex} (pure frames)"
                        else:
                            sync_status = f"? {header_hex} (check format)"
                        
                        print(f"  ✓ Created chunk_{chunk_num}.mp3 ({file_size/1024:.1f} KB) - {sync_status}")
                        chunk_num += 1
                    else:
                        print(f"  ✗ No MP3 frames found in chunk_{chunk_num}")
                        os.remove(chunk_file)
                else:
                    print(f"  ✗ Failed to create chunk_{chunk_num}: {result.stderr}")
                    
            finally:
                # Clean up temporary file
                os.unlink(temp_pcm_path)
        
        print(f"\n✓ Created {chunk_num} pure MP3 frame chunks")
        print("✓ Removed all headers - chunks contain only raw MP3 frames")
        print("✓ Ready for binary concatenation")
        
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
