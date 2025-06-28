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

def segment_pcm_and_convert_to_mp3(pcm_file_path, output_dir, segment_duration=1.0, sample_rate=16000, channels=1, bit_depth=16):
    """
    Segment a PCM file into chunks and convert each chunk to MP3
    
    Args:
        pcm_file_path (str): Path to the input PCM file
        output_dir (str): Directory to save MP3 chunks
        segment_duration (float): Duration of each segment in seconds (default: 1.0)
        sample_rate (int): Sample rate in Hz (default: 16000)
        channels (int): Number of channels (default: 1 for mono)
        bit_depth (int): Bit depth (default: 16)
    """
    try:
        print(f"PCM file parameters:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {channels}")
        print(f"  Bit depth: {bit_depth}")
        print(f"  Segment duration: {segment_duration} seconds")
        
        # Read the entire PCM file to get total info
        with open(pcm_file_path, 'rb') as pcm_file:
            pcm_data = pcm_file.read()
        
        total_bytes = len(pcm_data)
        bytes_per_second = sample_rate * channels * (bit_depth // 8)
        total_duration = total_bytes / bytes_per_second
        expected_chunks = int(total_duration / segment_duration) + (1 if total_duration % segment_duration > 0 else 0)
        
        print(f"\nFile info:")
        print(f"  Total file size: {total_bytes} bytes")
        print(f"  Total duration: {total_duration:.2f} seconds")
        print(f"  Expected chunks: {expected_chunks}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Use ffmpeg segment muxer for gapless chunks
        output_pattern = os.path.join(output_dir, "chunk_%d.mp3")
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-f', 's16le',  # Input format: signed 16-bit little-endian
            '-ar', str(sample_rate),  # Sample rate
            '-ac', str(channels),  # Number of channels
            '-i', pcm_file_path,  # Input file
            '-f', 'segment',  # Use segment muxer
            '-segment_time', str(segment_duration),  # Segment duration
            '-segment_format', 'mp3',  # Output format
            '-c:a', 'mp3',  # Audio codec
            '-q:a', '4',  # VBR quality (0=best, 9=worst, 4=good for voice)
            '-af', 'lowpass=f=8000',  # Low-pass filter at 8kHz (good for speech)
            '-compression_level', '2',  # LAME compression level (0-9, 2=high quality/small size)
            '-avoid_negative_ts', 'disabled',  # Don't add padding
            '-break_non_keyframes', '1',  # Allow breaking at non-keyframes for precise timing
            output_pattern  # Output pattern
        ]
        
        print(f"\nSegmenting PCM to MP3 chunks...")
        print(f"Output pattern: {output_pattern}")
        
        # Run ffmpeg command
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Count created files
            created_files = []
            chunk_num = 0
            while True:
                chunk_file = os.path.join(output_dir, f"chunk_{chunk_num}.mp3")
                if os.path.exists(chunk_file):
                    created_files.append(chunk_file)
                    file_size = os.path.getsize(chunk_file)
                    print(f"  ✓ Created chunk_{chunk_num}.mp3 ({file_size/1024:.1f} KB)")
                    chunk_num += 1
                else:
                    break
            
            print(f"\n✓ Segmentation complete! Created {len(created_files)} MP3 chunks in {output_dir}")
            print("✓ Chunks are optimized for gapless playback")
            
        else:
            print(f"✗ Error during segmentation: {result.stderr}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output paths
    pcm_dir = os.path.join(script_dir, 'pcm')
    mp3_dir = os.path.join(script_dir, 'mp3_chunks')
    
    # Define file path
    pcm_file = os.path.join(pcm_dir, 'mayun_zh.pcm')
    
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
    
    # Convert the file with specified parameters
    segment_pcm_and_convert_to_mp3(
        pcm_file_path=pcm_file,
        output_dir=mp3_dir,
        segment_duration=1,  # 1 second per segment
        sample_rate=16000,     # 16kHz
        channels=1,            # mono
        bit_depth=16           # 16-bit
    )

if __name__ == "__main__":
    main()
