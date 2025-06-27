

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

def segment_pcm_and_convert_to_mp3(pcm_file_path, output_dir, segment_duration=3.0, sample_rate=16000, channels=1, bit_depth=16):
    """
    Segment a PCM file into chunks and convert each chunk to MP3
    
    Args:
        pcm_file_path (str): Path to the input PCM file
        output_dir (str): Directory to save MP3 chunks
        segment_duration (float): Duration of each segment in seconds (default: 3.0)
        sample_rate (int): Sample rate in Hz (default: 16000)
        channels (int): Number of channels (default: 1 for mono)
        bit_depth (int): Bit depth (default: 16)
    """
    try:
        # Calculate bytes per second and segment size
        bytes_per_second = sample_rate * channels * (bit_depth // 8)
        segment_bytes = int(bytes_per_second * segment_duration)
        
        print(f"PCM file parameters:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {channels}")
        print(f"  Bit depth: {bit_depth}")
        print(f"  Bytes per second: {bytes_per_second}")
        print(f"  Segment duration: {segment_duration} seconds")
        print(f"  Bytes per segment: {segment_bytes}")
        
        # Read the entire PCM file
        with open(pcm_file_path, 'rb') as pcm_file:
            pcm_data = pcm_file.read()
        
        total_bytes = len(pcm_data)
        total_duration = total_bytes / bytes_per_second
        expected_chunks = (total_bytes + segment_bytes - 1) // segment_bytes  # Ceiling division
        
        print(f"\nFile info:")
        print(f"  Total file size: {total_bytes} bytes")
        print(f"  Total duration: {total_duration:.2f} seconds")
        print(f"  Expected chunks: {expected_chunks}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        chunk_count = 0
        offset = 0
        
        while offset < total_bytes:
            chunk_count += 1
            
            # Calculate end position for this chunk
            end_pos = min(offset + segment_bytes, total_bytes)
            chunk_data = pcm_data[offset:end_pos]
            chunk_size = len(chunk_data)
            chunk_duration = chunk_size / bytes_per_second
            
            print(f"\nProcessing chunk {chunk_count}:")
            print(f"  Offset: {offset} bytes")
            print(f"  Chunk size: {chunk_size} bytes")
            print(f"  Chunk duration: {chunk_duration:.2f} seconds")
            
            # Create temporary PCM file for this chunk
            with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as temp_pcm:
                temp_pcm.write(chunk_data)
                temp_pcm_path = temp_pcm.name
            
            try:
                # Define output MP3 file path
                mp3_filename = f"chunk_{chunk_count}.mp3"
                mp3_path = os.path.join(output_dir, mp3_filename)
                
                # Convert PCM chunk to MP3 using ffmpeg
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output files
                    '-f', 's16le',  # Input format: signed 16-bit little-endian
                    '-ar', str(sample_rate),  # Sample rate
                    '-ac', str(channels),  # Number of channels
                    '-i', temp_pcm_path,  # Input file
                    '-codec:a', 'mp3',  # Audio codec
                    '-b:a', '128k',  # Audio bitrate
                    mp3_path  # Output file
                ]
                
                # Run ffmpeg command
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"  ✓ Successfully created {mp3_filename}")
                else:
                    print(f"  ✗ Error creating {mp3_filename}: {result.stderr}")
                
            finally:
                # Clean up temporary PCM file
                if os.path.exists(temp_pcm_path):
                    os.unlink(temp_pcm_path)
            
            # Move to next chunk
            offset = end_pos
        
        print(f"\n✓ Conversion complete! Created {chunk_count} MP3 chunks in {output_dir}")
        
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
        segment_duration=3.0,  # 3 seconds per segment
        sample_rate=16000,     # 16kHz
        channels=1,            # mono
        bit_depth=16           # 16-bit
    )

if __name__ == "__main__":
    main()
