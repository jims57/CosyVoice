import os
import subprocess

def convert_pcm_to_mp3(pcm_file_path, mp3_file_path, sample_rate=16000, channels=1, bit_depth=16, bitrate='128k'):
    """
    Convert a PCM file to MP3 format
    
    Args:
        pcm_file_path (str): Path to the input PCM file
        mp3_file_path (str): Path to the output MP3 file
        sample_rate (int): Sample rate in Hz (default: 16000)
        channels (int): Number of channels (default: 1 for mono)
        bit_depth (int): Bit depth (default: 16)
        bitrate (str): MP3 bitrate (default: '128k')
    """
    try:
        print(f"Converting PCM to MP3:")
        print(f"  Input: {pcm_file_path}")
        print(f"  Output: {mp3_file_path}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {channels}")
        print(f"  Bit depth: {bit_depth}")
        print(f"  MP3 bitrate: {bitrate}")
        
        # Check if input file exists
        if not os.path.exists(pcm_file_path):
            print(f"Error: Input file {pcm_file_path} not found!")
            return False
        
        # Get input file size
        input_size = os.path.getsize(pcm_file_path)
        bytes_per_second = sample_rate * channels * (bit_depth // 8)
        duration = input_size / bytes_per_second
        
        print(f"  Input file size: {input_size} bytes")
        print(f"  Estimated duration: {duration:.2f} seconds")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(mp3_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert PCM to MP3 using ffmpeg
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 's16le',  # Input format: signed 16-bit little-endian
            '-ar', str(sample_rate),  # Sample rate
            '-ac', str(channels),  # Number of channels
            '-i', pcm_file_path,  # Input file
            '-codec:a', 'mp3',  # Audio codec
            '-b:a', bitrate,  # Audio bitrate
            '-id3v2_version', '3',  # Use ID3v2.3 tags
            '-write_id3v1', '1',  # Also write ID3v1 tags
            mp3_file_path  # Output file
        ]
        
        print(f"\nRunning ffmpeg command...")
        
        # Run ffmpeg command
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            output_size = os.path.getsize(mp3_file_path)
            compression_ratio = (1 - output_size / input_size) * 100
            
            print(f"✓ Conversion successful!")
            print(f"✓ Output file: {mp3_file_path}")
            print(f"✓ Output size: {output_size/1024:.1f} KB")
            print(f"✓ Compression ratio: {compression_ratio:.1f}%")
            
            # Try to get actual duration from the MP3 file
            try:
                probe_cmd = [
                    'ffprobe',
                    '-v', 'quiet',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    mp3_file_path
                ]
                duration_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                if duration_result.returncode == 0:
                    actual_duration = float(duration_result.stdout.strip())
                    print(f"✓ Actual duration: {actual_duration:.2f} seconds")
            except:
                pass  # Duration check failed, but that's okay
            
            return True
        else:
            print(f"✗ Error during conversion:")
            print(f"✗ {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error converting file: {e}")
        return False

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output paths
    pcm_dir = os.path.join(script_dir, 'pcm')
    
    # Define file paths
    pcm_file = os.path.join(pcm_dir, 'mayun_zh.pcm')
    mp3_file = os.path.join(script_dir, 'mayun_zh.mp3')  # Save in same directory as script
    
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
    
    # Convert the PCM file to MP3
    success = convert_pcm_to_mp3(
        pcm_file_path=pcm_file,
        mp3_file_path=mp3_file,
        sample_rate=16000,  # 16kHz
        channels=1,         # mono
        bit_depth=16,       # 16-bit
        bitrate='128k'      # 128 kbps MP3
    )
    
    if success:
        print(f"\n🎵 Conversion completed successfully!")
        print(f"🎵 You can now play the MP3 file: {mp3_file}")
    else:
        print(f"\n❌ Conversion failed!")

if __name__ == "__main__":
    main()
