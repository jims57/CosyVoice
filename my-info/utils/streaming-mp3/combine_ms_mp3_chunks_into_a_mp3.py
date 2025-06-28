import os
import subprocess
import tempfile
import re

def combine_mp3_chunks(chunks_dir, output_dir, output_filename="combined_audio.mp3"):
    """
    Combine all MP3 chunks into a single MP3 file
    
    Args:
        chunks_dir (str): Directory containing MP3 chunks
        output_dir (str): Directory to save the combined MP3
        output_filename (str): Name of the output MP3 file
    """
    try:
        # Get all MP3 files in the chunks directory
        mp3_files = []
        for filename in os.listdir(chunks_dir):
            if filename.endswith('.mp3') and filename.startswith('audio_'):
                mp3_files.append(filename)
        
        if not mp3_files:
            print(f"No MP3 chunk files found in {chunks_dir}")
            return
        
        # Sort files by timestamp to ensure correct order
        def extract_timestamp(filename):
            match = re.search(r'audio_(\d+)\.mp3', filename)
            return int(match.group(1)) if match else 0
        
        mp3_files.sort(key=extract_timestamp)
        
        # Validate MP3 files and filter out corrupted ones
        valid_mp3_files = []
        print(f"Validating {len(mp3_files)} MP3 chunks...")
        
        for filename in mp3_files:
            file_path = os.path.join(chunks_dir, filename)
            file_size = os.path.getsize(file_path)
            
            # Quick validation using ffprobe
            try:
                probe_cmd = [
                    'ffprobe',
                    '-v', 'quiet',
                    '-show_entries', 'format=format_name',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    file_path
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and 'mp3' in result.stdout.lower():
                    valid_mp3_files.append(filename)
                    print(f"  ✓ {filename} ({file_size/1024:.1f} KB)")
                else:
                    print(f"  ✗ {filename} ({file_size/1024:.1f} KB) - Invalid format")
            except:
                print(f"  ✗ {filename} ({file_size/1024:.1f} KB) - Validation failed")
        
        if not valid_mp3_files:
            print("No valid MP3 files found!")
            return
            
        print(f"\nFound {len(valid_mp3_files)} valid MP3 chunks")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output file path
        output_path = os.path.join(output_dir, output_filename)
        
        # Delete existing output file if it exists
        if os.path.exists(output_path):
            print(f"Deleting existing file: {output_path}")
            os.unlink(output_path)
        
        print(f"\nCombining chunks into: {output_path}")
        
        # Build ffmpeg command with all input files
        ffmpeg_cmd = ['ffmpeg', '-y']  # Start with ffmpeg and overwrite flag
        
        # Add all input files
        for filename in valid_mp3_files:
            file_path = os.path.join(chunks_dir, filename)
            ffmpeg_cmd.extend(['-i', file_path])
        
        # Add filter_complex to concatenate all inputs
        filter_parts = []
        for i in range(len(valid_mp3_files)):
            filter_parts.append(f"[{i}:0]")
        
        concat_filter = f"{''.join(filter_parts)}concat=n={len(valid_mp3_files)}:v=0:a=1[out]"
        ffmpeg_cmd.extend(['-filter_complex', concat_filter, '-map', '[out]', output_path])
        
        # Run ffmpeg command
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            output_size = os.path.getsize(output_path)
            print(f"✓ Successfully combined {len(valid_mp3_files)} chunks")
            print(f"✓ Output file: {output_path}")
            print(f"✓ Output size: {output_size/1024:.1f} KB")
            
            # Calculate and display total duration if possible
            try:
                # Get duration using ffprobe
                probe_cmd = [
                    'ffprobe',
                    '-v', 'quiet',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    output_path
                ]
                duration_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                if duration_result.returncode == 0:
                    duration = float(duration_result.stdout.strip())
                    print(f"✓ Total duration: {duration:.2f} seconds")
            except:
                pass  # Duration calculation failed, but that's okay
                
        else:
            print(f"✗ Error combining chunks: {result.stderr}")
                
    except Exception as e:
        print(f"Error combining MP3 chunks: {e}")

def remove_lame_padding(input_path, output_path):
    """
    Remove LAME padding (55555555 patterns) from MP3 file
    
    Args:
        input_path (str): Input MP3 file path
        output_path (str): Output cleaned MP3 file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(input_path, 'rb') as input_file:
            data = input_file.read()
        
        # Find the last occurrence of meaningful audio data
        # Look for the end of real MP3 frames before LAME padding starts
        padding_pattern = b'\x55\x55\x55\x55'
        
        # Find where padding starts (multiple consecutive 0x55 bytes)
        padding_start = -1
        i = len(data) - 1
        consecutive_55_count = 0
        
        # Scan backwards to find where padding begins
        while i >= 0:
            if data[i] == 0x55:
                consecutive_55_count += 1
                if consecutive_55_count >= 20:  # Found significant padding
                    padding_start = i + 20
                    break
            else:
                consecutive_55_count = 0
            i -= 1
        
        if padding_start > 0:
            # Remove padding and write cleaned data
            cleaned_data = data[:padding_start]
            with open(output_path, 'wb') as output_file:
                output_file.write(cleaned_data)
            return True
        else:
            # No significant padding found, copy original
            with open(output_path, 'wb') as output_file:
                output_file.write(data)
            return True
            
    except Exception as e:
        print(f"  Error cleaning {input_path}: {e}")
        return False

def extract_mp3_frames(data):
    """
    Extract raw MP3 audio frames from MP3 data, skipping ID3 tags and metadata
    
    Args:
        data (bytes): MP3 file data
        
    Returns:
        list: List of MP3 audio frame bytes
    """
    frames = []
    i = 0
    
    print(f"  Analyzing {len(data)} bytes for MP3 frames...")
    
    while i < len(data) - 4:
        # Look for MP3 frame sync word: 0xFF followed by 0xFX (where X >= 0xE)
        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
            # Found potential MP3 frame header
            try:
                # Parse MP3 frame header to get frame length
                frame_length = get_mp3_frame_length(data[i:i+4])
                
                if frame_length > 0 and i + frame_length <= len(data):
                    # Verify next frame sync or end of data
                    next_pos = i + frame_length
                    if (next_pos >= len(data) - 4 or 
                        (data[next_pos] == 0xFF and (data[next_pos + 1] & 0xE0) == 0xE0)):
                        # Extract the complete frame
                        frame = data[i:i + frame_length]
                        frames.append(frame)
                        i = next_pos
                        continue
                        
                i += 1
            except:
                i += 1
        else:
            i += 1
    
    print(f"  Found {len(frames)} MP3 frames")
    return frames

def get_mp3_frame_length(header):
    """
    Calculate MP3 frame length from 4-byte header
    
    Args:
        header (bytes): 4-byte MP3 frame header
        
    Returns:
        int: Frame length in bytes, or 0 if invalid
    """
    if len(header) < 4:
        return 0
        
    # Parse MP3 header bits
    if header[0] != 0xFF or (header[1] & 0xE0) != 0xE0:
        return 0
    
    # MPEG version
    version_bits = (header[1] >> 3) & 0x03
    if version_bits == 1:  # Reserved
        return 0
        
    # Layer
    layer_bits = (header[1] >> 1) & 0x03
    if layer_bits == 0:  # Reserved
        return 0
    
    # Bitrate index
    bitrate_index = (header[2] >> 4) & 0x0F
    if bitrate_index == 0 or bitrate_index == 15:  # Free or bad
        return 0
    
    # Sample rate index  
    sample_rate_index = (header[2] >> 2) & 0x03
    if sample_rate_index == 3:  # Reserved
        return 0
        
    # Padding bit
    padding = (header[2] >> 1) & 0x01
    
    # Bitrate tables (kbps)
    if version_bits == 3:  # MPEG-1
        if layer_bits == 1:  # Layer III
            bitrates = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
            sample_rates = [44100, 48000, 32000, 0]
        else:
            return 0
    elif version_bits == 2:  # MPEG-2
        if layer_bits == 1:  # Layer III  
            bitrates = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]
            sample_rates = [22050, 24000, 16000, 0]
        else:
            return 0
    else:
        return 0
        
    bitrate = bitrates[bitrate_index] * 1000  # Convert to bps
    sample_rate = sample_rates[sample_rate_index]
    
    if bitrate == 0 or sample_rate == 0:
        return 0
    
    # Calculate frame length
    if layer_bits == 1:  # Layer III
        if version_bits == 3:  # MPEG-1
            frame_length = int((144 * bitrate) / sample_rate) + padding
        else:  # MPEG-2
            frame_length = int((72 * bitrate) / sample_rate) + padding
    else:
        return 0
        
    return frame_length if frame_length > 4 else 0

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output paths
    chunks_dir = os.path.join(script_dir, 'ms_mp3_chunks')
    output_dir = script_dir
    
    # Check if chunks directory exists
    if not os.path.exists(chunks_dir):
        print(f"Error: Chunks directory {chunks_dir} not found!")
        return
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not in PATH!")
        print("Please install ffmpeg: https://ffmpeg.org/download.html")
        return
    
    # Combine the MP3 chunks
    combine_mp3_chunks(chunks_dir, output_dir, "combined_audio.mp3")

if __name__ == "__main__":
    main()
