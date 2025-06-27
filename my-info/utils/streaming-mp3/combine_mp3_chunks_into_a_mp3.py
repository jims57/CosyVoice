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
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output file path
        output_path = os.path.join(output_dir, output_filename)
        
        # Delete existing output file if it exists
        if os.path.exists(output_path):
            print(f"Deleting existing file: {output_path}")
            os.unlink(output_path)
        
        # Create a temporary file list for ffmpeg concat demuxer
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file_path = temp_file.name
            for filename in mp3_files:
                file_path = os.path.join(chunks_dir, filename)
                # Write file path with proper escaping for ffmpeg
                temp_file.write(f"file '{os.path.abspath(file_path)}'\n")
        
        try:
            print(f"\nCombining chunks into: {output_path}")
            
            # Use ffmpeg concat demuxer to combine MP3 files
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-f', 'concat',  # Use concat demuxer
                '-safe', '0',  # Allow absolute paths
                '-i', temp_file_path,  # Input file list
                '-c', 'copy',  # Copy streams without re-encoding
                output_path  # Output file
            ]
            
            # Run ffmpeg command
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                output_size = os.path.getsize(output_path)
                print(f"✓ Successfully combined {len(mp3_files)} chunks")
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
                
        finally:
            # Clean up temporary file list
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except Exception as e:
        print(f"Error combining MP3 chunks: {e}")

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output paths
    chunks_dir = os.path.join(script_dir, 'mp3_chunks')
    output_dir = os.path.join(script_dir, 'mp3_combined')
    
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
