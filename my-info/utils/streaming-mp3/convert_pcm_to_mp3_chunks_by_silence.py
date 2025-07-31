"""
Convert PCM to MP3 Chunks by Silence Detection Utility

This script segments a PCM audio file into chunks based on volume levels and converts each chunk to MP3 format.
It uses a simple silence detection approach to find optimal split points at quiet samples.

Usage:
    python convert_pcm_to_mp3_chunks_by_silence.py

Configuration:
    - volume_dB: Volume threshold in dB for split detection (default: -40.0 dB)
    - min_samples_window: Minimum window size for volume analysis (default: 2400 samples)
    - input_pcm_sample_rate: Input PCM sample rate (default: 48000 Hz)
    - output_mp3_sample_rate: Output MP3 sample rate (default: 16000 Hz)

Examples:
    # My usage examples:
    python convert_pcm_to_mp3_chunks_by_silence.py --pcm-file pcm/pcm_chunks_combined.pcm  --volume-dB -40.0 --min-samples-window 960  --input-pcm-sample-rate 16000 --output-mp3-sample-rate 16000

    # Basic usage with default settings
    python convert_pcm_to_mp3_chunks_by_silence.py
    
    # Custom volume threshold and window size
    python convert_pcm_to_mp3_chunks_by_silence.py --volume-dB -30.0 --min-samples-window 2400
    
    # Custom sample rates
    python convert_pcm_to_mp3_chunks_by_silence.py --input-pcm-sample-rate 48000 --output-mp3-sample-rate 16000

Requirements:
    - ffmpeg must be installed and available in PATH
    - Input PCM file should be raw PCM data (no WAV header)
    - Output directory will be created automatically

Output:
    - Creates numbered MP3 chunks: chunk_0.mp3, chunk_1.mp3, etc.
    - Combines all chunks into a single MP3 file in mp3_combined folder
"""


import os
import subprocess
import tempfile
import argparse
import glob

def segment_pcm_by_volume_and_convert_to_mp3(pcm_file_path, output_dir, volume_dB=-40.0, min_samples_window=4800, input_pcm_sample_rate=48000, output_mp3_sample_rate=16000, channels=1, bit_depth=16):
    """
    Split PCM by volume levels using simple frame-based silence detection and convert chunks to MP3
    """
    try:
        print(f"PCM input parameters:")
        print(f"  Input sample rate: {input_pcm_sample_rate} Hz")
        print(f"  Channels: {channels} (Mono)")
        print(f"  Bit depth: {bit_depth}")
        print(f"")
        print(f"Simple silence detection parameters:")
        print(f"  Volume threshold: {volume_dB} dB")
        print(f"  Min silence duration: {min_samples_window / input_pcm_sample_rate:.3f} seconds")
        print(f"  Output MP3 sample rate: {output_mp3_sample_rate} Hz")
        
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
        
        # Read the entire PCM file as bytes (safer than numpy)
        with open(pcm_file_path, 'rb') as f:
            pcm_data = f.read()
        
        # Calculate frame parameters
        bytes_per_sample = (bit_depth // 8) * channels
        total_samples = len(pcm_data) // bytes_per_sample
        frame_size_seconds = 0.02  # 20ms frames
        frame_samples = int(input_pcm_sample_rate * frame_size_seconds)
        frame_bytes = frame_samples * bytes_per_sample
        min_silence_frames = int(min_samples_window / frame_samples)
        
        print(f"\nInput file analysis:")
        print(f"  Total bytes: {len(pcm_data)}")
        print(f"  Total samples: {total_samples}")
        print(f"  Duration: {total_samples / input_pcm_sample_rate:.2f} seconds")
        print(f"  Frame size: {frame_samples} samples ({frame_size_seconds*1000:.0f}ms)")
        print(f"  Min silence frames: {min_silence_frames}")
        
        print(f"\nProcessing frames for silence detection...")
        
        # Simple silence detection without numpy
        silent_regions = []
        current_silence_start = -1
        silence_frame_count = 0
        
        # Convert volume_dB to amplitude threshold
        # For 16-bit PCM: -40dB ≈ 327, -30dB ≈ 1033, -20dB ≈ 3276
        max_amplitude = 32767  # Maximum for 16-bit signed
        volume_threshold = max_amplitude * (10 ** (volume_dB / 20.0))
        
        print(f"  Using amplitude threshold: {volume_threshold:.0f} (from {volume_dB} dB)")
        
        for frame_idx in range(0, len(pcm_data), frame_bytes):
            frame_data = pcm_data[frame_idx:frame_idx + frame_bytes]
            if len(frame_data) < frame_bytes:
                break
            
            # Simple silence detection: check for very small values
            # Convert bytes to 16-bit integers manually for silence check
            is_silent = True
            for i in range(0, len(frame_data), 2):
                if i + 1 < len(frame_data):
                    # Read 16-bit little-endian sample
                    sample = int.from_bytes(frame_data[i:i+2], byteorder='little', signed=True)
                    # Use calculated volume threshold instead of hardcoded 1000
                    if abs(sample) > volume_threshold:
                        is_silent = False
                        break
            
            if is_silent:
                if current_silence_start == -1:
                    current_silence_start = frame_idx
                    silence_frame_count = 1
                else:
                    silence_frame_count += 1
            else:
                # End of silence region
                if current_silence_start != -1 and silence_frame_count >= min_silence_frames:
                    silence_end = current_silence_start + (silence_frame_count * frame_bytes)
                    silent_regions.append((current_silence_start, silence_end))
                    duration = silence_frame_count * frame_size_seconds
                    print(f"  Found silence: bytes {current_silence_start}-{silence_end} ({silence_frame_count} frames, {duration:.3f}s)")
                
                current_silence_start = -1
                silence_frame_count = 0
        
        # Handle final silence
        if current_silence_start != -1 and silence_frame_count >= min_silence_frames:
            silence_end = len(pcm_data)
            silent_regions.append((current_silence_start, silence_end))
            duration = silence_frame_count * frame_size_seconds
            print(f"  Found final silence: bytes {current_silence_start}-{silence_end} ({silence_frame_count} frames, {duration:.3f}s)")
        
        print(f"\nSplitting audio into chunks...")
        
        # Split PCM at silent regions but preserve all samples
        chunk_num = 0
        start_byte = 0
        
        for silence_start, silence_end in silent_regions:
            # Create chunk ending at silence end (include the silence)
            end_byte = min(silence_end, len(pcm_data))
            
            if end_byte > start_byte:
                chunk_data = pcm_data[start_byte:end_byte]
                actual_frames = len(chunk_data) // bytes_per_sample
                chunk_duration = actual_frames / input_pcm_sample_rate
                
                print(f"  Chunk {chunk_num}: bytes {start_byte}-{end_byte} ({actual_frames} samples, duration: {chunk_duration:.2f}s)")
                
                # Convert chunk to MP3
                if len(chunk_data) > bytes_per_sample:  # At least one sample
                    success = convert_chunk_to_mp3_bytes(chunk_data, chunk_num, output_dir, input_pcm_sample_rate, output_mp3_sample_rate, channels, bit_depth)
                    if success:
                        chunk_num += 1
            
            # Set start of next chunk right after current chunk (no gap)
            start_byte = end_byte
        
        # Create final chunk with remaining data
        if start_byte < len(pcm_data):
            final_data = pcm_data[start_byte:]
            actual_frames = len(final_data) // bytes_per_sample
            final_duration = actual_frames / input_pcm_sample_rate
            
            print(f"  Final chunk {chunk_num}: bytes {start_byte}-{len(pcm_data)} ({actual_frames} samples, duration: {final_duration:.2f}s)")
            
            if len(final_data) > bytes_per_sample:
                success = convert_chunk_to_mp3_bytes(final_data, chunk_num, output_dir, input_pcm_sample_rate, output_mp3_sample_rate, channels, bit_depth)
                if success:
                    chunk_num += 1
        
        print(f"\n✓ Created {chunk_num} silence-based MP3 chunks")
        
        # Combine all chunks into single MP3
        combine_chunks_to_single_mp3(output_dir, chunk_num)
        
    except Exception as e:
        print(f"Error processing file: {e}")

def convert_chunk_to_mp3_bytes(chunk_data, chunk_num, output_dir, input_sample_rate, output_sample_rate, channels, bit_depth):
    """
    Convert PCM chunk bytes to MP3 file (without numpy)
    """
    try:
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
                return False
            else:
                # Get output file size
                mp3_size = os.path.getsize(chunk_file)
                bytes_per_sample = 2  # 16-bit
                actual_frames = len(chunk_data) // bytes_per_sample
                duration = actual_frames / input_sample_rate
                
                print(f"    ✓ Created chunk_{chunk_num}.mp3 ({mp3_size/1024:.1f} KB, {actual_frames} samples, ~{duration:.2f}s)")
                return True
        
        finally:
            # Clean up temporary PCM file
            os.unlink(temp_pcm_path)
    
    except Exception as e:
        print(f"Error converting chunk {chunk_num}: {e}")
        return False

def combine_chunks_to_single_mp3(chunks_dir, total_chunks):
    """
    Combine all MP3 chunks into a single MP3 file
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mp3_combined_dir = os.path.join(script_dir, 'mp3_combined')
        os.makedirs(mp3_combined_dir, exist_ok=True)
        
        output_path = os.path.join(mp3_combined_dir, 'combined_output.mp3')
        
        print(f"\nCombining {total_chunks} chunks into single MP3...")
        
        # Create temporary file list for ffmpeg concat
        concat_file = os.path.join(script_dir, 'temp_concat_list.txt')
        with open(concat_file, 'w') as f:
            for i in range(total_chunks):
                chunk_file = os.path.join(chunks_dir, f"chunk_{i}.mp3")
                if os.path.exists(chunk_file):
                    rel_path = os.path.relpath(chunk_file, script_dir)
                    f.write(f"file '{rel_path}'\n")
        
        try:
            # Use ffmpeg to concatenate MP3 files
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-f', 'concat',  # Use concat demuxer
                '-safe', '0',  # Allow unsafe file names
                '-i', concat_file,  # Input file list
                '-c', 'copy',  # Copy streams without re-encoding
                output_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error combining chunks: {result.stderr}")
                return
            
            # Get output file size
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path)
                print(f"✓ Successfully combined chunks into: {os.path.basename(output_path)}")
                print(f"✓ Combined file size: {output_size/1024/1024:.1f} MB")
                print(f"✓ Saved to: {mp3_combined_dir}")
            else:
                print("Error: Combined output file was not created")
        
        finally:
            # Clean up temporary concat file
            if os.path.exists(concat_file):
                os.unlink(concat_file)
    
    except Exception as e:
        print(f"Error combining chunks: {e}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert PCM to MP3 Chunks by Volume-based Silence Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python convert_pcm_to_mp3_chunks_by_silence.py
  
  # Custom volume threshold
  python convert_pcm_to_mp3_chunks_by_silence.py --volume-dB 8.0
  
  # Custom window size and sample rates
  python convert_pcm_to_mp3_chunks_by_silence.py --min-samples-window 2400 --input-pcm-sample-rate 48000
        """
    )
    
    parser.add_argument('--pcm-file', 
                       default=None,
                       help='Path to input PCM file (default: pcm/a_man_die_48khz.pcm)')
    
    parser.add_argument('--output-dir', 
                       default='mp3_chunks_split_by_silence',
                       help='Output directory for MP3 chunks (default: mp3_chunks_split_by_silence)')
    
    parser.add_argument('--volume-dB', 
                       type=float, 
                       default=-40.0,
                       help='Volume threshold in dB for split detection (default: -40.0)')
    
    parser.add_argument('--min-samples-window', 
                       type=int, 
                       default=2400,
                       help='Minimum window size for volume analysis (default: 2400)')
    
    parser.add_argument('--input-pcm-sample-rate', 
                       type=int, 
                       default=48000,
                       help='Input PCM sample rate in Hz (default: 48000)')
    
    parser.add_argument('--output-mp3-sample-rate', 
                       type=int, 
                       default=16000,
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
    segment_pcm_by_volume_and_convert_to_mp3(
        pcm_file_path=pcm_file,
        output_dir=mp3_dir,
        volume_dB=args.volume_dB,
        min_samples_window=args.min_samples_window,
        input_pcm_sample_rate=args.input_pcm_sample_rate,
        output_mp3_sample_rate=args.output_mp3_sample_rate,
        channels=args.channels,
        bit_depth=args.bit_depth
    )

if __name__ == "__main__":
    main()
