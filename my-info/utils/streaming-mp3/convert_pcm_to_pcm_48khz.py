"""
Convert PCM to 48kHz Mono PCM Utility

This script converts a PCM audio file to 48kHz mono PCM format.
This is useful for standardizing audio format before further processing.

Usage:
    python convert_pcm_to_pcm_48khz.py

Examples:
    # Convert default file
    python convert_pcm_to_pcm_48khz.py
    
    # Convert specific PCM file
    python convert_pcm_to_pcm_48khz.py --pcm-file pcm/mayun_zh.pcm
    
    # Convert with custom input sample rate
    python convert_pcm_to_pcm_48khz.py --pcm-file pcm/a_man_die.pcm --input-sample-rate 16000

Requirements:
    - ffmpeg must be installed and available in PATH
    - Input PCM file should be raw PCM data (no WAV header)
    - Output will be saved in pcm folder

Output:
    - Creates 48kHz mono PCM file with _48khz suffix
"""


import os
import subprocess
import tempfile
import argparse

def convert_pcm_to_48khz_mono(pcm_file_path, input_sample_rate=16000, channels=1, bit_depth=16):
    """
    Convert PCM file to 48kHz mono PCM format
    """
    try:
        print(f"PCM input parameters:")
        print(f"  Input sample rate: {input_sample_rate} Hz")
        print(f"  Channels: {channels} (Mono)")
        print(f"  Bit depth: {bit_depth}")
        print(f"")
        print(f"PCM output parameters:")
        print(f"  Output sample rate: 48000 Hz")
        print(f"  Output channels: 1 (Mono)")
        
        # Generate output file path in pcm folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pcm_dir = os.path.join(script_dir, 'pcm')
        os.makedirs(pcm_dir, exist_ok=True)
        
        # Create output filename with _48khz suffix
        base_name = os.path.splitext(os.path.basename(pcm_file_path))[0]
        if base_name.endswith('_16khz'):
            base_name = base_name[:-6]  # Remove _16khz suffix
        output_filename = f"{base_name}_48khz.pcm"
        output_path = os.path.join(pcm_dir, output_filename)
        
        print(f"\nConverting PCM to 48kHz mono...")
        print(f"Input: {pcm_file_path}")
        print(f"Output: {output_path}")
        
        # Use ffmpeg to convert PCM sample rate
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-f', 's16le',  # Input format
            '-ar', str(input_sample_rate),  # Input sample rate
            '-ac', str(channels),  # Input channels
            '-i', pcm_file_path,  # Input file
            '-f', 's16le',  # Output format
            '-ar', '48000',  # Output sample rate (48kHz)
            '-ac', '1',  # Output channels (mono)
            output_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error converting PCM: {result.stderr}")
            return
        
        # Get output file size
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path)
            print(f"\n✓ Successfully converted to 48kHz mono PCM")
            print(f"✓ Output file: {output_filename}")
            print(f"✓ Output size: {output_size/1024/1024:.1f} MB")
        else:
            print("Error: Output file was not created")
        
    except Exception as e:
        print(f"Error processing file: {e}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert PCM to 48kHz Mono PCM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python convert_pcm_to_pcm_48khz.py
  
  # Use specific PCM file
  python convert_pcm_to_pcm_48khz.py --pcm-file pcm/mayun_zh.pcm
  
  # Custom input sample rate
  python convert_pcm_to_pcm_48khz.py --pcm-file pcm/a_man_die.pcm --input-sample-rate 16000
        """
    )
    
    parser.add_argument('--pcm-file', 
                       default=None,
                       help='Path to input PCM file (default: pcm/a_man_die.pcm)')
    
    parser.add_argument('--input-sample-rate', 
                       type=int, 
                       default=16000,
                       help='Input PCM sample rate in Hz (default: 16000)')
    
    parser.add_argument('--channels', 
                       type=int, 
                       default=1,
                       help='Number of input channels (default: 1)')
    
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
        pcm_file = os.path.join(script_dir, 'pcm', 'a_man_die.pcm')
    
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
    
    # Convert with specified parameters
    convert_pcm_to_48khz_mono(
        pcm_file_path=pcm_file,
        input_sample_rate=args.input_sample_rate,
        channels=args.channels,
        bit_depth=args.bit_depth
    )

if __name__ == "__main__":
    main()
