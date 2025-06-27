import wave
import os

def convert_pcm_to_wav(pcm_file_path, wav_file_path, sample_rate=16000, channels=1, sample_width=2):
    """
    Convert a PCM file to WAV format
    
    Args:
        pcm_file_path (str): Path to the input PCM file
        wav_file_path (str): Path to the output WAV file
        sample_rate (int): Sample rate in Hz (default: 16000)
        channels (int): Number of channels (default: 1 for mono)
        sample_width (int): Sample width in bytes (default: 2 for 16-bit)
    """
    try:
        # Read PCM data
        with open(pcm_file_path, 'rb') as pcm_file:
            pcm_data = pcm_file.read()
        
        # Calculate number of frames
        frames = len(pcm_data) // (sample_width * channels)
        
        print(f"PCM file info:")
        print(f"  File size: {len(pcm_data)} bytes")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {channels}")
        print(f"  Sample width: {sample_width} bytes")
        print(f"  Calculated frames: {frames}")
        
        # Create WAV file
        with wave.open(wav_file_path, 'wb') as wav_file:
            # Set WAV parameters
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.setnframes(frames)
            wav_file.setcomptype('NONE', 'not compressed')
            
            # Write PCM data to WAV file
            wav_file.writeframes(pcm_data)
        
        print(f"Successfully converted {pcm_file_path} to {wav_file_path}")
        
        # Calculate duration
        duration = frames / sample_rate
        print(f"Audio duration: {duration:.2f} seconds")
        
    except Exception as e:
        print(f"Error converting file: {e}")

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output paths
    pcm_dir = os.path.join(script_dir, 'pcm')
    wav_dir = os.path.join(script_dir, 'wav')
    
    # Create wav directory if it doesn't exist
    os.makedirs(wav_dir, exist_ok=True)
    
    # Define file paths
    pcm_file = os.path.join(pcm_dir, 'mayun_zh.pcm')
    wav_file = os.path.join(wav_dir, 'converted_from_pcm.wav')
    
    # Check if input file exists
    if not os.path.exists(pcm_file):
        print(f"Error: Input file {pcm_file} not found!")
        return
    
    # Convert the file with specified parameters: 16kHz, mono, 16-bit
    convert_pcm_to_wav(pcm_file, wav_file, sample_rate=16000, channels=1, sample_width=2)

if __name__ == "__main__":
    main()
