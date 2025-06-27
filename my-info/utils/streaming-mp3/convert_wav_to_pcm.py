import wave
import os

def convert_wav_to_pcm(wav_file_path, pcm_file_path):
    """
    Convert a WAV file to PCM format (raw audio data)
    
    Args:
        wav_file_path (str): Path to the input WAV file
        pcm_file_path (str): Path to the output PCM file
    """
    try:
        # Open the WAV file
        with wave.open(wav_file_path, 'rb') as wav_file:
            # Get audio parameters
            frames = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            
            print(f"Audio info:")
            print(f"  Frames: {frames}")
            print(f"  Sample width: {sample_width} bytes")
            print(f"  Frame rate: {framerate} Hz")
            print(f"  Channels: {channels}")
            
            # Read all frames (raw PCM data)
            pcm_data = wav_file.readframes(frames)
            
            # Write PCM data to file
            with open(pcm_file_path, 'wb') as pcm_file:
                pcm_file.write(pcm_data)
                
            print(f"Successfully converted {wav_file_path} to {pcm_file_path}")
            print(f"PCM file size: {len(pcm_data)} bytes")
            
    except Exception as e:
        print(f"Error converting file: {e}")

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output paths
    wav_dir = os.path.join(script_dir, 'wav')
    pcm_dir = os.path.join(script_dir, 'pcm')
    
    # Create pcm directory if it doesn't exist
    os.makedirs(pcm_dir, exist_ok=True)
    
    # Define file paths
    wav_file = os.path.join(wav_dir, 'mayun_zh.wav')
    pcm_file = os.path.join(pcm_dir, 'mayun_zh.pcm')
    
    # Check if input file exists
    if not os.path.exists(wav_file):
        print(f"Error: Input file {wav_file} not found!")
        return
    
    # Convert the file
    convert_wav_to_pcm(wav_file, pcm_file)

if __name__ == "__main__":
    main()
