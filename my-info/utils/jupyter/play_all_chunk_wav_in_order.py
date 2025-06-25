from IPython.display import Audio, display
import os
import glob
import torchaudio
import torch

def play_all_chunks_in_order():
    """
    Play all wav chunks in order from the generated_wavs folder
    """
    # Use expanduser to get the full path like in play_zero_shot_wav.py
    generated_wavs_dir = os.path.expanduser('~/CosyVoice/generated_wavs')
    
    # Check if the directory exists
    if not os.path.exists(generated_wavs_dir):
        print(f"Directory {generated_wavs_dir} does not exist!")
        return
    
    # Get all chunk files and sort them numerically
    chunk_pattern = os.path.join(generated_wavs_dir, "chunk_*.wav")
    chunk_files = glob.glob(chunk_pattern)
    
    if not chunk_files:
        print(f"No chunk files found in {generated_wavs_dir}")
        return
    
    # Sort files by chunk number
    def extract_chunk_number(filename):
        basename = os.path.basename(filename)
        # Extract number from "chunk_X.wav"
        return int(basename.replace("chunk_", "").replace(".wav", ""))
    
    chunk_files.sort(key=extract_chunk_number)
    
    print(f"Found {len(chunk_files)} chunk files:")
    for chunk_file in chunk_files:
        print(f"  {os.path.basename(chunk_file)}")
    
    # Display each chunk individually with its own player
    print("\n=== Individual Chunk Players ===")
    for i, chunk_file in enumerate(chunk_files):
        print(f"\nChunk {i+1}: {os.path.basename(chunk_file)}")
        
        # Check if file exists before playing
        if os.path.exists(chunk_file):
            print(f"Playing audio from: {chunk_file}")
            display(Audio(chunk_file))
        else:
            print(f"Error: Audio file not found at {chunk_file}")
    
    # Combine all chunks into a single wav file
    print("\n=== Combining All Chunks ===")
    combined_audio_chunks = []
    sample_rate = None
    
    for chunk_file in chunk_files:
        # Load audio with torchaudio to properly handle wav headers
        audio, sr = torchaudio.load(chunk_file)
        
        # Store sample rate from first file
        if sample_rate is None:
            sample_rate = sr
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        combined_audio_chunks.append(audio)
        print(f"Loaded chunk: {os.path.basename(chunk_file)}")
    
    if combined_audio_chunks:
        # Concatenate all audio chunks
        print("Concatenating all chunks...")
        combined_audio = torch.cat(combined_audio_chunks, dim=1)
        
        # Save combined audio to current directory (not generated_wavs)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        combined_output_path = os.path.join(current_dir, "combined_chunks.wav")
        
        torchaudio.save(combined_output_path, combined_audio, sample_rate)
        print(f"Combined audio saved to: {combined_output_path}")
        
        # Display the combined audio player
        print("\n=== Combined Audio Player ===")
        display(Audio(combined_output_path))
    
    return chunk_files

# Call the function directly when running in Jupyter
play_all_chunks_in_order()
