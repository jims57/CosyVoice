import os
import torchaudio
import torch

def combine_wav_files():
    """
    Combine all wav files in generated_wavs folder in numerical order
    and save the combined file in the current directory
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generated_wavs_dir = os.path.join(script_dir, "generated_wavs")
    
    # Check if generated_wavs folder exists
    if not os.path.exists(generated_wavs_dir):
        print(f"Error: {generated_wavs_dir} folder not found!")
        return
    
    # Get all wav files and sort them numerically
    wav_files = []
    for filename in os.listdir(generated_wavs_dir):
        if filename.startswith("chunk_") and filename.endswith(".wav"):
            wav_files.append(filename)
    
    # Sort files numerically (chunk_1.wav, chunk_2.wav, etc.)
    wav_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    
    if not wav_files:
        print("No chunk wav files found in generated_wavs folder!")
        return
    
    print(f"Found {len(wav_files)} wav files to combine:")
    for file in wav_files:
        print(f"  {file}")
    
    # Load and combine all wav files
    combined_audio = None
    sample_rate = None
    
    for i, filename in enumerate(wav_files):
        file_path = os.path.join(generated_wavs_dir, filename)
        print(f"Loading {filename}...")
        
        # Load the audio file
        audio, sr = torchaudio.load(file_path)
        
        # Check if sample rates match
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            print(f"Warning: Sample rate mismatch in {filename} ({sr} vs {sample_rate})")
            continue
        
        # Concatenate audio tensors
        if combined_audio is None:
            combined_audio = audio
        else:
            combined_audio = torch.cat([combined_audio, audio], dim=1)
        
        print(f"  Added {filename} (duration: {audio.shape[1]/sr:.2f}s)")
    
    if combined_audio is not None:
        # Save the combined audio
        output_path = os.path.join(script_dir, "combined_audio.wav")
        torchaudio.save(output_path, combined_audio, sample_rate)
        
        total_duration = combined_audio.shape[1] / sample_rate
        print(f"\nSuccessfully combined {len(wav_files)} files!")
        print(f"Total duration: {total_duration:.2f} seconds")
        print(f"Saved as: {output_path}")
    else:
        print("No audio files were successfully loaded!")

if __name__ == "__main__":
    combine_wav_files()
