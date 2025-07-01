import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def remove_silence_from_wav(input_wav_path, output_wav_path, min_silence_len=100, silence_thresh=-40, keep_silence=50):
    """
    Remove silence from WAV file and keep only human voice audio
    
    Args:
        input_wav_path: Path to input WAV file
        output_wav_path: Path to output WAV file with silence removed
        min_silence_len: Minimum length of silence to be considered silence (ms)
        silence_thresh: Silence threshold in dBFS (lower values = more sensitive)
        keep_silence: Amount of silence to keep at the beginning and end of each segment (ms)
    """
    print(f"Loading audio from: {input_wav_path}")
    audio = AudioSegment.from_wav(input_wav_path)
    
    print(f"Detecting non-silent segments...")
    print(f"Parameters: min_silence_len={min_silence_len}ms, silence_thresh={silence_thresh}dBFS")
    
    # Detect non-silent segments
    nonsilent_segments = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    
    if not nonsilent_segments:
        print("No voice segments detected!")
        return
    
    print(f"Found {len(nonsilent_segments)} voice segments")
    
    # Combine all non-silent segments
    combined_audio = AudioSegment.empty()
    
    for i, (start, end) in enumerate(nonsilent_segments):
        # Add some silence padding if specified
        segment_start = max(0, start - keep_silence)
        segment_end = min(len(audio), end + keep_silence)
        
        segment = audio[segment_start:segment_end]
        combined_audio += segment
        
        print(f"Segment {i+1}: {start}ms to {end}ms (with padding: {segment_start}ms to {segment_end}ms)")
    
    # Export the combined audio
    combined_audio.export(output_wav_path, format="wav")
    print(f"Silence removed audio saved: {output_wav_path}")
    print(f"Original duration: {len(audio)/1000:.2f}s, New duration: {len(combined_audio)/1000:.2f}s")

def main():
    # Adjustable parameters for fine-tuning
    PARAMETERS = {
        'min_silence_len': 100,     # Minimum silence length in ms (increase to ignore short pauses)
        'silence_thresh': -40,      # Silence threshold in dBFS (decrease for more sensitive detection)
        'keep_silence': 50          # Amount of silence to keep around each segment in ms
    }
    
    # Define file paths
    input_wav_path = os.path.join("combined_wav", "combined_audio.wav")
    output_wav_path = os.path.join("combined_wav", "silence_removed.wav")
    
    print("=== Silence Removal Parameters ===")
    print(f"min_silence_len: {PARAMETERS['min_silence_len']}ms - Minimum silence duration to remove")
    print(f"silence_thresh: {PARAMETERS['silence_thresh']}dBFS - Volume threshold for silence detection")
    print(f"keep_silence: {PARAMETERS['keep_silence']}ms - Silence padding around voice segments")
    print("=====================================")
    
    remove_silence_from_wav(
        input_wav_path, 
        output_wav_path,
        min_silence_len=PARAMETERS['min_silence_len'],
        silence_thresh=PARAMETERS['silence_thresh'],
        keep_silence=PARAMETERS['keep_silence']
    )

if __name__ == "__main__":
    main()
