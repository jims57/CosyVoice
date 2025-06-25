from IPython.display import Audio, display
import os

# Construct the full path to the audio file
# The '~' (tilde) for home directory often needs to be expanded
# os.path.expanduser() is perfect for this.
# audio_file_path = os.path.expanduser('~/CosyVoice/zero_shot_0.wav')
audio_file_path = os.path.expanduser('~/CosyVoice/test.wav')

# Check if the file exists before trying to play it
if os.path.exists(audio_file_path):
    print(f"Playing audio from: {audio_file_path}")
    display(Audio(audio_file_path))
else:
    print(f"Error: Audio file not found at {audio_file_path}")
    print("Please ensure the file path is correct and the file exists.")