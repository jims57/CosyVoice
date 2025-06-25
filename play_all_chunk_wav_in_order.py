import os

def play_all_chunks_in_order():
    """
    Play all wav chunks in order from the generated_wavs folder
    """
    # Get the directory where this Python file is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # generated_wavs is in the same directory as this script
    generated_wavs_dir = os.path.join(script_dir, "generated_wavs")
    
    # Check if the directory exists
    if not os.path.exists(generated_wavs_dir):
        print(f"Directory {generated_wavs_dir} does not exist!")
        return 