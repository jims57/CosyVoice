from pydub import AudioSegment
audio = AudioSegment.from_wav("chunk.wav")
nonsilent = detect_nonsilent(audio, min_silence_len=100, silence_thresh=-40)
start, end = nonsilent[0]  # First voice segment
trimmed_audio = audio[start:end]