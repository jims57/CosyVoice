To segment PCM data from a WAV file, convert segments to MP3, and play them seamlessly:

1. Segmenting PCM Data
Read WAV header (typically 44 bytes) to locate raw PCM data start.

Calculate segment boundaries using:

python
segment_duration = 5.0  # seconds per segment
bytes_per_second = sample_rate * channels * (bit_depth // 8)
segment_bytes = int(bytes_per_second * segment_duration)
Extract segments as byte arrays, ensuring alignment to frame boundaries (e.g., 4-byte alignment for 16-bit stereo).

2. Converting Segments to MP3
Use FFmpeg to encode each PCM segment:

bash
ffmpeg -f s16le -ar 44100 -ac 2 -i segment.pcm segment.mp3
Flags:

-f s16le: Signed 16-bit little-endian PCM input

-ar 44100: Sample rate (adjust to match source)

-ac 2: Stereo channels (use -ac 1 for mono)

3. Gapless Playback
Encode with LAME: Add --gapless to minimize gaps:

bash
lame --gapless -y -m s -s 44.1 segment.pcm segment.mp3
Playback: Use a player supporting gapless concatenation (e.g., VLC with --gapless-playback).

Precise timing: Decode MP3s to PCM and stream sequentially via audio APIs (e.g., Web Audio API):

javascript
function playSequentially(audioFiles) {
  let current = new Audio(audioFiles[0]);
  current.play();
  current.onended = () => playSequentially(audioFiles.slice(1));
}
Key Considerations
Header Handling: Prepend a WAV header to each PCM segment if your encoder requires it.

Bit Depth: Match source WAV’s bit depth (e.g., 16-bit: s16le, 32-bit: s32le).

Testing: Verify alignment by checking for audio artifacts at segment boundaries.

Example Workflow
Split input.wav into segment1.pcm, segment2.pcm...

Convert segments:

bash
ffmpeg -f s16le -ar 48000 -ac 1 -i segment1.pcm segment1.mp3
Play with gapless:

bash
vlc --gapless-playback segment1.mp3 segment2.mp3 ...
Note: For programmatic playback, decode MP3s to PCM and buffer sequentially to avoid decoder-induced gaps.