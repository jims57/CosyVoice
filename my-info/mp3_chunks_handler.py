"""
MP3 Chunks Handler Module

This module encapsulates the silence detection and PCM to MP3 conversion logic
for streaming applications. It processes PCM data in a buffer and generates
MP3 chunks based on silence detection.

Usage:
    from mp3_chunks_handler import MP3ChunksHandler
    
    handler = MP3ChunksHandler(
        input_sample_rate=16000,
        output_sample_rate=16000,
        volume_dB=-40.0,
        min_samples_window=960
    )
    
    # Add PCM data to buffer
    handler.add_pcm_data(pcm_bytes)
    
    # Get available MP3 chunks
    mp3_chunks = handler.get_available_chunks()
    
    # Finalize and get last chunk
    final_chunk = handler.finalize()

Created: December 19, 2024
"""

import os
import subprocess
import tempfile
import glob
from typing import List, Optional, Tuple


class MP3ChunksHandler:
    """
    Handles PCM data buffering and silence-based MP3 chunk generation
    """
    
    def __init__(self, input_sample_rate=16000, output_sample_rate=16000, 
                 volume_dB=-40.0, min_samples_window=960, channels=1, bit_depth=16):
        """
        Initialize MP3 chunks handler
        
        Args:
            input_sample_rate: Input PCM sample rate (Hz)
            output_sample_rate: Output MP3 sample rate (Hz)
            volume_dB: Volume threshold for silence detection (dB)
            min_samples_window: Minimum samples for silence window
            channels: Audio channels (1 for mono)
            bit_depth: Bit depth (16 for 16-bit)
        """
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.volume_dB = volume_dB
        self.min_samples_window = min_samples_window
        self.channels = channels
        self.bit_depth = bit_depth
        
        # Buffer for accumulating PCM data
        self.pcm_buffer = bytearray()
        
        # Silence detection parameters
        self.bytes_per_sample = (bit_depth // 8) * channels
        self.frame_size_seconds = 0.02  # 20ms frames
        self.frame_samples = int(input_sample_rate * self.frame_size_seconds)
        self.frame_bytes = self.frame_samples * self.bytes_per_sample
        self.min_silence_frames = int(min_samples_window / self.frame_samples)
        
        # Silence detection state
        self.last_processed_byte = 0
        self.current_silence_start = -1
        self.silence_frame_count = 0
        self.silent_regions = []
        
        print(f"[MP3Handler] Initialized with:")
        print(f"  Input rate: {input_sample_rate} Hz")
        print(f"  Output rate: {output_sample_rate} Hz") 
        print(f"  Volume threshold: {volume_dB} dB")
        print(f"  Min silence window: {min_samples_window} samples")
        print(f"  Frame size: {self.frame_samples} samples ({self.frame_size_seconds*1000:.0f}ms)")
    
    def add_pcm_data(self, pcm_data: bytes) -> None:
        """
        Add new PCM data to the buffer
        
        Args:
            pcm_data: Raw PCM bytes to add to buffer
        """
        self.pcm_buffer.extend(pcm_data)
        print(f"[MP3Handler] Added {len(pcm_data)} bytes, buffer size: {len(self.pcm_buffer)} bytes")
    
    def _detect_new_silence_regions(self) -> None:
        """
        Detect silence regions in newly added PCM data
        """
        # Process frames from last processed position
        start_byte = self.last_processed_byte
        
        for frame_idx in range(start_byte, len(self.pcm_buffer), self.frame_bytes):
            frame_data = self.pcm_buffer[frame_idx:frame_idx + self.frame_bytes]
            if len(frame_data) < self.frame_bytes:
                break  # Incomplete frame, wait for more data
            
            # Simple silence detection
            is_silent = True
            for i in range(0, len(frame_data), 2):
                if i + 1 < len(frame_data):
                    sample = int.from_bytes(frame_data[i:i+2], byteorder='little', signed=True)
                    if abs(sample) > 1000:  # Threshold for silence
                        is_silent = False
                        break
            
            if is_silent:
                if self.current_silence_start == -1:
                    self.current_silence_start = frame_idx
                    self.silence_frame_count = 1
                else:
                    self.silence_frame_count += 1
            else:
                # End of silence region
                if self.current_silence_start != -1 and self.silence_frame_count >= self.min_silence_frames:
                    silence_end = self.current_silence_start + (self.silence_frame_count * self.frame_bytes)
                    self.silent_regions.append((self.current_silence_start, silence_end))
                    duration = self.silence_frame_count * self.frame_size_seconds
                    print(f"[MP3Handler] Found silence: bytes {self.current_silence_start}-{silence_end} ({self.silence_frame_count} frames, {duration:.3f}s)")
                
                self.current_silence_start = -1
                self.silence_frame_count = 0
            
            # Update last processed position
            self.last_processed_byte = frame_idx + self.frame_bytes
    
    def get_available_chunks(self) -> List[bytes]:
        """
        Get available MP3 chunks based on current buffer and silence detection
        
        Returns:
            List of MP3 chunk bytes ready to be sent
        """
        # Detect new silence regions
        self._detect_new_silence_regions()
        
        chunks = []
        chunks_generated = 0
        
        # Process complete silence regions
        while self.silent_regions:
            silence_start, silence_end = self.silent_regions[0]
            
            # Check if we have enough buffer data up to this silence region
            if silence_end <= len(self.pcm_buffer):
                # Find start of this chunk (end of previous chunk or beginning)
                if hasattr(self, '_last_chunk_end'):
                    chunk_start = self._last_chunk_end
                else:
                    chunk_start = 0
                
                # Create chunk ending at silence start
                chunk_end = silence_start
                
                if chunk_end > chunk_start:
                    chunk_data = bytes(self.pcm_buffer[chunk_start:chunk_end])
                    actual_frames = len(chunk_data) // self.bytes_per_sample
                    chunk_duration = actual_frames / self.input_sample_rate
                    
                    print(f"[MP3Handler] Creating chunk: bytes {chunk_start}-{chunk_end} ({actual_frames} samples, {chunk_duration:.2f}s)")
                    
                    # Convert to MP3
                    mp3_data = self._convert_pcm_to_mp3(chunk_data)
                    if mp3_data:
                        chunks.append(mp3_data)
                        chunks_generated += 1
                        print(f"[MP3Handler] Generated MP3 chunk {chunks_generated}: {len(mp3_data)} bytes")
                
                # Update last chunk end position
                self._last_chunk_end = silence_end
                
                # Remove processed silence region
                self.silent_regions.pop(0)
            else:
                # Not enough data yet, wait for more
                break
        
        return chunks
    
    def finalize(self) -> Optional[bytes]:
        """
        Finalize processing and get the last chunk from remaining buffer data
        
        Returns:
            Final MP3 chunk bytes, or None if no remaining data
        """
        # Handle any remaining silence at the end
        if self.current_silence_start != -1 and self.silence_frame_count >= self.min_silence_frames:
            silence_end = len(self.pcm_buffer)
            self.silent_regions.append((self.current_silence_start, silence_end))
            duration = self.silence_frame_count * self.frame_size_seconds
            print(f"[MP3Handler] Found final silence: bytes {self.current_silence_start}-{silence_end} ({self.silence_frame_count} frames, {duration:.3f}s)")
        
        # Process any remaining chunks
        final_chunks = self.get_available_chunks()
        
        # Handle final remaining data
        if hasattr(self, '_last_chunk_end'):
            final_start = self._last_chunk_end
        else:
            final_start = 0
            
        if final_start < len(self.pcm_buffer):
            final_data = bytes(self.pcm_buffer[final_start:])
            actual_frames = len(final_data) // self.bytes_per_sample
            final_duration = actual_frames / self.input_sample_rate
            
            print(f"[MP3Handler] Creating final chunk: bytes {final_start}-{len(self.pcm_buffer)} ({actual_frames} samples, {final_duration:.2f}s)")
            
            if len(final_data) > self.bytes_per_sample:
                mp3_data = self._convert_pcm_to_mp3(final_data)
                if mp3_data:
                    print(f"[MP3Handler] Generated final MP3 chunk: {len(mp3_data)} bytes")
                    return mp3_data
        
        # Return the first chunk if we have any, or None
        return final_chunks[0] if final_chunks else None
    
    def _convert_pcm_to_mp3(self, pcm_data: bytes) -> Optional[bytes]:
        """
        Convert PCM data to MP3 bytes using ffmpeg
        
        Args:
            pcm_data: Raw PCM bytes
            
        Returns:
            MP3 bytes or None if conversion failed
        """
        try:
            # Create temporary PCM file
            with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as temp_pcm:
                temp_pcm.write(pcm_data)
                temp_pcm_path = temp_pcm.name
            
            try:
                # Create temporary MP3 output file
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
                    temp_mp3_path = temp_mp3.name
                
                # Convert using ffmpeg
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output files
                    '-f', 's16le',  # Input format
                    '-ar', str(self.input_sample_rate),  # Input sample rate
                    '-ac', str(self.channels),  # Input channels
                    '-i', temp_pcm_path,  # Input file
                    '-c:a', 'libmp3lame',  # MP3 encoder
                    '-b:a', '320k',  # 320 kbps
                    '-ar', str(self.output_sample_rate),  # Output sample rate
                    '-ac', '1',  # Mono output
                    '-write_id3v1', '0',  # No ID3v1
                    '-write_id3v2', '0',  # No ID3v2
                    '-id3v2_version', '0',  # No ID3v2
                    '-write_xing', '0',  # No Xing header
                    '-fflags', '+bitexact',
                    temp_mp3_path
                ]
                
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Read MP3 data
                    with open(temp_mp3_path, 'rb') as f:
                        mp3_data = f.read()
                    return mp3_data
                else:
                    print(f"[MP3Handler] FFmpeg error: {result.stderr}")
                    return None
            
            finally:
                # Clean up temporary files
                if os.path.exists(temp_pcm_path):
                    os.unlink(temp_pcm_path)
                if 'temp_mp3_path' in locals() and os.path.exists(temp_mp3_path):
                    os.unlink(temp_mp3_path)
        
        except Exception as e:
            print(f"[MP3Handler] Error converting PCM to MP3: {e}")
            return None
    
    def reset(self) -> None:
        """
        Reset the handler state for a new session
        """
        self.pcm_buffer.clear()
        self.last_processed_byte = 0
        self.current_silence_start = -1
        self.silence_frame_count = 0
        self.silent_regions.clear()
        if hasattr(self, '_last_chunk_end'):
            delattr(self, '_last_chunk_end')
        print("[MP3Handler] Handler reset for new session")
