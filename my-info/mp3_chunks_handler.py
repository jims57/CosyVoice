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
    
    def __init__(self, input_sample_rate=24000, output_sample_rate=24000, 
                 volume_dB=-40.0, min_samples_window=960, channels=1, bit_depth=16):
        """
        Initialize MP3 chunks handler - Updated January 23, 2025
        
        Args:
            input_sample_rate: Input PCM sample rate (Hz) - Updated to 24000
            output_sample_rate: Output MP3 sample rate (Hz) - Updated to 24000
            volume_dB: Volume threshold for silence detection (dB) - Updated to -40.0
            min_samples_window: Minimum samples for silence window - Updated to 960
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
        
        # Convert volume_dB to amplitude threshold (same as working script)
        max_amplitude = 32767  # Maximum for 16-bit signed
        self.volume_threshold = max_amplitude * (10 ** (volume_dB / 20.0))
        
        # Silence detection state (same as working script)
        self.processed_samples = 0
        self.chunk_start_sample = 0
        self.silent_regions = []
        
        print(f"[MP3Handler] Initialized with:")
        print(f"  Input rate: {input_sample_rate} Hz")
        print(f"  Output rate: {output_sample_rate} Hz") 
        print(f"  Volume threshold: {self.volume_threshold:.0f} (from {volume_dB} dB)")
        print(f"  Min silence window: {min_samples_window} samples")
    
    def add_pcm_data(self, pcm_data: bytes) -> List[bytes]:
        """
        Add new PCM data and return any ready MP3 chunks (exact same logic as working script)
        
        Args:
            pcm_data: Raw PCM bytes to add to buffer
            
        Returns:
            List of MP3 chunks that are ready to be sent (always returns list, never None)
        """
        # Add new data to buffer
        self.pcm_buffer.extend(pcm_data)
        print(f"[MP3Handler] Added {len(pcm_data)} bytes, buffer size: {len(self.pcm_buffer)} bytes")
        
        mp3_chunks = []  # Always initialize as empty list
        
        # Process complete samples (exact same logic as working script)
        while len(self.pcm_buffer) >= self.min_samples_window * 2:
            # Extract samples for silence detection (exact same window as working script)
            window_start = self.processed_samples * 2
            window_end = window_start + (self.min_samples_window * 2)
            
            if window_end > len(self.pcm_buffer):
                break  # Not enough data for complete window
            
            samples_data = self.pcm_buffer[window_start:window_end]
            
            # Simple silence detection (exact same logic as working script)
            is_silent_region = True
            lowest_volume_sample = 0
            
            for i in range(0, len(samples_data), 2):
                if i + 1 < len(samples_data):
                    # Read 16-bit little-endian sample
                    sample = int.from_bytes(samples_data[i:i+2], byteorder='little', signed=True)
                    # Use exact same volume threshold logic as working script
                    if abs(sample) > self.volume_threshold:
                        is_silent_region = False
                        break
                    if abs(sample) < abs(lowest_volume_sample):
                        lowest_volume_sample = sample
            
            if is_silent_region:
                # Found silence region, create MP3 chunk ending here (preserve all samples)
                chunk_end_sample = self.processed_samples + self.min_samples_window
                
                # Calculate chunk boundaries in bytes (exact same as working script)
                chunk_start_byte = self.chunk_start_sample * 2
                chunk_end_byte = chunk_end_sample * 2
                
                # Extract chunk data (preserve all samples including silence)
                if chunk_end_byte <= len(self.pcm_buffer):
                    chunk_data = bytes(self.pcm_buffer[chunk_start_byte:chunk_end_byte])
                    
                    if len(chunk_data) > 2:  # At least one sample
                        # Convert chunk to MP3 using exact same logic as working script
                        mp3_data = self._convert_pcm_to_mp3(chunk_data)
                        
                        if mp3_data:
                            chunk_duration = len(chunk_data) / (self.input_sample_rate * 2)
                            mp3_chunks.append(mp3_data)
                            print(f"[MP3Handler] Generated MP3 chunk (silence): {len(mp3_data)} bytes ({chunk_duration:.3f}s)")
                        
                        # Update chunk start for next chunk (exact same as working script)
                        self.chunk_start_sample = chunk_end_sample
                        
                        # Remove processed data from buffer (exact same as working script)
                        self.pcm_buffer = self.pcm_buffer[chunk_end_byte:]
                        self.processed_samples = 0  # Reset relative to new buffer start
                        
                        continue  # Continue processing for more chunks
            
            # Move to next window (exact same as working script)
            self.processed_samples += self.min_samples_window
            
            # If no silence found in this window, move the window (exact same as working script)
            if not is_silent_region:
                # Continue to next window without splitting
                continue
        
        return mp3_chunks  # Always return list (empty if no chunks ready)
    
    def finalize(self) -> Optional[bytes]:
        """
        Finalize processing and get the last chunk from remaining buffer data (same as working script)
        
        Returns:
            Final MP3 chunk bytes, or None if no remaining data
        """
        # Convert any remaining PCM data to final MP3 chunk (preserve all remaining samples)
        if len(self.pcm_buffer) > 2:
            final_chunk_data = bytes(self.pcm_buffer)
            
            # Convert final chunk to MP3
            mp3_data = self._convert_pcm_to_mp3(final_chunk_data)
            
            if mp3_data:
                final_duration = len(final_chunk_data) / (self.input_sample_rate * 2)
                print(f"[MP3Handler] Generated final MP3 chunk: {len(mp3_data)} bytes ({final_duration:.3f}s)")
                return mp3_data
        
        return None

    def _convert_pcm_to_mp3(self, pcm_data: bytes) -> Optional[bytes]:
        """
        Convert PCM data to MP3 bytes using ffmpeg (same as working script)
        
        Args:
            pcm_data: Raw PCM bytes
            
        Returns:
            MP3 bytes or None if conversion failed
        """
        try:
            # Use ffmpeg to convert PCM to MP3 (same logic as working script)
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output files
                '-f', 's16le',  # Input format: 16-bit little-endian PCM
                '-ar', str(self.input_sample_rate),  # Input sample rate
                '-ac', str(self.channels),  # Input channels
                '-i', 'pipe:0',  # Read from stdin
                '-c:a', 'libmp3lame',  # MP3 encoder
                '-b:a', '320k',  # 320 kbps
                '-ar', str(self.output_sample_rate),  # Output sample rate
                '-ac', '1',  # Mono output
                '-write_id3v1', '0',  # No ID3v1
                '-write_id3v2', '0',  # No ID3v2
                '-id3v2_version', '0',  # No ID3v2
                '-write_xing', '0',  # No Xing header
                '-fflags', '+bitexact',
                '-f', 'mp3',  # MP3 format
                'pipe:1'  # Output to stdout
            ]
            
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            mp3_data, error = process.communicate(input=pcm_data)
            
            if process.returncode == 0:
                return mp3_data
            else:
                print(f"[MP3Handler] FFmpeg error: {error.decode()}")
                return None
        
        except Exception as e:
            print(f"[MP3Handler] Error converting PCM to MP3: {e}")
            return None
    
    def reset(self) -> None:
        """
        Reset the handler state for a new session
        """
        self.pcm_buffer.clear()
        self.processed_samples = 0
        self.chunk_start_sample = 0
        self.silent_regions.clear()
        print("[MP3Handler] Handler reset for new session")
