import sys
sys.path.append('third_party/Matcha-TTS')

# Register vllm model
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

import torch
import numpy as np
import io
import asyncio
import json
import base64
import os
import glob
from typing import Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import time
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import subprocess
import threading
from queue import Queue

# Valid API Keys for WebSocket authentication
VALID_API_KEYS = {
    "sk-5z6y7x8w9v0u1t2s3r4q5p6o7n8m9l0k1j2i3h4g",
    "sk-3a4b5c6d7e8f9g0h1i2j3k4l5m6n7o8p9q0r1s2t",
    "sk-9m8n7b6v5c4x3z2a1s0d9f8g7h6j5k4l3p2o1i0u",
    "sk-7u6y5t4r3e2w1q0a9s8d7f6g5h4j3k2l1z0x9c8v"
}

# API model for TTS request
class TTSRequest(BaseModel):
    text: str
    speaker_id: Optional[int] = 0
    language: Optional[str] = "ZH"
    speed: Optional[float] = 1.0
    audio_format: Optional[str] = "mp3"  # wav or mp3
    sdp_ratio: Optional[float] = 0.2
    noise_scale: Optional[float] = 0.6
    noise_scale_w: Optional[float] = 0.8

# Initialize FastAPI app
app = FastAPI()

# Global variables to store model and prompt
global_cosyvoice = None
global_prompt_speech_16k = None
global_prompt_text = None
global_normalized_prompt_text = None

# Speaker cache for different speakerId values
speaker_cache = {}

def get_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def remove_silence_from_first_pcm_chunk(pcm_data, sample_rate):
    """Remove silence from the first PCM chunk to improve response time"""
    try:
        # Convert PCM bytes to AudioSegment (same logic as remove_silence_from_pcm_and_save_to_a_single_wav.py)
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
        
        # Create a temporary file-like object
        import io
        pcm_io = io.BytesIO(pcm_data)
        
        # Load PCM data as AudioSegment (same parameters as working script)
        audio = AudioSegment.from_raw(
            pcm_io,
            sample_width=2,     # 16-bit = 2 bytes
            frame_rate=sample_rate,
            channels=1          # mono
        )
        
        # Use the same successful parameters from remove_silence_from_pcm_and_save_to_a_single_wav.py
        min_silence_len = 200    # Minimum silence length in ms
        silence_thresh = -50     # Silence threshold in dBFS
        keep_silence = 100       # Amount of silence to keep around segments in ms
        
        print(f"[WS-TTS] Detecting non-silent segments in first chunk...")
        print(f"[WS-TTS] Parameters: min_silence_len={min_silence_len}ms, silence_thresh={silence_thresh}dBFS")
        
        # Detect non-silent segments
        nonsilent_segments = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        
        if not nonsilent_segments:
            print("[WS-TTS] No voice segments detected in first chunk, keeping original")
            return pcm_data
        
        print(f"[WS-TTS] Found {len(nonsilent_segments)} voice segments in first chunk")
        
        # Take the first voice segment (which should be the human voice)
        start, end = nonsilent_segments[0]
        
        # Add some silence padding if specified
        segment_start = max(0, start - keep_silence)
        segment_end = min(len(audio), end + keep_silence)
        
        # Extract the first voice segment
        trimmed_audio = audio[segment_start:segment_end]
        
        print(f"[WS-TTS] First voice segment: {start}ms to {end}ms (with padding: {segment_start}ms to {segment_end}ms)")
        print(f"[WS-TTS] Original duration: {len(audio)/1000:.2f}s, Trimmed duration: {len(trimmed_audio)/1000:.2f}s")
        
        # Convert back to PCM bytes
        trimmed_pcm_io = io.BytesIO()
        trimmed_audio.export(trimmed_pcm_io, format="raw")
        trimmed_pcm_data = trimmed_pcm_io.getvalue()
        
        print(f"[WS-TTS] First chunk silence removal: {len(pcm_data)} → {len(trimmed_pcm_data)} bytes")
        
        return trimmed_pcm_data
        
    except Exception as e:
        print(f"[WS-TTS] Error removing silence from first chunk: {e}")
        return pcm_data  # Return original data if error occurs

def generate_audio_chunks(text, request_start_time, audio_format):
    """Generate audio chunks for streaming (similar to vllm-demo.py)"""
    global global_cosyvoice, global_prompt_speech_16k, global_normalized_prompt_text
    
    try:
        # Time before inference
        pre_inference_time = time.time()
        pre_processing_duration = (pre_inference_time - request_start_time) * 1000
        print(f"[TTS] Pre-processing time: {pre_processing_duration:.2f}ms")
        
        # Run TTS inference with streaming
        inference_start_time = time.time()
        print(f"[TTS] Starting inference at: {time.strftime('%H:%M:%S.%f')[:-3]}")
        
        first_chunk_generated = False
        first_chunk_time = None
        chunk_count = 0
        
        # Pre-normalize input text with splitting enabled for faster first chunk
        text_norm_start = time.time()
        normalized_text_chunks = global_cosyvoice.frontend.text_normalize(text, split=True, text_frontend=False)
        text_norm_time = (time.time() - text_norm_start) * 1000
        print(f"[TTS] Text normalization time: {text_norm_time:.2f}ms, got {len(normalized_text_chunks)} chunks")
        
        # Process all text chunks to stay within TRT limits
        for chunk_idx, text_chunk in enumerate(normalized_text_chunks):
            print(f"[TTS] Processing text chunk {chunk_idx + 1}/{len(normalized_text_chunks)}: {text_chunk[:50]}...")
            
            # Use inference_zero_shot for each text chunk to maintain proper ordering
            for i, j in enumerate(global_cosyvoice.inference_zero_shot(
                text_chunk, 
                global_normalized_prompt_text, 
                global_prompt_speech_16k, 
                zero_shot_spk_id='cached_prompt_spk', 
                stream=True
            )):
                chunk_start_time = time.time()
                chunk_count += 1  # Keep incrementing chunk_count across all text chunks
                
                # Record first chunk timing (only for the very first chunk)
                if not first_chunk_generated:
                    first_chunk_time = (chunk_start_time - inference_start_time) * 1000
                    first_chunk_since_request = (chunk_start_time - request_start_time) * 1000
                    print(f"[TTS] First chunk generated time: {first_chunk_time:.2f}ms")
                    print(f"[TTS] First chunk generated since request arrival: {first_chunk_since_request:.2f}ms")
                    first_chunk_generated = True
                
                # Convert audio tensor to appropriate format for streaming
                if audio_format.lower() == "wav":
                    buffer = io.BytesIO()
                    torchaudio.save(buffer, j['tts_speech'], global_cosyvoice.sample_rate, format="wav")
                    audio_bytes = buffer.getvalue()
                    buffer.close()
                elif audio_format.lower() == "mp3":
                    # Convert to MP3 for each chunk
                    wav_buffer = io.BytesIO()
                    torchaudio.save(wav_buffer, j['tts_speech'], global_cosyvoice.sample_rate, format="wav")
                    wav_buffer.seek(0)
                    
                    # Convert to MP3
                    waveform, sample_rate = torchaudio.load(wav_buffer)
                    mp3_buffer = io.BytesIO()
                    torchaudio.save(mp3_buffer, waveform, sample_rate, format="mp3")
                    audio_bytes = mp3_buffer.getvalue()
                    wav_buffer.close()
                    mp3_buffer.close()
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported audio format: {audio_format}")
                
                chunk_processing_time = (time.time() - chunk_start_time) * 1000
                total_time_so_far = (time.time() - request_start_time) * 1000
                
                print(f"[TTS] Chunk {chunk_count} (text chunk {chunk_idx + 1}) processed in {chunk_processing_time:.2f}ms, total time: {total_time_so_far:.2f}ms")
                
                # Yield each chunk to client immediately
                yield audio_bytes
        
    except Exception as e:
        error_time = (time.time() - request_start_time) * 1000
        print(f"[TTS] Error after {error_time:.2f}ms: {str(e)}")
        yield b"Error: TTS generation failed"

def load_and_cache_speaker(speaker_id):
    """Load and cache speaker data for given speakerId"""
    global speaker_cache
    
    if speaker_id in speaker_cache:
        print(f"[Speaker Cache] Using cached speaker {speaker_id}")
        return speaker_cache[speaker_id]
    
    print(f"[Speaker Cache] Loading new speaker {speaker_id}")
    
    # Load speaker audio file from speaker-specific folder
    speaker_wav_path = f'./asset/speakerId-{speaker_id}/speakerId-{speaker_id}.wav'
    speaker_txt_path = f'./asset/speakerId-{speaker_id}/speakerId-{speaker_id}.txt'
    
    try:
        prompt_speech_16k = load_wav(speaker_wav_path, 16000)
        
        # Read prompt text from corresponding txt file
        with open(speaker_txt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        
        print(f"[Speaker Cache] Loaded prompt text for speaker {speaker_id}: {prompt_text[:50]}...")
        
        # Pre-normalize the prompt text
        normalized_prompt_text = global_cosyvoice.frontend.text_normalize(prompt_text, split=False, text_frontend=True)
        
        # Pre-compute and cache speaker in CosyVoice
        cache_key = f'cached_prompt_spk_{speaker_id}'
        global_cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, cache_key)
        
        # Cache speaker data
        speaker_data = {
            'prompt_speech_16k': prompt_speech_16k,
            'prompt_text': prompt_text,
            'normalized_prompt_text': normalized_prompt_text,
            'cache_key': cache_key
        }
        
        speaker_cache[speaker_id] = speaker_data
        print(f"[Speaker Cache] Cached speaker {speaker_id} with key {cache_key}")
        
        return speaker_data
        
    except Exception as e:
        print(f"[Speaker Cache] Error loading speaker {speaker_id}: {e}")
        # Fallback to default speaker (speakerId 1)
        if speaker_id != 1:
            print(f"[Speaker Cache] Falling back to default speaker 1")
            return load_and_cache_speaker(1)
        else:
            raise e

@app.get("/")
async def root():
    return {"message": "CosyVoice API is running"}

@app.websocket("/cosy-tts")
async def websocket_tts(websocket: WebSocket):
    """
    WebSocket TTS endpoint that streams audio chunks as they're generated
    """
    await websocket.accept()
    print(f"[WS-TTS] WebSocket connection established")
    
    # Check X-API-Key header for authorization
    api_key = websocket.headers.get("x-api-key")
    if not api_key or api_key not in VALID_API_KEYS:
        print(f"[WS-TTS] Unauthorized access attempt with API key: {api_key}")
        await websocket.send_text(json.dumps({"error": "Unauthorized: Invalid or missing X-API-Key header"}))
        await websocket.close(code=4001, reason="Unauthorized")
        return
    
    print(f"[WS-TTS] Authorized connection with valid API key")
    
    try:
        while True:
            # Receive TTS request from client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Start timing
            start_time = time.time()
            print(f"WS-TTS start time: {time.strftime('%H:%M:%S.%f')[:-3]}")
            
            # Extract parameters with defaults
            text = request_data.get("text", "")
            speaker_id = request_data.get("speakerId", 1)  # Default to speakerId 1
            save_audio_files = request_data.get("saveAudioFiles", False)
            output_sample_rate = request_data.get("outputSampleRate", 22050)  # Default 22050 Hz (CosyVoice native)
            audio_format = request_data.get("audioFormat", "mp3")  # Default to mp3, can be "mp3" or "pcm"
            
            if not text:
                await websocket.send_text(json.dumps({"error": "Text is required"}))
                continue
            
            if global_cosyvoice is None:
                await websocket.send_text(json.dumps({"error": "Model not initialized"}))
                continue
            
            # Load and cache speaker data
            try:
                speaker_data = load_and_cache_speaker(speaker_id)
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"Failed to load speaker {speaker_id}: {str(e)}"}))
                continue
            
            # Setup folder for saving audio chunks if requested
            chunk_save_folder = None
            chunk_file_counter = 0
            if save_audio_files:
                chunk_save_folder = os.path.join(os.path.dirname(__file__), "cosy_mp3_chunks")
                os.makedirs(chunk_save_folder, exist_ok=True)
                print(f"[WS-TTS] Created/verified chunk save folder: {chunk_save_folder}")
            
            # Initialize buffer for accumulating audio data
            if audio_format.lower() == "pcm":
                # For PCM format, no conversion needed
                pcm_buffer = bytearray()
                pcm_chunk_counter = 0
                is_first_chunk = True  # Track first chunk for silence removal
                # Calculate target PCM bytes per chunk (approximate)
                segment_duration = 1.0  # 1 second chunks
                target_pcm_bytes_per_chunk = int(output_sample_rate * 2 * segment_duration)  # 2 bytes per sample
                print(f"[WS-TTS] Audio format: PCM, target PCM bytes per chunk: {target_pcm_bytes_per_chunk}")
            else:
                # For MP3 format (default), use existing logic
                pcm_buffer = bytearray()
                mp3_chunk_counter = 0
                # Calculate target PCM bytes per MP3 chunk (approximate)
                segment_duration = 1.0  # 1 second chunks
                target_pcm_bytes_per_chunk = int(output_sample_rate * 2 * segment_duration)  # 2 bytes per sample
                print(f"[WS-TTS] Audio format: MP3, target PCM bytes per chunk: {target_pcm_bytes_per_chunk}")
            
            print(f"[WS-TTS] CosyVoice native sample rate: {global_cosyvoice.sample_rate} Hz")
            print(f"[WS-TTS] Output sample rate: {output_sample_rate} Hz")
            if global_cosyvoice.sample_rate == output_sample_rate:
                print(f"[WS-TTS] ✓ No resampling needed - rates match (optimal performance)")
            else:
                print(f"[WS-TTS] ⚠ Resampling will be applied: {global_cosyvoice.sample_rate} Hz → {output_sample_rate} Hz")
            
            # Generate audio
            print(f"Generating audio for text: {text[:50]}{'...' if len(text) > 50 else ''}")
            if save_audio_files:
                print(f"[WS-TTS] Will save {audio_format.upper()} chunks to files")

            # Helper function to convert PCM data to MP3 chunk (only used for MP3 format)
            async def convert_pcm_to_mp3_chunk(pcm_data, sample_rate):
                """Convert PCM data to MP3 chunk using ffmpeg and trim padding"""
                try:
                    # Use ffmpeg to convert PCM to MP3
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output files
                        '-f', 's16le',  # Input format: 16-bit little-endian PCM
                        '-ar', str(sample_rate),  # Input sample rate
                        '-ac', '1',  # Mono
                        '-i', 'pipe:0',  # Read from stdin
                        '-c:a', 'libmp3lame',  # MP3 encoder
                        '-b:a', '320k',  # 320 kbps
                        '-ar', str(output_sample_rate),  # Output sample rate
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
                    
                    if process.returncode != 0:
                        print(f"[WS-TTS] FFmpeg error: {error.decode()}")
                        return None
                    
                    # Trim padding from the end of MP3 chunk
                    trimmed_mp3_data = trim_mp3_padding(mp3_data)
                    
                    return trimmed_mp3_data
                except Exception as e:
                    print(f"[WS-TTS] Error converting PCM to MP3: {e}")
                    return None

            def trim_mp3_padding(mp3_data):
                """Remove padding bytes from the end of MP3 chunk to ensure clean frame boundaries"""
                if len(mp3_data) < 4:
                    return mp3_data
                
                # Convert to bytearray for easier manipulation
                data = bytearray(mp3_data)
                original_length = len(data)
                
                # Look for repetitive padding patterns at the end
                # Common MP3 padding patterns: 0x55, 0xAA, 0x00, etc.
                padding_patterns = [0x55, 0xAA, 0x00]
                
                # Find the last non-padding byte
                end_pos = len(data)
                
                for pattern in padding_patterns:
                    # Check if we have repetitive padding pattern at the end
                    consecutive_count = 0
                    pos = len(data) - 1
                    
                    # Count consecutive padding bytes from the end
                    while pos >= 0 and data[pos] == pattern:
                        consecutive_count += 1
                        pos -= 1
                    
                    # If we found significant padding (more than 16 consecutive bytes)
                    if consecutive_count > 16:
                        potential_end = pos + 1
                        if potential_end < end_pos:
                            end_pos = potential_end
                            print(f"[WS-TTS] Detected {consecutive_count} bytes of 0x{pattern:02X} padding, trimming to position {end_pos}")
                
                # Additional check: look for MP3 frame sync patterns to avoid cutting in the middle of frames
                # MP3 frame sync is 0xFFF (first 11 bits), so we look for 0xFF followed by 0xF*
                if end_pos < original_length:
                    # Try to align to the last valid MP3 frame boundary
                    for i in range(end_pos - 1, max(0, end_pos - 100), -1):  # Look back up to 100 bytes
                        if i + 1 < len(data) and data[i] == 0xFF and (data[i + 1] & 0xF0) == 0xF0:
                            # Found potential MP3 frame sync, this might be a better cut point
                            # Look for the end of this frame
                            frame_start = i
                            # MP3 frame header is 4 bytes, try to find frame length
                            if frame_start + 4 <= len(data):
                                # For now, just cut here as it's a frame boundary
                                end_pos = min(end_pos, frame_start + 4)
                                print(f"[WS-TTS] Aligned to MP3 frame boundary at position {end_pos}")
                                break
                
                # Trim the data
                trimmed_data = bytes(data[:end_pos])
                
                if end_pos < original_length:
                    bytes_removed = original_length - end_pos
                    print(f"[WS-TTS] Trimmed {bytes_removed} padding bytes from MP3 chunk ({original_length} -> {end_pos} bytes)")
                
                return trimmed_data

            before_inference_time = time.time()
            elapsed_since_start = (before_inference_time - start_time) * 1000
            print(f"Time before inference: {elapsed_since_start:.2f} ms since start")
            
            try:
                # === DETAILED PERFORMANCE LOGGING ===
                # Time before inference
                pre_inference_time = time.time()
                pre_processing_duration = (pre_inference_time - start_time) * 1000
                print(f"[WS-TTS] 📊 Pre-processing time: {pre_processing_duration:.2f}ms")
                
                # Run TTS inference with streaming
                inference_start_time = time.time()
                print(f"[WS-TTS] 🚀 Starting inference at: {time.strftime('%H:%M:%S.%f')[:-3]}")
                
                first_chunk_generated = False
                first_chunk_sent = False
                chunk_count = 0
                
                # === TEXT NORMALIZATION TIMING ===
                text_norm_start = time.time()
                normalized_text_chunks = global_cosyvoice.frontend.text_normalize(text, split=True, text_frontend=False)
                text_norm_time = (time.time() - text_norm_start) * 1000
                print(f"[WS-TTS] 📝 Text normalization time: {text_norm_time:.2f}ms, got {len(normalized_text_chunks)} chunks")
                
                # Process all text chunks to stay within TRT limits
                for chunk_idx, text_chunk in enumerate(normalized_text_chunks):
                    print(f"[WS-TTS] 🔄 Processing text chunk {chunk_idx + 1}/{len(normalized_text_chunks)}: {text_chunk[:50]}...")
                    
                    # === ZERO-SHOT INFERENCE TIMING ===
                    zeroshot_start = time.time()
                    
                    # Use inference_zero_shot for each text chunk to maintain proper ordering
                    for i, j in enumerate(global_cosyvoice.inference_zero_shot(
                        text_chunk, 
                        speaker_data['normalized_prompt_text'], 
                        speaker_data['prompt_speech_16k'], 
                        zero_shot_spk_id=speaker_data['cache_key'], 
                        stream=True
                    )):
                        chunk_start_time = time.time()
                        chunk_count += 1  # Keep incrementing chunk_count across all text chunks
                        
                        # Record first chunk timing (only for the very first chunk)
                        if not first_chunk_generated:
                            first_chunk_time = (chunk_start_time - inference_start_time) * 1000
                            first_chunk_since_request = (chunk_start_time - start_time) * 1000
                            zeroshot_time = (chunk_start_time - zeroshot_start) * 1000
                            print(f"[WS-TTS] ⚡ First chunk generated time: {first_chunk_time:.2f}ms")
                            print(f"[WS-TTS] ⚡ First chunk since request arrival: {first_chunk_since_request:.2f}ms")
                            print(f"[WS-TTS] ⚡ Zero-shot inference time: {zeroshot_time:.2f}ms")
                            first_chunk_generated = True
                        
                        # === AUDIO TO PCM CONVERSION ===
                        audio_convert_start = time.time()
                        
                        # Convert audio tensor to PCM data (16-bit signed integers)
                        audio_tensor = j['tts_speech']
                        
                        # Resample to target output sample rate if needed
                        if global_cosyvoice.sample_rate != output_sample_rate:
                            resample_start = time.time()
                            audio_tensor = torchaudio.functional.resample(
                                audio_tensor, 
                                global_cosyvoice.sample_rate, 
                                output_sample_rate
                            )
                            resample_time = (time.time() - resample_start) * 1000
                            print(f"[WS-TTS] 🔄 Resampled chunk {chunk_count}: {resample_time:.2f}ms ({global_cosyvoice.sample_rate} → {output_sample_rate} Hz)")
                        else:
                            print(f"[WS-TTS] ✓ Chunk {chunk_count}: No resampling needed (optimal)")
                        
                        # Convert to 16-bit PCM
                        audio_np = audio_tensor.numpy()
                        # Normalize to [-1, 1] range if needed
                        if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                            audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))
                        # Convert to 16-bit PCM
                        pcm_data = (audio_np * 32767).astype(np.int16).tobytes()
                        
                        # Apply silence removal to first chunk if PCM format
                        if audio_format.lower() == "pcm" and is_first_chunk:
                            silence_removal_start = time.time()
                            # Replace pcm_data with silence-removed version - only this processed data goes to clients
                            pcm_data = remove_silence_from_first_pcm_chunk(pcm_data, output_sample_rate)
                            silence_removal_time = (time.time() - silence_removal_start) * 1000
                            print(f"[WS-TTS] 🎯 First chunk silence removal time: {silence_removal_time:.2f}ms")
                            print(f"[WS-TTS] 🎯 First chunk: sending only silence-removed PCM data to clients")
                            is_first_chunk = False  # Mark that we've processed the first chunk
                        
                        # Add PCM data to buffer (for first chunk, this is the silence-removed data)
                        pcm_buffer.extend(pcm_data)
                        
                        audio_convert_time = (time.time() - audio_convert_start) * 1000
                        print(f"[WS-TTS] 🎵 Audio to PCM conversion time: {audio_convert_time:.2f}ms, PCM buffer size: {len(pcm_buffer)} bytes")
                        
                        # === PROCESS PCM BUFFER INTO CHUNKS ===
                        if audio_format.lower() == "pcm":
                            # For PCM format, send raw PCM data directly
                            while len(pcm_buffer) >= target_pcm_bytes_per_chunk:
                                send_start = time.time()
                                
                                # Extract PCM chunk from buffer
                                pcm_chunk = bytes(pcm_buffer[:target_pcm_bytes_per_chunk])
                                pcm_buffer = pcm_buffer[target_pcm_bytes_per_chunk:]
                                
                                # Send PCM chunk immediately
                                await websocket.send_bytes(pcm_chunk)
                                send_time = (time.time() - send_start) * 1000
                                
                                print(f"[WS-TTS] 📦 PCM chunk {pcm_chunk_counter} sent: {len(pcm_chunk)} bytes, send time: {send_time:.2f}ms")
                                
                                # Save PCM chunk to file if requested (silence already removed)
                                if save_audio_files and chunk_save_folder:
                                    chunk_filename = f"chunk_{pcm_chunk_counter}.pcm"
                                    chunk_filepath = os.path.join(chunk_save_folder, chunk_filename)
                                    try:
                                        with open(chunk_filepath, 'wb') as f:
                                            f.write(pcm_chunk)
                                        print(f"[WS-TTS] Saved {chunk_filename} ({len(pcm_chunk)} bytes)")
                                    except Exception as save_error:
                                        print(f"[WS-TTS] Error saving chunk file: {save_error}")
                                
                                pcm_chunk_counter += 1
                                
                                # Track first chunk sent timing
                                if not first_chunk_sent:
                                    first_chunk_sent_time = time.time()
                                    elapsed_since_start = (first_chunk_sent_time - start_time) * 1000
                                    elapsed_since_before = (first_chunk_sent_time - before_inference_time) * 1000
                                    print(f"[WS-TTS] 🎯 Time to send first PCM chunk: {elapsed_since_start:.2f} ms since start, {elapsed_since_before:.2f} ms since before inference")
                                    first_chunk_sent = True
                                
                                await asyncio.sleep(0)  # Allow other tasks
                        else:
                            # For MP3 format, use existing MP3 conversion logic
                            while len(pcm_buffer) >= target_pcm_bytes_per_chunk:
                                mp3_convert_start = time.time()
                                
                                # Extract PCM chunk from buffer
                                pcm_chunk = bytes(pcm_buffer[:target_pcm_bytes_per_chunk])
                                pcm_buffer = pcm_buffer[target_pcm_bytes_per_chunk:]
                                
                                # Convert PCM chunk to MP3
                                mp3_data = await convert_pcm_to_mp3_chunk(pcm_chunk, output_sample_rate)
                                
                                if mp3_data:
                                    # Send MP3 chunk immediately
                                    send_start = time.time()
                                    await websocket.send_bytes(mp3_data)
                                    send_time = (time.time() - send_start) * 1000
                                    
                                    mp3_convert_time = (time.time() - mp3_convert_start) * 1000
                                    
                                    print(f"[WS-TTS] 📦 MP3 chunk {mp3_chunk_counter} sent: {len(mp3_data)} bytes, convert time: {mp3_convert_time:.2f}ms, send time: {send_time:.2f}ms")
                                    
                                    # Save MP3 chunk to file if requested
                                    if save_audio_files and chunk_save_folder:
                                        chunk_filename = f"chunk_{mp3_chunk_counter}.mp3"
                                        chunk_filepath = os.path.join(chunk_save_folder, chunk_filename)
                                        try:
                                            with open(chunk_filepath, 'wb') as f:
                                                f.write(mp3_data)
                                            print(f"[WS-TTS] Saved {chunk_filename} ({len(mp3_data)} bytes)")
                                        except Exception as save_error:
                                            print(f"[WS-TTS] Error saving chunk file: {save_error}")
                                    
                                    mp3_chunk_counter += 1
                                    
                                    # Track first chunk sent timing
                                    if not first_chunk_sent:
                                        first_chunk_sent_time = time.time()
                                        elapsed_since_start = (first_chunk_sent_time - start_time) * 1000
                                        elapsed_since_before = (first_chunk_sent_time - before_inference_time) * 1000
                                        print(f"[WS-TTS] 🎯 Time to send first MP3 chunk: {elapsed_since_start:.2f} ms since start, {elapsed_since_before:.2f} ms since before inference")
                                        first_chunk_sent = True
                                    
                                    await asyncio.sleep(0)  # Allow other tasks
                                else:
                                    print(f"[WS-TTS] Failed to convert PCM chunk to MP3")
                        
                        chunk_processing_time = (time.time() - chunk_start_time) * 1000
                        total_time_so_far = (time.time() - start_time) * 1000
                        
                        print(f"[WS-TTS] 📊 Audio chunk {chunk_count} (text chunk {chunk_idx + 1}) processed in {chunk_processing_time:.2f}ms, total time: {total_time_so_far:.2f}ms")
                
                # === PROCESS REMAINING PCM DATA ===
                if len(pcm_buffer) > 0:
                    print(f"[WS-TTS] 🔄 Processing remaining PCM data: {len(pcm_buffer)} bytes")
                    remaining_start = time.time()
                    
                    if audio_format.lower() == "pcm":
                        # For PCM format, send remaining PCM data directly
                        remaining_pcm_data = bytes(pcm_buffer)
                        await websocket.send_bytes(remaining_pcm_data)
                        
                        remaining_time = (time.time() - remaining_start) * 1000
                        print(f"[WS-TTS] 📦 Final PCM chunk {pcm_chunk_counter} sent: {len(remaining_pcm_data)} bytes, process time: {remaining_time:.2f}ms")
                        
                        # Save final PCM chunk to file if requested
                        if save_audio_files and chunk_save_folder:
                            chunk_filename = f"chunk_{pcm_chunk_counter}.pcm"
                            chunk_filepath = os.path.join(chunk_save_folder, chunk_filename)
                            try:
                                with open(chunk_filepath, 'wb') as f:
                                    f.write(remaining_pcm_data)
                                print(f"[WS-TTS] Saved final {chunk_filename} ({len(remaining_pcm_data)} bytes)")
                            except Exception as save_error:
                                print(f"[WS-TTS] Error saving final chunk file: {save_error}")
                        
                        pcm_chunk_counter += 1
                    else:
                        # For MP3 format, convert remaining PCM data to MP3
                        remaining_pcm_data = bytes(pcm_buffer)
                        mp3_data = await convert_pcm_to_mp3_chunk(remaining_pcm_data, output_sample_rate)
                        
                        if mp3_data:
                            # Send final MP3 chunk
                            await websocket.send_bytes(mp3_data)
                            
                            remaining_time = (time.time() - remaining_start) * 1000
                            print(f"[WS-TTS] 📦 Final MP3 chunk {mp3_chunk_counter} sent: {len(mp3_data)} bytes, process time: {remaining_time:.2f}ms")
                            
                            # Save final MP3 chunk to file if requested
                            if save_audio_files and chunk_save_folder:
                                chunk_filename = f"chunk_{mp3_chunk_counter}.mp3"
                                chunk_filepath = os.path.join(chunk_save_folder, chunk_filename)
                                try:
                                    with open(chunk_filepath, 'wb') as f:
                                        f.write(mp3_data)
                                    print(f"[WS-TTS] Saved final {chunk_filename} ({len(mp3_data)} bytes)")
                                except Exception as save_error:
                                    print(f"[WS-TTS] Error saving final chunk file: {save_error}")
                            
                            mp3_chunk_counter += 1
                        else:
                            print(f"[WS-TTS] Failed to convert remaining PCM data to MP3")
                    
                    # Clear buffer
                    pcm_buffer.clear()
                
                # Log completion timing
                after_inference_time = time.time()
                elapsed_since_start = (after_inference_time - start_time) * 1000
                elapsed_since_before = (after_inference_time - before_inference_time) * 1000
                print(f"[WS-TTS] ✅ Time after inference: {elapsed_since_start:.2f} ms since start, {elapsed_since_before:.2f} ms since before inference")
                
                generation_time = time.time() - start_time
                print(f"[WS-TTS] 🏁 Audio generated in {generation_time:.2f} seconds")
                
                # === REQUEST SUMMARY ===
                print(f"[WS-TTS] 📋 REQUEST SUMMARY:")
                print(f"[WS-TTS] 📋   Audio Format: {audio_format}")
                print(f"[WS-TTS] 📋   Sample Rate: {output_sample_rate} Hz")
                print(f"[WS-TTS] 📋   Speaker ID: {speaker_id}")
                print(f"[WS-TTS] 📋   Total Generation Time: {generation_time:.2f}s")
                
                if audio_format.lower() == "pcm":
                    print(f"[WS-TTS] 📊 Total PCM chunks sent: {pcm_chunk_counter}")
                    if save_audio_files:
                        print(f"[WS-TTS] Saved {pcm_chunk_counter} PCM chunk files to {chunk_save_folder}")
                else:
                    print(f"[WS-TTS] 📊 Total MP3 chunks sent: {mp3_chunk_counter}")
                    if save_audio_files:
                        print(f"[WS-TTS] Saved {mp3_chunk_counter} MP3 chunk files to {chunk_save_folder}")
                
            except Exception as inference_error:
                print(f"Inference error: {str(inference_error)}")
                print(f"Error type: {type(inference_error)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                await websocket.send_text(json.dumps({"error": f"TTS generation failed: {str(inference_error)}"}))
            
    except WebSocketDisconnect:
        print(f"[WS-TTS] WebSocket connection disconnected")
    except Exception as e:
        print(f"[WS-TTS] WebSocket error: {str(e)}")
        try:
            await websocket.send_text(json.dumps({"error": f"WebSocket error: {str(e)}"}))
        except:
            pass

@app.post("/tts")
async def generate_tts(request: TTSRequest):
    """
    TTS API endpoint that streams audio chunks as they're generated
    """
    global global_cosyvoice, global_prompt_speech_16k
    
    # Start timing
    request_start_time = time.time()
    print(f"[TTS] Request received at: {time.strftime('%H:%M:%S.%f')[:-3]}")
    
    if global_cosyvoice is None or global_prompt_speech_16k is None:
        return {"error": "Model not initialized"}
    
    # Determine media type based on format
    if request.audio_format.lower() == "wav":
        media_type = "audio/wav"
        filename = "tts_stream.wav"
    elif request.audio_format.lower() == "mp3":
        media_type = "audio/mpeg"
        filename = "tts_stream.mp3"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {request.audio_format}")
    
    # Return streaming response
    return StreamingResponse(
        generate_audio_chunks(request.text, request_start_time, request.audio_format),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    global global_cosyvoice, global_prompt_speech_16k, global_prompt_text, global_normalized_prompt_text
    
    device = get_device()
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    
    print("Loading CosyVoice2 model with vllm...")
    # Initialize CosyVoice2 with vllm enabled
    global_cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)
    
    # Add this debug line to check the actual sample rate
    print(f"🔍 DEBUG: CosyVoice2-0.5B reported sample rate: {global_cosyvoice.sample_rate} Hz")
    
    # Load prompt speech for default speaker (speakerId 1) from folder structure
    global_prompt_speech_16k = load_wav('./asset/speakerId-1/speakerId-1.wav', 16000)
    
    # Read prompt text from speakerId-1.txt
    with open('./asset/speakerId-1/speakerId-1.txt', 'r', encoding='utf-8') as f:
        global_prompt_text = f.read().strip()
    
    print(f"[Startup] Loaded default prompt text: {global_prompt_text[:50]}...")
    
    # Pre-normalize the prompt text and cache default speaker
    global_normalized_prompt_text = global_cosyvoice.frontend.text_normalize(global_prompt_text, split=False, text_frontend=True)
    global_cosyvoice.add_zero_shot_spk(global_prompt_text, global_prompt_speech_16k, 'cached_prompt_spk')
    
    # Cache default speaker (speakerId 1) for consistency
    speaker_cache[1] = {
        'prompt_speech_16k': global_prompt_speech_16k,
        'prompt_text': global_prompt_text,
        'normalized_prompt_text': global_normalized_prompt_text,
        'cache_key': 'cached_prompt_spk'
    }
    print(f"[Speaker Cache] Default speaker (ID 1) cached")
    
    # Pre-warm the model with a dummy inference AND pre-normalize prompt text
    print("Pre-warming model...")
    warmup_start = time.time()
    try:
        # Pre-normalize the prompt text to avoid repeated processing (same as vllm-demo.py)
        global_normalized_prompt_text = global_cosyvoice.frontend.text_normalize(global_prompt_text, split=False, text_frontend=True)
        
        # Pre-compute prompt audio processing and cache it as a speaker (same as vllm-demo.py)
        print("Pre-computing prompt audio processing...")
        cache_start = time.time()
        global_cosyvoice.add_zero_shot_spk(global_prompt_text, global_prompt_speech_16k, 'cached_prompt_spk')
        cache_time = (time.time() - cache_start) * 1000
        print(f"Prompt audio cached in {cache_time:.2f}ms")
        
        # Enhanced warm-up: Run multiple inference cycles to fully initialize VLLM
        print("Enhanced VLLM warm-up...")
        warmup_texts = [
            "Hello world",
            "青石板上泛着水光，雨丝斜斜地织着帘子。",
            "Did you guys see the video of that dude who was at the gym who took his earbuds and just smacked them against the wall because they would not stay in his ear during his set?",
            "This is a longer text to warm up the VLLM engine with various text lengths and patterns for better first chunk performance."
        ]
        
        for idx, warmup_text in enumerate(warmup_texts):
            print(f"VLLM warm-up cycle {idx + 1}/{len(warmup_texts)}: {warmup_text[:30]}...")
            cycle_start = time.time()
            chunk_count = 0
            for i, j in enumerate(global_cosyvoice.inference_zero_shot(
                warmup_text, 
                global_normalized_prompt_text, 
                global_prompt_speech_16k, 
                zero_shot_spk_id='cached_prompt_spk', 
                stream=True
            )):
                chunk_count += 1
                if chunk_count >= 2:  # Process a few chunks to fully warm up
                    break
            cycle_time = (time.time() - cycle_start) * 1000
            print(f"Warm-up cycle {idx + 1} completed in {cycle_time:.2f}ms, generated {chunk_count} chunks")
        
        warmup_time = (time.time() - warmup_start) * 1000
        print(f"Enhanced model pre-warming completed in {warmup_time:.2f}ms")
    except Exception as e:
        print(f"Warmup failed: {e}")
    
    print("CosyVoice model loaded successfully!")

    # === GPU/CUDA/CUDNN Acceleration Diagnostics ===
    print("\n" + "="*50)
    print("🚀 GPU ACCELERATION STATUS")
    print("="*50)
    
    # CUDA/CUDNN Info
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ CUDNN Version: {torch.backends.cudnn.version()}")
    print(f"✅ CUDNN Enabled: {torch.backends.cudnn.enabled}")
    if torch.cuda.is_available():
        print(f"✅ GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # ONNX Runtime Providers
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"✅ ONNX Providers: {providers}")
    if 'CUDAExecutionProvider' in providers:
        print("✅ ONNX CUDA Acceleration: ENABLED")
    else:
        print("❌ ONNX CUDA Acceleration: DISABLED")
    
    # VLLM Status
    if hasattr(global_cosyvoice.model.llm, 'vllm'):
        print("✅ VLLM Acceleration: ENABLED")
        print("✅ VLLM GPU Memory Utilization: Configured")
    else:
        print("❌ VLLM Acceleration: DISABLED")
    
    print("="*50)
    print("🎯 Ready for high-performance TTS inference!")
    print("="*50 + "\n")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9003)
