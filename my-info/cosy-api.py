"""
CosyVoice API Server - Text-to-Speech WebSocket & REST API

USAGE INSTRUCTIONS:
==================

1. PERMISSION SETTINGS:
   - Ensure the script has execute permissions:
     chmod +x cosy-api.py

2. PACKAGE INSTALLATION:
   - Install required dependencies:
     pip install torch torchaudio
     pip install fastapi uvicorn websockets
     pip install pydub
     pip install onnxruntime-gpu  # or onnxruntime for CPU
     pip install vllm
   
   - Ensure ffmpeg is installed for MP3 conversion:
     # Ubuntu/Debian:
     sudo apt-get install ffmpeg
     # macOS:
     brew install ffmpeg
     # Windows: Download from https://ffmpeg.org/

3. MODEL SETUP:
   - Ensure CosyVoice2 model is downloaded to: pretrained_models/CosyVoice2-0.5B/
   - Ensure speaker files are in: ./asset/speakerId-{ID}/speakerId-{ID}.wav and .txt

4. USAGE EXAMPLES:
   - Run with default port 9003:
     python cosy-api.py
   
   - Run with custom port 8080:
     python cosy-api.py --port 8080
   
   - Run with custom port 9004:
     python cosy-api.py --port 9004
   
   - Get help:
     python cosy-api.py --help

5. API ENDPOINTS:
   - WebSocket TTS: ws://localhost:{port}/cosy-tts
   - REST TTS: POST http://localhost:{port}/tts
   - Health check: GET http://localhost:{port}/

6. AUTHENTICATION:
   - WebSocket requires X-API-Key header with valid API key
   - See VALID_API_KEYS in code for accepted keys

7. SUPPORTED AUDIO FORMATS:
   - PCM (raw audio data)
   - MP3 (compressed audio)
   - WAV (REST API only)
"""

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
import argparse
from typing import Optional, List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Header
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import time
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import subprocess
import threading
from queue import Queue
import requests  # Add this import for HTTP requests
from collections import OrderedDict  # Add this import for LRU cache

# Import MP3 chunks handler - Added December 19, 2024
from mp3_chunks_handler import MP3ChunksHandler

# Configuration - Default Speaker IDs (configurable array)
DEFAULT_SPEAKER_IDS = [3, 2, 5, 8]  # Configurable array for default speakers

# Valid API Keys for WebSocket authentication
VALID_API_KEYS = {
    "sk-5z6y7x8w9v0u1t2s3r4q5p6o7n8m9l0k1j2i3h4g",
    "sk-3a4b5c6d7e8f9g0h1i2j3k4l5m6n7o8p9q0r1s2t",
    "sk-9m8n7b6v5c4x3z2a1s0d9f8g7h6j5k4l3p2o1i0u",
    "sk-7u6y5t4r3e2w1q0a9s8d7f6g5h4j3k2l1z0x9c8v"
}

# Add configurable domain at the top of the file
TTS_CLONE_DOMAIN = "http://tts-clone.watchfun.cn"  # Configurable domain

# Configurable memory settings
MAX_CACHED_SPEAKERS = 100  # Maximum number of speakers to keep in memory cache

# MP3 Chunks Configuration - Updated January 23, 2025
MP3_INPUT_PCM_SAMPLE_RATE = 24000  # Input PCM sample rate for MP3 chunking (Hz) - Updated to match working config
MP3_VOLUME_DB = -40.0  # Volume threshold for silence detection (dB)
MP3_MIN_SAMPLES_WINDOW = 960  # Minimum samples window for silence detection
MP3_OUTPUT_SAMPLE_RATE = 24000  # Output MP3 sample rate (Hz) - Updated to match working config

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

# Global cache for speaker data with LRU functionality
speaker_cache = OrderedDict()  # Changed from dict to OrderedDict for LRU support

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
        first_chunk_sent = False
        first_chunk_sent_since_request = None  # Track first chunk sent timing
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
    """Load and cache speaker data for given speakerId (supports both integer and 32-bit MD5 string)"""
    global speaker_cache
    
    # Convert speaker_id to string for consistent handling
    speaker_id_str = str(speaker_id)
    
    # Check if already cached - move to end (most recently used)
    if speaker_id_str in speaker_cache:
        speaker_cache.move_to_end(speaker_id_str)
        return speaker_cache[speaker_id_str]
    
    # Validate speaker_id type
    if isinstance(speaker_id, int) or (isinstance(speaker_id, str) and speaker_id.isdigit()):
        # Integer type speakerId - use existing logic
        speaker_type = "integer"
        actual_speaker_id = int(speaker_id)
        cache_key = actual_speaker_id
    elif isinstance(speaker_id, str) and len(speaker_id) == 32 and all(c in "0123456789abcdef" for c in speaker_id.lower()):
        # 32-bit MD5 string type speakerId - use new logic
        speaker_type = "md5"
        actual_speaker_id = speaker_id.lower()
        cache_key = actual_speaker_id
    else:
        # Invalid type
        raise ValueError({
            "errorCode": 400,
            "message": f"Invalid speaker_id format. Must be integer or 32-character MD5 string, got: {speaker_id}"
        })
    
    print(f"[Speaker Cache] Loading new speaker {cache_key} (type: {speaker_type})")
    
    try:
        if speaker_type == "integer":
            # Existing logic for integer speakerId
            speaker_wav_path = f'./asset/speakerId-{actual_speaker_id}/speakerId-{actual_speaker_id}.wav'
            speaker_txt_path = f'./asset/speakerId-{actual_speaker_id}/speakerId-{actual_speaker_id}.txt'
            
            prompt_speech_16k = load_wav(speaker_wav_path, 16000)
            
            # Read prompt text from corresponding txt file
            with open(speaker_txt_path, 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip()
            
            print(f"[Speaker Cache] Loaded prompt text for speaker {actual_speaker_id}: {prompt_text[:50]}...")
            
        elif speaker_type == "md5":
            # New logic for MD5 string speakerId
            try:
                # Get speaker info from TTS clone service
                speaker_info_url = f"{TTS_CLONE_DOMAIN}/getSpeakerInfo/{actual_speaker_id}"
                print(f"[Speaker Cache] Fetching speaker info from: {speaker_info_url}")
                
                response = requests.get(
                    speaker_info_url, 
                    headers={"x-api-key": "sk-5z6y7x8w9v0u1t2s3r4q5p6o7n8m9l0k1j2i3h4g"},
                    timeout=10
                )
                response.raise_for_status()
                
                speaker_info = response.json()
                
                # Check if response is successful
                if speaker_info.get("errorCode") != 0:
                    raise ValueError({
                        "errorCode": speaker_info.get("errorCode", 500),
                        "message": speaker_info.get("message", "Failed to get speaker info")
                    })
                
                # Get WAV URL and text from response
                wav_url = speaker_info.get("wavUrl")
                prompt_text = speaker_info.get("text")
                
                if not wav_url or not prompt_text:
                    raise ValueError({
                        "errorCode": 500,
                        "message": "Invalid speaker info response: missing wavUrl or text"
                    })
                
                print(f"[Speaker Cache] Got WAV URL: {wav_url}")
                print(f"[Speaker Cache] Got prompt text: {prompt_text[:50]}...")
                
                # Download WAV file
                print(f"[Speaker Cache] Downloading WAV file from: {wav_url}")
                wav_response = requests.get(wav_url, timeout=30)
                wav_response.raise_for_status()
                
                # Save WAV to temporary file and load
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                    temp_wav.write(wav_response.content)
                    temp_wav_path = temp_wav.name
                
                # Load WAV file
                prompt_speech_16k = load_wav(temp_wav_path, 16000)
                
                # Clean up temporary file
                os.unlink(temp_wav_path)
                
                print(f"[Speaker Cache] Successfully loaded WAV file for MD5 speaker {actual_speaker_id}")
                
            except requests.RequestException as e:
                raise ValueError({
                    "errorCode": 500,
                    "message": f"Failed to fetch speaker data from TTS clone service: {str(e)}"
                })
            except Exception as e:
                raise ValueError({
                    "errorCode": 500,
                    "message": f"Failed to process MD5 speaker data: {str(e)}"
                })
        
        # Pre-normalize the prompt text
        normalized_prompt_text = global_cosyvoice.frontend.text_normalize(prompt_text, split=False, text_frontend=True)
        
        # Pre-compute and cache speaker in CosyVoice
        cosyvoice_cache_key = f'cached_prompt_spk_{cache_key}'
        global_cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, cosyvoice_cache_key)
        
        # Cache speaker data
        speaker_data = {
            'prompt_speech_16k': prompt_speech_16k,
            'prompt_text': prompt_text,
            'normalized_prompt_text': normalized_prompt_text,
            'cache_key': cosyvoice_cache_key,
            'speaker_type': speaker_type,
            'original_speaker_id': speaker_id
        }
        
        # Cache the speaker data (add this before the existing return statement)
        speaker_cache[speaker_id_str] = speaker_data
        speaker_cache.move_to_end(speaker_id_str)  # Mark as most recently used
        
        # Check if cache size exceeds limit and remove oldest entries
        while len(speaker_cache) > MAX_CACHED_SPEAKERS:
            oldest_speaker_id = next(iter(speaker_cache))  # Get first (oldest) key
            removed_data = speaker_cache.pop(oldest_speaker_id)
            # Clean up memory - delete the removed speaker data
            del removed_data
            print(f"Removed oldest cached speaker {oldest_speaker_id} from memory cache")
        
        return speaker_data
        
    except ValueError as ve:
        # Re-raise ValueError with error dict
        raise ve
    except Exception as e:
        print(f"[Speaker Cache] Error loading speaker {speaker_id}: {e}")
        # Fallback to default speaker (speakerId 1) only for integer types
        if speaker_type == "integer" and actual_speaker_id != 1:
            print(f"[Speaker Cache] Falling back to default speaker 1")
            return load_and_cache_speaker(1)
        else:
            raise ValueError({
                "errorCode": 500,
                "message": f"Failed to load speaker {speaker_id}: {str(e)}"
            })

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "errorCode": 0,  # 0 typically indicates success
        "message": "CosyVoice API is running"
    }

@app.websocket("/tts")
async def websocket_tts(websocket: WebSocket):
    """
    WebSocket TTS endpoint that streams audio chunks as they're generated
    """
    # Configure WebSocket with extended timeout for long audio streams
    await websocket.accept()
    
    # Set extended timeout to prevent disconnection during long audio generation
    # Configure websocket to handle long-running operations without timeout
    if hasattr(websocket, '_connection') and hasattr(websocket._connection, 'transport'):
        try:
            # Set socket timeout for long operations (10 minutes)
            transport = websocket._connection.transport
            if hasattr(transport, 'get_extra_info'):
                sock = transport.get_extra_info('socket')
                if sock:
                    import socket
                    sock.settimeout(600)  # 10 minutes timeout
                    print(f"[WS-TTS] WebSocket timeout set to 10 minutes")
        except Exception as e:
            print(f"[WS-TTS] Could not set extended timeout: {e}")
    
    print(f"[WS-TTS] WebSocket connection established")
    
    # Optimize WebSocket for low latency
    try:
        # Access the underlying connection and optimize TCP settings
        if hasattr(websocket, '_connection') and hasattr(websocket._connection, 'transport'):
            transport = websocket._connection.transport
            if hasattr(transport, 'get_extra_info'):
                sock = transport.get_extra_info('socket')
                if sock:
                    import socket
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192)
                    print(f"[WS-TTS] WebSocket optimized: TCP_NODELAY enabled, small send buffer")
    except Exception as e:
        print(f"[WS-TTS] Could not optimize WebSocket: {e}")
    
    # Check X-API-Key header for authorization
    api_key = websocket.headers.get("x-api-key")
    if not api_key or api_key not in VALID_API_KEYS:
        print(f"[WS-TTS] Unauthorized access attempt with API key: {api_key}")
        await websocket.send_text(json.dumps({"error": "Unauthorized: Invalid or missing X-API-Key header"}))
        await websocket.close(code=4001, reason="Unauthorized")
        return
    
    print(f"[WS-TTS] Authorized connection with valid API key")

    def create_header_bytes(start_time_id, message_id):
        """
        创建固定长度的消息头部字节
        startTimeId: 8字节 (64位大端序整数)
        messageId: 4字节 (32位大端序整数)
        总计: 12字节
        """
        import struct
        # 使用大端序格式，便于Java解析
        # Q = 64位无符号整数，I = 32位无符号整数
        header = struct.pack('>QI', start_time_id, message_id)
        return header
    
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
            speaker_id = request_data.get("speakerId", 1)  # Can be int or MD5 string
            save_audio_files = request_data.get("saveAudioFiles", False)
            output_sample_rate = request_data.get("outputSampleRate", 22050)  # Default 22050 Hz (CosyVoice native)
            audio_format = request_data.get("audioFormat", "mp3")  # Default to mp3, can be "mp3" or "pcm"
            
            # 提取新增的消息标识参数
            start_time_id = request_data.get("startTimeId")
            message_id = request_data.get("messageId")
            
            # 判断是否需要添加消息头部
            has_message_headers = start_time_id is not None and message_id is not None
            
            if has_message_headers:
                print(f"[WS-TTS] Message headers detected - startTimeId: {start_time_id}, messageId: {message_id}")
                # 验证参数范围
                if not isinstance(start_time_id, int) or start_time_id < 0:
                    await websocket.send_text(json.dumps({"error": "startTimeId must be a non-negative integer"}))
                    continue
                if not isinstance(message_id, int) or message_id < 1 or message_id > 4294967295:
                    await websocket.send_text(json.dumps({"error": "messageId must be an integer between 1 and 4294967295"}))
                    continue
            else:
                print(f"[WS-TTS] No message headers - using standard PCM streaming")
            
            if not text:
                await websocket.send_text(json.dumps({"error": "Text is required"}))
                continue
            
            if global_cosyvoice is None:
                await websocket.send_text(json.dumps({"error": "Model not initialized"}))
                continue
            
            # Load and cache speaker data
            try:
                speaker_data = load_and_cache_speaker(speaker_id)
            except ValueError as ve:
                # Handle structured error response
                error_data = ve.args[0] if ve.args and isinstance(ve.args[0], dict) else {
                    "errorCode": 400,
                    "message": str(ve)
                }
                await websocket.send_text(json.dumps(error_data))
                continue
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "errorCode": 500,
                    "message": f"Failed to load speaker {speaker_id}: {str(e)}"
                }))
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
                # For MP3 format, use MP3 chunks handler - Updated January 23, 2025
                mp3_handler = MP3ChunksHandler(
                    input_sample_rate=output_sample_rate,  # Use actual resampled rate (16000 Hz)
                    output_sample_rate=output_sample_rate,  # Use user's outputSampleRate (16000 Hz)
                    volume_dB=MP3_VOLUME_DB,
                    min_samples_window=MP3_MIN_SAMPLES_WINDOW,
                    channels=1,
                    bit_depth=16
                )
                mp3_chunk_counter = 0
                print(f"[WS-TTS] Audio format: MP3, using silence-based chunking")
                print(f"[WS-TTS] MP3 config: input_rate={MP3_INPUT_PCM_SAMPLE_RATE}Hz, volume_dB={MP3_VOLUME_DB}, min_window={MP3_MIN_SAMPLES_WINDOW}")
            
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
                first_chunk_sent_since_request = None  # Track first chunk sent timing
                chunk_count = 0
                
                # === TEXT NORMALIZATION TIMING ===
                text_norm_start = time.time()
                normalized_text_chunks = global_cosyvoice.frontend.text_normalize(text, split=True, text_frontend=False)
                text_norm_time = (time.time() - text_norm_start) * 1000
                print(f"[WS-TTS] 📝 Text normalization time: {text_norm_time:.2f}ms, got {len(normalized_text_chunks)} chunks")
                
                # ====== ADD: TEXT SPLITTING BY PUNCTUATION LOGIC FROM MELO-API ======
                # Define punctuation markers for all supported languages
                punctuation_markers = [
                    # English
                    '.', '!', '?', ';', ',', ':',
                    # Spanish
                    '¡', '¿',
                    # French
                    '«', '»',
                    # Chinese
                    '。', '！', '？', '；', '，', '：', '、',
                    # Japanese  
                    # Korean (uses mostly English punctuation)
                ]
                
                # Function to split text by punctuation while keeping the punctuation
                def split_by_punctuation(text):
                    segments = []
                    current_segment = ""
                    
                    for char in text:
                        current_segment += char
                        if char in punctuation_markers:
                            if current_segment.strip():  # Only add non-empty segments
                                segments.append(current_segment.strip())
                            current_segment = ""
                    
                    # Add any remaining text
                    if current_segment.strip():
                        segments.append(current_segment.strip())
                    
                    # If we have no splits (no punctuation in text), use the whole text
                    if not segments:
                        segments = [text]
                    
                    # For Chinese text, ensure first segment isn't too long for fast response
                    # Note: We don't have language detection here, so we'll check if text contains Chinese characters
                    contains_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
                    if contains_chinese and segments and len(segments[0]) > 25:
                        # Extract a shorter first segment if it's Chinese and too long
                        # This helps get the first audio chunk to the client faster
                        first_part = segments[0][:25]
                        rest_part = segments[0][25:]
                        segments[0] = first_part
                        # Only insert the rest if it's not empty
                        if rest_part.strip():
                            segments.insert(1, rest_part)
                    
                    # Combine very short segments with the next segment for better quality
                    combined_segments = []
                    current_combined = ""
                    
                    for segment in segments:
                        # If current segment is short (less than 5 chars) or current_combined is empty
                        if len(segment) < 5 or not current_combined:
                            current_combined += " " + segment if current_combined else segment
                        else:
                            combined_segments.append(current_combined)
                            current_combined = segment
                    
                    # Add the last combined segment if it exists
                    if current_combined:
                        combined_segments.append(current_combined)
                    
                    return combined_segments
                
                # Apply punctuation splitting to each normalized text chunk
                final_text_chunks = []
                for norm_chunk in normalized_text_chunks:
                    punctuation_split_chunks = split_by_punctuation(norm_chunk)
                    final_text_chunks.extend(punctuation_split_chunks)
                
                # Replace the normalized_text_chunks with the punctuation-split chunks
                normalized_text_chunks = final_text_chunks
                
                print(f"[WS-TTS] 📝 After punctuation splitting: {len(normalized_text_chunks)} final text chunks")
                for i, chunk in enumerate(normalized_text_chunks):
                    print(f"[WS-TTS] Final chunk {i+1}: {chunk[:50]}{'...' if len(chunk) > 50 else ''}")
                # =====================================================================
                
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
                            
                            # IMMEDIATE FIRST CHUNK SEND - Don't wait for buffer accumulation
                            if len(pcm_data) > 0:
                                # Detailed timing around the send operation
                                pre_send_time = time.time()
                                print(f"[WS-TTS] 📊 About to send first chunk at: {time.strftime('%H:%M:%S.%f')[:-3]}")
                                
                                # 根据是否有消息头部决定发送格式
                                send_start = time.time()
                                if has_message_headers:
                                    # 添加消息头部到PCM数据前面
                                    header_bytes = create_header_bytes(start_time_id, message_id)
                                    data_to_send = header_bytes + pcm_data
                                    print(f"[WS-TTS] 📋 Adding header to first chunk: {len(header_bytes)} header + {len(pcm_data)} PCM = {len(data_to_send)} total bytes")
                                    await websocket.send_bytes(data_to_send)
                                else:
                                    # 直接发送PCM数据
                                    await websocket.send_bytes(pcm_data)
                                send_end = time.time()
                                
                                # Force any pending I/O to complete
                                await asyncio.sleep(0)
                                
                                post_send_time = time.time()
                                
                                send_time = (send_end - send_start) * 1000
                                flush_time = (post_send_time - send_end) * 1000
                                
                                # Track first chunk sent timing immediately after send
                                first_chunk_sent_time = post_send_time
                                elapsed_since_start = (first_chunk_sent_time - start_time) * 1000
                                elapsed_since_before = (first_chunk_sent_time - before_inference_time) * 1000
                                first_chunk_sent_since_request = elapsed_since_start  # Store for summary
                                first_chunk_sent = True
                                
                                print(f"[WS-TTS] 🚀 IMMEDIATE first PCM chunk sent: {len(pcm_data)} bytes")
                                print(f"[WS-TTS] 📊 Send operation: {send_time:.2f}ms, flush: {flush_time:.2f}ms")
                                print(f"[WS-TTS] 🎯 Time to send first PCM chunk: {elapsed_since_start:.2f} ms since start")
                                print(f"[WS-TTS] 📡 WebSocket send completed at: {time.strftime('%H:%M:%S.%f')[:-3]}")
                                print(f"[WS-TTS] 🔍 Network should deliver within 10-50ms to client")
                                
                                # Save first chunk to file if requested
                                if save_audio_files and chunk_save_folder:
                                    chunk_filename = f"chunk_{pcm_chunk_counter}.pcm"
                                    chunk_filepath = os.path.join(chunk_save_folder, chunk_filename)
                                    try:
                                        with open(chunk_filepath, 'wb') as f:
                                            # 保存与发送给客户端相同的数据格式
                                            if has_message_headers:
                                                # 保存带头部的数据
                                                header_bytes = create_header_bytes(start_time_id, message_id)
                                                f.write(header_bytes + pcm_data)
                                                print(f"[WS-TTS] Saved {chunk_filename} with header ({len(header_bytes + pcm_data)} bytes)")
                                            else:
                                                # 保存纯PCM数据
                                                f.write(pcm_data)
                                                print(f"[WS-TTS] Saved {chunk_filename} ({len(pcm_data)} bytes)")
                                    except Exception as save_error:
                                        print(f"[WS-TTS] Error saving chunk file: {save_error}")
                                
                                pcm_chunk_counter += 1
                                await asyncio.sleep(0)  # Allow other tasks
                            
                            is_first_chunk = False  # Mark that we've processed the first chunk
                            # Don't add to buffer for first chunk since we sent it immediately
                        else:
                            # Add PCM data to buffer (for non-first chunks)
                            if audio_format.lower() == "pcm":
                                pcm_buffer.extend(pcm_data)
                            # For MP3 format, PCM data will be added to mp3_handler in the processing section below
                        
                        audio_convert_time = (time.time() - audio_convert_start) * 1000
                        if audio_format.lower() == "pcm":
                            print(f"[WS-TTS] 🎵 Audio to PCM conversion time: {audio_convert_time:.2f}ms, PCM buffer size: {len(pcm_buffer)} bytes")
                        else:
                            print(f"[WS-TTS] 🎵 Audio to PCM conversion time: {audio_convert_time:.2f}ms, MP3 processing active")
                        
                        # === PROCESS PCM BUFFER INTO CHUNKS ===
                        if audio_format.lower() == "pcm":
                            # For PCM format, send raw PCM data directly
                            while len(pcm_buffer) >= target_pcm_bytes_per_chunk:
                                send_start = time.time()
                                
                                # Extract PCM chunk from buffer
                                pcm_chunk = bytes(pcm_buffer[:target_pcm_bytes_per_chunk])
                                pcm_buffer = pcm_buffer[target_pcm_bytes_per_chunk:]
                                
                                # 根据是否有消息头部决定发送格式
                                if has_message_headers:
                                    # 添加消息头部到PCM数据前面
                                    header_bytes = create_header_bytes(start_time_id, message_id)
                                    data_to_send = header_bytes + pcm_chunk
                                    await websocket.send_bytes(data_to_send)
                                    print(f"[WS-TTS] 📦 PCM chunk {pcm_chunk_counter} sent with header: {len(header_bytes)} header + {len(pcm_chunk)} PCM = {len(data_to_send)} total bytes")
                                else:
                                    # 直接发送PCM数据
                                    await websocket.send_bytes(pcm_chunk)
                                    print(f"[WS-TTS] 📦 PCM chunk {pcm_chunk_counter} sent: {len(pcm_chunk)} bytes")
                                
                                send_time = (time.time() - send_start) * 1000
                                print(f"[WS-TTS] Send time: {send_time:.2f}ms")
                                
                                # Save PCM chunk to file if requested (silence already removed)
                                if save_audio_files and chunk_save_folder:
                                    chunk_filename = f"chunk_{pcm_chunk_counter}.pcm"
                                    chunk_filepath = os.path.join(chunk_save_folder, chunk_filename)
                                    try:
                                        with open(chunk_filepath, 'wb') as f:
                                            # 保存与发送给客户端相同的数据格式
                                            if has_message_headers:
                                                # 保存带头部的数据
                                                header_bytes = create_header_bytes(start_time_id, message_id)
                                                f.write(header_bytes + pcm_chunk)
                                                print(f"[WS-TTS] Saved {chunk_filename} with header ({len(header_bytes + pcm_chunk)} bytes)")
                                            else:
                                                # 保存纯PCM数据
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
                                    first_chunk_sent_since_request = elapsed_since_start  # Store for summary
                                    print(f"[WS-TTS] 🎯 Time to send first PCM chunk: {elapsed_since_start:.2f} ms since start, {elapsed_since_before:.2f} ms since before inference")
                                    first_chunk_sent = True
                                
                                await asyncio.sleep(0)  # Allow other tasks
                        else:
                            # For MP3 format, use MP3 chunks handler - Updated January 23, 2025
                            mp3_convert_start = time.time()
                            
                            # Add PCM data to MP3 handler and get available chunks
                            available_mp3_chunks = mp3_handler.add_pcm_data(pcm_data)
                            
                            # Safety check: ensure we have a list to iterate over
                            if available_mp3_chunks is None:
                                available_mp3_chunks = []
                            
                            # Send each available MP3 chunk immediately
                            for mp3_data in available_mp3_chunks:
                                if mp3_data:
                                    await websocket.send_bytes(mp3_data)
                                    print(f"[WS-TTS] 📦 MP3 chunk {mp3_chunk_counter} sent: {len(mp3_data)} bytes")
                                    
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
                                        first_chunk_sent_since_request = elapsed_since_start
                                        print(f"[WS-TTS] 🎯 Time to send first MP3 chunk: {elapsed_since_start:.2f} ms since start")
                                        first_chunk_sent = True
                                    
                                    await asyncio.sleep(0)  # Allow other tasks
                            
                            mp3_convert_time = (time.time() - mp3_convert_start) * 1000
                            print(f"[WS-TTS] 🎵 MP3 processing time: {mp3_convert_time:.2f}ms")
                
                chunk_processing_time = (time.time() - chunk_start_time) * 1000
                total_time_so_far = (time.time() - start_time) * 1000
                
                print(f"[WS-TTS] 📊 Audio chunk {chunk_count} (text chunk {chunk_idx + 1}) processed in {chunk_processing_time:.2f}ms, total time: {total_time_so_far:.2f}ms")
                
                # === PROCESS REMAINING PCM DATA ===
                if audio_format.lower() == "pcm":
                    if len(pcm_buffer) > 0:
                        print(f"[WS-TTS] 🔄 Processing remaining PCM data: {len(pcm_buffer)} bytes")
                        remaining_start = time.time()
                        
                        # For PCM format, send remaining PCM data directly
                        remaining_pcm_data = bytes(pcm_buffer)
                        
                        # 根据是否有消息头部决定发送格式
                        if has_message_headers:
                            # 添加消息头部到PCM数据前面
                            header_bytes = create_header_bytes(start_time_id, message_id)
                            data_to_send = header_bytes + remaining_pcm_data
                            await websocket.send_bytes(data_to_send)
                            print(f"[WS-TTS] 📦 Final PCM chunk {pcm_chunk_counter} sent with header: {len(header_bytes)} header + {len(remaining_pcm_data)} PCM = {len(data_to_send)} total bytes")
                        else:
                            # 直接发送PCM数据
                            await websocket.send_bytes(remaining_pcm_data)
                            print(f"[WS-TTS] 📦 Final PCM chunk {pcm_chunk_counter} sent: {len(remaining_pcm_data)} bytes")
                        
                        remaining_time = (time.time() - remaining_start) * 1000
                        print(f"[WS-TTS] Process time: {remaining_time:.2f}ms")
                        
                        # Save final PCM chunk to file if requested
                        if save_audio_files and chunk_save_folder:
                            chunk_filename = f"chunk_{pcm_chunk_counter}.pcm"
                            chunk_filepath = os.path.join(chunk_save_folder, chunk_filename)
                            try:
                                with open(chunk_filepath, 'wb') as f:
                                    # 保存与发送给客户端相同的数据格式
                                    if has_message_headers:
                                        # 保存带头部的数据
                                        header_bytes = create_header_bytes(start_time_id, message_id)
                                        f.write(header_bytes + remaining_pcm_data)
                                        print(f"[WS-TTS] Saved final {chunk_filename} with header ({len(header_bytes + remaining_pcm_data)} bytes)")
                                    else:
                                        # 保存纯PCM数据
                                        f.write(remaining_pcm_data)
                                        print(f"[WS-TTS] Saved final {chunk_filename} ({len(remaining_pcm_data)} bytes)")
                            except Exception as save_error:
                                print(f"[WS-TTS] Error saving final chunk file: {save_error}")
                            
                            pcm_chunk_counter += 1
                        
                        # Clear buffer
                        pcm_buffer.clear()
                else:
                    # For MP3 format, finalize the MP3 handler - Updated December 19, 2024
                    print(f"[WS-TTS] 🔄 Finalizing MP3 processing...")
                    remaining_start = time.time()
                    
                    # Get final MP3 chunk from handler
                    final_mp3_data = mp3_handler.finalize()
                    
                    if final_mp3_data:
                        # Send final MP3 chunk
                        await websocket.send_bytes(final_mp3_data)
                        
                        remaining_time = (time.time() - remaining_start) * 1000
                        print(f"[WS-TTS] 📦 Final MP3 chunk {mp3_chunk_counter} sent: {len(final_mp3_data)} bytes, process time: {remaining_time:.2f}ms")
                        
                        # Save final MP3 chunk to file if requested
                        if save_audio_files and chunk_save_folder:
                            chunk_filename = f"chunk_{mp3_chunk_counter}.mp3"
                            chunk_filepath = os.path.join(chunk_save_folder, chunk_filename)
                            try:
                                with open(chunk_filepath, 'wb') as f:
                                    f.write(final_mp3_data)
                                print(f"[WS-TTS] Saved final {chunk_filename} ({len(final_mp3_data)} bytes)")
                            except Exception as save_error:
                                print(f"[WS-TTS] Error saving final chunk file: {save_error}")
                        
                        mp3_chunk_counter += 1
                    else:
                        print(f"[WS-TTS] No final MP3 chunk generated")
                
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
                print(f"[WS-TTS] 📋   Message Headers: {'Yes' if has_message_headers else 'No'}")
                if has_message_headers:
                    print(f"[WS-TTS] 📋   StartTimeId: {start_time_id}")
                    print(f"[WS-TTS] 📋   MessageId: {message_id}")
                print(f"[WS-TTS] 📋   Total Generation Time: {generation_time:.2f}s")
                
                # Add first chunk timing summary
                if first_chunk_generated:
                    print(f"[WS-TTS] 📋   First Chunk Inference Time: {first_chunk_time:.2f}ms")
                    print(f"[WS-TTS] 📋   First Chunk Since Request: {first_chunk_since_request:.2f}ms")
                    if first_chunk_sent_since_request is not None:
                        print(f"[WS-TTS] 📋   First Chunk Sent Since Request: {first_chunk_sent_since_request:.2f}ms")
                else:
                    print(f"[WS-TTS] 📋   First Chunk: Not generated")
                
                if audio_format.lower() == "pcm":
                    print(f"[WS-TTS] 📊 Total PCM chunks sent: {pcm_chunk_counter}")
                    if save_audio_files:
                        print(f"[WS-TTS] Saved {pcm_chunk_counter} PCM chunk files to {chunk_save_folder}")
                else:
                    print(f"[WS-TTS] 📊 Total MP3 chunks sent: {mp3_chunk_counter}")
                    if save_audio_files:
                        print(f"[WS-TTS] Saved {mp3_chunk_counter} MP3 chunk files to {chunk_save_folder}")
                
                # Send an empty chunk to signal completion (for both MP3 and PCM)
                if has_message_headers:
                    # 发送空的完成信号时也要添加消息头部
                    header_bytes = create_header_bytes(start_time_id, message_id)
                    completion_signal = header_bytes + b''
                    await websocket.send_bytes(completion_signal)
                    print(f"[WS-TTS] 📡 Empty completion chunk sent with header for {audio_format.upper()} format")
                else:
                    # 标准的空完成信号
                    await websocket.send_bytes(b'')
                    print(f"[WS-TTS] 📡 Empty completion chunk sent for {audio_format.upper()} format")
                
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

@app.get("/getDefaultSpeakerList")
async def get_default_speaker_list(request: Request, x_api_key: Optional[str] = Header(None)):
    """
    Get Default Speaker List API
    
    Parameters:
    - x_api_key: API key in header
    
    Returns:
    - JSON response with default speaker list and their mp3 URLs
    """
    # Verify API key
    if x_api_key not in VALID_API_KEYS:
        return {
            "errorCode": 401,
            "message": "Invalid API key"
        }
    
    try:
        speaker_list = []
        
        # Process each speaker ID in the configured order
        for speaker_id in DEFAULT_SPEAKER_IDS:
            # Construct file paths
            speaker_folder = f'./asset/speakerId-{speaker_id}'
            wav_path = os.path.join(speaker_folder, f'speakerId-{speaker_id}.wav')
            mp3_path = os.path.join(speaker_folder, f'speakerId-{speaker_id}.mp3')
            
            # Check if WAV file exists
            if not os.path.exists(wav_path):
                print(f"[Default Speakers] WAV file not found for speaker {speaker_id}: {wav_path}")
                continue
            
            # Convert WAV to MP3 if MP3 doesn't exist
            if not os.path.exists(mp3_path):
                try:
                    print(f"[Default Speakers] Converting WAV to MP3 for speaker {speaker_id}")
                    # Use ffmpeg to convert WAV to MP3 with 16kHz and mono
                    cmd = [
                        'ffmpeg',
                        '-i', wav_path,
                        '-ar', '16000',
                        '-ac', '1',
                        '-ab', '128k',
                        '-y',
                        mp3_path
                    ]
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                    print(f"[Default Speakers] Successfully converted WAV to MP3 for speaker {speaker_id}")
                except subprocess.CalledProcessError as e:
                    print(f"[Default Speakers] Failed to convert WAV to MP3 for speaker {speaker_id}: {e}")
                    continue
                except FileNotFoundError:
                    print(f"[Default Speakers] ffmpeg not found, cannot convert speaker {speaker_id}")
                    continue
            
            # Build MP3 URL
            host = request.url.hostname
            port = request.url.port
            scheme = request.url.scheme
            if port is None or (port == 80 and scheme == 'http') or (port == 443 and scheme == 'https'):
                mp3_url = f"{scheme}://{host}/speaker/{speaker_id}.mp3"
            else:
                mp3_url = f"{scheme}://{host}:{port}/speaker/{speaker_id}.mp3"
            
            # Add to speaker list
            speaker_list.append({
                "speakerId": speaker_id,
                "mp3Url": mp3_url
            })
        
        return {
            "errorCode": 0,
            "message": "success",
            "speakerList": speaker_list
        }
        
    except Exception as e:
        return {
            "errorCode": 500,
            "message": f"Internal server error: {str(e)}"
        }

@app.get("/speaker/{speaker_id}.mp3")
async def get_speaker_mp3(speaker_id: int):
    """
    Serve speaker MP3 files
    
    Parameters:
    - speaker_id: Integer speaker ID
    
    Returns:
    - MP3 file response
    """
    try:
        # Construct file paths
        speaker_folder = f'./asset/speakerId-{speaker_id}'
        wav_path = os.path.join(speaker_folder, f'speakerId-{speaker_id}.wav')
        mp3_path = os.path.join(speaker_folder, f'speakerId-{speaker_id}.mp3')
        
        # Check if WAV file exists
        if not os.path.exists(wav_path):
            raise HTTPException(
                status_code=404,
                detail={
                    "errorCode": 404,
                    "message": f"Speaker {speaker_id} not found"
                }
            )
        
        # Convert WAV to MP3 if MP3 doesn't exist
        if not os.path.exists(mp3_path):
            try:
                print(f"[Speaker MP3] Converting WAV to MP3 for speaker {speaker_id}")
                # Use ffmpeg to convert WAV to MP3 with 16kHz and mono
                cmd = [
                    'ffmpeg',
                    '-i', wav_path,
                    '-ar', '16000',
                    '-ac', '1',
                    '-ab', '128k',
                    '-y',
                    mp3_path
                ]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"[Speaker MP3] Successfully converted WAV to MP3 for speaker {speaker_id}")
            except subprocess.CalledProcessError as e:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "errorCode": 500,
                        "message": "Failed to convert WAV to MP3"
                    }
                )
            except FileNotFoundError:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "errorCode": 500,
                        "message": "ffmpeg not found"
                    }
                )
        
        # Serve the MP3 file
        return FileResponse(mp3_path, media_type="audio/mpeg", filename=f"speakerId-{speaker_id}.mp3")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "errorCode": 500,
                "message": f"Internal server error: {str(e)}"
            }
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
    parser = argparse.ArgumentParser(description='CosyVoice API Server')
    parser.add_argument('--port', type=int, default=9003, help='Port number to run the server on (default: 9003)')
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
