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

def get_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

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

@app.get("/")
async def root():
    return {"message": "CosyVoice API is running"}

@app.websocket("/ws-tts")
async def websocket_tts(websocket: WebSocket):
    """
    WebSocket TTS endpoint that streams audio chunks as they're generated
    """
    await websocket.accept()
    print(f"[WS-TTS] WebSocket connection established")
    
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
            audio_format = "wav"  # Always use WAV for WebSocket
            
            if not text:
                await websocket.send_text(json.dumps({"error": "Text is required"}))
                continue
            
            if global_cosyvoice is None or global_prompt_speech_16k is None:
                await websocket.send_text(json.dumps({"error": "Model not initialized"}))
                continue
            
            # Generate audio
            print(f"Generating audio for text: {text[:50]}{'...' if len(text) > 50 else ''}")
            
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
                            first_chunk_since_request = (chunk_start_time - start_time) * 1000
                            zeroshot_time = (chunk_start_time - zeroshot_start) * 1000
                            print(f"[WS-TTS] ⚡ First chunk generated time: {first_chunk_time:.2f}ms")
                            print(f"[WS-TTS] ⚡ First chunk since request arrival: {first_chunk_since_request:.2f}ms")
                            print(f"[WS-TTS] ⚡ Zero-shot inference time: {zeroshot_time:.2f}ms")
                            first_chunk_generated = True
                        
                        # === AUDIO CONVERSION TIMING ===
                        audio_convert_start = time.time()
                        
                        # Convert audio tensor to WAV format for WebSocket streaming
                        buffer = io.BytesIO()
                        torchaudio.save(buffer, j['tts_speech'], global_cosyvoice.sample_rate, format="wav")
                        wav_bytes = buffer.getvalue()
                        buffer.close()
                        
                        audio_convert_time = (time.time() - audio_convert_start) * 1000
                        chunk_processing_time = (time.time() - chunk_start_time) * 1000
                        total_time_so_far = (time.time() - start_time) * 1000
                        
                        print(f"[WS-TTS] 🎵 Audio conversion time: {audio_convert_time:.2f}ms")
                        print(f"[WS-TTS] 📦 Chunk {chunk_count} (text chunk {chunk_idx + 1}) processed in {chunk_processing_time:.2f}ms, total time: {total_time_so_far:.2f}ms")
                        
                        # === WEBSOCKET SEND TIMING ===
                        send_start = time.time()
                        await websocket.send_bytes(wav_bytes)
                        await asyncio.sleep(0)
                        send_time = (time.time() - send_start) * 1000
                        
                        print(f"[WS-TTS] 📡 WebSocket send time: {send_time:.2f}ms")
                        
                        # Track first chunk sent timing
                        if not first_chunk_sent:
                            first_chunk_sent_time = time.time()
                            elapsed_since_start = (first_chunk_sent_time - start_time) * 1000
                            elapsed_since_before = (first_chunk_sent_time - before_inference_time) * 1000
                            print(f"[WS-TTS] 🎯 Time to send first chunk: {elapsed_since_start:.2f} ms since start, {elapsed_since_before:.2f} ms since before inference")
                            first_chunk_sent = True
                
                # Log completion timing
                after_inference_time = time.time()
                elapsed_since_start = (after_inference_time - start_time) * 1000
                elapsed_since_before = (after_inference_time - before_inference_time) * 1000
                print(f"[WS-TTS] ✅ Time after inference: {elapsed_since_start:.2f} ms since start, {elapsed_since_before:.2f} ms since before inference")
                
                generation_time = time.time() - start_time
                print(f"[WS-TTS] 🏁 Audio generated in {generation_time:.2f} seconds")
                
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
    
    # Load prompt speech
    global_prompt_speech_16k = load_wav('./asset/man-short.wav', 16000)
    global_prompt_text = "Did you guys see the video of that dude who was at the gym who took his earbuds and just smacked them against the wall because they would not stay in his ear during his set? "
    
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
