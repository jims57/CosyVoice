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
from typing import Optional
from fastapi import FastAPI, HTTPException
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
                    print(f"[TTS] First chunk generated time: {first_chunk_time:.2f}ms")
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
        
        # Warm up with longer text to prime VLLM for better performance (same as vllm-demo.py)
        for i, j in enumerate(global_cosyvoice.inference_zero_shot("Did you guys see the video of that dude who was at the gym who took his earbuds and just smacked them against the wall because they would not stay in his ear during his set?", global_normalized_prompt_text, global_prompt_speech_16k, zero_shot_spk_id='cached_prompt_spk', stream=True)):
            break  # Just run once to warm up
        warmup_time = (time.time() - warmup_start) * 1000
        print(f"Model pre-warmed in {warmup_time:.2f}ms")
    except Exception as e:
        print(f"Warmup failed: {e}")
    
    print("CosyVoice model loaded successfully!")

@app.get("/")
async def root():
    return {"message": "CosyVoice API is running"}

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9003)
