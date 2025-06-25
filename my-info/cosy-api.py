import sys
sys.path.append('third_party/Matcha-TTS')

# Register vllm model
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

import torch
import numpy as np
import io
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
        # Pre-normalize the prompt text to avoid repeated processing
        global_normalized_prompt_text = global_cosyvoice.frontend.text_normalize(global_prompt_text, split=False, text_frontend=True)
        
        # Pre-compute prompt audio processing and cache it as a speaker
        print("Pre-computing prompt audio processing...")
        cache_start = time.time()
        global_cosyvoice.add_zero_shot_spk(global_prompt_text, global_prompt_speech_16k, 'cached_prompt_spk')
        cache_time = (time.time() - cache_start) * 1000
        print(f"Prompt audio cached in {cache_time:.2f}ms")
        
        # Warm up with longer text to prime VLLM for better performance
        for i, j in enumerate(global_cosyvoice.inference_zero_shot("Did you guys see the video of that dude who was at the gym who took his earbuds and just smacked them against the wall because they would not stay in his ear during his set?", global_prompt_text, global_prompt_speech_16k, stream=True)):
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
    global global_cosyvoice, global_prompt_speech_16k, global_normalized_prompt_text
    
    start_time = time.time()
    print(f"API start time: {time.strftime('%H:%M:%S.%f')[:-3]}")
    
    if global_cosyvoice is None or global_prompt_speech_16k is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Generate audio
        print(f"Generating audio for text: {request.text[:50]}{'...' if len(request.text) > 50 else ''}")
        
        before_inference_time = time.time()
        elapsed_since_start = (before_inference_time - start_time) * 1000
        print(f"Time before inference: {elapsed_since_start:.2f} ms since start")
        
        try:
            # Collect all audio chunks
            audio_chunks = []
            for i, j in enumerate(global_cosyvoice.inference_zero_shot(
                request.text, 
                global_normalized_prompt_text, 
                global_prompt_speech_16k, 
                zero_shot_spk_id='cached_prompt_spk', 
                stream=True
            )):
                audio_chunks.append(j['tts_speech'])
            
            # Concatenate all chunks
            if audio_chunks:
                audio = torch.cat(audio_chunks, dim=1)
            else:
                raise HTTPException(status_code=500, detail="No audio generated")
                
        except Exception as inference_error:
            print(f"Inference error: {str(inference_error)}")
            print(f"Error type: {type(inference_error)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise
        
        after_inference_time = time.time()
        elapsed_since_start = (after_inference_time - start_time) * 1000
        elapsed_since_last = (after_inference_time - before_inference_time) * 1000
        print(f"Time after inference: {elapsed_since_start:.2f} ms since start, {elapsed_since_last:.2f} ms since before inference")
        
        # Create in-memory file
        audio_io = io.BytesIO()
        
        if request.audio_format.lower() == "wav":
            torchaudio.save(audio_io, audio, global_cosyvoice.sample_rate, format="WAV")
            media_type = "audio/wav"
            filename = "output.wav"
        elif request.audio_format.lower() == "mp3":
            # First write as WAV to memory
            wav_io = io.BytesIO()
            
            # Check if audio is valid
            if audio.numel() == 0 or torch.isnan(audio).any():
                raise HTTPException(status_code=500, detail="Generated audio is invalid or empty")
            
            torchaudio.save(wav_io, audio, global_cosyvoice.sample_rate, format="WAV")
            wav_io.seek(0)
            
            # Try MP3 conversion up to 3 times
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Convert to MP3 using torchaudio
                    waveform, sample_rate = torchaudio.load(wav_io)
                    
                    # Convert to MP3
                    mp3_io = io.BytesIO()
                    torchaudio.save(mp3_io, waveform, sample_rate, format="mp3")
                    audio_io = mp3_io
                    media_type = "audio/mpeg"
                    filename = "output.mp3"
                    break  # Success, exit the retry loop
                except RuntimeError as e:
                    print(f"MP3 conversion attempt {attempt+1}/{max_attempts} failed: {str(e)}")
                    if attempt < max_attempts - 1:
                        # Reset WAV IO for next attempt
                        wav_io.seek(0)
                    else:
                        # All attempts failed, raise an appropriate error
                        print(f"All MP3 conversion attempts failed")
                        raise HTTPException(status_code=500, detail="Failed to generate MP3 audio after multiple attempts")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported audio format: {request.audio_format}")
        
        audio_io.seek(0)
        
        first_byte_time = time.time()
        elapsed_since_start = (first_byte_time - start_time) * 1000
        elapsed_since_last = (first_byte_time - after_inference_time) * 1000
        print(f"Time to send first byte: {elapsed_since_start:.2f} ms since start, {elapsed_since_last:.2f} ms since after inference")
        
        generation_time = time.time() - start_time
        print(f"Audio generated in {generation_time:.2f} seconds")
        
        return StreamingResponse(
            audio_io, 
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except Exception as e:
        print(f"Error in generate_tts: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
