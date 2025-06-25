# 今天天气真不错，阳光明媚，适合出去散步。
# 我喜欢在周末的时候和朋友一起看电影，分享快乐时光。
# 学习新知识是一件很有趣的事情，让人感到充实和满足。
# 妈妈做的饭菜总是那么美味，充满了家的温暖和爱意。
# 春天来了，花儿绽放，小鸟歌唱，大自然充满了生机。
# 收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。
# 忙碌了一整周之后，这个周末我终于有机会好好放松了一下，去了几个城市里平时没空去的小众景点，还尝试了好几家新开的咖啡馆和旧货店，感觉整个人都充满了能量，已经开始期待下次的随心所欲的旅行了。

# The weather is beautiful today, with bright sunshine perfect for a walk.
# I enjoy watching movies with friends on weekends, sharing happy moments together.
# Learning new knowledge is very interesting and makes me feel fulfilled and satisfied.
# Mom's cooking is always so delicious, full of warmth and love from home.
# Spring has arrived, flowers are blooming, birds are singing, and nature is full of life.
# After spending the entire weekend exploring various hidden gems around the city, trying out new cafes and vintage shops, I'm feeling incredibly refreshed and already planning my next spontaneous adventure for sure.
# Did you guys see the video of that dude who was at the gym who took his earbuds and just smacked them against the wall because they would not stay in his ear during his set? I feel that man's pain on a personal level. I can't keep earbuds in my ear to save my life, especially earbuds. I have four pairs of them and I hate all of them cuz they all fucking suck. That's why I got these instead. These are the Lenovo Eraser X15 Pros. They are the best over-the-ear earbud that I've ever had. Now, I don't really like calling them buds cuz they're not like actual buds that go inside your ear. They they look just like this. They just kind of rest right over the top of your ear, just like that. And the cool part is, no matter what you're doing, they're not going to fall off your head. So if you like to go to the gym all the time, you won't have to worry about them falling out during any of your sets. And they're waterproof, so you don't have to worry about sweat. You can literally go swimming or take a shower in these. They're on sale right now for like $25 on Tik Tok, but with the right coupons, you can get them for like 10 bucks.


import sys
sys.path.append('third_party/Matcha-TTS')

# Register vllm model (from vllm_example.py)
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
import os
import uuid
import time
import io

# Global variables for model and prompt (initialized once)
app = FastAPI()
cosyvoice = None
prompt_speech_16k = None
prompt_text = "Did you guys see the video of that dude who was at the gym who took his earbuds and just smacked them against the wall because they would not stay in his ear during his set? "
normalized_prompt_text = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model and prompt speech once when the app starts"""
    global cosyvoice, prompt_speech_16k
    
    print("Loading CosyVoice2 model with vllm...")
    # Initialize CosyVoice2 with vllm enabled
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)
    
    # Load prompt speech
    prompt_speech_16k = load_wav('./asset/man-short.wav', 16000)
    
    # Pre-warm the model with a dummy inference AND pre-normalize prompt text
    print("Pre-warming model...")
    warmup_start = time.time()
    try:
        # Pre-normalize the prompt text to avoid repeated processing
        global normalized_prompt_text
        normalized_prompt_text = cosyvoice.frontend.text_normalize(prompt_text, split=False, text_frontend=True)
        
        for i, j in enumerate(cosyvoice.inference_zero_shot("Hello world", "Hello", prompt_speech_16k, stream=False)):
            break  # Just run once to warm up
        warmup_time = (time.time() - warmup_start) * 1000
        print(f"Model pre-warmed in {warmup_time:.2f}ms")
    except Exception as e:
        print(f"Warmup failed: {e}")
    
    print("Model loaded successfully!")

def cleanup_file(file_path: str):
    """Background task to clean up temporary files"""
    if os.path.exists(file_path):
        os.remove(file_path)

@app.get("/tts")
async def text_to_speech(text_content: str):
    """
    TTS API endpoint that streams audio chunks as they're generated
    """
    global cosyvoice, prompt_speech_16k, prompt_text
    
    # Start timing
    request_start_time = time.time()
    print(f"[TTS] Request received at: {time.strftime('%H:%M:%S.%f')[:-3]}")
    
    if cosyvoice is None or prompt_speech_16k is None:
        return {"error": "Model not initialized"}
    
    def generate_audio_chunks():
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
            
            # Create generated_wavs folder if it doesn't exist
            os.makedirs("generated_wavs", exist_ok=True)
            
            for i, j in enumerate(cosyvoice.inference_zero_shot(text_content, normalized_prompt_text, prompt_speech_16k, stream=True)):
                chunk_start_time = time.time()
                chunk_count += 1
                
                # Save each chunk as separate file in generated_wavs folder
                chunk_filename = f"generated_wavs/chunk_{i+1}.wav"
                torchaudio.save(chunk_filename, j['tts_speech'], cosyvoice.sample_rate)
                
                # Record first chunk timing
                if not first_chunk_generated:
                    first_chunk_time = (chunk_start_time - inference_start_time) * 1000
                    print(f"[TTS] First chunk generated time: {first_chunk_time:.2f}ms")
                    first_chunk_generated = True
                
                # Convert audio tensor to wav bytes for streaming
                buffer = io.BytesIO()
                torchaudio.save(buffer, j['tts_speech'], cosyvoice.sample_rate, format="wav")
                wav_bytes = buffer.getvalue()
                buffer.close()
                
                chunk_processing_time = (time.time() - chunk_start_time) * 1000
                total_time_so_far = (time.time() - request_start_time) * 1000
                
                print(f"[TTS] Chunk {i+1} processed in {chunk_processing_time:.2f}ms, total time: {total_time_so_far:.2f}ms")
                print(f"[TTS] Chunk {i+1} saved as {chunk_filename}")
                
                # Yield the chunk immediately to client
                yield wav_bytes
            
            # End of inference timing
            inference_end_time = time.time()
            total_inference_time = (inference_end_time - inference_start_time) * 1000
            total_request_time = (inference_end_time - request_start_time) * 1000
            
            print(f"[TTS] Inference completed in: {total_inference_time:.2f}ms")
            print(f"[TTS] Total chunks generated: {chunk_count}")
            print(f"[TTS] Total request time: {total_request_time:.2f}ms")
            print(f"[TTS] Response completed at: {time.strftime('%H:%M:%S.%f')[:-3]}")
            
        except Exception as e:
            error_time = (time.time() - request_start_time) * 1000
            print(f"[TTS] Error after {error_time:.2f}ms: {str(e)}")
            yield b"Error: TTS generation failed"
    
    # Return streaming response
    return StreamingResponse(
        generate_audio_chunks(),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=tts_stream.wav"}
    )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "CosyVoice TTS API is running"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
