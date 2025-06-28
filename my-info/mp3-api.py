import asyncio
import json
import os
import glob
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "MP3 Streaming API is running"}

@app.websocket("/mp3-stream")
async def websocket_mp3_stream(websocket: WebSocket):
    """
    WebSocket endpoint that streams MP3 files from mp3_chunks folder in order
    """
    await websocket.accept()
    print(f"[MP3-STREAM] WebSocket connection established")
    
    try:
        while True:
            # Wait for client message before starting streaming
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            print(f"[MP3-STREAM] Received request: {request_data}")
            
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            mp3_chunks_dir = os.path.join(script_dir, "mp3_chunks")
            
            print(f"[MP3-STREAM] Looking for MP3 files in: {mp3_chunks_dir}")
            
            if not os.path.exists(mp3_chunks_dir):
                await websocket.send_text(json.dumps({"error": f"mp3_chunks directory not found at {mp3_chunks_dir}"}))
                continue
            
            # Get all MP3 files and sort them by chunk number
            mp3_pattern = os.path.join(mp3_chunks_dir, "chunk_*.mp3")
            mp3_files = glob.glob(mp3_pattern)
            
            print(f"[MP3-STREAM] Found {len(mp3_files)} MP3 files with pattern 'chunk_*.mp3'")
            
            if not mp3_files:
                await websocket.send_text(json.dumps({"error": "No MP3 chunk files found in mp3_chunks directory"}))
                continue
            
            # Sort files by chunk number (extract number from filename)
            def extract_chunk_number(filename):
                try:
                    basename = os.path.basename(filename)
                    # Extract number between 'chunk_' and '.mp3'
                    num_str = basename.replace('chunk_', '').replace('.mp3', '')
                    return int(num_str)
                except Exception as e:
                    print(f"[MP3-STREAM] Error extracting number from {basename}: {e}")
                    return float('inf')  # Put invalid files at the end
            
            mp3_files.sort(key=extract_chunk_number)
            
            print(f"[MP3-STREAM] After sorting, will stream {len(mp3_files)} MP3 chunks")
            print(f"[MP3-STREAM] First few files: {[os.path.basename(f) for f in mp3_files[:5]]}")
            print(f"[MP3-STREAM] Last few files: {[os.path.basename(f) for f in mp3_files[-5:]]}")
            
            # Stream each MP3 file in order
            successful_streams = 0
            for i, mp3_file in enumerate(mp3_files):
                try:
                    start_time = time.time()
                    
                    # Read MP3 file
                    with open(mp3_file, 'rb') as f:
                        mp3_data = f.read()
                    
                    read_time = (time.time() - start_time) * 1000
                    
                    # Send MP3 data immediately
                    send_start = time.time()
                    await websocket.send_bytes(mp3_data)
                    send_time = (time.time() - send_start) * 1000
                    
                    file_size_kb = len(mp3_data) / 1024
                    total_time = read_time + send_time
                    successful_streams += 1
                    
                    print(f"[MP3-STREAM] Streamed {os.path.basename(mp3_file)} ({file_size_kb:.1f}KB) - Read: {read_time:.2f}ms, Send: {send_time:.2f}ms, Total: {total_time:.2f}ms")
                    
                    # Small async sleep to allow other tasks
                    await asyncio.sleep(0)
                    
                except Exception as file_error:
                    print(f"[MP3-STREAM] Error processing {mp3_file}: {str(file_error)}")
                    print(f"[MP3-STREAM] Error type: {type(file_error)}")
                    import traceback
                    print(f"[MP3-STREAM] Traceback: {traceback.format_exc()}")
                    await websocket.send_text(json.dumps({"error": f"Error processing {os.path.basename(mp3_file)}: {str(file_error)}"}))
                    # Continue with next file instead of breaking
                    continue
            
            print(f"[MP3-STREAM] Completed streaming {successful_streams}/{len(mp3_files)} MP3 chunks")
            
    except WebSocketDisconnect:
        print(f"[MP3-STREAM] WebSocket connection disconnected")
    except Exception as e:
        print(f"[MP3-STREAM] WebSocket error: {str(e)}")
        print(f"[MP3-STREAM] Error type: {type(e)}")
        import traceback
        print(f"[MP3-STREAM] Traceback: {traceback.format_exc()}")
        try:
            await websocket.send_text(json.dumps({"error": f"MP3 streaming error: {str(e)}"}))
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9003)
