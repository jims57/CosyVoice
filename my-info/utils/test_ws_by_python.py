import asyncio
import websockets
import json
import sys
import time

async def test_websocket():
    uri = "ws://localhost:9003/ws-tts"
    async with websockets.connect(uri) as websocket:
        # Record start time
        start_time = time.time()
        print(f"Request start time: {time.strftime('%H:%M:%S.%f')[:-3]}")
        
        # Send test message
        message = {
            "text": "青石板上泛着水光，雨丝斜斜地织着帘子。我撑一把油纸伞，踩着湿润的石板路，听脚步声在巷子里轻轻回响。"
        }
        await websocket.send(json.dumps(message))
        print(f"Sent: {message}")
        
        # Receive audio chunks
        chunk_count = 0
        first_chunk_time = None
        last_chunk_time = None
        
        while True:
            try:
                # Add timeout to detect when no more chunks are coming
                audio_data = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                chunk_receive_end = time.time()
                
                chunk_count += 1
                
                # Calculate timing
                time_since_request = (chunk_receive_end - start_time) * 1000
                
                # Record first chunk timing
                if chunk_count == 1:
                    first_chunk_time = time_since_request
                    print(f"First chunk arrived in: {first_chunk_time:.2f}ms since request")
                
                last_chunk_time = time_since_request
                
                print(f"Received chunk {chunk_count}, size: {len(audio_data)} bytes, total time: {time_since_request:.2f}ms")
                
            except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
                # Exit when connection closes or no chunks for 3 seconds
                break
        
        # Final summary
        if first_chunk_time and last_chunk_time:
            print(f"\n=== Summary ===")
            print(f"First chunk arrival time: {first_chunk_time:.2f}ms")
            print(f"Total chunks received: {chunk_count}")
            print(f"Total time for all chunks: {last_chunk_time:.2f}ms")

if __name__ == "__main__":
    asyncio.run(test_websocket())