import asyncio
import websockets
import json
import sys

async def test_websocket():
    uri = "ws://localhost:9003/ws-tts"
    async with websockets.connect(uri) as websocket:
        # Send test message
        message = {
            "text": "青石板上泛着水光，雨丝斜斜地织着帘子。我撑一把油纸伞，踩着湿润的石板路，听脚步声在巷子里轻轻回响。"
        }
        await websocket.send(json.dumps(message))
        print(f"Sent: {message}")
        
        # Receive audio chunks
        chunk_count = 0
        while True:
            try:
                audio_data = await websocket.recv()
                chunk_count += 1
                print(f"Received chunk {chunk_count}, size: {len(audio_data)} bytes")
            except websockets.exceptions.ConnectionClosed:
                break

if __name__ == "__main__":
    asyncio.run(test_websocket())