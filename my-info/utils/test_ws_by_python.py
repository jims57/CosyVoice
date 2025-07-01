"""
WebSocket TTS Test Script

Usage:
    python test_ws_by_python.py           # Test with 2 requests (default)
    python test_ws_by_python.py 1         # Test with 1 request
    python test_ws_by_python.py 5         # Test with 5 requests
"""

import asyncio
import websockets
import json
import sys
import time
import random

# Pool of Chinese sentences for testing
CHINESE_SENTENCES = [
    "青石板上泛着水光，雨丝斜斜地织着帘子。我撑一把油纸伞，踩着湿润的石板路，听脚步声在巷子里轻轻回响。",
    "梧桐叶落满庭院，秋风送来阵阵桂花香。我坐在古色古香的凉亭里，品一壶清茶，看云卷云舒。",
    "夕阳西下时分，渔船归来泊靠码头。海鸥在晚霞中翱翔，渔民们唱着悠扬的渔歌，收拾着一天的收获。",
    "春雨绵绵润如酥，柳絮飞舞似雪花。我漫步在小桥流水边，听着潺潺水声，感受着江南的诗意。",
    "月光如水洒在竹林里，清风徐来带着淡淡竹香。我独自走在幽静的小径上，享受着这份难得的宁静。",
    "晨雾缭绕在青山间，鸟儿在枝头欢快歌唱。我沿着蜿蜒的山路慢慢攀登，呼吸着清新的山间空气。",
    "古刹钟声悠远绵长，香烟袅袅升向天空。我在殿堂里静静祈祷，心灵在这庄严的氛围中得到净化。",
    "荷塘月色美如画，荷花盛开香阵阵。我泛舟湖上观赏美景，感受着夏夜的清凉和荷香的怡人。",
    "雪花纷飞覆盖大地，梅花独自傲雪开放。我踏雪寻梅来到园中，被这坚韧不拔的精神深深感动。",
    "星空璀璨银河横贯，夜风轻抚大地安详。我仰望浩瀚的夜空，思考着人生的意义和宇宙的奥秘。"
]

async def test_websocket():
    # Get requestTimes from command line argument, default to 2
    requestTimes = 2
    if len(sys.argv) > 1:
        try:
            requestTimes = int(sys.argv[1])
        except ValueError:
            print("Invalid requestTimes argument, using default 2")
            requestTimes = 2
    
    print(f"Testing with {requestTimes} requests")
    
    # Ensure we have enough unique sentences
    if requestTimes > len(CHINESE_SENTENCES):
        print(f"Warning: Only {len(CHINESE_SENTENCES)} unique sentences available, but {requestTimes} requests requested")
    
    # Select unique sentences for each request
    selected_sentences = random.sample(CHINESE_SENTENCES, min(requestTimes, len(CHINESE_SENTENCES)))
    if requestTimes > len(CHINESE_SENTENCES):
        # If more requests than sentences, extend with random selection
        additional_needed = requestTimes - len(CHINESE_SENTENCES)
        selected_sentences.extend(random.choices(CHINESE_SENTENCES, k=additional_needed))
    
    uri = "ws://localhost:9003/cosy-tts"
    headers = {
        "x-api-key": "sk-5z6y7x8w9v0u1t2s3r4q5p6o7n8m9l0k1j2i3h4g"
    }
    async with websockets.connect(uri, additional_headers=headers) as websocket:
        print(f"WebSocket connection established")
        
        # Process multiple requests through the same connection
        for request_num in range(1, requestTimes + 1):
            print(f"\n=== Request {request_num}/{requestTimes} ===")
            
            # Record start time for this request
            start_time = time.time()
            print(f"Request {request_num} start time: {time.strftime('%H:%M:%S.%f')[:-3]}")
            
            # Send test message with unique sentence
            message = {
                "text": selected_sentences[request_num - 1]
            }
            await websocket.send(json.dumps(message))
            print(f"Request {request_num} sent: {message}")
            
            # Receive audio chunks for this request
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
                        print(f"Request {request_num} - First chunk arrived in: {first_chunk_time:.2f}ms since request")
                    
                    last_chunk_time = time_since_request
                    
                    print(f"Request {request_num} - Received chunk {chunk_count}, size: {len(audio_data)} bytes, total time: {time_since_request:.2f}ms")
                    
                except asyncio.TimeoutError:
                    # No more chunks for this request, move to next request
                    break
            
            # Summary for this request
            if first_chunk_time and last_chunk_time:
                print(f"\n=== Request {request_num} Summary ===")
                print(f"First chunk arrival time: {first_chunk_time:.2f}ms")
                print(f"Total chunks received: {chunk_count}")
                print(f"Total time for all chunks: {last_chunk_time:.2f}ms")

if __name__ == "__main__":
    asyncio.run(test_websocket())