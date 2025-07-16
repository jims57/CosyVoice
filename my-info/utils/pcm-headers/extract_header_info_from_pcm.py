# 读取保存的PCM文件前12字节
with open('chunk_0.pcm', 'rb') as f:
    header_bytes = f.read(12)
    
import struct
start_time_id = struct.unpack('>Q', header_bytes[:8])[0]  # 前8字节
message_id = struct.unpack('>I', header_bytes[8:12])[0]   # 后4字节
print(f"StartTimeId: {start_time_id}")
print(f"MessageId: {message_id}")