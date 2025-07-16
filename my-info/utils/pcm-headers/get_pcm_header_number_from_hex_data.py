import struct

# 将十六进制转换回数字
header_hex = "00000198112BCC9800BC614E"
header_bytes = bytes.fromhex(header_hex)

# 解析（大端序格式）
parsed_start_time_id = struct.unpack('>Q', header_bytes[:8])[0]
parsed_message_id = struct.unpack('>I', header_bytes[8:12])[0]

print(f"Parsed startTimeId: {parsed_start_time_id}")  # 应该是 1752634739864
print(f"Parsed messageId: {parsed_message_id}")      # 应该是 12345678