//
// PCM音频流消息头部提取工具类
// 
// 作者: Jimmy Gan
// 日期: 2025-07-16
// 
// 功能说明:
// - 从PCM音频数据流中提取消息头部信息
// - 支持解析startTimeId（8字节时间戳）和messageId（4字节消息ID）
// - 纯内存操作，无IO操作
// - 使用大端序字节序，与CosyVoice API服务器兼容
//

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;

// PCM音频消息头部提取器
public class PcmHeaderExtractor {
    
    // 消息头部长度（固定12字节）
    public static final int HEADER_LENGTH = 12;
    
    // startTimeId字段长度（8字节）
    public static final int START_TIME_ID_LENGTH = 8;
    
    // messageId字段长度（4字节）
    public static final int MESSAGE_ID_LENGTH = 4;
    
    // 消息头部信息封装类
    public static class MessageHeaders {
        // 开始时间ID（时间戳）
        private final long startTimeId;
        
        // 消息ID（1-4294967295范围内）
        private final int messageId;
        
        // 纯PCM音频数据（移除头部后）
        private final byte[] pcmData;
        
        // 构造函数
        // @param startTimeId 开始时间ID
        // @param messageId 消息ID
        // @param pcmData 纯PCM音频数据
        public MessageHeaders(long startTimeId, int messageId, byte[] pcmData) {
            this.startTimeId = startTimeId;
            this.messageId = messageId;
            this.pcmData = pcmData;
        }
        
        // 获取开始时间ID
        // @return 开始时间ID（时间戳）
        public long getStartTimeId() {
            return startTimeId;
        }
        
        // 获取消息ID
        // @return 消息ID
        public int getMessageId() {
            return messageId;
        }
        
        // 获取纯PCM音频数据
        // @return PCM音频数据字节数组
        public byte[] getPcmData() {
            return pcmData;
        }
        
        // 获取PCM数据长度
        // @return PCM数据字节长度
        public int getPcmDataLength() {
            return pcmData != null ? pcmData.length : 0;
        }
        
        @Override
        public String toString() {
            return String.format("MessageHeaders{startTimeId=%d, messageId=%d, pcmDataLength=%d}", 
                               startTimeId, messageId, getPcmDataLength());
        }
    }
    
    // 检查数据是否包含消息头部
    // @param data 待检查的数据
    // @return 如果数据长度足够包含完整头部则返回true
    public static boolean hasHeaders(byte[] data) {
        return data != null && data.length >= HEADER_LENGTH;
    }

    // 从PCM数据流中快速提取startTimeId（不创建完整对象）
    // @param data 包含消息头部的PCM数据流
    // @return startTimeId时间戳
    // @throws IllegalArgumentException 如果数据为空或长度不足
    public static long extractStartTimeId(byte[] data) {
        if (data == null || data.length < START_TIME_ID_LENGTH) {
            throw new IllegalArgumentException("数据长度不足以提取startTimeId");
        }
        
        return ByteBuffer.wrap(data, 0, START_TIME_ID_LENGTH)
                        .order(ByteOrder.BIG_ENDIAN)
                        .getLong();
    }
    
    // 从PCM数据流中快速提取messageId（不创建完整对象）
    // @param data 包含消息头部的PCM数据流
    // @return messageId消息标识
    // @throws IllegalArgumentException 如果数据为空或长度不足
    public static int extractMessageId(byte[] data) {
        if (data == null || data.length < HEADER_LENGTH) {
            throw new IllegalArgumentException("数据长度不足以提取messageId");
        }
        
        return ByteBuffer.wrap(data, START_TIME_ID_LENGTH, MESSAGE_ID_LENGTH)
                        .order(ByteOrder.BIG_ENDIAN)
                        .getInt();
    }
    
    // 从PCM数据流中提取纯PCM音频数据（不创建完整对象）
    // @param data 包含消息头部的PCM数据流
    // @return 纯PCM音频数据字节数组（移除头部后）
    // @throws IllegalArgumentException 如果数据为空或长度不足
    public static byte[] extractPcmData(byte[] data) {
        if (data == null || data.length < HEADER_LENGTH) {
            throw new IllegalArgumentException("数据长度不足以提取PCM数据");
        }
        
        // 创建纯PCM数据数组（移除前12字节头部）
        byte[] pcmData = new byte[data.length - HEADER_LENGTH];
        System.arraycopy(data, HEADER_LENGTH, pcmData, 0, pcmData.length);
        
        return pcmData;
    }
    
    // 使用示例和测试方法
    public static void main(String[] args) {
        // 从同一文件夹读取真实的PCM文件进行测试
        String pcmFileName = "chunk_1.pcm";
        
        try {
            // 读取PCM文件
            byte[] pcmFileData = Files.readAllBytes(Paths.get(pcmFileName));
            
            // 检查是否包含头部
            if (!hasHeaders(pcmFileData)) {
                System.out.println("文件数据长度不足，无法包含消息头部");
                return;
            }

            // 使用新的extractPcmData方法提取纯PCM数据
            byte[] pcmData = extractPcmData(pcmFileData);
            
            // 测试快速提取头部信息
            long quickStartTimeId = extractStartTimeId(pcmFileData);
            int quickMessageId = extractMessageId(pcmFileData);
            System.out.println("快速提取 - StartTimeId: " + quickStartTimeId);
            System.out.println("快速提取 - MessageId: " + quickMessageId);

            // 显示提取的纯PCM数据信息
            System.out.println("PCM数据长度: " + pcmData.length + " 字节");
           
            // 将PCM数据转换为十六进制格式并显示
            StringBuilder hexString = new StringBuilder();
            for (byte b : pcmData) {
                hexString.append(String.format("%02X ", b));
            }
            System.out.println("PCM数据（十六进制）: " + hexString.toString().trim());
            
        } catch (IOException e) {
            System.err.println("读取文件出错: " + e.getMessage());
            System.err.println("请确保 " + pcmFileName + " 文件存在于当前目录");
        } catch (Exception e) {
            System.err.println("测试出错: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
