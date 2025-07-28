# 作者: Jimmy Gan
# 日期: 2025-07-28
# 功能: 将同一文件夹中的所有WAV文件转换为PCM格式

import os
import subprocess

def convert_wav_to_pcm(wav_file_path, pcm_file_path):
    """将WAV文件转换为PCM文件"""
    try:
        # 使用ffmpeg将WAV转换为PCM，提取原始音频数据
        command = [
            'ffmpeg',
            '-i', wav_file_path,
            '-f', 's16le',  # 16位有符号小端PCM格式
            '-acodec', 'pcm_s16le',  # PCM 16位编解码器
            '-y',  # 覆盖输出文件
            pcm_file_path
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"转换成功: {wav_file_path} -> {pcm_file_path}")
        else:
            print(f"转换失败 {wav_file_path}: {result.stderr}")
            
    except Exception as e:
        print(f"转换失败 {wav_file_path}: {str(e)}")

def main():
    """主函数：转换当前文件夹中的所有WAV文件为PCM格式"""
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 遍历当前目录中的所有文件
    for filename in os.listdir(current_dir):
        if filename.lower().endswith('.wav'):
            # 构建完整的文件路径
            wav_path = os.path.join(current_dir, filename)
            
            # 构建PCM文件路径（相同文件名，但扩展名为.pcm）
            pcm_filename = os.path.splitext(filename)[0] + '.pcm'
            pcm_path = os.path.join(current_dir, pcm_filename)
            
            # 转换WAV到PCM
            convert_wav_to_pcm(wav_path, pcm_path)

if __name__ == "__main__":
    main()
