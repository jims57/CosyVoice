"""
Author: Jimmy Gan - 2025-07-22
将asr_wav文件夹中的所有wav文件转换为44kHz单声道格式

使用说明 (Usage Instructions) - 2025-07-22:
===========================================

1. 权限设置 (Permission Setting):
   chmod +x connvert_24khz_wav_to_44kh_wav.py

2. 依赖安装 (Package Installation):
   # 安装ffmpeg
   - Ubuntu/Debian: sudo apt-get install ffmpeg
   - macOS: brew install ffmpeg
   - Windows: 下载ffmpeg并添加到PATH环境变量

3. 使用方法 (Usage):
   # 直接运行脚本（处理当前目录所有wav文件）
   python connvert_24khz_wav_to_44kh_wav.py
   
   # 或者使脚本可执行后直接运行
   ./connvert_24khz_wav_to_44kh_wav.py

4. 参数说明 (Parameters):
   - 无需传入参数，脚本会自动处理当前目录下所有.wav文件
   - 输出文件会自动添加"_44khz"后缀

5. 输出结果 (Output):
   - 原文件: example.wav -> 新文件: example_44khz.wav
   - 格式: 44kHz采样率，单声道(mono)

===========================================
"""

import os
import glob
import subprocess
from pathlib import Path

def convert_wav_to_44khz_mono(input_file, output_file):
    """
    将wav文件转换为44kHz单声道格式
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
    """
    try:
        # 使用ffmpeg转换音频格式
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-ar', '44100',      # 设置采样率为44kHz
            '-ac', '1',          # 设置为单声道
            '-y',                # 覆盖输出文件
            output_file
        ]
        
        # 执行ffmpeg命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 转换完成: {input_file} -> {output_file}")
        else:
            print(f"❌ 转换失败 {input_file}: {result.stderr}")
        
    except Exception as e:
        print(f"❌ 转换失败 {input_file}: {str(e)}")

def main():
    """
    主函数：批量转换当前文件夹中的所有wav文件
    """
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    
    # 查找所有wav文件
    wav_files = glob.glob(os.path.join(current_dir, "*.wav"))
    
    if not wav_files:
        print("❌ 未找到任何wav文件")
        return
    
    print(f"📁 找到 {len(wav_files)} 个wav文件")
    
    # 转换每个文件
    for wav_file in wav_files:
        # 生成输出文件名（添加44khz关键词）
        file_path = Path(wav_file)
        output_name = f"{file_path.stem}_44khz{file_path.suffix}"
        output_file = file_path.parent / output_name
        
        # 跳过已经包含44khz的文件
        if "44khz" in file_path.stem.lower():
            print(f"⏭️ 跳过已转换文件: {wav_file}")
            continue
            
        convert_wav_to_44khz_mono(wav_file, str(output_file))
    
    print("🎉 批量转换完成!")

if __name__ == "__main__":
    main()
