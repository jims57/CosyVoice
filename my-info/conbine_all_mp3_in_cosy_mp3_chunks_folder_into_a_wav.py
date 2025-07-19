# 作者: Jimmy Gan
# 日期: 2024-12-19
# 功能: 将cosy_mp3_chunks文件夹中的所有PCM文件按顺序合并为单个WAV文件
#
# 使用说明:
# 1. 安装依赖包:
#    pip install numpy soundfile
#
# 2. 文件权限设置:
#    chmod +x conbine_all_mp3_in_cosy_mp3_chunks_folder_into_a_wav.py
#
# 3. 目录结构要求:
#    my-info/
#    ├── conbine_all_mp3_in_cosy_mp3_chunks_folder_into_a_wav.py
#    └── cosy_mp3_chunks/
#        ├── chunk_0.pcm
#        ├── chunk_1.pcm
#        ├── chunk_2.pcm
#        └── ...
#
# 4. 使用方法:
#    方法1 - 直接运行（使用默认参数）:
#    python conbine_all_mp3_in_cosy_mp3_chunks_folder_into_a_wav.py
#
#    方法2 - 在代码中调用函数（可自定义参数）:
#    from conbine_all_mp3_in_cosy_mp3_chunks_folder_into_a_wav import combine_pcm_files_to_wav
#    combine_pcm_files_to_wav(
#        input_folder="/path/to/pcm/files",
#        output_wav_path="/path/to/output.wav",
#        input_sample_rate=16000,    # 输入PCM文件采样率
#        output_sample_rate=16000    # 输出WAV文件采样率
#    )
#
# 5. 输出文件:
#    - 默认输出: combined_output.wav
#    - 位置: 与脚本同目录
#
# 6. 支持的PCM格式:
#    - 16位PCM (int16)
#    - 文件名格式: chunk_0.pcm, chunk_1.pcm, chunk_2.pcm, ...
#    - 按数字顺序自动排序合并

import os
import glob
import numpy as np
import soundfile as sf
from pathlib import Path

def combine_pcm_files_to_wav(input_folder, output_wav_path, 
                           input_sample_rate=16000, output_sample_rate=16000):
    """
    将指定文件夹中的所有PCM文件按顺序合并为单个WAV文件
    
    参数:
        input_folder: PCM文件所在文件夹路径
        output_wav_path: 输出WAV文件路径
        input_sample_rate: 输入PCM文件的采样率，默认16kHz
        output_sample_rate: 输出WAV文件的采样率，默认16kHz
    """
    # 获取所有PCM文件并按名称排序
    pcm_files = glob.glob(os.path.join(input_folder, "chunk_*.pcm"))
    pcm_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    if not pcm_files:
        print(f"在文件夹 {input_folder} 中未找到PCM文件")
        return
    
    print(f"找到 {len(pcm_files)} 个PCM文件:")
    for file in pcm_files:
        print(f"  - {os.path.basename(file)}")
    
    # 读取并合并所有PCM文件
    combined_audio = []
    total_samples = 0
    
    for pcm_file in pcm_files:
        print(f"正在处理: {os.path.basename(pcm_file)}")
        
        # 读取PCM文件
        with open(pcm_file, 'rb') as f:
            pcm_data = f.read()
        
        # 将字节数据转换为numpy数组 (假设16位PCM)
        audio_data = np.frombuffer(pcm_data, dtype=np.int16)
        combined_audio.append(audio_data)
        total_samples += len(audio_data)
        
        print(f"  样本数: {len(audio_data)}")
    
    # 合并所有音频数据
    final_audio = np.concatenate(combined_audio)
    print(f"合并完成，总样本数: {total_samples}")
    
    # 如果需要重采样
    if input_sample_rate != output_sample_rate:
        print(f"重采样: {input_sample_rate}Hz -> {output_sample_rate}Hz")
        # 这里可以添加重采样逻辑，如果需要的话
        # 暂时保持原采样率
    
    # 保存为WAV文件
    sf.write(output_wav_path, final_audio, output_sample_rate, subtype='PCM_16')
    print(f"WAV文件已保存: {output_wav_path}")
    
    # 显示文件信息
    duration = len(final_audio) / output_sample_rate
    print(f"音频时长: {duration:.2f} 秒")

def main():
    # 设置路径
    current_dir = Path(__file__).parent
    input_folder = current_dir / "cosy_mp3_chunks"
    output_wav_path = current_dir / "combined_output.wav"
    
    # 检查输入文件夹是否存在
    if not input_folder.exists():
        print(f"错误: 文件夹 {input_folder} 不存在")
        return
    
    # 合并PCM文件为WAV
    combine_pcm_files_to_wav(
        input_folder=str(input_folder),
        output_wav_path=str(output_wav_path),
        input_sample_rate=16000,  # 输入PCM文件采样率
        output_sample_rate=16000  # 输出WAV文件采样率
    )

if __name__ == "__main__":
    main()
