# 作者: Jimmy Gan
# 日期: 2024-12-19
# 功能: 将同一文件夹中的所有WAV文件按顺序合并为单个WAV文件
#
# 使用说明:
# 1. 安装依赖包:
#    pip install numpy soundfile wave
#
# 2. 文件权限设置:
#    chmod +x combine_all_wavs_in_same_folder_into_a_single_wav.py
#
# 3. 目录结构要求:
#    my-info/
#    ├── combine_all_wavs_in_same_folder_into_a_single_wav.py
#    ├── zero_shot_0.wav
#    ├── zero_shot_1.wav
#    ├── zero_shot_2.wav
#    └── ...
#
# 4. 使用方法:
#    方法1 - 直接运行（使用默认参数）:
#    python combine_all_wavs_in_same_folder_into_a_single_wav.py
#
#    方法2 - 在代码中调用函数（可自定义参数）:
#    from combine_all_wavs_in_same_folder_into_a_single_wav import combine_wav_files
#    combine_wav_files(
#        input_folder="/path/to/wav/files",
#        output_wav_path="/path/to/output.wav",
#        file_pattern="zero_shot_*.wav"
#    )
#
# 5. 输出文件:
#    - 默认输出: combined_wav_output.wav
#    - 位置: 与脚本同目录
#
# 6. 处理逻辑:
#    - 按文件名数字顺序排序（zero_shot_0.wav, zero_shot_1.wav, ...）
#    - 移除每个WAV文件的头部信息
#    - 合并音频数据
#    - 为合并后的文件添加正确的WAV头部信息

import os
import glob
import wave
import numpy as np
import soundfile as sf
from pathlib import Path

def combine_wav_files(input_folder, output_wav_path, file_pattern="zero_shot_*.wav"):
    """
    将指定文件夹中的所有WAV文件按顺序合并为单个WAV文件
    
    参数:
        input_folder: WAV文件所在文件夹路径
        output_wav_path: 输出WAV文件路径
        file_pattern: 文件匹配模式，默认"zero_shot_*.wav"
    """
    # 获取所有WAV文件并按名称排序
    wav_files = glob.glob(os.path.join(input_folder, file_pattern))
    
    # 修复排序逻辑，正确提取数字部分
    def extract_number(filename):
        basename = os.path.basename(filename)
        # 从 "zero_shot_0.wav" 中提取 "0"
        parts = basename.split('_')
        if len(parts) >= 3:
            number_part = parts[2].split('.')[0]  # 获取 "0" from "0.wav"
            try:
                return int(number_part)
            except ValueError:
                return 0
        return 0
    
    wav_files.sort(key=extract_number)
    
    if not wav_files:
        print(f"在文件夹 {input_folder} 中未找到WAV文件")
        return
    
    print(f"找到 {len(wav_files)} 个WAV文件:")
    for file in wav_files:
        print(f"  - {os.path.basename(file)}")
    
    # 读取并合并所有WAV文件
    combined_audio = []
    sample_rate = None
    channels = None
    total_samples = 0
    
    for wav_file in wav_files:
        print(f"正在处理: {os.path.basename(wav_file)}")
        
        # 使用soundfile读取WAV文件（自动处理头部信息）
        audio_data, sr = sf.read(wav_file)
        
        # 检查采样率和声道数是否一致
        if sample_rate is None:
            sample_rate = sr
            channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
        elif sr != sample_rate:
            print(f"警告: 文件 {wav_file} 的采样率 {sr}Hz 与第一个文件 {sample_rate}Hz 不一致")
            continue
        
        # 确保音频数据是二维数组（单声道或多声道）
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)
        
        combined_audio.append(audio_data)
        total_samples += len(audio_data)
        
        print(f"  样本数: {len(audio_data)}, 采样率: {sr}Hz, 声道数: {audio_data.shape[1]}")
    
    if not combined_audio:
        print("没有有效的音频数据可以合并")
        return
    
    # 合并所有音频数据
    final_audio = np.vstack(combined_audio)
    print(f"合并完成，总样本数: {total_samples}")
    
    # 如果是单声道，转换为1D数组
    if final_audio.shape[1] == 1:
        final_audio = final_audio.flatten()
    
    # 保存为WAV文件
    sf.write(output_wav_path, final_audio, sample_rate, subtype='PCM_16')
    print(f"WAV文件已保存: {output_wav_path}")
    
    # 显示文件信息
    duration = len(final_audio) / sample_rate
    print(f"音频时长: {duration:.2f} 秒")
    print(f"采样率: {sample_rate}Hz")
    print(f"声道数: {1 if final_audio.ndim == 1 else final_audio.shape[1]}")

def main():
    # 设置路径
    current_dir = Path(__file__).parent
    input_folder = str(current_dir)  # 当前脚本所在目录
    output_wav_path = current_dir / "combined_wav_output.wav"
    
    # 合并WAV文件
    combine_wav_files(
        input_folder=input_folder,
        output_wav_path=str(output_wav_path),
        file_pattern="zero_shot_*.wav"
    )

if __name__ == "__main__":
    main()
