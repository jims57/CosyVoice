#!/bin/bash

# 作者: Jimmy Gan
# 日期: 2025-07-31

# 步骤1：删除pcm_chunks目录下所有pcm文件
rm -rf pcm_chunks/*

# 步骤2：复制/Users/mac/jims57/EdgeDownload目录下所有pcm文件到pcm_chunks目录
# 只移动今天创建且文件名为chunk_*.pcm的文件到pcm_chunks目录
today=$(date +%Y-%m-%d)
find /Users/mac/jims57/EdgeDownload/ -maxdepth 1 -type f -name "chunk_*.pcm" -newermt "$today" ! -newermt "$today +1 day" -exec mv {} pcm_chunks/ \;

# 步骤3：合并所有pcm文件为一个pcm文件
python combine_all_pcms_into_a_pcm.py

# 步骤4：将pcm文件转换为wav文件
python convert_pcm_to_wav.py --pcm-file pcm/pcm_chunks_combined.pcm --wav-file wav/pcm_chunks_combined.wav

# 步骤5：删除mp3_chunks_split_by_silence目录下所有mp3文件
rm -rf mp3_chunks_split_by_silence/*

# 步骤6：按静音分割pcm文件为mp3片段
python convert_pcm_to_mp3_chunks_by_silence.py --pcm-file pcm/pcm_chunks_combined.pcm  --volume-dB -40.0 --min-samples-window 960  --input-pcm-sample-rate 16000 --output-mp3-sample-rate 16000

# 步骤7：合并所有mp3片段为一个mp3文件
python combine_all_mp3_chunk_into_a_mp3.py --chunks-dir mp3_chunks_split_by_silence --output-name combined_from_mp3_chunks_split_by_silence.mp3