#!/bin/bash

# 作者: Jimmy Gan
# 日期: 2025-07-31

# 步骤1：删除mp3_chunks_split_by_silence目录下所有mp3文件
rm -rf mp3_chunks_split_by_silence/*

# 步骤2：复制/Users/mac/jims57/EdgeDownload目录下所有pcm文件到pcm_chunks目录
# 只移动今天创建且文件名为chunk_*.pcm的文件到pcm_chunks目录
today=$(date +%Y-%m-%d)
find /Users/mac/jims57/EdgeDownload/ -maxdepth 1 -type f -name "chunk_*.mp3" -newermt "$today" ! -newermt "$today +1 day" -exec mv {} mp3_chunks_split_by_silence/ \;

# 步骤3：合并所有mp3文件为一个mp3文件
python combine_all_mp3_chunk_into_a_mp3.py --chunks-dir mp3_chunks_split_by_silence
