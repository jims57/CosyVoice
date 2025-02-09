import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch
import gc

torch.cuda.empty_cache()

# Force garbage collection
gc.collect()

# Check available memory
print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        print(f"Current GPU device: {torch.cuda.current_device()}")
        device = "cuda"
    except Exception as e:
        print(f"Error initializing CUDA: {e}")
        print("Falling back to CPU")
        device = "cpu"
else:
    device = "cpu"

print(f"==== Using device: {device} ====")

# Print GPU info
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory Usage:")
print(f"  Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
print(f"  Cached:    {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")


# ==== CosyVoice2 ====
# [CPU] 
# cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# [GPU]
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, fp16=True)

# Clone voice[小猪配奇]
prompt_speech_16k = load_wav('./audios/小猪配奇1_16k.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('他在旧书店找到一本泛黄的笔记本，封面上写着一个熟悉的名字。', '再见妈妈，努力工作哦！杨女士，你要工作吗？还是只要玩？', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# Clone voice[哆啦A梦]
prompt_speech_16k = load_wav('./audios/哆啦A梦1_16k.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('他在旧书店找到一本泛黄的笔记本，封面上写着一个熟悉的名字。', '地址就是这里没有啊，你看，云遊旅馆的招牌就挂在那里，没有吧？', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# Clone voice[樊登]
# prompt_speech_16k = load_wav('./audios/fd-3.MP3', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('他在旧书店找到一本泛黄的笔记本，封面上写着一个熟悉的名字。', '想来想去，不要，这份工作我不要了。就就，打算拒绝这份工作。后来这个作者听说这种事以后就特别担心。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)



# ==== CosyVoice ====
# 300M model
# cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M') 
# # cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-25Hz') 
# # prompt_speech_16k = load_wav('./audios/fd-3.MP3', 16000)
# # for i, j in enumerate(cosyvoice.inference_zero_shot('他在旧书店找到一本泛黄的笔记本，封面上写着一个熟悉的名字。', '想来想去，不要，这份工作我不要了。就就，打算拒绝这份工作。后来这个作者听说这种事以后就特别担心。', prompt_speech_16k, stream=False)):
# #     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


# prompt_speech_16k = load_wav('./audios/小猪配奇1.mp3', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('他在旧书店找到一本泛黄的笔记本，封面上写着一个熟悉的名字。', '再见妈妈，努力工作哦！杨女士，你要工作吗？还是只要玩？', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)