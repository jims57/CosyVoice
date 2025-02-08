import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch

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

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# Clone voice[小猪配奇]
# prompt_speech_16k = load_wav('./audios/小猪配奇1.mp3', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('他在旧书店找到一本泛黄的笔记本，封面上写着一个熟悉的名字。', '再见妈妈，努力工作哦！杨女士，你要工作吗？还是只要玩？', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# Clone voice[哆啦A梦]
prompt_speech_16k = load_wav('./audios/哆啦A梦1.mp3', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('他在旧书店找到一本泛黄的笔记本，封面上写着一个熟悉的名字。', '地址就是这里没有啊，你看，云遊旅馆的招牌就挂在那里，没有吧？', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)