import os
import sys
import torch
import librosa
import torchaudio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > 0.8:  # max_val = 0.8
        speech = speech / speech.abs().max() * 0.8
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

# Initialize the model
cosyvoice = CosyVoice("iic/CosyVoice-300M")

# Load and process the reference voice audio
prompt_speech_16k = load_wav('audios/小猪配奇1.mp3', 16000)
prompt_speech_16k = postprocess(prompt_speech_16k)

# The text that matches the content in the reference audio file
prompt_text = "再见妈妈，努力工作哦！杨女士，你要工作吗？还是只要玩？"

# The new text you want to synthesize (you can change this to whatever you want to say)
text_to_speak = "你好，我是一个AI语音助手"

# Generate speech
for i in cosyvoice.inference_zero_shot(text_to_speak, prompt_text, prompt_speech_16k, stream=False):
    speech = i['tts_speech']
    
# Save the output
torchaudio.save('output.wav', speech, cosyvoice.sample_rate)