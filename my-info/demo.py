# 今天天气真不错，阳光明媚，适合出去散步。
# 我喜欢在周末的时候和朋友一起看电影，分享快乐时光。
# 学习新知识是一件很有趣的事情，让人感到充实和满足。
# 妈妈做的饭菜总是那么美味，充满了家的温暖和爱意。
# 春天来了，花儿绽放，小鸟歌唱，大自然充满了生机。
# 收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。
# 忙碌了一整周之后，这个周末我终于有机会好好放松了一下，去了几个城市里平时没空去的小众景点，还尝试了好几家新开的咖啡馆和旧货店，感觉整个人都充满了能量，已经开始期待下次的随心所欲的旅行了。

# The weather is beautiful today, with bright sunshine perfect for a walk.
# I enjoy watching movies with friends on weekends, sharing happy moments together.
# Learning new knowledge is very interesting and makes me feel fulfilled and satisfied.
# Mom's cooking is always so delicious, full of warmth and love from home.
# Spring has arrived, flowers are blooming, birds are singing, and nature is full of life.
# After spending the entire weekend exploring various hidden gems around the city, trying out new cafes and vintage shops, I'm feeling incredibly refreshed and already planning my next spontaneous adventure for sure.
# Did you guys see the video of that dude who was at the gym who took his earbuds and just smacked them against the wall because they would not stay in his ear during his set? I feel that man's pain on a personal level. I can't keep earbuds in my ear to save my life, especially earbuds. I have four pairs of them and I hate all of them cuz they all fucking suck. That's why I got these instead. These are the Lenovo Eraser X15 Pros. They are the best over-the-ear earbud that I've ever had. Now, I don't really like calling them buds cuz they're not like actual buds that go inside your ear. They they look just like this. They just kind of rest right over the top of your ear, just like that. And the cool part is, no matter what you're doing, they're not going to fall off your head. So if you like to go to the gym all the time, you won't have to worry about them falling out during any of your sets. And they're waterproof, so you don't have to worry about sweat. You can literally go swimming or take a shower in these. They're on sale right now for like $25 on Tik Tok, but with the right coupons, you can get them for like 10 bucks.


import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio


cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# Speaker prompt audio
# prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)  # 希望你以后能够做的比我还好呦。
# prompt_speech_16k = load_wav('./asset/cross_lingual_prompt.wav', 16000) # 在那之后，我们全资收购了那家公司，从而保持了管理层的一致性，确保利益与即将并入家族的资产保持一致。这就是我们有时不全盘买下的原因。
# prompt_speech_16k = load_wav('./asset/情感女声.wav', 16000)  # 我相信很多听友听到这首歌，应该是在96年90年代的那个夏天。
prompt_speech_16k = load_wav('./asset/man-short.wav', 16000) # Did you guys see the video of that dude who was at the gym who took his earbuds and just smacked them against the wall because they would not stay in his ear during his set? 

# ==== Example 1 ====
for i, j in enumerate(cosyvoice.inference_zero_shot("Did you guys see the video of that dude who was at the gym who took his earbuds and just smacked them against the wall because they would not stay in his ear during his set? I feel that man's pain on a personal level. I can't keep earbuds in my ear to save my life, especially earbuds. I have four pairs of them and I hate all of them cuz they all fucking suck. That's why I got these instead. These are the Lenovo Eraser X15 Pros. They are the best over-the-ear earbud that I've ever had. Now, I don't really like calling them buds cuz they're not like actual buds that go inside your ear. They they look just like this. They just kind of rest right over the top of your ear, just like that. And the cool part is, no matter what you're doing, they're not going to fall off your head. So if you like to go to the gym all the time, you won't have to worry about them falling out during any of your sets. And they're waterproof, so you don't have to worry about sweat. You can literally go swimming or take a shower in these. They're on sale right now for like $25 on Tik Tok, but with the right coupons, you can get them for like 10 bucks.", 'Did you guys see the video of that dude who was at the gym who took his earbuds and just smacked them against the wall because they would not stay in his ear during his set? ', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


# ==== Example 2 ====
# assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech_16k, 'my_zero_shot_spk') is True
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# cosyvoice.save_spkinfo()



