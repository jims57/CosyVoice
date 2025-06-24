CosyVoice2 vllm Usage
If you want to use vllm for inference, please install vllm==v0.9.0. Older vllm version do not support CosyVoice2 inference.

Notice that vllm==v0.9.0 has a lot of specific requirements, for example torch==2.7.0. You can create a new env to in case your hardward do not support vllm and old env is corrupted.

conda create -n cosyvoice_vllm --clone cosyvoice
conda activate cosyvoice_vllm
pip install vllm==v0.9.0 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
python vllm_example.py