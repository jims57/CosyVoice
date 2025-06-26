(cosyvoice_vllm) root@11e5cf377461:~/CosyVoice# nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0
(cosyvoice_vllm) root@11e5cf377461:~/CosyVoice# python -c "
import onnxruntime as ort
print('Available providers:', ort.get_available_providers())
if 'CUDAExecutionProvider' in ort.get_available_providers():
    print('CUDA provider available')
else:
    print('CUDA provider NOT available')
"
Available providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
CUDA provider available
(cosyvoice_vllm) root@11e5cf377461:~/CosyVoice# python -c "
import torch
print(f'PyTorch CUDNN: {torch.backends.cudnn.version()}')
print(f'PyTorch CUDA: {torch.version.cuda}')
print(f'CUDNN enabled: {torch.backends.cudnn.enabled}')
"
PyTorch CUDNN: 90100
PyTorch CUDA: 12.1
CUDNN enabled: True