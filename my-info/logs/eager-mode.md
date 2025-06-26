INFO 06-26 16:55:19 [worker.py:291] Memory profiling takes 0.71 seconds
INFO 06-26 16:55:19 [worker.py:291] the current vLLM instance can use total_gpu_memory (23.65GiB) x gpu_memory_utilization (0.40) = 9.46GiB
INFO 06-26 16:55:19 [worker.py:291] model weights take 0.70GiB; non_torch_memory takes 0.08GiB; PyTorch activation peak memory takes 1.12GiB; the rest of the memory reserved for KV Cache is 7.57GiB.
INFO 06-26 16:55:19 [executor_base.py:112] # cuda blocks: 41345, # CPU blocks: 21845
INFO 06-26 16:55:19 [executor_base.py:117] Maximum concurrency for 32768 tokens per request: 20.19x
INFO 06-26 16:55:22 [model_runner.py:1512] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100%|██████████████████████████████████████| 70/70 [00:34<00:00,  2.02it/s]
INFO 06-26 16:55:56 [model_ru