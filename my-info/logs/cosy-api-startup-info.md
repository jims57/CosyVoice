CUDA available: True
CUDA version: 12.6
GPU device name: NVIDIA GeForce RTX 4090
Using device: cuda:0
Loading CosyVoice2 model with vllm...
/root/cosyvoice_vllm/lib/python3.10/site-packages/diffusers/models/lora.py:393: FutureWarning: `LoRACompatibleLinear` is deprecated and will be removed in version 1.0.0. Use of `LoRACompatibleLinear` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`.
  deprecate("LoRACompatibleLinear", "1.0.0", deprecation_message)
2025-06-26 17:45:53,997 INFO input frame rate=25
/root/cosyvoice_vllm/lib/python3.10/site-packages/torch/nn/utils/weight_norm.py:143: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
2025-06-26 17:45:55.513999882 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 8 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.
2025-06-26 17:45:55.516425743 [W:onnxruntime:, session_state.cc:1166 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
2025-06-26 17:45:55.516441003 [W:onnxruntime:, session_state.cc:1168 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
text.cc: festival_Text_init
open voice lang map failed
INFO 06-26 17:45:58 [__init__.py:31] Available plugins for group vllm.general_plugins:
INFO 06-26 17:45:58 [__init__.py:33] - lora_filesystem_resolver -> vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver
INFO 06-26 17:45:58 [__init__.py:36] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 06-26 17:45:58 [config.py:793] This model supports multiple tasks: {'reward', 'score', 'generate', 'embed', 'classify'}. Defaulting to 'generate'.
WARNING 06-26 17:45:58 [arg_utils.py:1583] --enable-prompt-embeds is not supported by the V1 Engine. Falling back to V0. 
INFO 06-26 17:45:58 [llm_engine.py:230] Initializing a V0 LLM engine (v0.9.0) with config: model='pretrained_models/CosyVoice2-0.5B/vllm', speculative_config=None, tokenizer='pretrained_models/CosyVoice2-0.5B/vllm', skip_tokenizer_init=True, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=pretrained_models/CosyVoice2-0.5B/vllm, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=None, chunked_prefill_enabled=False, use_async_output_proc=True, pooler_config=None, compilation_config={"compile_sizes": [], "inductor_compile_config": {"enable_auto_functionalized_v2": false}, "cudagraph_capture_sizes": [256, 248, 240, 232, 224, 216, 208, 200, 192, 184, 176, 168, 160, 152, 144, 136, 128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 4, 2, 1], "max_capture_size": 256}, use_cached_outputs=False, 
INFO 06-26 17:45:58 [cuda.py:292] Using Flash Attention backend.
INFO 06-26 17:45:59 [parallel_state.py:1064] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
INFO 06-26 17:45:59 [model_runner.py:1170] Starting to load model pretrained_models/CosyVoice2-0.5B/vllm...
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  9.19it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  9.18it/s]

INFO 06-26 17:45:59 [default_loader.py:280] Loading weights took 0.13 seconds
INFO 06-26 17:45:59 [model_runner.py:1202] Model loading took 0.6951 GiB and 0.204926 seconds
INFO 06-26 17:46:00 [worker.py:291] Memory profiling takes 0.69 seconds
INFO 06-26 17:46:00 [worker.py:291] the current vLLM instance can use total_gpu_memory (23.65GiB) x gpu_memory_utilization (0.40) = 9.46GiB
INFO 06-26 17:46:00 [worker.py:291] model weights take 0.70GiB; non_torch_memory takes 0.07GiB; PyTorch activation peak memory takes 1.12GiB; the rest of the memory reserved for KV Cache is 7.57GiB.
INFO 06-26 17:46:00 [executor_base.py:112] # cuda blocks: 41366, # CPU blocks: 21845
INFO 06-26 17:46:00 [executor_base.py:117] Maximum concurrency for 32768 tokens per request: 20.20x
INFO 06-26 17:46:03 [model_runner.py:1512] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.


2025-06-26 17:46:49,952 INFO synthesis text This is a longer text to warm up the V L L M engine with various text lengths and patterns for better first chunk performance.
WARNING 06-26 17:46:49 [preprocess.py:63] Using None for EOS token id because tokenizer is not initialized
2025-06-26 17:46:50,357 INFO yield speech len 1.8, rtf 0.22481772634718153
2025-06-26 17:46:50,573 INFO yield speech len 1.0, rtf 0.21644902229309082
  0%|                                                                                                             | 0/1 [00:00<?, ?it/s]
Warm-up cycle 4 completed in 657.81ms, generated 2 chunks
Enhanced model pre-warming completed in 10653.12ms
CosyVoice model loaded successfully!

==================================================
🚀 GPU ACCELERATION STATUS
==================================================
✅ CUDA Available: True
✅ CUDA Version: 12.6
✅ CUDNN Version: 90501
✅ CUDNN Enabled: True
✅ GPU Device: NVIDIA GeForce RTX 4090
✅ GPU Memory: 23.6 GB
✅ ONNX Providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
✅ ONNX CUDA Acceleration: ENABLED
✅ VLLM Acceleration: ENABLED
✅ VLLM GPU Memory Utilization: Configured
==================================================
🎯 Ready for high-performance TTS inference!
==================================================

INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9003 (Press CTRL+C to quit)