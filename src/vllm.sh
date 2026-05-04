#!/bin/bash

if [ "$1" == "stop" ]; then
    echo "停止 vLLM 服务..."
    pkill -f "vllm.entrypoints.api_server"
    echo "已停止"
    exit 0
fi

MODEL_PATH="$1"

if [ -z "$MODEL_PATH" ]; then
    echo "用法: $0 <模型路径>"
    echo "或: $0 stop"
    exit 1
fi

for i in {0..3}; do
    gpu_start=$((i * 2))
    gpu_end=$((gpu_start + 1))
    port=$((4100 + i))
    
    CUDA_VISIBLE_DEVICES=${gpu_start},${gpu_end} python -m vllm.entrypoints.api_server \
        --model ${MODEL_PATH} \
        --served-model-name $(basename "$MODEL_PATH") \
        --tensor-parallel-size 2 \
        --host 127.0.0.1 \
        --port ${port} \
        --trust-remote-code \
	      --swap-space 0 \
        --no-enable-chunked-prefill &
done

echo "所有vLLM服务已启动"