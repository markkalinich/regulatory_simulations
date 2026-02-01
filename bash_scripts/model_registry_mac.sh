#!/bin/bash
# Model Registry for Mac (original configuration)
# This file is sourced by run_all_models.sh

FAMILIES=(gemma qwen llama)
GEMMA_SIZES=(270m 1b 4b 12b 27b)
QWEN_SIZES=(0.6b 1.7b 4b 8b 14b 32b)
LLAMA_SIZES=(1b 8b 70b)

MODEL_REGISTRY=$(cat <<'EOF'
gemma|270m|3|google.gemma-3-270m
gemma|1b|3|google/gemma-3-1b
gemma|4b|3|google/gemma-3-4b
gemma|12b|3|google/gemma-3-12b
gemma|27b|3|google/gemma-3-27b

qwen|0.6b|3|qwen3-0.6b
qwen|1.7b|3|qwen/qwen3-1.7b
qwen|4b|3|qwen/qwen3-4b
qwen|8b|3|qwen/qwen3-8b
qwen|14b|3|qwen/qwen3-14b
qwen|32b|3|qwen/qwen3-32b

llama|1b|3.2|llama-3.2-1b-instruct
llama|8b|3|meta-llama-3-8b-instruct
llama|70b|3|meta-llama-3-70b-instruct
EOF
)

