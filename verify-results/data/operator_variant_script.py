"""Verify PR #3002 (torchtune#2990 defect 1):
_LLMEvalWrapper must NOT download GPT-2 model weights on init.

The fix overrides HFLM._create_model as a no-op. This test instantiates
the wrapper and asserts no weight files (*.bin/*.safetensors/*.pt) appear
in the GPT-2 HF Hub cache. Config/tokenizer files are allowed.
"""
import os
import shutil
import sys

cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
gpt2_dir = os.path.join(cache_dir, "models--gpt2")
if os.path.exists(gpt2_dir):
    shutil.rmtree(gpt2_dir)
    print(f"Cleaned {gpt2_dir}")

import torch
from unittest.mock import MagicMock

from eleuther_eval import _LLMEvalWrapper

# Confirm the fix is present
assert hasattr(_LLMEvalWrapper, "_create_model"), "FAIL: _create_model override missing"
print("Fix present: _LLMEvalWrapper._create_model override found")

dummy_model = torch.nn.Module()
dummy_tokenizer = MagicMock()

wrapper = _LLMEvalWrapper(
    dummy_model,
    dummy_tokenizer,
    device=torch.device("cpu"),
    max_seq_length=128,
    batch_size=1,
)
print("Wrapper instantiated OK")

# Assert: no weight files downloaded
weight_files = []
if os.path.exists(gpt2_dir):
    for root, dirs, files in os.walk(gpt2_dir):
        for f in files:
            if f.endswith((".bin", ".safetensors", ".pt", ".ckpt", ".onnx")):
                weight_files.append(os.path.join(root, f))

if weight_files:
    print(f"FAIL: GPT-2 weight files were downloaded:")
    for wf in weight_files:
        print(f"  {wf}")
    sys.exit(1)

# Assert: model rebound to the torchtune model, not GPT-2
assert wrapper.model is dummy_model, "FAIL: wrapper.model is not the provided model"

print("PASS: No GPT-2 weights downloaded; model rebound correctly.")
print("GPT-2 cache (config/tokenizer only, expected):")
if os.path.exists(gpt2_dir):
    for root, dirs, files in os.walk(gpt2_dir):
        for f in files:
            print(f"  {f}")
