"""Verify PR #3003 (torchtune#2990 defect 2):
_VLMEvalWrapper._model_multimodal_generate must return the FULL sequence
(prompt + generated tokens), matching the lm-eval harness contract.

The harness (HFMultimodalLM.generate_until) does:
    cont_toks = cont_toks[context_enc.shape[1]:]
expecting prompt+generated. The old code returned only generated tokens,
so the slice produced an empty list.

This test mocks the VLM model/transform/KV-cache and asserts:
  1. Return shape is (1, prompt_len + num_generated)
  2. First prompt_len tokens equal the prompt
  3. Harness slice [prompt_len:] yields exactly the generated tokens (non-empty)
"""
import sys
import torch
from unittest.mock import MagicMock, patch

from eleuther_eval import _VLMEvalWrapper

# ---- Build wrapper bypassing __init__ (no VLM checkpoint needed) ----
wrapper = _VLMEvalWrapper.__new__(_VLMEvalWrapper)

GEN_ID = 5
STOP_ID = 999
VOCAB = 1000

state = {"calls": 0}

def fake_model(x, **kwargs):
    """Return (bsz, seq, vocab) logits; argmax is GEN_ID 3 times, then STOP_ID."""
    bsz = x.shape[0]
    seq = x.shape[1] if x.dim() > 1 else 1
    logits = torch.full((bsz, seq, VOCAB), -1e9)
    tid = STOP_ID if state["calls"] >= 3 else GEN_ID
    state["calls"] += 1
    logits[:, :, tid] = 0.0
    return logits

wrapper._model = fake_model
wrapper._device = torch.device("cpu")
wrapper._dtype = torch.float32
wrapper._batch_size = 1
wrapper._max_seq_length = 4096
wrapper._max_images_per_sample = 7

mock_transform = MagicMock()
mock_transform.image_seq_len = 10
mock_transform.stop_tokens = {STOP_ID}
wrapper._transform = mock_transform

# ---- Fake batch ----
prompt_len = 50
prompt = torch.randint(0, VOCAB, (1, prompt_len))
batch = {
    "input_ids": prompt.clone(),
    "encoder_mask": torch.ones(1, 1, prompt_len),
}

# ---- Run ----
with patch("eleuther_eval.local_kv_cache"):
    result = wrapper._model_multimodal_generate(batch, max_length=10, stop=[])

print(f"Return shape: {tuple(result.shape)}")
# 3x GEN_ID then STOP_ID => 4 generated tokens
expected_gen = [GEN_ID, GEN_ID, GEN_ID, STOP_ID]
assert result.shape[0] == 1, f"FAIL: batch dim {result.shape}"
assert result.shape[1] == prompt_len + len(expected_gen), \
    f"FAIL: expected seq len {prompt_len + len(expected_gen)}, got {result.shape[1]}"

# Prompt prefix intact
assert torch.equal(result[0, :prompt_len], prompt[0]), "FAIL: prompt prefix corrupted"
print("Prompt prefix intact OK")

# Generated suffix correct
gen_part = result[0, prompt_len:].tolist()
assert gen_part == expected_gen, f"FAIL: generated part {gen_part} != {expected_gen}"
print(f"Generated suffix OK: {gen_part}")

# ---- Simulate harness slice ----
sliced = result[0].tolist()[prompt_len:]
assert sliced == expected_gen, f"FAIL: harness slice gave {sliced}"
assert len(sliced) > 0, "FAIL: harness slice is empty (the original bug)"
print(f"Harness slice [:{prompt_len}:] -> {sliced} (non-empty, correct)")

print("PASS: _model_multimodal_generate returns full sequence; harness contract satisfied.")
