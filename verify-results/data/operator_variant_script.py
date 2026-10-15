"""Verify PR #3004 (torchtune#2990 defect 3):
_LLMEvalWrapper._model_generate must respect the harness's per-request
max_gen_toks, communicated via max_length (= context_len + max_gen_toks).

The old code hardcoded max_generated_tokens=256. This test patches
torchtune.generation.generate to capture the kwarg and asserts the
derived budget matches the requested one (64/256/1024), plus the 256
fallback when max_length is absent.
"""
import sys
import torch
from unittest.mock import MagicMock, patch

from eleuther_eval import _LLMEvalWrapper

# ---- Build wrapper bypassing __init__ (no HF download needed) ----
wrapper = _LLMEvalWrapper.__new__(_LLMEvalWrapper)
wrapper._model = torch.nn.Module()
tok = MagicMock()
tok.eos_id = 1
tok.pad_id = 0
tok.stop_tokens = {1}
wrapper._tokenizer = tok
wrapper._max_seq_length = 4096
wrapper._batch_size = 1
wrapper._dtype = torch.float32
wrapper._enable_kv_cache = True
wrapper._device = torch.device("cpu")

captured = {}

def fake_generate(model, input, max_generated_tokens, **kwargs):
    captured["max_generated_tokens"] = max_generated_tokens
    bsz, seq = input.shape
    # Return fake output long enough; content doesn't matter for this test
    return torch.zeros(bsz, seq + max_generated_tokens, dtype=torch.long), None

with patch("eleuther_eval.local_kv_cache"), \
     patch("eleuther_eval.generate", side_effect=fake_generate):
    for requested in [64, 256, 1024]:
        captured.clear()
        seq_len = 100
        context = torch.zeros(1, seq_len, dtype=torch.long)
        # Harness sets: kwargs["max_length"] = context_len + max_gen_toks
        wrapper._model_generate(context, max_length=seq_len + requested)
        got = captured.get("max_generated_tokens")
        assert got == requested, f"FAIL: requested {requested}, got {got}"
        print(f"  requested={requested} -> max_generated_tokens={got} OK")

    # Fallback when max_length not provided
    captured.clear()
    wrapper._model_generate(torch.zeros(1, 100, dtype=torch.long))
    got = captured.get("max_generated_tokens")
    assert got == 256, f"FAIL: fallback expected 256, got {got}"
    print(f"  no max_length -> fallback 256 OK")

print("PASS: _model_generate respects harness max_gen_toks via max_length.")
