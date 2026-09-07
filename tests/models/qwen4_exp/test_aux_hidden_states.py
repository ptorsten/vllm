# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EAGLE-3 auxiliary hidden states of Qwen4Exp: the per-layer feature is the
hyper-connected multi-stream state averaged over the streams, and the default
layer set spans the stack."""

from types import SimpleNamespace

import torch

from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForCausalLM, Qwen4ExpModel


def test_aux_feature_averages_the_hc_streams() -> None:
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    model.config = SimpleNamespace(hc_count=4)
    streams = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 12)
    feature = model._aux_feature(streams)
    assert feature.shape == (2, 3)
    torch.testing.assert_close(feature, streams.view(2, 4, 3).mean(dim=1))


def test_default_aux_layers_span_the_stack() -> None:
    lm = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    lm.model = SimpleNamespace(layers=[None] * 48, aux_hidden_state_layers=())
    assert lm.get_eagle3_aux_hidden_state_layers() == (8, 18, 28, 38, 48)
    lm.set_aux_hidden_state_layers((4, 14))
    assert lm.model.aux_hidden_state_layers == (4, 14)
