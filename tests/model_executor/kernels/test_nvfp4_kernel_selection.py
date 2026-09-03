# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A4 NVFP4 layers must not be served by a weight-only (W4A16) kernel while a W4A4
kernel is available. On sm12x the registry's first entry (CuTe-DSL W4A4) is sm100-only
and the second (CuTe-DSL W4A16) accepts any config, so registry order alone silently
skipped activation quantization there."""

from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCuteDslNvFp4LinearKernel,
    FlashInferCuteDslNvFp4W4A16LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
from vllm.platforms import PlatformEnum
from vllm.platforms.interface import DeviceCapability


def _supported(monkeypatch, kernel, ok: bool):
    monkeypatch.setattr(
        kernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (ok, None if ok else "no")),
    )
    monkeypatch.setattr(
        kernel, "can_implement", classmethod(lambda cls, config: (True, None))
    )


def _sm12x_registry(monkeypatch):
    import vllm.model_executor.kernels.linear as linear_mod

    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(
        linear_mod.current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(12, 1),
    )
    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "auto")
    monkeypatch.setattr(linear_mod.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setitem(
        linear_mod._POSSIBLE_NVFP4_KERNELS,
        PlatformEnum.CUDA,
        [
            FlashInferCuteDslNvFp4LinearKernel,  # sm100-only W4A4
            FlashInferCuteDslNvFp4W4A16LinearKernel,  # weight-only, sm100/sm12x
            FlashInferCutlassNvFp4LinearKernel,  # W4A4, sm100+ incl. sm12x
            MarlinNvFp4LinearKernel,  # weight-only fallback
        ],
    )
    _supported(monkeypatch, FlashInferCuteDslNvFp4LinearKernel, False)
    _supported(monkeypatch, FlashInferCuteDslNvFp4W4A16LinearKernel, True)
    _supported(monkeypatch, FlashInferCutlassNvFp4LinearKernel, True)
    _supported(monkeypatch, MarlinNvFp4LinearKernel, True)


def test_w4a4_prefers_a_w4a4_kernel_over_weight_only(monkeypatch):
    _sm12x_registry(monkeypatch)
    kernel = init_nvfp4_linear_kernel(use_a16=False)
    assert isinstance(kernel, FlashInferCutlassNvFp4LinearKernel)


def test_w4a16_keeps_its_explicit_weight_only_choice(monkeypatch):
    # W4A16 schemes never reach the registry walk: they take the explicit
    # branch that prefers CuTe-DSL W4A16 on sm100/103 and Marlin elsewhere.
    _sm12x_registry(monkeypatch)
    kernel = init_nvfp4_linear_kernel(use_a16=True)
    assert isinstance(kernel, MarlinNvFp4LinearKernel)


def test_w4a4_falls_back_to_weight_only_when_nothing_else_fits(
    monkeypatch, caplog_vllm
):
    _sm12x_registry(monkeypatch)
    _supported(monkeypatch, FlashInferCutlassNvFp4LinearKernel, False)
    kernel = init_nvfp4_linear_kernel(use_a16=False)
    assert isinstance(kernel, FlashInferCuteDslNvFp4W4A16LinearKernel)
    assert "activations are not quantized" in caplog_vllm.text
