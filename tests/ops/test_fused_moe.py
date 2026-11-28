"""Tests for FusedMoE, the routed MoE FFN operator."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._ops.moe import (
    FusedMoEFwdOp,
    FusedTopKFwdOp,
)
from tileops._utils import get_shared_memory_optin
from workloads.device import run_device
from workloads.moe import FusedMoEWorkload, moe_verification
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts as _vllm_fused_experts,
    )

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


def _vllm_runs_on(device) -> bool:
    """Whether vLLM is installed and its platform is *device*'s type."""
    if not _VLLM_AVAILABLE:
        return False
    from vllm.platforms import current_platform

    return current_platform.device_type == torch.device(device).type


def _workload(T, E, K, H, F, dtype="bfloat16", **params) -> FusedMoEWorkload:
    """One ``FusedMoEFwdOp`` call; *params* are its other manifest parameters and ``some``."""
    call = manifest_call("FusedMoEFwdOp", {"D": dtype}, T=T, E=E, H=H, F=F, top_k=K, **params)
    return FusedMoEWorkload(call)


class Qwen3Fixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " scoring_func, renormalize, dtype",
            [
                pytest.param(
                    32,
                    8,
                    2,
                    64,
                    32,
                    "softmax",
                    False,
                    "bfloat16",
                    marks=pytest.mark.smoke,
                    id="smoke-softmax-bf16",
                ),
                pytest.param(
                    32,
                    8,
                    2,
                    64,
                    32,
                    "softmax",
                    True,
                    "float16",
                    marks=pytest.mark.smoke,
                    id="smoke-softmax-renorm-fp16",
                ),
                pytest.param(
                    512,
                    128,
                    8,
                    2048,
                    1024,
                    "softmax",
                    False,
                    "bfloat16",
                    marks=pytest.mark.full,
                    id="qwen3-small",
                ),
                pytest.param(
                    2048,
                    128,
                    8,
                    2048,
                    1024,
                    "softmax",
                    False,
                    "bfloat16",
                    marks=pytest.mark.full,
                    id="qwen3-medium",
                ),
                pytest.param(
                    512,
                    256,
                    8,
                    2048,
                    1024,
                    "sigmoid",
                    True,
                    "bfloat16",
                    marks=pytest.mark.full,
                    id="deepseek-small",
                ),
            ],
        )
    ]


@Qwen3Fixture
def test_fused_moe_qwen3(
    num_tokens,
    num_experts,
    top_k,
    hidden_size,
    ffn_size,
    scoring_func,
    renormalize,
    dtype,
) -> None:
    workload = _workload(
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        ffn_size,
        dtype,
        scoring_func=scoring_func,
        renormalize=renormalize,
    )
    inputs = workload.gen_inputs()
    op = FusedMoEFwdOp(**workload.arguments())
    TestBase.check(workload, op, *inputs)

    hidden, gating, w_gate_up, w_down, *_ = inputs
    # vLLM's fp32 Triton tiles take up to 128 KB of shared memory, over SM86/SM89's 99 KB.
    if (
        _vllm_runs_on(hidden.device)
        and hidden.is_cuda
        and scoring_func == "softmax"
        and get_shared_memory_optin(hidden.device.index) >= 128 * 1024
    ):
        topk_weights, topk_ids = FusedTopKFwdOp(top_k, scoring_func, renormalize)(gating)
        out_vllm = _vllm_fused_experts(
            hidden.float(),
            w_gate_up.float(),
            w_down.float(),
            topk_weights,
            topk_ids,
        ).to(hidden.dtype)
        compare_outputs(op(*inputs).float(), out_vllm.float(), moe_verification(2))


# The grouped-GEMM epilogue race these cases guard shows only inside the full FusedMoE pipeline
# and on a small fraction of calls, so the nightly case repeats the op many times. Shapes are
# (T, E, top_k, H, F).
_DET_CASES = [
    pytest.param(
        dict(shape=(32, 8, 2, 64, 32), reps=3),
        marks=pytest.mark.smoke,
        id="smoke",
    ),
    pytest.param(
        dict(shape=(2048, 128, 8, 2048, 1024), reps=200),
        marks=pytest.mark.nightly,
        id="qwen3-medium",
    ),
]


@pytest.mark.parametrize("case", _DET_CASES)
def test_fused_moe_deterministic(case):
    """Every call on fixed inputs matches the reference and is bitwise-identical to the first.

    Guards the cooperative grouped-GEMM epilogue, which must order each warp group's
    register-to-``C_shared`` write before the async TMA store reads it.
    """
    reps = case["reps"]
    workload = _workload(*case["shape"])
    inputs = workload.gen_inputs()
    op = FusedMoEFwdOp(**workload.arguments())
    TestBase.check(workload, op, *inputs)
    first = op(*inputs)
    for i in range(reps):
        assert torch.equal(op(*inputs), first), f"call {i + 1}/{reps} differs from the first"


@pytest.mark.smoke
def test_fused_moe_gpt_oss_swigluoai_and_expert_biases() -> None:
    """gpt-oss's routing (softmax with renormalize), swigluoai and both expert biases meet the
    reference."""
    call = manifest_call(
        "FusedMoEFwdOp",
        {"D": "bfloat16"},
        T=32,
        E=8,
        H=64,
        F=32,
        top_k=2,
        scoring_func="softmax",
        renormalize=True,
        activation="swigluoai",
        some=["b_gate_up", "b_down"],
    )
    workload = FusedMoEWorkload(call)
    TestBase.check(workload, FusedMoEFwdOp(**call.arguments()), *workload.gen_inputs())


class KimiFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size,"
            " routed_scaling_factor, with_correction_bias, dtype",
            [
                pytest.param(
                    32,
                    8,
                    2,
                    64,
                    32,
                    1.0,
                    False,
                    "bfloat16",
                    marks=pytest.mark.smoke,
                    id="smoke-nobias-bf16",
                ),
                pytest.param(
                    32,
                    8,
                    2,
                    64,
                    32,
                    2.827,
                    True,
                    "float16",
                    marks=pytest.mark.smoke,
                    id="smoke-bias-scale-fp16",
                ),
                pytest.param(
                    64,
                    384,
                    8,
                    64,
                    32,
                    2.827,
                    True,
                    "bfloat16",
                    marks=pytest.mark.full,
                    id="kimi-k2-64-tokens-bf16",
                ),
                pytest.param(
                    512,
                    384,
                    8,
                    256,
                    128,
                    2.827,
                    True,
                    "bfloat16",
                    marks=pytest.mark.full,
                    id="kimi-k2-small-bf16",
                ),
                pytest.param(
                    2048,
                    384,
                    8,
                    256,
                    128,
                    2.827,
                    True,
                    "bfloat16",
                    marks=pytest.mark.full,
                    id="kimi-k2-medium-bf16",
                ),
            ],
        )
    ]


@KimiFixture
def test_fused_moe_kimi(
    num_tokens,
    num_experts,
    top_k,
    hidden_size,
    ffn_size,
    routed_scaling_factor,
    with_correction_bias,
    dtype,
) -> None:
    workload = _workload(
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        ffn_size,
        dtype,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=routed_scaling_factor,
        some=["correction_bias"] if with_correction_bias else [],
    )
    TestBase.check(workload, FusedMoEFwdOp(**workload.arguments()), *workload.gen_inputs())


@pytest.mark.smoke
def test_grouped_routing_scales_the_output_once() -> None:
    """DeepSeek-V3 grouped routing scales the routed output by the scaling factor once."""
    call = manifest_call(
        "FusedMoEFwdOp",
        {"D": "bfloat16"},
        T=32,
        E=256,
        H=64,
        F=32,
        top_k=8,
        num_expert_group=8,
        topk_group=4,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
        some=["correction_bias"],
    )
    workload = FusedMoEWorkload(call)
    op = FusedMoEFwdOp(**call.arguments())
    TestBase.check(workload, op, *workload.gen_inputs())


class VllmFixture(FixtureBase):
    PARAMS = [
        (
            "num_tokens, num_experts, top_k, hidden_size, ffn_size, routed_scaling_factor, dtype",
            [
                pytest.param(
                    32,
                    8,
                    2,
                    64,
                    32,
                    1.0,
                    "bfloat16",
                    marks=pytest.mark.smoke,
                    id="smoke-bf16",
                ),
                pytest.param(
                    32,
                    8,
                    2,
                    64,
                    32,
                    2.827,
                    "bfloat16",
                    marks=pytest.mark.full,
                    id="scale-bf16",
                ),
                pytest.param(
                    64,
                    384,
                    8,
                    64,
                    32,
                    2.827,
                    "bfloat16",
                    marks=pytest.mark.full,
                    id="kimi-k2-64-tokens-bf16",
                ),
                pytest.param(
                    512,
                    384,
                    8,
                    256,
                    128,
                    2.827,
                    "bfloat16",
                    marks=pytest.mark.full,
                    id="kimi-k2-small-bf16",
                ),
            ],
        )
    ]


@VllmFixture
def test_fused_moe_vs_vllm(
    num_tokens,
    num_experts,
    top_k,
    hidden_size,
    ffn_size,
    routed_scaling_factor,
    dtype,
) -> None:
    if not _vllm_runs_on(run_device()):
        pytest.skip("vLLM is not installed or has no platform for the run device")

    workload = _workload(
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        ffn_size,
        dtype,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=routed_scaling_factor,
        some=["correction_bias"],
    )
    hidden, gating, w_gate_up, w_down, correction_bias, *_ = workload.gen_inputs()
    topk_weights, topk_ids = FusedTopKFwdOp(top_k, "sigmoid", True)(gating, correction_bias)
    op = FusedMoEFwdOp(**workload.arguments())
    out_tileops = op(hidden, gating, w_gate_up, w_down, correction_bias)

    out_vllm = _vllm_fused_experts(hidden, w_gate_up, w_down, topk_weights, topk_ids)
    if routed_scaling_factor != 1.0:
        out_vllm = out_vllm * routed_scaling_factor

    compare_outputs(out_tileops.float(), out_vllm.float(), moe_verification(2))


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_the_routed_weights_are_priced_from_the_experts_stage() -> None:
    """The in-tree call prices the experts its routed-experts stage read; with no stage call
    the price falls to the data-independent bound of top_k experts."""
    T, E, K, H, F_ = 32, 8, 2, 64, 32
    hidden, gating, w_gate_up, w_down, *_ = _workload(T, E, K, H, F_).gen_inputs()

    op = FusedMoEFwdOp(top_k=K)
    op(hidden, gating, w_gate_up, w_down)
    _, topk_ids = FusedTopKFwdOp(K)(gating)
    active = topk_ids.unique().numel()
    per_expert = 3 * F_ * H * 2
    fixed = 2 * T * H * 2 + T * E * 4
    flops, nbytes = op.eval_roofline()
    # Softmax routing as FusedTopKFwdOp prices it, then per route the two GEMMs, the gated
    # activation and the weighted combine.
    assert flops == T * (4 * E + K + K * E) + T * K * (6 * F_ * H + 6 * F_ + 2 * H)
    assert nbytes == active * per_expert + fixed
    assert op.roofline_data_terms() == {"active_experts": active}
