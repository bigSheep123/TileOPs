"""Tests for FusedMoESharedExpertFwdOp, FusedMoE with a shared expert."""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops._kernels.gemm.call_spec import GEMMCall
from tileops._kernels.gemm.dense import GEMMTMAKernel
from tileops._kernels.moe import SharedExpertMLPKernel
from tileops._ops.moe import FusedMoESharedExpertFwdOp, SharedExpertMLPFwdOp
from tileops._ops.moe.fused_moe import FusedMoEFwdOp
from tileops.backend import _registry as registry
from tileops.backend._registry import register_kernel_type
from workloads.device import run_device
from workloads.moe import FusedMoESharedExpertWorkload, SharedExpertMLPWorkload
from workloads.workload_base import manifest_call


def _workload(T, E, K, H, F, S=None, **params) -> FusedMoESharedExpertWorkload:
    """One bfloat16 ``FusedMoESharedExpertFwdOp`` call, with the shared expert's weights when
    *S* is given; *params* are its other manifest parameters."""
    if S is not None:
        params.update(S=S, some=["shared_w_gate_up", "shared_w_down"])
    call = manifest_call(
        "FusedMoESharedExpertFwdOp", {"D": "bfloat16"}, T=T, E=E, H=H, F=F, top_k=K, **params
    )
    return FusedMoESharedExpertWorkload(call)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_the_shared_expert_matches_its_reference(dtype):
    workload = SharedExpertMLPWorkload(
        manifest_call("SharedExpertMLPFwdOp", {"D": dtype}, T=32, H=256, S=128)
    )
    inputs = workload.gen_inputs()
    TestBase.check(workload, SharedExpertMLPFwdOp(), *inputs)


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "tokens, hidden, ffn, dtype",
    [
        # Fused gate/up on 192-wide tiles; the down GEMM on 224-wide tiles staged in halves.
        pytest.param(1024, 256, 512, "bfloat16", marks=pytest.mark.smoke, id="fused-gate-up"),
        # Fused gate/up on 192-wide tiles over three waves, the last row and column tiles
        # ragged.
        pytest.param(600, 256, 7384, "float16", marks=pytest.mark.smoke, id="fused-waves-ragged"),
        # Separate gate/up on 256-wide tiles.
        pytest.param(4096, 256, 512, "bfloat16", marks=pytest.mark.full, id="separate"),
        # Separate gate/up over more than one wave.
        pytest.param(4096, 256, 2048, "bfloat16", marks=pytest.mark.full, id="separate-two-waves"),
        # An expert narrower than the model: only the gate/up GEMM, fused on 224-wide tiles,
        # on the template.
        pytest.param(512, 512, 256, "float16", marks=pytest.mark.full, id="narrow-expert"),
        # An expert width the fused epilogue cannot store keeps the separate activation.
        pytest.param(
            512, 1024, 1020, "bfloat16", marks=pytest.mark.full, id="narrow-expert-unfusable"
        ),
    ],
)
def test_the_shared_expert_runs_the_dense_template_past_its_threshold(tokens, hidden, ffn, dtype):
    """A shared expert past ``template_min_m`` runs its gate/up GEMM, and its down GEMM when
    the expert is at least as wide as the model, on the SM90 dense template."""
    workload = SharedExpertMLPWorkload(
        manifest_call("SharedExpertMLPFwdOp", {"D": dtype}, T=tokens, H=hidden, S=ffn)
    )
    op = SharedExpertMLPFwdOp()
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_a_published_threshold_decomposes_the_next_call_again():
    """``template_min_m`` decides which GEMMs the expert runs, so publishing it decomposes the
    next call from the new value."""
    workload = SharedExpertMLPWorkload(
        manifest_call("SharedExpertMLPFwdOp", {"D": "bfloat16"}, T=1024, H=256, S=512)
    )
    op = SharedExpertMLPFwdOp()
    inputs = workload.gen_inputs()
    TestBase.check(workload, op, *inputs)
    (kernel,) = op.iter_kernels()
    (templated,) = kernel._generation.programs.values()
    kernel.config = {**kernel.config, "template_min_m": 2048}
    TestBase.check(workload, op, *inputs)
    decomposed = [p for p in kernel._generation.programs.values() if isinstance(p, tuple)]
    assert [type(k) for k in decomposed[0][:2]] != [type(k) for k in templated[:2]]


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "tokens, hidden, ffn",
    [
        # The selector serves the down GEMM with the stream tile.
        pytest.param(70, 8456, 1048, marks=pytest.mark.smoke, id="stream"),
        # The selector serves the down GEMM with a two-tile fill cluster, which a 128-row
        # tile cannot launch.
        pytest.param(100, 2048, 1024, marks=pytest.mark.full, id="fill-cluster"),
    ],
)
def test_a_published_tile_runs_as_the_basic_gemm_it_states(tokens, hidden, ffn):
    """A published GEMM tile naming no structure runs each dense GEMM as that basic tile, also
    one the selector serves with a config outside the basic schema on a 132-SM H200."""
    workload = SharedExpertMLPWorkload(
        manifest_call("SharedExpertMLPFwdOp", {"D": "bfloat16"}, T=tokens, H=hidden, S=ffn)
    )
    op = SharedExpertMLPFwdOp()
    inputs = workload.gen_inputs()
    TestBase.check(workload, op, *inputs)
    (kernel,) = op.iter_kernels()
    (selected,) = [p for p in kernel._generation.programs.values() if isinstance(p, tuple)]
    call = GEMMCall(
        m=tokens, n=hidden, k=ffn, dtype=torch.bfloat16, trans_b=True, device=run_device()
    )
    basic_keys = GEMMTMAKernel._MODAL_CONFIG.keys()
    if call.calibration == "h200" and call.sm_count == 132:
        assert selected[1].config.keys() != basic_keys
    tile = {"block_m": 128, "block_n": 64, "block_k": 64, "num_stages": 4}
    kernel.config = {**kernel.config, **tile}
    TestBase.check(workload, op, *inputs)
    (published,) = [p for p in kernel._generation.programs.values() if isinstance(p, tuple)]
    for gemm in published[:2]:
        assert gemm.config.keys() == basic_keys
        assert {key: gemm.config[key] for key in tile} == tile


@pytest.mark.sm90
@pytest.mark.parametrize(
    "hidden, ffn",
    [
        pytest.param(2048, 1408, marks=pytest.mark.smoke, id="fused-gate-up"),
        pytest.param(1024, 6144, marks=pytest.mark.full, id="down"),
    ],
)
def test_the_shared_expert_splits_k_at_32_tokens(hidden, ffn):
    """At 32 tokens a long contraction runs split over K: the gate/up product fused with
    its activation, or the down product on the calibrated small-M band."""
    workload = SharedExpertMLPWorkload(
        manifest_call("SharedExpertMLPFwdOp", {"D": "bfloat16"}, T=32, H=hidden, S=ffn)
    )
    TestBase.check(workload, SharedExpertMLPFwdOp(), *workload.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "num_tokens",
    [pytest.param(32, marks=pytest.mark.smoke), pytest.param(512, marks=pytest.mark.full)],
)
def test_fused_moe_shared_expert_basic(num_tokens):
    """Both halves meet the reference, and the routed half is FusedMoE's output."""
    K = 2
    workload = _workload(num_tokens, 8, K, 64, 32, S=16)
    inputs = workload.gen_inputs()
    op = FusedMoESharedExpertFwdOp(**workload.arguments())
    TestBase.check(workload, op, *inputs)

    _, routed_out = op(*inputs)
    assert torch.equal(routed_out, FusedMoEFwdOp(top_k=K)(*inputs[:4]))


@pytest.mark.smoke
def test_fused_moe_shared_expert_none():
    """Without the shared weights, shared_out is None."""
    T, E, K, H, F = 16, 4, 2, 32, 16
    workload = _workload(T, E, K, H, F)
    shared_out, routed_out = FusedMoESharedExpertFwdOp(top_k=K)(*workload.gen_inputs())

    assert shared_out is None
    assert routed_out.shape == (T, H)


@pytest.mark.smoke
def test_fused_moe_shared_expert_refuses_swigluoai():
    """The shared expert computes silu_and_mul and takes no expert bias, so the routed
    experts' gpt-oss activation is refused at construction."""
    with pytest.raises(ValueError, match="swigluoai"):
        FusedMoESharedExpertFwdOp(top_k=2, activation="swigluoai")


@pytest.mark.smoke
def test_the_shared_expert_refuses_a_non_silu_activation():
    """The shared-expert kernel applies silu_and_mul; with gelu routed experts the two halves
    would disagree, so a call passing the shared weights is refused."""
    T, E, H, F, S = 4, 4, 128, 128, 128
    op = FusedMoESharedExpertFwdOp(2, activation="gelu_and_mul")
    hidden, gating, w_gate_up, w_down, *_ = _workload(T, E, 2, H, F).gen_inputs()
    args = (
        hidden,
        gating,
        w_gate_up,
        w_down,
        None,
        torch.randn(2 * S, H, dtype=torch.bfloat16, device=run_device()),
        torch.randn(H, S, dtype=torch.bfloat16, device=run_device()),
    )
    with pytest.raises(ValueError, match="silu_and_mul"):
        op(*args)


@pytest.mark.smoke
def test_fused_moe_shared_expert_tp():
    """Each rank's shared output is its shard's partial sum, and the routed output does not
    depend on the rank."""
    K = 2
    for tp_rank in range(2):
        workload = _workload(32, 8, K, 64, 32, S=16, tp_size=2, tp_rank=tp_rank)
        inputs = workload.gen_inputs()
        op = FusedMoESharedExpertFwdOp(**workload.arguments())
        TestBase.check(workload, op, *inputs)

        _, routed_out = op(*inputs)
        assert torch.equal(routed_out, FusedMoEFwdOp(top_k=K)(*inputs[:4]))


@pytest.mark.smoke
def test_fused_moe_shared_expert_tp_rejects_local_shards():
    """TP contract: the op shards complete weights itself, so a TP-local shard paired with a
    complete weight disagrees on the shared width and is refused."""
    T, E, K, H, F, F_s, tp_size = 32, 8, 2, 64, 32, 16, 2
    dtype = torch.bfloat16
    dev = run_device()

    op = FusedMoESharedExpertFwdOp(
        top_k=K,
        scoring_func="softmax",
        renormalize=False,
        tp_size=tp_size,
        tp_rank=0,
    )

    hidden, gating, w_gate_up, w_down, *_ = _workload(T, E, K, H, F).gen_inputs()

    shard_size = F_s // tp_size

    # Pass TP-local gate_up shard instead of full weights → must raise
    bad_gate_up = torch.randn(2 * shard_size, H, dtype=dtype, device=dev)
    good_w_down = torch.randn(H, F_s, dtype=dtype, device=dev)
    with pytest.raises(ValueError, match="shared_w"):
        op(
            hidden,
            gating,
            w_gate_up,
            w_down,
            shared_w_gate_up=bad_gate_up,
            shared_w_down=good_w_down,
        )

    # Pass TP-local down shard instead of full weights → must raise
    good_gate_up = torch.randn(2 * F_s, H, dtype=dtype, device=dev)
    bad_w_down = torch.randn(H, shard_size, dtype=dtype, device=dev)
    with pytest.raises(ValueError, match="shared_w"):
        op(
            hidden,
            gating,
            w_gate_up,
            w_down,
            shared_w_gate_up=good_gate_up,
            shared_w_down=bad_w_down,
        )


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_a_replaced_shared_expert_kernel_is_the_one_built():
    """An implementation registered for the shared half's sub-op is the one built."""
    built = []

    class Replacement(SharedExpertMLPKernel):
        preferred_over = frozenset({"shared_expert_mlp"})

        def __init__(self, **kwargs):
            built.append(kwargs)
            super().__init__(**kwargs)

    T, E, K, H, F, F_s = 32, 8, 2, 64, 32, 16
    state = registry.snapshot()
    try:
        register_kernel_type("SharedExpertMLPFwdOp", "shared_expert_mlp_replacement", Replacement)
        op = FusedMoESharedExpertFwdOp(top_k=K)
    finally:
        registry.restore(state)

    op(*_workload(T, E, K, H, F, S=F_s).gen_inputs())
    assert built, "the replacement was never constructed"


@pytest.mark.smoke
def test_grouped_routing_with_the_shared_expert() -> None:
    """DeepSeek-V3 grouped routing with the shared expert scales only the routed output, once."""
    call = manifest_call(
        "FusedMoESharedExpertFwdOp",
        {"D": "bfloat16"},
        T=32,
        E=256,
        H=64,
        F=32,
        S=32,
        top_k=8,
        num_expert_group=8,
        topk_group=4,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
        some=["correction_bias", "shared_w_gate_up", "shared_w_down"],
    )
    workload = FusedMoESharedExpertWorkload(call)
    op = FusedMoESharedExpertFwdOp(**call.arguments())
    TestBase.check(workload, op, *workload.gen_inputs())
