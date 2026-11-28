"""Case builders and launch checks shared by the LayerNorm and AdaLayerNorm tests."""

import pytest
import torch
from tilelang import tvm

from tileops._kernels.norm._config import make_raw_sums_hold
from tileops._utils import WARP_LANES, get_sm_count, get_sm_version


def far_case(rows, n, kernel, dtype, row, mark):
    """A far-from-zero case: *rows* as `outlier_rows` reads them, *row* a `far_from_zero` row."""
    return pytest.param(
        rows, n, kernel, dtype, row, marks=mark, id=f"{kernel}-{n}-{str(dtype)[6:]}-{row}"
    )


def outlier_case(rows, n, kernel, leads, mark):
    """An outlier-lead case: *rows* as `outlier_rows` reads them, *leads* as `warp_leads` or
    ``"row"``."""
    return pytest.param(rows, n, kernel, leads, marks=mark, id=f"{kernel}-{n}-{leads}")


def outlier_rows(rows) -> int:
    """Rows of a case: *rows* itself, rows an SM when a float, or one CTA past a full wave."""
    if rows == "wave+1":
        return get_sm_count() + 1
    return int(rows * get_sm_count()) if isinstance(rows, float) else rows


def assert_raw_sums_fail(x: torch.Tensor) -> None:
    """Assert that every row of *x* fails `make_raw_sums_hold` at one add deep, the loosest
    bound any kernel takes, so the kernel takes the shift-robust rule the case tests."""
    rows = x.double().reshape(-1, x.shape[-1])
    var = rows.var(dim=-1, unbiased=False)
    square_mean = rows.square().mean(dim=-1)
    # The row nearest to holding: the least mean square for its variance.
    i = int((square_mean / var).argmin())
    holds = make_raw_sums_hold(str(x.dtype)[6:], 1)(float(square_mean[i]), float(var[i]))
    assert not bool(tvm.arith.Analyzer().simplify(holds))


def warp_leads(held, m: int, n: int, dtype: torch.dtype) -> tuple:
    """Each warp's first element in every cluster CTA's share of a row, as *held*, a
    `LayerNormHeldKernel`-like class, places *m* rows of *n*: the elements its warps take
    their first sums less."""
    sm_count, arch = get_sm_count(), get_sm_version()
    cluster = held._cluster(n, dtype, m, arch, sm_count)
    vec = held._vec(n, dtype, cluster)
    threads = held._threads(n, cluster, vec, m, sm_count)
    return tuple(
        cta * (n // cluster) + warp * WARP_LANES * vec
        for cta in range(cluster)
        for warp in range(threads // WARP_LANES)
    )


def check_held_launch(held, kernel: str, m: int, n: int, dtype: torch.dtype) -> None:
    """Assert that *held*'s rule places *m* rows of *n* as *kernel* names: "held" alone on
    its SM, "queued" with CTAs queued, "split" or "queued-split" likewise across a cluster,
    "fragment" past a thread's registers; skip a split where clusters do not exist."""
    sm_count, arch = get_sm_count(), get_sm_version()
    cluster = held._cluster(n, dtype, m, arch, sm_count)
    if "split" in kernel and arch < 90:
        pytest.skip("a row splits across a cluster on SM90 only")
    if kernel == "fragment":
        assert held._queued_past_registers(n, dtype, m, arch, sm_count)
        return
    assert not held._queued_past_registers(n, dtype, m, arch, sm_count)
    assert (m * cluster > sm_count) == kernel.startswith("queued")
    assert (cluster > 1) == ("split" in kernel)
