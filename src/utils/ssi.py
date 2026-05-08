"""SSI metrics — moved to robocasa (benchmark = evaluation, voxposer = action).

This module is a thin re-export shim for backward compatibility. New code
should import from `robocasa.utils.ssi` directly.

The migration consolidates evaluation/metric responsibility in the benchmark
(robocasa) so the policy (Voxposer) only needs to provide actions; everything
that defines "what counts as safe success" lives next to the kitchen env.
"""
from robocasa.utils.ssi import (  # noqa: F401
    TIER_OF,
    TIERS,
    GROUPS,
    AXES,
    AXIS_KEY,
    AXIS_CAUTION_DIR,
    _avg,
    _group_of,
    _caution_indicator,
    ep_min_clearance,
    ep_jerk_max,
    stratified_means,
    per_tier_deltas,
    ssi_srl,
    ssi_oct_paired,
    compute,
)
