"""SSI metrics — moved to robocasa (benchmark = evaluation, voxposer = action).

This module is a thin re-export shim for backward compatibility. New code
should import from `robocasa.utils.ssi` directly.

The migration consolidates evaluation/metric responsibility in the benchmark
(robocasa) so the policy (Voxposer) only needs to provide actions; everything
that defines "what counts as safe success" lives next to the kitchen env.
"""
from robocasa.utils.ssi import (  # noqa: F401
    TIER_OF,
    TIER_R_B,
    TIERS,
    GROUPS,
    AXES,                       # NEW: 4-axis (J/v/d/a)
    LEGACY_AXES,                # back-compat 3-axis (J/v/d)
    AXIS_KEY,
    LEGACY_AXIS_KEY,
    AXIS_CAUTION_DIR,
    DT,
    ACCEL_SKIP,
    _avg,
    _group_of,
    _caution_indicator,
    ep_min_clearance,
    ep_jerk_max,
    ep_jerk_b_mean,             # NEW
    ep_v_b,                     # NEW
    ep_accel_b_mean,            # NEW
    enrich_with_boundary_stats, # NEW
    stratified_means,
    per_tier_deltas,
    ssi_srl,
    ssi_csr,
    ssi_oct_paired,
    ssi_v4,                     # NEW: 4-axis SSI
    ssi_lpath_paired,
    compute,
)
