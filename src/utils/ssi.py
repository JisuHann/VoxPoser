"""SSI metrics — implemented in robocasa (benchmark = evaluation, policy = action).

Thin re-export shim. New code should import `robocasa.utils.ssi` directly.

Trimmed to what is actually imported through it: `compute` (run_LMP) and `_avg`
(scripts/merge_workers.py). The previous list re-exported 27 names, of which 25
had no consumer, and four of those — ssi_srl, ssi_csr, ssi_lpath_paired and the
LEGACY_* axis maps — no longer exist: they were built on safe_success, which
ANDed boundary proximity with obstacle contact. Those are separate metrics now
(violation_ratio and collision_free_success), so a rate combining them measures
neither.

Re-exporting a name nobody imports is not free: it makes the shim look like the
module's public surface, so deleting anything upstream appears to break a
consumer that does not exist.
"""
from robocasa.utils.ssi import (  # noqa: F401
    _avg,
    compute,
)
