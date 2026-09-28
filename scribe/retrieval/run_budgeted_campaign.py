#!/usr/bin/env python3
"""Generation driver of same-pool runs: loads the K_cap file, installs the pool backend's generation
half and runs run_campaign_pool. Usage: python run_budgeted_campaign.py --phase generation
<run_campaign options>."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "harness" / "runners"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    import run_campaign_pool as RCP
    if RCP._phase_of(argv) != "generation":
        sys.exit("### cap budget: run_budgeted_campaign.py serves --phase generation; the acquisition runs "
                 "run_acquisition.py")
    import budget as PBK
    try:
        kc = PBK.load_kcap()
    except PBK.KcapConfigError as e:
        sys.exit(f"### cap budget: refused before any unit: {e}")
    print(f"### K_cap file {kc['path']} sha256 {kc['sha256'][:16]} ({len(kc['tasks'])} tasks, rule "
          f"{PBK.CAP_RULE})", flush=True)
    try:
        PBK.install_generation_kcap(kc)
    except BaseException as e:
        sys.exit(f"### cap budget: generation refused before any unit: {type(e).__name__}: {e}")
    RCP.main(argv)


if __name__ == "__main__":
    main()
