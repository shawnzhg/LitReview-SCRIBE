"""Tests same_pool_prepare pin, materialize and verify on a synthetic acquisition at seeds 0, 1 and 2."""

from __future__ import annotations

import json
import sys

import pytest

from harness_env import ROOT

sys.path.insert(0, str(ROOT / "scribe" / "launchers"))
import same_pool_prepare as CP

T = "pmcid_TESTREUSE"


def _src_unit(root, system, seed=0):
    u = root / "level5" / system / "native_chain" / T / f"seed{seed}"
    u.mkdir(parents=True)
    (u / "evidence_bundle.json").write_text(json.dumps({"papers": [{"paper_id": "1"}], "evidence": [],
                                                        "content_hash": "sha256:x", "retrieval_status": "ok"}))
    (u / "pool_doccache.jsonl").write_text('{"paper_id": "1"}\n')
    (u / "task_spec.json").write_text("{}")
    (u / "trace.jsonl").write_text('{"window": "acquisition", "seq": 0}\n{"window": "synthesis", "seq": 1}\n')
    (u / "manifests.jsonl").write_text('{"window": "acquisition"}\n{"window": "synthesis"}\n')
    (u / "synthesis_graph.json").write_text("{}")
    return u


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pin_materialize_verify(seed, capsys, tmp_path):
    src_system = "SCRIBE"
    base = tmp_path / f"reuse_{seed}"
    src, dst = base / "src", base / "dst"
    _src_unit(src, src_system, seed)
    (dst / "level6").mkdir(parents=True)
    tasks = base / "tasks.txt"
    tasks.write_text(T + "\n")
    pins = base / "pins.json"
    CP.main(["pin", "--src-root", str(src / "level5"), "--src-system", src_system, "--seed", str(seed),
             "--tasks", str(tasks), "--out", str(pins)])
    p = json.loads(pins.read_text())
    assert p["src_system"] == src_system and p["seed"] == seed
    assert "synthesis_graph.json" not in p["tasks"][T]["files"]
    prov = base / "prov.json"
    CP.main(["materialize", "--pins", str(pins), "--tasks", str(tasks), "--level", "6", "--runs-root", str(dst),
             "--out", str(prov)])
    d = dst / "level6" / "SCRIBE" / "native_chain" / T / f"seed{seed}"
    assert d.is_dir() and not (d / "synthesis_graph.json").exists()
    assert (d / "manifests.jsonl").read_text() == '{"window": "acquisition"}\n'
    pv = json.loads(prov.read_text())
    assert pv["system"] == "SCRIBE" and pv["seed"] == seed and pv["tasks"][T]["dest_unit"] == str(d)
    with pytest.raises(SystemExit) as e:
        CP.main(["verify", "--provenance", str(prov), "--phase", "pre", "--out", str(base / "verify.json")])
    assert e.value.code == 0 and json.loads((base / "verify.json").read_text())["ok"] is True
