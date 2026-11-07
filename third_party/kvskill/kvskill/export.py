"""Exports the tensors of a KV-Skill carrier to a safetensors payload for the vLLM server. Usage:
python -m kvskill.export --artifact_path <dir> --out <payload.safetensors>."""

from __future__ import annotations

import argparse
import json


def export_tensors(*, cn_a, cn_b, cn_k, cn_v, cn_w, cn_bl, meta: dict, out: str) -> str:
    from safetensors.torch import save_file

    t = {"cn_a": cn_a, "cn_b": cn_b, "cn_k": cn_k, "cn_v": cn_v, "cn_w": cn_w, "cn_bl": cn_bl}
    t = {k: v.detach().float().contiguous().cpu() for k, v in t.items()}
    hdr = {
        "cn": "1",
        "cn_depths": json.dumps([int(x) for x in meta["cn_depths"]]),
        "cn_ds": str(int(meta["cn_ds"])),
        "cn_r": str(int(meta["cn_r"])),
        "cn_dm": str(int(meta.get("cn_dm") or t["cn_a"].shape[-1])),
        "cn_nlayers": str(int(meta["cn_nlayers"])),
        "base_model": str(meta.get("base_model", "")),
        "stage": str(meta.get("stage", "")),
    }
    save_file(t, out, metadata=hdr)
    return out


def export_from_theta(theta, out: str) -> str:
    return export_tensors(cn_a=theta.A, cn_b=theta.B, cn_k=theta.K, cn_v=theta.V,
                          cn_w=theta.wg, cn_bl=theta.bl, meta=theta.meta, out=out)


def export_from_artifact(path: str, out: str) -> str:
    from kvskill.artifact import KVSkillArtifact

    art = KVSkillArtifact.load(path)
    assert art.meta.get("is_kvskill"), f"{path} is not a KV-Skill artifact"
    return export_tensors(cn_a=art.cn_a, cn_b=art.cn_b, cn_k=art.cn_k, cn_v=art.cn_v,
                          cn_w=art.cn_w, cn_bl=art.cn_bl, meta=art.meta, out=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_path", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = export_from_artifact(a.artifact_path, a.out)
    from safetensors import safe_open
    with safe_open(p, framework="pt", device="cpu") as f:
        print(f"[cn_export] {p} meta={f.metadata()}", flush=True)


if __name__ == "__main__":
    main()
