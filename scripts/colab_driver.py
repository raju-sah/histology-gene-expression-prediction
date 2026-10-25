"""Drive a Colab-CLI session sample by sample, downloading each sample's outputs as soon as it is done.

    .venv/bin/python scripts/colab_driver.py --session hestfm-p2 --script /path/phase2_cli.py

Why: VM disk is lost if Colab reclaims the VM, and `colab exec` websockets drop on long runs. So each sample runs
in its own `exec` (with SAMPLES / VISIUM overridden in the generated script), the sample's files are tarred,
downloaded into data/embeddings/ and extracted. If an exec drops, the driver waits until the kernel is idle and
re-runs it; the notebook code skips finished steps. Samples already complete locally are skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hestfm import load_config  # noqa: E402

REMOTE = "/content/hest_fm_project/embeddings"
NOISE = ("new version", "colab update", "enable_update_check")


def colab(*args, stdin=None, timeout=None) -> tuple[int, str]:
    try:
        p = subprocess.run(["colab", *args], input=stdin, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:   # a hung client is handled like a dropped connection
        return 124, "Connection was lost (client timeout)"
    out = "\n".join(l for l in (p.stdout + p.stderr).splitlines() if not any(n in l for n in NOISE))
    return p.returncode, out


def py(session, code, timeout=600) -> str:
    return colab("exec", "-s", session, stdin=code, timeout=timeout)[1]


def wait_idle(session, max_min=180):
    for _ in range(max_min * 2):
        rc, out = colab("status", "-s", session, timeout=120)
        if "not found" in out:
            raise SystemExit(f"session {session} is gone (VM reclaimed); outputs downloaded so far are kept locally")
        if "IDLE" in out:
            return
        time.sleep(30)
    raise SystemExit("kernel stayed busy too long")


def local_done(dataset, sid, subdirs) -> bool:
    base = ROOT / "data" / "embeddings" / dataset
    q = base / "qc" / f"{sid}.json"
    if not q.exists() or json.load(open(q)).get("errors"):
        return False
    return all((base / d / f"{sid}.h5").exists() for d in subdirs)


def make_script(template: str, samples, visium) -> str:
    s = re.sub(r"^SAMPLES = \[.*\]$", f"SAMPLES = {samples!r}", template, count=1, flags=re.M)
    s = re.sub(r"^VISIUM = \[.*\]$", f"VISIUM = {visium!r}", s, count=1, flags=re.M)
    assert f"SAMPLES = {samples!r}" in s and f"VISIUM = {visium!r}" in s, "config lines not found"
    return s


def fetch(session, dataset, sid, subdirs, tmp: Path):
    """Tar the sample's files on the VM, download, extract into data/embeddings/<dataset>/."""
    paths = [f"{dataset}/{d}/{sid}.h5" for d in subdirs] + [f"{dataset}/qc/{sid}.json"]
    code = (f"import os, tarfile\nos.chdir({REMOTE!r})\n"
            f"ps = [p for p in {paths!r} if os.path.exists(p)]\n"
            f"with tarfile.open('/content/{sid}.tar', 'w') as t:\n    [t.add(p) for p in ps]\n"
            f"print('TARRED', len(ps))\n")
    out = py(session, code)
    if "TARRED" not in out:
        raise RuntimeError(f"tar failed for {sid}: {out[-500:]}")
    local = tmp / f"{sid}.tar"
    for attempt in range(6):   # the Colab contents proxy returns transient 503s
        local.unlink(missing_ok=True)
        rc, out = colab("download", "-s", session, f"/content/{sid}.tar", str(local), timeout=3600)
        if rc == 0 and local.exists() and "failed" not in out.lower():
            break
        print(f"[{sid}] download attempt {attempt + 1} failed: {out.strip().splitlines()[-1][:120] if out.strip() else rc}",
              flush=True)
        time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError(f"download failed for {sid} after retries")
    with tarfile.open(local) as t:
        for m in t.getmembers():   # only plain files under the expected dataset folder
            if not m.isfile() or not m.name.startswith(dataset + "/") or ".." in m.name:
                raise RuntimeError(f"unexpected member {m.name}")
        t.extractall(ROOT / "data" / "embeddings", filter="data")
    local.unlink()
    py(session, f"import os; os.remove('/content/{sid}.tar')")


def run_one(session, template, samples, visium, tmp, label, retries=4, exec_timeout=45 * 60):
    script = tmp / f"run_{label}.py"
    script.write_text(make_script(template, samples, visium))
    for attempt in range(retries):
        t0 = time.time()
        rc, out = colab("exec", "-s", session, "-f", str(script), timeout=exec_timeout)
        tail = "\n".join(l for l in out.splitlines() if re.search(r"==|embedded|pseudo-spots|cross-check|ok|Error|lost", l))
        print(f"[{label}] exec rc={rc} after {(time.time() - t0) / 60:.1f} min\n{tail[-1500:]}", flush=True)
        if "Connection was lost" in out or rc != 0:
            wait_idle(session)
            continue
        return
    raise RuntimeError(f"{label}: failed after {retries} attempts")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--script", required=True, help="CLI script made by nb_to_cli_script.py")
    ap.add_argument("--samples", nargs="*", default=None)
    ap.add_argument("--visium", nargs="*", default=None)
    a = ap.parse_args(argv)
    cfg = load_config()
    xs = a.samples if a.samples is not None else ["TENX189", "TENX188", "TENX187", "NCBI887", "NCBI888", "TENX160"]
    vs = a.visium if a.visium is not None else list(cfg["visium_organ"])
    template = Path(a.script).read_text()
    tmp = ROOT / "data" / "tmp_driver"
    tmp.mkdir(parents=True, exist_ok=True)
    xsub = ["uni2_h", "virchow2", "conch_v15", "uni2_h_c56", "expression", "expression_w050", "expression_w200", "cells"]
    for sid in xs:
        if local_done("phase2/xenium5k", sid, xsub):
            print(f"[{sid}] already downloaded, skip", flush=True)
            continue
        run_one(a.session, template, [sid], [], tmp, sid)
        fetch(a.session, "phase2/xenium5k", sid, xsub, tmp)
        print(f"[{sid}] downloaded; done locally: {local_done('phase2/xenium5k', sid, xsub)}", flush=True)
    # Visium needs the Xenium 5k gene union on the VM; upload it from the local expression files.
    import h5py
    genes = set()
    for sid in xs:
        p = ROOT / "data/embeddings/phase2/xenium5k/expression" / f"{sid}.h5"
        if p.exists():
            with h5py.File(p) as f:
                genes |= set(f["genes"][:].astype(str))
    (tmp / "genes_5k.txt").write_text("\n".join(sorted(genes)))
    colab("upload", "-s", a.session, str(tmp / "genes_5k.txt"), "/content/genes_5k.txt", timeout=600)
    vsub = ["uni2_h", "virchow2", "conch_v15", "expression"]
    for sid in vs:
        if local_done("phase4/visium", sid, vsub):
            print(f"[{sid}] already downloaded, skip", flush=True)
            continue
        run_one(a.session, template, [], [sid], tmp, sid)
        fetch(a.session, "phase4/visium", sid, vsub, tmp)
        print(f"[{sid}] downloaded; done locally: {local_done('phase4/visium', sid, vsub)}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
