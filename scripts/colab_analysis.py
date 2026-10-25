"""Run the Phase 2-5 analysis pipeline on a Colab VM (large RAM) and fetch the reports.

    .venv/bin/python scripts/colab_analysis.py launch --session hestfm-p2b
    .venv/bin/python scripts/colab_analysis.py status --session hestfm-p2b
    .venv/bin/python scripts/colab_analysis.py fetch  --session hestfm-p2b

`launch` uploads src/, scripts/, configs/ and starts scripts/remote_pipeline.sh detached (nohup), so dropped
CLI connections do not interrupt it; `status` prints the log tail; `fetch` downloads /content/reports into
reports/ (files only, no data). The data are the Colab outputs already on the VM disk.
"""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from colab_driver import ROOT, colab, py

REMOTE_CODE = "/content/hestfm"
EMB = "/content/hest_fm_project/embeddings"
REP = "/content/reports"
LOG = "/content/pipeline.log"


def launch(session):
    tmp = ROOT / "data" / "tmp_driver"
    tmp.mkdir(parents=True, exist_ok=True)
    tar = tmp / "code.tar"
    with tarfile.open(tar, "w") as t:
        for d in ("src", "scripts", "configs", "pyproject.toml"):
            t.add(ROOT / d, arcname=d, filter=lambda m: None if "__pycache__" in m.name else m)
    rc, out = colab("upload", "-s", session, str(tar), "/content/code.tar", timeout=600)
    assert rc == 0, out
    code = f"""
import os, subprocess, tarfile
os.makedirs({REMOTE_CODE!r}, exist_ok=True)
tarfile.open('/content/code.tar').extractall({REMOTE_CODE!r}, filter='data')
subprocess.Popen(['bash', '-c', 'cd {REMOTE_CODE} && PYTHONPATH={REMOTE_CODE}/src nohup bash scripts/remote_pipeline.sh {EMB} {REP} >> {LOG} 2>&1 &'],
                 start_new_session=True)
print('LAUNCHED')
"""
    print(py(session, code)[-500:])


def status(session, n=30):
    print(py(session, f"import os\nprint(open({LOG!r}).read()[-6000:] if os.path.exists({LOG!r}) else 'no log yet')")[-6000:])


def fetch(session):
    out = py(session, f"import tarfile\nt = tarfile.open('/content/reports.tar', 'w'); t.add({REP!r}, arcname='reports'); t.close(); print('TARRED')")
    assert "TARRED" in out, out
    tmp = ROOT / "data" / "tmp_driver" / "reports.tar"
    rc, out = colab("download", "-s", session, "/content/reports.tar", str(tmp), timeout=1800)
    assert rc == 0 and tmp.exists(), out
    with tarfile.open(tmp) as t:
        for m in t.getmembers():
            if not m.name.startswith("reports") or ".." in m.name:
                raise RuntimeError(f"unexpected member {m.name}")
        t.extractall(ROOT, filter="data")
    tmp.unlink()
    print("fetched into", ROOT / "reports")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["launch", "status", "fetch"])
    ap.add_argument("--session", required=True)
    a = ap.parse_args()
    {"launch": launch, "status": status, "fetch": fetch}[a.cmd](a.session)
