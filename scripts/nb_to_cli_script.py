"""Turn a generated Colab notebook into a plain script for `colab exec -f` (Colab CLI, no Drive, no Secrets).

    .venv/bin/python scripts/nb_to_cli_script.py notebooks/colab_phase1_benchmark.ipynb /tmp/out.py [--root /content/hest_fm_project]

- `%%writefile` cells become Python that writes the same file;
- cells that mount Drive / read Colab Secrets, `!pip` and `!du` are dropped (packages: `colab install`;
  HF auth: a token already cached on the VM by `huggingface_hub.login` in the Colab UI);
- the Drive root is replaced by a local path on the VM, downloaded afterwards with `colab download`.
"""
from __future__ import annotations

import argparse

import nbformat

DROP = ("google.colab", "!pip")
DRIVE = "/content/drive/MyDrive/hest_fm_project"


def convert(nb_path, root):
    nb = nbformat.read(nb_path, 4)
    out = ["import os, sys",
           "os.environ.setdefault('HF_TOKEN_PATH', '/content/.hf_token')   # uploaded by the user; never read here",
           "os.makedirs('/content/lib', exist_ok=True)", "sys.path.insert(0, '/content/lib')",
           "from huggingface_hub import whoami",
           "print('HF login ok:', bool(whoami().get('name')))   # token cached on the VM by the user", ""]
    for c in nb.cells:
        if c.cell_type != "code" or any(d in c.source for d in DROP):
            continue
        src = c.source.replace(DRIVE, root)
        if src.startswith("%%writefile"):
            first, body = src.split("\n", 1)
            path = first.split(maxsplit=1)[1].strip()
            out.append(f"open({path!r}, 'w').write({body!r})")
            continue
        lines = []
        for l in src.splitlines():
            s = l.lstrip()
            if s.startswith("!nvidia-smi"):
                lines.append(l[: len(l) - len(s)] + "os.system('nvidia-smi --query-gpu=name,memory.total --format=csv')")
            elif s.startswith("!du"):
                lines.append(l[: len(l) - len(s)] + f"os.system('du -sh {root}/embeddings/*')")
            elif s.startswith(("!", "%")):
                raise ValueError(f"unhandled magic: {l}")
            else:
                lines.append(l)
        out += ["", "\n".join(lines), "sys.stdout.flush()"]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("out")
    ap.add_argument("--root", default="/content/hest_fm_project")
    a = ap.parse_args()
    open(a.out, "w").write(convert(a.notebook, a.root))
    print("wrote", a.out)
