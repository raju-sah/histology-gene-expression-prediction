"""Build notebooks/colab_*.ipynb from code in this script and src/hestfm (single source of truth).

    .venv/bin/python scripts/build_notebooks.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hestfm"


def md(s):
    return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())


def code(s):
    return nbf.v4.new_code_cell(textwrap.dedent(s).strip())


def inline_module(name: str, as_name: str | None = None):
    """A cell that writes src/hestfm/<name>.py into /content/lib/<as_name>.py so the notebook can import it."""
    body = (SRC / f"{name}.py").read_text()
    return nbf.v4.new_code_cell(f"%%writefile /content/lib/{as_name or name}.py\n{body}")


SETUP_GPU = code("""
    !nvidia-smi --query-gpu=name,memory.total --format=csv
    import os, sys
    os.makedirs('/content/lib', exist_ok=True)
    sys.path.insert(0, '/content/lib')
""")

MOUNT_AND_LOGIN = code("""
    from google.colab import drive, userdata
    drive.mount('/content/drive')
    from huggingface_hub import login
    # Token comes from Colab Secrets (key icon on the left -> HF_TOKEN, notebook access ON). Never printed.
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
    from huggingface_hub import whoami
    print('HF login ok:', bool(whoami().get('name')))
""")

COMMON_HELPERS = code("""
    import json, time, shutil, gc
    import numpy as np, h5py, torch

    def atomic_h5(path, write_fn):
        tmp = path + '.tmp'
        write_fn(tmp)
        os.replace(tmp, path)

    def save_embeddings(path, emb, barcodes, coords):
        def w(p):
            with h5py.File(p, 'w') as f:
                f.create_dataset('embeddings', data=np.asarray(emb, dtype=np.float16))
                f.create_dataset('barcodes', data=np.asarray(barcodes).astype('S'))
                f.create_dataset('coords', data=np.asarray(coords))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        atomic_h5(path, w)

    def read_patch_meta(h5_path):
        with h5py.File(h5_path, 'r') as f:
            bkey = 'barcode' if 'barcode' in f else 'barcodes'
            ikey = img_key(f)
            bc = f[bkey][:].reshape(-1)
            bc = np.array([b.decode() if isinstance(b, bytes) else str(b) for b in bc])
            return bc, f['coords'][:], dict(f[ikey].attrs), f[ikey].shape[0]

    def img_key(f):
        return next(k for k in ('img', 'imgs', 'images') if k in f)

    def embed_patch_file(h5_path, encoders, chunk=512):
        # One pass over the patch file; every encoder sees each chunk. Returns {name: (N, D) float16}.
        out = {e.name: [] for e in encoders}
        with h5py.File(h5_path, 'r') as f:
            imgs = f[img_key(f)]; n = imgs.shape[0]
            for i in range(0, n, chunk):
                batch = imgs[i:i + chunk]
                for e in encoders:
                    out[e.name].append(embed(batch, e, amp_dtype=AMP))
        return {k: np.concatenate(v) for k, v in out.items()}

    AMP = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print('autocast dtype:', AMP)
""")


def phase1():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
        # Phase 1: reproduce HEST-Benchmark (ResNet50 + UNI2-h) on IDC, LUNG, COAD

        **Runtime:** GPU **L4** (A100 also fine). High-RAM not needed. Expected ~30-45 min on L4 (~3-4 compute units).

        **Before running:** add your HuggingFace token as a Colab Secret named `HF_TOKEN` (key icon, enable notebook access).
        The token is read with `userdata.get` and never printed or saved.

        **What it does, per task and sample (one at a time):** download the HEST-Benchmark patch file + expression file,
        embed all patches with ResNet50 (HEST truncated variant, 1024-d) and UNI2-h (1536-d), save float16 embeddings and
        the raw counts of the task's gene lists to Drive, delete the raw files. Resumable: finished samples are skipped.

        **Output (Drive):** `MyDrive/hest_fm_project/embeddings/phase1/<TASK>/{resnet50,uni2_h,expression}/<sample>.h5`
        plus `MyDrive/hest_fm_project/embeddings/phase1/run_summary.json` and `sanity_hest_ridge.json`.

        **Afterwards download** the folder `MyDrive/hest_fm_project/embeddings/phase1` (~0.4 GB) to `data/embeddings/phase1`.
        """),
        SETUP_GPU,
        code("!pip -q install 'timm>=1.0.9' 'huggingface_hub>=0.24' h5py anndata"),
        MOUNT_AND_LOGIN,
        code("""
        DRIVE_ROOT = '/content/drive/MyDrive/hest_fm_project'
        OUT = f'{DRIVE_ROOT}/embeddings/phase1'
        SCRATCH = '/content/scratch'
        TASKS = ['IDC', 'LUNG', 'COAD']
        ENCODERS = ['resnet50', 'uni2_h']
        GENE_LISTS = ['var_50genes.json', 'mean_50genes.json']
        os.makedirs(OUT, exist_ok=True); os.makedirs(SCRATCH, exist_ok=True)
        """),
        inline_module("encoders"),
        code("from encoders import build_encoder, embed"),
        COMMON_HELPERS,
        code("""
        import pandas as pd, anndata as ad
        from huggingface_hub import hf_hub_download

        BENCH = 'MahmoodLab/hest-bench'

        def bench_file(task, rel):
            return hf_hub_download(BENCH, f'{task}/{rel}', repo_type='dataset', local_dir=SCRATCH)

        def task_meta(task):
            splits, i = [], 0
            while True:
                try:
                    tr = pd.read_csv(bench_file(task, f'splits/train_{i}.csv'))
                    te = pd.read_csv(bench_file(task, f'splits/test_{i}.csv'))
                except Exception:
                    break
                splits.append((tr, te)); i += 1
            rows = pd.concat([pd.concat([a, b]) for a, b in splits]).drop_duplicates('sample_id')
            genes = []
            for gl in GENE_LISTS:
                try:
                    genes += json.load(open(bench_file(task, gl)))['genes']
                except Exception:
                    pass
            return rows, list(dict.fromkeys(genes)), len(splits)

        def save_expression(path, adata_path, genes, barcodes, coords):
            a = ad.read_h5ad(adata_path)
            genes = [g for g in genes if g in a.var_names]
            a = a[:, genes]
            pos = {b: i for i, b in enumerate(a.obs_names)}
            keep = np.array([b in pos for b in barcodes])
            X = a.X[[pos[b] for b in barcodes[keep]]]
            X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
            def w(p):
                with h5py.File(p, 'w') as f:
                    f.create_dataset('counts', data=X.astype(np.float32))
                    f.create_dataset('barcodes', data=barcodes[keep].astype('S'))
                    f.create_dataset('genes', data=np.array(genes).astype('S'))
                    f.create_dataset('coords', data=coords[keep])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            atomic_h5(path, w)
            return int(keep.sum())
        """),
        code("""
        encoders = [build_encoder(n) for n in ENCODERS]
        print('loaded', [(e.name, e.dim) for e in encoders])
        """),
        code("""
        summary = {'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'tasks': {}}
        t_all = time.time()
        for task in TASKS:
            rows, genes, n_folds = task_meta(task)
            print(f'== {task}: {len(rows)} samples, {n_folds} folds, {len(genes)} genes')
            summary['tasks'][task] = {'samples': {}, 'n_folds': n_folds}
            for _, r in rows.iterrows():
                sid = r['sample_id']
                outs = {e.name: f'{OUT}/{task}/{e.name}/{sid}.h5' for e in encoders}
                expr_out = f'{OUT}/{task}/expression/{sid}.h5'
                if all(os.path.exists(p) for p in outs.values()) and os.path.exists(expr_out):
                    print(f'  {sid}: done, skipping'); continue
                t0 = time.time()
                patches = bench_file(task, r['patches_path'])
                adata = bench_file(task, r['expr_path'])
                bc, coords, attrs, n = read_patch_meta(patches)
                todo = [e for e in encoders if not os.path.exists(outs[e.name])]
                t1 = time.time()
                embs = embed_patch_file(patches, todo)
                t_emb = time.time() - t1
                for e in todo:
                    save_embeddings(outs[e.name], embs[e.name], bc, coords)
                n_expr = save_expression(expr_out, adata, genes, bc, coords)
                os.remove(patches); os.remove(adata)
                summary['tasks'][task]['samples'][sid] = {'n_patches': int(n), 'n_with_expr': n_expr,
                    'embed_s': round(t_emb, 1), 'total_s': round(time.time() - t0, 1),
                    'patches_per_s': round(n / max(t_emb, 1e-6), 1)}
                print(f'  {sid}: {n} patches, {n_expr} with expression, embed {t_emb:.0f}s '
                      f'({n / max(t_emb, 1e-6):.0f} patch/s, all encoders), total {time.time() - t0:.0f}s')
                gc.collect(); torch.cuda.empty_cache()
        summary['wall_s'] = round(time.time() - t_all, 1)
        json.dump(summary, open(f'{OUT}/run_summary.json', 'w'), indent=2)
        print(json.dumps(summary, indent=2))
        """),
        md("""
        ## Sanity check: HEST-Benchmark ridge on the saved embeddings (CPU, a few minutes)
        Same protocol as `hest.bench`: log1p counts of `var_50genes`, StandardScaler -> PCA(256) -> Ridge(lsqr,
        alpha = 100/(256*50), no intercept), Pearson per gene averaged over genes then folds. The authoritative
        evaluation is run locally (`scripts/run_phase1_eval.py`); this cell just gives you an immediate number.
        """),
        code("""
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline

        PUBLISHED = {'resnet50': {'IDC': 0.4739, 'LUNG': 0.4949, 'COAD': 0.2500},
                     'uni2_h': {'IDC': 0.5898, 'LUNG': 0.5587, 'COAD': 0.3015}}

        def load(task, enc, sid, genes):
            with h5py.File(f'{OUT}/{task}/{enc}/{sid}.h5') as f:
                X = f['embeddings'][:].astype(np.float32); bc = f['barcodes'][:].astype(str)
            with h5py.File(f'{OUT}/{task}/expression/{sid}.h5') as f:
                Y = f['counts'][:]; xb = f['barcodes'][:].astype(str); g = list(f['genes'][:].astype(str))
            pos = {b: i for i, b in enumerate(xb)}; keep = np.array([b in pos for b in bc])
            return X[keep], np.log1p(Y[[pos[b] for b in bc[keep]]][:, [g.index(x) for x in genes]])

        sanity = {}
        for task in TASKS:
            genes = json.load(open(bench_file(task, 'var_50genes.json')))['genes']
            _, _, n_folds = task_meta(task)
            for enc in ENCODERS:
                fold_r = []
                for i in range(n_folds):
                    tr = pd.read_csv(bench_file(task, f'splits/train_{i}.csv'))['sample_id']
                    te = pd.read_csv(bench_file(task, f'splits/test_{i}.csv'))['sample_id']
                    Xtr, Ytr = map(np.concatenate, zip(*[load(task, enc, s, genes) for s in tr]))
                    Xte, Yte = map(np.concatenate, zip(*[load(task, enc, s, genes) for s in te]))
                    pipe = Pipeline([('s', StandardScaler()), ('p', PCA(256, random_state=1))])
                    Ztr, Zte = pipe.fit_transform(Xtr), pipe.transform(Xte)
                    reg = Ridge(solver='lsqr', alpha=100 / (256 * 50), fit_intercept=False, random_state=1, max_iter=1000)
                    P = reg.fit(Ztr, Ytr).predict(Zte)
                    r = [np.corrcoef(Yte[:, k], P[:, k])[0, 1] for k in range(len(genes))]
                    fold_r.append(float(np.nanmean(r)))
                sanity[f'{task}/{enc}'] = {'ours': round(float(np.mean(fold_r)), 4), 'published': PUBLISHED[enc][task],
                                           'folds': [round(x, 4) for x in fold_r]}
                print(task, enc, sanity[f'{task}/{enc}'])
        json.dump(sanity, open(f'{OUT}/sanity_hest_ridge.json', 'w'), indent=2)
        """),
        code("""
        !du -sh {OUT}
        print('Done. Download', OUT, '-> data/embeddings/phase1')
        """),
    ]
    return nb


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    builders = {"colab_phase1_benchmark": phase1}
    try:
        from build_notebooks_phase2 import BUILDERS as more   # later phases
        builders.update(more)
    except ImportError:
        pass
    for name, fn in builders.items():
        path = ROOT / "notebooks" / f"{name}.ipynb"
        nb = fn()
        nb.metadata.update({"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                            "kernelspec": {"name": "python3", "display_name": "Python 3"}})
        nbf.write(nb, path)
        print("wrote", path.relative_to(ROOT), len(nb.cells), "cells")
