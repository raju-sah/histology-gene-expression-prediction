"""Phase 2 (+ Phase 4 Visium) Colab notebook. Imported by build_notebooks.py."""
from __future__ import annotations

import nbformat as nbf

from build_notebooks import COMMON_HELPERS, MOUNT_AND_LOGIN, SETUP_GPU, code, inline_module, md

XENIUM5K = ["TENX189", "TENX188", "TENX187", "NCBI887", "NCBI888", "TENX160"]   # small -> large; TENX160 last
VISIUM = ["TENX62", "TENX72", "MISC13", "MISC14", "MISC15", "MISC16", "MEND85", "MEND86", "MEND89", "MEND90",
          "TENX65", "TENX51", "TENX50"]


def phase2():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"""
        # Phase 2: Xenium 5k predictability atlas: embeddings + patch-aligned pseudo-spots (+ Phase 4 Visium)

        **Runtime:** GPU **L4, High-RAM ON** (TENX160 has ~1-2 B transcripts; the cell x gene matrix needs ~10-20 GB RAM).
        A100 also works (faster GPU part, but the transcript streaming is CPU/network-bound, so L4 is the better value).
        Expected ~2-3 h on L4 (~4-6 compute units at ~1.5 CU/h). Every step is resumable: if the runtime disconnects, just
        re-run all cells and finished work is skipped.

        **Secrets:** `HF_TOKEN` in Colab Secrets (never printed or saved).

        **Section A (Xenium 5k, {len(XENIUM5K)} samples, one at a time):**
        1. download HEST `patches/<id>.h5` (224 px at 0.5 um/px, 112 um footprint), embed with UNI2-h, Virchow2, CONCH v1.5,
           H-optimus-1 (only if access works; otherwise skipped and recorded) and UNI2-h on the central 56 um
           (`uni2_h_c56`, patch-size sensitivity); delete the patch file;
        2. download `transcripts/<id>_transcripts.parquet` to the VM disk (deleted right after), read it row group by row group,
           build the cell x gene matrix (qv >= 20, real genes, assigned cells), cell centroids in H&E pixels, coarse
           marker-based cell types, and pseudo-spot counts = sum of cells whose centroid lies in the patch footprint
           (window scale 1.0) plus 0.5x and 2x windows for Phase 5;
        3. download HEST `st/<id>.h5ad` only to cross-check our pseudo-spots against HEST's (QC numbers saved), then delete.

        TENX190 is skipped on purpose: HEST labels it Xenium 5k but it is the Xenium v1 (541-gene) run.

        **Section B (Phase 4, {len(VISIUM)} Visium samples of lung / ovary / cervix):** embeddings with the same encoders
        and expression restricted to the Xenium 5k genes.

        **Output (Drive):** `MyDrive/hest_fm_project/embeddings/phase2/xenium5k/...` and `.../embeddings/phase4/visium/...`

        **Afterwards download into `data/embeddings/` (keeping the folder structure):**
        - `phase2/xenium5k/{{uni2_h,virchow2,conch_v15,hoptimus1}}/`, `expression/`, `cells/`, `qc/`  (~3 GB)
        - `phase4/visium/` (~0.5 GB)
        - optional for Phase 5: `phase2/xenium5k/{{uni2_h_c56,expression_w050,expression_w200}}/` (~2 GB)
        """),
        SETUP_GPU,
        code("!pip -q install 'timm>=1.0.9' 'huggingface_hub>=0.24' h5py anndata pyarrow einops einops-exts 'transformers==4.46.3'"),
        MOUNT_AND_LOGIN,
        code(f"""
        DRIVE_ROOT = '/content/drive/MyDrive/hest_fm_project'
        OUT = f'{{DRIVE_ROOT}}/embeddings/phase2/xenium5k'
        OUT_VIS = f'{{DRIVE_ROOT}}/embeddings/phase4/visium'
        SCRATCH = '/content/scratch'
        SAMPLES = {XENIUM5K!r}
        VISIUM = {VISIUM!r}
        ENCODERS = ['uni2_h', 'virchow2', 'conch_v15', 'hoptimus1', 'uni2_h_c56']
        VISIUM_ENCODERS = ['uni2_h', 'virchow2', 'conch_v15', 'hoptimus1']
        WINDOWS = {{'expression': 1.0, 'expression_w050': 0.5, 'expression_w200': 2.0}}
        MIN_QV = 20
        HEST = 'MahmoodLab/hest'
        for d in (OUT, OUT_VIS, SCRATCH):
            os.makedirs(d, exist_ok=True)
        """),
        inline_module("encoders"),
        inline_module("data", "hestfm_data"),
        inline_module("pseudospots"),
        inline_module("celltypes"),
        code("""
        from encoders import build_encoder, embed
        from hestfm_data import write_expression
        from pseudospots import TranscriptAggregator, orientation_check, window_assignment, pseudospot_counts, COLUMNS
        from celltypes import annotate, type_profiles, tau, composition
        """),
        COMMON_HELPERS,
        code("""
        import pandas as pd, pyarrow.parquet as pq, scipy.sparse as sp, anndata as ad, traceback
        from huggingface_hub import hf_hub_download, HfFileSystem

        def hest_file(rel):
            return hf_hub_download(HEST, rel, repo_type='dataset', local_dir=SCRATCH)

        def qc_path(sid, root=OUT):
            return f'{root}/qc/{sid}.json'

        def load_qc(sid, root=OUT):
            p = qc_path(sid, root)
            return json.load(open(p)) if os.path.exists(p) else {}

        def save_qc(sid, qc, root=OUT):
            os.makedirs(f'{root}/qc', exist_ok=True)
            json.dump(qc, open(qc_path(sid, root) + '.tmp', 'w'), indent=2)
            os.replace(qc_path(sid, root) + '.tmp', qc_path(sid, root))

        CONTROL_PREFIX = ('NegControl', 'BLANK', 'Unassigned', 'Deprecated', 'Intergenic', 'antisense', 'Genomic')
        """),
        md("## Load encoders (H-optimus-1 is optional: if access is still pending it is skipped)"),
        code("""
        skipped, loaded = {}, {}
        for n in ENCODERS:
            try:
                if n.startswith('uni2_h_c') and 'uni2_h' in loaded:
                    import copy
                    e = copy.copy(loaded['uni2_h']); e.name = n; e.center_crop_px = int(round(int(n.split('_c')[-1]) / 0.5))
                else:
                    e = build_encoder(n)
                loaded[n] = e
            except Exception as ex:
                skipped[n] = f'{type(ex).__name__}: {str(ex)[:200]}'
                print(f'SKIP encoder {n}: {skipped[n]}')
        encoders = [loaded[n] for n in ENCODERS if n in loaded]
        print('encoders:', [(e.name, e.dim) for e in encoders], '| skipped:', list(skipped))
        print(torch.cuda.memory_allocated() / 1e9, 'GB on GPU')
        """),
        md("## Section A: Xenium 5k"),
        code("""
        def embed_sample(sid, encs, out_root, qc):
            todo = [e for e in encs if not os.path.exists(f'{out_root}/{e.name}/{sid}.h5')]
            if not todo:
                return
            t0 = time.time()
            p = hest_file(f'patches/{sid}.h5')
            bc, coords, attrs, n = read_patch_meta(p)
            embs = embed_patch_file(p, todo)
            for e in todo:
                save_embeddings(f'{out_root}/{e.name}/{sid}.h5', embs[e.name], bc, coords)
            qc.update({'n_patches': int(n), 'patch_size_src': float(attrs.get('patch_size_src', np.nan)),
                       'pixel_size': float(attrs.get('pixel_size', np.nan))})
            qc.setdefault('embed_s', {})['+'.join(e.name for e in todo)] = round(time.time() - t0, 1)
            os.remove(p)
            print(f'  {sid}: embedded {n} patches with {[e.name for e in todo]} in {time.time() - t0:.0f}s')

        def patch_meta_from_outputs(sid, qc):
            ref = next(e.name for e in encoders if os.path.exists(f'{OUT}/{e.name}/{sid}.h5'))
            with h5py.File(f'{OUT}/{ref}/{sid}.h5') as f:
                bc = f['barcodes'][:].astype(str); coords = f['coords'][:]
            return bc, coords, qc['patch_size_src']

        def stream_cells(sid):
            # Download the whole transcript parquet to VM disk (much faster than remote row-group reads),
            # aggregate it row group by row group, then delete it before the next sample.
            t0 = time.time()
            local = hest_file(f'transcripts/{sid}_transcripts.parquet')
            print(f'    downloaded transcripts ({os.path.getsize(local) / 1e9:.1f} GB) in {(time.time() - t0) / 60:.1f} min')
            try:
                pf = pq.ParquetFile(local)
                n_rg = pf.metadata.num_row_groups
                agg = TranscriptAggregator(min_qv=MIN_QV)
                t1 = time.time()
                for i in range(n_rg):
                    agg.add(pf.read_row_group(i, columns=COLUMNS))
                    if i % 50 == 0 or i == n_rg - 1:
                        el = time.time() - t1
                        print(f'    row group {i + 1}/{n_rg}  kept {agg.n_kept / 1e6:.1f}M / seen {agg.n_seen / 1e6:.1f}M '
                              f'transcripts, {len(agg.cell_index)} cells, {el / 60:.1f} min (eta {el / (i + 1) * (n_rg - i - 1) / 60:.0f} min)')
                return agg.finalize(), agg
            finally:
                os.remove(local)

        def pseudospots_sample(sid, qc):
            outs = {k: f'{OUT}/{k}/{sid}.h5' for k in WINDOWS}
            if all(os.path.exists(p) for p in outs.values()) and os.path.exists(f'{OUT}/cells/{sid}.h5'):
                return
            bc, coords, S = patch_meta_from_outputs(sid, qc)
            t0 = time.time()
            cells, agg = stream_cells(sid)
            keep = np.array([not g.startswith(CONTROL_PREFIX) for g in cells.genes])
            cells.counts, cells.genes = cells.counts[:, keep].tocsr(), cells.genes[keep]
            ori = orientation_check(cells.centroids, coords, S)
            swap = ori['swapped'] > ori['xy']
            qc.update({'transcripts_seen': int(agg.n_seen), 'transcripts_kept': int(agg.n_kept),
                       'n_cells': int(cells.counts.shape[0]), 'n_genes': int(len(cells.genes)),
                       'orientation_frac_inside': ori, 'swap_xy': bool(swap), 'stream_min': round((time.time() - t0) / 60, 1),
                       'median_transcripts_per_cell': float(np.median(cells.n_transcripts))})
            if max(ori.values()) < 0.5:
                print(f'  WARNING {sid}: only {max(ori.values()):.2f} of cells fall inside patches; check alignment')
            labels, types, used = annotate(cells.counts, cells.genes)
            prof, n_per = type_profiles(cells.counts, labels, types)
            for key, scale in WINDOWS.items():
                A = window_assignment(cells.centroids, coords, S, scale, swap)
                X = pseudospot_counts(A, cells)
                extra = {'n_cells': np.asarray(A.sum(1)).ravel().astype(np.int32)}
                if scale == 1.0:
                    comp, cols = composition(A, labels, types)
                    extra.update({'composition': comp.astype(np.int32), 'composition_cols': np.array(cols).astype('S')})
                tmp = outs[key] + '.tmp'
                write_expression(tmp, X, bc, cells.genes, coords, **extra); os.replace(tmp, outs[key])
                qc.setdefault('windows', {})[key] = {'scale': scale, 'mean_cells': float(extra['n_cells'].mean()),
                                                     'mean_counts': float(X.sum() / X.shape[0])}
            os.makedirs(f'{OUT}/cells', exist_ok=True)
            def w(p):
                with h5py.File(p, 'w') as f:
                    f['type_profiles'] = prof.astype(np.float32); f['types'] = np.array(types).astype('S')
                    f['n_per_type'] = n_per; f['genes'] = cells.genes.astype('S'); f['tau'] = tau(prof).astype(np.float32)
                    f['gene_total_counts'] = np.asarray(cells.counts.sum(0)).ravel()
                    f['gene_cells_detected'] = np.asarray((cells.counts > 0).sum(0)).ravel()
                    f.attrs['markers'] = json.dumps(used)
            atomic_h5(f'{OUT}/cells/{sid}.h5', w)
            qc['cell_types'] = dict(zip(*np.unique(labels, return_counts=True)))
            qc['cell_types'] = {k: int(v) for k, v in qc['cell_types'].items()}
            del cells, agg; gc.collect()
            print(f'  {sid}: pseudo-spots done in {(time.time() - t0) / 60:.1f} min, swap_xy={swap}, orientation={ori}')

        def hest_crosscheck(sid, qc):
            if 'hest_crosscheck' in qc:
                return
            p = hest_file(f'st/{sid}.h5ad')
            a = ad.read_h5ad(p)
            with h5py.File(f'{OUT}/expression/{sid}.h5') as f:
                bc = f['barcodes'][:].astype(str); genes = f['genes'][:].astype(str)
                g = f['counts_csr']; M = sp.csr_matrix((g['data'][:].astype(np.float32), g['indices'][:].astype(np.int32), g['indptr'][:]), shape=tuple(g['shape'][:]))
            vset, oset = set(a.var_names), set(a.obs_names)
            common = [x for x in genes if x in vset]
            okb = np.array([b in oset for b in bc])
            H = a[bc[okb], common].X
            H = H.toarray() if hasattr(H, 'toarray') else np.asarray(H)
            gpos = {x: i for i, x in enumerate(genes)}
            gi = [gpos[x] for x in common]
            O = M[okb][:, gi].toarray()
            tot_r = float(np.corrcoef(O.sum(1), H.sum(1))[0, 1])
            lo, lh = np.log1p(O), np.log1p(H)
            r = [np.corrcoef(lo[:, k], lh[:, k])[0, 1] for k in range(len(common)) if lo[:, k].std() > 0 and lh[:, k].std() > 0]
            qc['hest_crosscheck'] = {'n_spots': int(okb.sum()), 'n_genes': len(common), 'total_counts_pearson': round(tot_r, 4),
                                     'per_gene_log1p_pearson_median': round(float(np.median(r)), 4),
                                     'ratio_ours_to_hest_counts': round(float(O.sum() / max(H.sum(), 1)), 3)}
            os.remove(p)
            print(f'  {sid}: HEST cross-check {qc["hest_crosscheck"]}')
        """),
        code("""
        t_all = time.time()
        for sid in SAMPLES:
            qc = load_qc(sid)
            print(f'== {sid}')
            for step in (lambda: embed_sample(sid, encoders, OUT, qc), lambda: pseudospots_sample(sid, qc),
                         lambda: hest_crosscheck(sid, qc)):
                try:
                    step()
                except Exception:
                    traceback.print_exc()
                    qc.setdefault('errors', []).append(traceback.format_exc()[-800:])
                    save_qc(sid, qc)
                    break
                save_qc(sid, qc)
            gc.collect(); torch.cuda.empty_cache()
        print(f'Section A wall time {(time.time() - t_all) / 60:.1f} min')
        """),
        md("## Section B (Phase 4): Visium lung / ovary / cervix, same encoders, Xenium 5k genes only"),
        code("""
        genes_5k = set()
        for sid in SAMPLES:
            p = f'{OUT}/expression/{sid}.h5'
            if os.path.exists(p):
                with h5py.File(p) as f:
                    genes_5k |= set(f['genes'][:].astype(str))
        print(len(genes_5k), 'Xenium 5k genes (union)')
        vis_encoders = [e for e in encoders if e.name in VISIUM_ENCODERS]
        for sid in VISIUM:
            qc = load_qc(sid, OUT_VIS)
            try:
                embed_sample(sid, vis_encoders, OUT_VIS, qc)
                xo = f'{OUT_VIS}/expression/{sid}.h5'
                if not os.path.exists(xo):
                    with h5py.File(f'{OUT_VIS}/{vis_encoders[0].name}/{sid}.h5') as f:
                        bc = f['barcodes'][:].astype(str); coords = f['coords'][:]
                    a = ad.read_h5ad(hest_file(f'st/{sid}.h5ad'))
                    a.var_names_make_unique()
                    g = sorted(genes_5k & set(a.var_names))
                    oset = set(a.obs_names)
                    okb = np.array([b in oset for b in bc])
                    X = a[bc[okb], g].X
                    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
                    tmp = xo + '.tmp'
                    write_expression(tmp, X, bc[okb], np.array(g), coords[okb],
                                     total_counts_all_genes=np.asarray(a[bc[okb]].X.sum(1)).ravel().astype(np.float32))
                    os.replace(tmp, xo)
                    qc.update({'n_spots_expr': int(okb.sum()), 'n_genes_5k': len(g)})
                    os.remove(f'{SCRATCH}/st/{sid}.h5ad')
                print(f'  {sid}: ok {qc.get("n_spots_expr")}')
            except Exception:
                traceback.print_exc(); qc.setdefault('errors', []).append(traceback.format_exc()[-800:])
            save_qc(sid, qc, OUT_VIS)
        """),
        md("## Summary"),
        code("""
        rows = []
        for sid in SAMPLES:
            q = load_qc(sid)
            rows.append({'sample': sid, 'patches': q.get('n_patches'), 'cells': q.get('n_cells'), 'genes': q.get('n_genes'),
                         'swap_xy': q.get('swap_xy'), 'inside_frac': max(q.get('orientation_frac_inside', {'x': np.nan}).values()),
                         'mean_cells_per_spot': q.get('windows', {}).get('expression', {}).get('mean_cells'),
                         'hest_total_r': q.get('hest_crosscheck', {}).get('total_counts_pearson'),
                         'hest_gene_r_median': q.get('hest_crosscheck', {}).get('per_gene_log1p_pearson_median'),
                         'errors': len(q.get('errors', []))})
        summary = pd.DataFrame(rows); print(summary.to_string())
        vis = pd.DataFrame([{'sample': s, **{k: load_qc(s, OUT_VIS).get(k) for k in ('n_patches', 'n_spots_expr', 'n_genes_5k')},
                             'errors': len(load_qc(s, OUT_VIS).get('errors', []))} for s in VISIUM]); print(vis.to_string())
        json.dump({'xenium5k': rows, 'visium': vis.to_dict('records'), 'skipped_encoders': skipped,
                   'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'},
                  open(f'{OUT}/run_summary.json', 'w'), indent=2, default=str)
        """),
        code("!du -sh {OUT}/* {OUT_VIS}"),
    ]
    return nb


BUILDERS = {"colab_phase2_xenium5k": phase2}
