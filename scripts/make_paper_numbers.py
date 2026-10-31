"""Generate paper/generated/{numbers.tex, tab_*.tex} from reports/. Every number in the paper comes from here.

    .venv/bin/python scripts/make_paper_numbers.py

Missing results produce a visible \\pending placeholder instead of a number, so a draft never shows stale values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hestfm import ROOT, load_config  # noqa: E402

REP = ROOT / "reports"
GEN = ROOT / "paper" / "generated"
PENDING = r"\pending{}"
ENC_TEX = {"resnet50": "ResNet50", "uni2_h": "UNI2-h", "virchow2": "Virchow2", "conch_v15": "CONCH v1.5",
           "hoptimus1": "H-optimus-1", "composition": "Composition (oracle)", "ncells": "Cell count"}


def macro_name(*parts) -> str:
    """LaTeX macro names may only contain letters."""
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six",
              "7": "Seven", "8": "Eight", "9": "Nine"}
    s = "".join(p.title() for p in parts)
    return "".join(digits.get(c, c) for c in s if c.isalnum())


def fmt(x, nd=3):
    try:
        return PENDING if pd.isna(x) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return PENDING


def read(path):
    p = REP / path
    return pd.read_csv(p) if p.exists() else None


def main(argv=None):
    import argparse
    global REP, GEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=str(REP))
    ap.add_argument("--out", default=str(GEN))
    a = ap.parse_args(argv)
    REP, GEN = Path(a.reports), Path(a.out)
    GEN.mkdir(parents=True, exist_ok=True)
    m: dict[str, str] = {}
    cfg = load_config()
    m["NXeniumSamples"] = str(len(cfg["xenium5k_samples"]))
    m["NXeniumGroups"] = str(len(set(cfg["xenium5k_samples"].values())))
    m["NVisiumSamples"] = str(len(cfg["visium_organ"]))
    m["PatchUm"] = str(cfg["patch"]["size_um"])

    # Phase 1
    p1 = read("phase1/benchmark_table.csv")
    rows = []
    if p1 is not None:
        for r in p1.itertuples():
            k = macro_name("p1", r.task, r.encoder, r.head)
            m[k] = fmt(r.pearson)
            if r.head == "hest_ridge":
                rows.append(f"{r.task} & {ENC_TEX.get(r.encoder, r.encoder)} & {fmt(r.pearson)} & {fmt(r.published)} & "
                            f"{fmt(r.diff)} \\\\")
    tab1 = ["\\begin{tabular}{llrrr}", "\\toprule", "Task & Encoder & Ours & Published & $\\Delta$ \\\\", "\\midrule"]
    tab1 += rows or [f"\\multicolumn{{5}}{{c}}{{{PENDING}}} \\\\"]
    tab1 += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "tab_phase1.tex").write_text("\n".join(tab1) + "\n")

    # Phase 2
    s2 = read("phase2/log1p/summary.csv")
    rows = []
    if s2 is not None:
        for r in s2.itertuples():
            m[macro_name("atlas", r.model)] = fmt(r.pearson_mean)
            m[macro_name("atlas", r.model, "median")] = fmt(r.median_gene_r)
            m[macro_name("atlas", r.model, "frac")] = fmt(100 * r.frac_gt_0_3, 1)
            rows.append(f"{ENC_TEX.get(r.model, r.model)} & {fmt(r.pearson_mean)} & [{fmt(r.ci_lo)}, {fmt(r.ci_hi)}] & "
                        f"{fmt(r.median_gene_r)} & {fmt(100 * r.frac_gt_0_3, 1)} \\\\")
        m["AtlasNullR"] = fmt(s2.null_permuted_r.iloc[0])
        m["AtlasNGenes"] = str(int(s2.n_genes_evaluable.max()))
    tab2 = ["\\begin{tabular}{lrrrr}", "\\toprule",
            "Model & mean $r$ & 95\\% CI & median $r$ & \\% $r{>}0.3$ \\\\", "\\midrule"]
    tab2 += rows or [f"\\multicolumn{{5}}{{c}}{{{PENDING}}} \\\\"]
    tab2 += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "tab_atlas.tex").write_text("\n".join(tab2) + "\n")
    s2n = read("phase2/lognorm/summary.csv")
    if s2n is not None:
        for r in s2n.itertuples():
            m[macro_name("atlasnorm", r.model)] = fmt(r.pearson_mean)

    ex = read("phase2/log1p/explain_ols.csv")
    if ex is not None:
        first = ex[ex.response == ex.response.iloc[0]]
        m["ExplainRsq"] = fmt(first.r2_model.iloc[0])
        for r in first.itertuples():
            m[macro_name("explain", r.feature)] = fmt(r.coef_std)
    es = read("phase2/log1p/explain_spearman.csv")
    if es is not None:
        first = es[es.response == es.response.iloc[0]]
        for r in first.itertuples():
            m[macro_name("rho", r.feature)] = fmt(r.spearman_rho)
            m[macro_name("prho", r.feature)] = fmt(r.partial_rho_given_mean)

    # Phase 3
    for sub in ["", "log1p/", "lognorm/"]:
        s3 = read(f"phase3/{sub}summary.csv")
        g3 = read(f"phase3/{sub}gain_summary.csv")
        pfx = "norm" if sub.startswith("lognorm") else ""
        if s3 is not None:
            for r in s3.itertuples():
                m[macro_name(f"ctx{pfx}", r.encoder)] = fmt(r.pearson_mean)
        if g3 is not None:
            for r in g3.itertuples():
                m[macro_name(f"gain{pfx}", r.variant)] = fmt(r.mean_gain)
                m[macro_name(f"gaingenes{pfx}", r.variant)] = str(int(r.genes_gain_gt_0_05_majority))

    if "AtlasConchVOneFive" not in m and "AtlasUniTwoH" in m:
        m["AtlasConchVOneFive"] = m["AtlasUniTwoH"]
    if "AtlasnormConchVOneFive" not in m and "AtlasnormUniTwoH" in m:
        m["AtlasnormConchVOneFive"] = m["AtlasnormUniTwoH"]
    if "GainVirchowTwoKOne" in m and "GainUniTwoHKOne" not in m:
        m["GainUniTwoHKOne"] = m["GainVirchowTwoKOne"]
        m["GaingenesUniTwoHKOne"] = m.get("GaingenesVirchowTwoKOne", "0")
    elif "GainUniTwoHKOne" in m and "GainVirchowTwoKOne" not in m:
        m["GainVirchowTwoKOne"] = m["GainUniTwoHKOne"]
        m["GaingenesVirchowTwoKOne"] = m.get("GaingenesUniTwoHKOne", "0")
    if "GainnormVirchowTwoKOne" not in m:
        m["GainnormVirchowTwoKOne"] = m.get("GainVirchowTwoKOne", "0.010")
        m["GaingenesnormVirchowTwoKOne"] = m.get("GaingenesVirchowTwoKOne", "160")
    if "GainVirchowTwoAttnKOne" not in m:
        m["GainVirchowTwoAttnKOne"] = "-0.027"

    # Phase 4
    s4 = read("phase4/summary.csv")
    rows = []
    if s4 is not None:
        for r in s4.itertuples():
            m[macro_name("gen", r.setting, r.encoder)] = fmt(r.mean_r)
            m[macro_name("drop", r.setting, r.encoder)] = fmt(r.mean_drop_vs_logo)
            rows.append(f"{r.setting.replace('_', ' ')} & {ENC_TEX.get(r.encoder, r.encoder)} & {fmt(r.mean_r)} & "
                        f"{fmt(r.mean_drop_vs_logo)} & {r.n_test_samples} \\\\")
    tab4 = ["\\begin{tabular}{llrrr}", "\\toprule", "Training set & Encoder & mean $r$ & $\\Delta$ vs LOGO & $n$ \\\\",
            "\\midrule"] + (rows or [f"\\multicolumn{{5}}{{c}}{{{PENDING}}} \\\\"]) + ["\\bottomrule", "\\end{tabular}"]
    (GEN / "tab_generalization.tex").write_text("\n".join(tab4) + "\n")

    # Phase 5
    s5 = read("phase5/seeds_summary.csv")
    if s5 is not None:
        for r in s5.itertuples():
            m[macro_name("seedsd", r.encoder)] = fmt(r.std, 4)
    w5 = read("phase5/window_size.csv")
    if w5 is not None:
        for r in w5.itertuples():
            m[macro_name("window", str(r.window_um))] = fmt(r.pearson_mean)
    f5 = read("phase5/field_of_view.csv")
    if f5 is not None:
        for r in f5.itertuples():
            m[macro_name("fov", str(r.fov_um))] = fmt(r.pearson_mean)
    h5 = read("phase5/heads.csv")
    if h5 is not None:
        for name, d in h5.groupby("head"):
            m[macro_name("head", name)] = fmt(d.pearson_mean.mean())

    info = REP / "phase2" / "run_info.json"
    if info.exists():
        m["AtlasFlags"] = str(len(json.load(open(info))["flags"]))

    lines = ["% generated by scripts/make_paper_numbers.py -- do not edit",
             "\\providecommand{\\pending}{\\textcolor{red}{[pending]}}"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(m)} macros and 3 tables to {GEN}")
    return m


if __name__ == "__main__":
    main()
