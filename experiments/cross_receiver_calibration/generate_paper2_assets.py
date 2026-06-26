#!/usr/bin/env python3
"""Generate Paper 2 paper-ready tables and figures from frozen results."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / "experiments/cross_receiver_calibration"
P2 = ROOT / "docs/paper2_rcpa"
FIG = P2 / "figures"
TAB = P2 / "tables"


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_mean(s: str) -> float:
    return float(s.split("±")[0].strip())


def table1_baseline(out: Path) -> None:
    tta = {r["method"]: r for r in load_csv(CAL / "results/tta_negative_quick_20260626_1744/summary_tta_negative.csv")}
    main = load_csv(CAL / "results/paper2_main/paper2_main_table.csv")
    rows = []
    for d in ["rx1_to_rx2", "rx2_to_rx1"]:
        cls = next(r for r in main if r["direction"] == d and "Source classifier" in r["method"])
        proto = next(r for r in main if r["direction"] == d and "RCPA-S" in r["method"] and r["K"] == "0")
        rows.append({"direction": d, "source_classifier": cls["file_acc"], "source_prototype": proto["file_acc"]})
    rows.append({
        "direction": "RX1→RX2 (quick)",
        "entropy_min_tta": f"{float(tta['entropy_min_tta']['file_acc'])*100:.1f}",
        "pseudo_proto_tta": f"{float(tta['pseudo_proto_tta']['file_acc'])*100:.1f}",
    })
    fields = ["direction", "source_classifier", "source_prototype", "entropy_min_tta", "pseudo_proto_tta"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, d in enumerate(["rx1_to_rx2", "rx2_to_rx1"]):
            w.writerow({"direction": d, "source_classifier": rows[i]["source_classifier"], "source_prototype": rows[i]["source_prototype"],
                        "entropy_min_tta": "20.8 ± —" if d == "rx1_to_rx2" else "—", "pseudo_proto_tta": "8.3 ± —" if d == "rx1_to_rx2" else "—"})
    # tex
    tex = TAB / "table1_baseline.tex"
    tex.write_text(r"""\begin{table}[t]
\caption{Cross-receiver source-only baselines (file-level accuracy, \%). TTA values from quick RX1$\rightarrow$RX2 run; full-direction RCPA aggregates in Table~\ref{tab:rcpa}.}
\label{tab:baseline}
\centering
\begin{tabular}{lcccc}
\toprule
Direction & Source cls. & Source proto. & Entropy TTA & Pseudo-proto TTA \\
\midrule
RX1$\rightarrow$RX2 & 19.4$\pm$3.4 & 15.3$\pm$7.3 & 20.8 & 8.3 \\
RX2$\rightarrow$RX1 & 20.8$\pm$8.8 & 13.0$\pm$3.1 & --- & --- \\
\bottomrule
\end{tabular}
\end{table}
""", encoding="utf-8")


def table2_rcpa(out: Path) -> None:
    pooled = load_csv(CAL / "results/paper2_main/paper2_rcpa_t_pooled.csv")
    cls_mean = 20.1  # from paper2_main
    tex_lines = [r"\begin{table}[t]", r"\caption{RCPA-T shot curve (file-level accuracy, \%). Mean$\pm$std over 3 seeds$\times$3 splits per direction.}",
                 r"\label{tab:rcpa}", r"\centering", r"\begin{tabular}{ccccc}", r"\toprule",
                 r"$K$ & RX1$\rightarrow$RX2 & RX2$\rightarrow$RX1 & Pooled & $\Delta$ vs cls. \\", r"\midrule"]
    main = load_csv(CAL / "results/paper2_main/paper2_main_table.csv")
    for k in [1, 3, 5, 10, 20]:
        r1 = next(r for r in main if r["direction"] == "rx1_to_rx2" and "RCPA-T" in r["method"] and r["K"] == str(k))
        r2 = next(r for r in main if r["direction"] == "rx2_to_rx1" and "RCPA-T" in r["method"] and r["K"] == str(k))
        p = next(r for r in pooled if int(r["K"]) == k)
        delta = parse_mean(p["RCPA-T mean±std"]) - cls_mean
        tex_lines.append(f"{k} & {r1['file_acc']} & {r2['file_acc']} & {p['RCPA-T mean±std']} & {delta:+.1f} pp \\\\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "table2_rcpa_shotcurve.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def table3_ablation(out: Path) -> None:
    main = load_csv(CAL / "results/paper2_main/paper2_main_table.csv")
    oob = load_csv(CAL / "results/oob_eq_quick_20260626_1731/probe_before_after.csv")
    tex = TAB / "table3_ablation.tex"
    k5_s = next(r for r in main if "RCPA-S" in r["method"] and r["K"] == "5" and r["direction"] == "rx1_to_rx2")["file_acc"]
    k5_t = next(r for r in main if "RCPA-T" in r["method"] and r["K"] == "5" and r["direction"] == "rx1_to_rx2")["file_acc"]
    k5_b = next(r for r in main if "RCPA-B" in r["method"] and r["K"] == "5" and r["direction"] == "rx1_to_rx2")["file_acc"]
    rx_before = float(next(r for r in oob if r["repr"] == "oob_only" and r["phase"] == "before")["receiver_probe_acc"]) * 100
    rx_after = float(next(r for r in oob if r["repr"] == "oob_only" and r["eq_method"] == "coral")["receiver_probe_acc"]) * 100
    tex.write_text(f"""\\begin{{table}}[t]
\\caption{{Ablation and auxiliary analyses (RX1$\\rightarrow$RX2 full-mode means at $K$=5 unless noted).}}
\\label{{tab:ablation}}
\\centering
\\begin{{tabular}}{{lc}}
\\toprule
Method / analysis & File-acc or probe \\\\
\\midrule
RCPA-S (source prototype) & {k5_s} \\\\
RCPA-T (primary) & {k5_t} \\\\
RCPA-B (blend ablation) & {k5_b} \\\\
OOB-Eq CORAL: RX probe before/after & {rx_before:.1f}\\% / {rx_after:.1f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""", encoding="utf-8")


def fig2_rcpa_shotcurve() -> None:
    pooled = load_csv(CAL / "results/paper2_main/paper2_rcpa_t_pooled.csv")
    main = load_csv(CAL / "results/paper2_main/paper2_main_table.csv")
    ks = [1, 3, 5, 10, 20]
    fig, ax = plt.subplots(figsize=(7, 4))
    for direction, color, label in [("rx1_to_rx2", "C0", "RX1→RX2"), ("rx2_to_rx1", "C1", "RX2→RX1")]:
        ys, stds = [], []
        for k in ks:
            r = next(x for x in main if x["direction"] == direction and "RCPA-T" in x["method"] and x["K"] == str(k))
            parts = r["file_acc"].split("±")
            ys.append(float(parts[0].strip()))
            stds.append(float(parts[1].strip()) if len(parts) > 1 else 0)
        ax.errorbar(ks, ys, yerr=stds, marker="o", capsize=3, color=color, label=label)
    p_ys = [parse_mean(next(x for x in pooled if int(x["K"]) == k)["RCPA-T mean±std"]) for k in ks]
    ax.plot(ks, p_ys, "k--", marker="s", label="Pooled mean")
    ax.axhline(20.1, color="gray", ls=":", label="Source classifier (~20%)")
    ax.set_xlabel("K (labeled calibration windows per device)")
    ax.set_ylabel("File-level accuracy (%)")
    ax.set_title("RCPA-T shot curve (frozen full-mode results)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_rcpa_shotcurve.pdf")
    fig.savefig(FIG / "fig2_rcpa_shotcurve.png", dpi=150)
    plt.close(fig)


def fig1_diagnosis_summary() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    # Panel A: receiver probe by path
    paths = ["main", "oob", "fused"]
    rx_probe = [0.624, 0.727, 0.502]
    axes[0, 0].bar(paths, [v * 100 for v in rx_probe], color=["C0", "C2", "C1"])
    axes[0, 0].set_ylabel("Receiver probe acc (%)")
    axes[0, 0].set_title("(a) Receiver probe by embedding path")
    axes[0, 0].set_ylim(0, 80)

    # Panel B: OOB spectrum bias
    axes[0, 1].bar(["RX2/RX1 OOB\nenergy ratio"], [1.442], color="C3")
    axes[0, 1].axhline(1.0, color="gray", ls="--")
    axes[0, 1].set_ylabel("Ratio")
    axes[0, 1].set_title("(b) OOB spectral receiver bias")

    # Panel C: distance ratio
    models = ["CNN-IQ", "Ours fused"]
    ratios = [1.251, 0.221]
    colors = ["C3" if r > 1 else "C2" for r in ratios]
    axes[1, 0].bar(models, ratios, color=colors)
    axes[1, 0].axhline(1.0, color="gray", ls="--")
    axes[1, 0].set_ylabel("same-dev cross-RX / diff-dev ratio")
    axes[1, 0].set_title("(c) Embedding distance ratio")

    # Panel D: collapse
    methods = ["CNN\nRX1→RX2", "Ours\nRX1→RX2"]
    top1 = [95.8, 20.8]
    axes[1, 1].bar(methods, top1, color=["C3", "C1"])
    axes[1, 1].set_ylabel("Top-1 prediction mass (%)")
    axes[1, 1].set_title("(d) Confusion collapse (seed0 file-level)")
    axes[1, 1].set_ylim(0, 100)

    fig.suptitle("Cross-receiver failure diagnosis summary", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_diagnosis_summary.pdf")
    fig.savefig(FIG / "fig1_diagnosis_summary.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    table1_baseline(TAB / "table1_baseline.csv")
    table2_rcpa(TAB / "table2_rcpa.csv")
    table3_ablation(TAB / "table3_ablation.csv")
    fig1_diagnosis_summary()
    fig2_rcpa_shotcurve()
    print(f"Generated tables in {TAB} and figures in {FIG}")


if __name__ == "__main__":
    main()
