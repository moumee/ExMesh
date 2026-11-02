"""Markdown tables + paired comparisons for run_enhanced_ablation.py results.

Reads outputs/DTU/<scene>/enhanced/evaluation/main_<protocol>_<budget>/{summary.json,
per_view_metrics.csv}. Paired differences use per-view values with a cluster
bootstrap over source cameras (DTU protocol: the +/-5 deg pair of each capture
camera is one cluster; orbit protocol: each view is its own cluster).
Writes deliverables/enhanced_simplification/simplification_results.{md,csv}
(every condition) and simplification_summary.md (headline tables for the guide).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "deliverables/enhanced_simplification"
NAMES = {"wild_ss1": "Wild (Sobel 없음)", "sobel_ss1": "Sobel λ4", "ppq_ss1": "**PPQ (신규)**",
         "vis_ss1": "가시성", "wild_ss3": "Wild + SS3", "wild_ss1_tex": "Wild + 정제(tex)",
         "wild_ss1_full": "Wild + 정제(full)", "sobel_ss1_full": "Sobel + 정제(full)",
         "ppq_ss1_full": "**PPQ + 정제(full)**"}
PAIRS = [("ppq_ss1", "wild_ss1"), ("sobel_ss1", "wild_ss1"), ("ppq_ss1", "sobel_ss1"), ("vis_ss1", "wild_ss1"),
         ("wild_ss3", "wild_ss1"), ("wild_ss1_tex", "wild_ss1"), ("wild_ss1_full", "wild_ss1"),
         ("ppq_ss1_full", "ppq_ss1"), ("ppq_ss1_full", "wild_ss1_full"), ("ppq_ss1_full", "sobel_ss1_full")]
COST_PAIRS = [("ppq_ss1", "wild_ss1"), ("sobel_ss1", "wild_ss1"), ("ppq_ss1", "sobel_ss1")]
TEXTURE_PAIRS = [("wild_ss3", "wild_ss1"), ("wild_ss1_tex", "wild_ss1"), ("wild_ss1_full", "wild_ss1"),
                 ("ppq_ss1_full", "ppq_ss1")]
BUDGETS = (20000, 10000, 5000)
SCENES = ("scan24", "scan83")
PROTOCOLS = {"dtu": "촬영 카메라 ±5°", "orbit": "전 방위 고정 60뷰"}
EVALUATION = "evaluation"


def paired(rows, a, b, metric, seed=20261029):
    """Mean improvement of a over b (LPIPS sign flipped), 95% cluster-bootstrap CI, cluster win rate."""
    groups = sorted({r["source_view"] for r in rows})
    sign = -1 if metric == "lpips" else 1
    values = np.array([np.mean([sign * (float(r[f"{a}_{metric}"]) - float(r[f"{b}_{metric}"]))
                                for r in rows if r["source_view"] == g]) for g in groups])
    boots = np.mean(values[np.random.default_rng(seed).integers(0, len(values), (4000, len(values)))], axis=1)
    return float(values.mean()), np.percentile(boots, [2.5, 97.5]).tolist(), float(np.mean(values > 0))


def load(scene, protocol, budget, prefix="main"):
    directory = ROOT / "outputs/DTU" / scene / "enhanced" / EVALUATION / f"{prefix}_{protocol}_{budget}"
    if not (directory / "summary.json").exists():
        return None, None
    summary = json.loads((directory / "summary.json").read_text())
    with (directory / "per_view_metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if protocol == "orbit":
        for r in rows:
            r["source_view"] = r["view"]
    return summary, rows


def ci(value, interval, digits):
    mark = "" if interval[0] <= 0 <= interval[1] else "*"
    return f"{value:+.{digits}f}{mark} [{interval[0]:+.{digits}f}, {interval[1]:+.{digits}f}]"


def full_tables(records):
    lines = []
    for scene in SCENES:
        for protocol, title in PROTOCOLS.items():
            lines += [f"#### {scene} — {title}", "",
                      "| 목표 면 | 방법 | 면 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 평균 거리 % ↓ | HD95 % ↓ |",
                      "|---:|---|---:|---:|---:|---:|---:|---:|"]
            pair_lines = []
            for budget in BUDGETS:
                summary, rows = load(scene, protocol, budget)
                if summary is None:
                    continue
                for label, name in NAMES.items():
                    if label not in summary["render_mean"]:
                        continue
                    render, geometry = summary["render_mean"][label], summary["geometry_mean"][label]
                    lines.append(f"| {budget:,} | {name} | {summary['candidates'][label]['faces']:,} | "
                                 f"{render['psnr']:.3f} | {render['ssim']:.4f} | {render['lpips']:.4f} | "
                                 f"{geometry['mean_surface_distance_pct_bbox']:.4f} | {geometry['hd95_pct_bbox']:.4f} |")
                    records.append(dict(scene=scene, protocol=protocol, budget=budget, label=label, views=len(rows),
                                        faces=summary["candidates"][label]["faces"], **render,
                                        mean_distance_pct=geometry["mean_surface_distance_pct_bbox"],
                                        hd95_pct=geometry["hd95_pct_bbox"],
                                        hausdorff_pct=geometry["sampled_hausdorff_pct_bbox"]))
                for a, b in PAIRS:
                    if a in summary["render_mean"] and b in summary["render_mean"]:
                        p, l = paired(rows, a, b, "psnr"), paired(rows, a, b, "lpips")
                        pair_lines.append(f"| {budget:,} | {NAMES[a]} − {NAMES[b]} | {ci(p[0], p[1], 3)} | {p[2]:.0%} | "
                                          f"{ci(l[0], l[1], 4)} | {l[2]:.0%} |")
            lines += ["", "쌍별 차이(양수 = 앞 방법이 좋음, * = 95% cluster bootstrap 구간이 0을 포함하지 않음, "
                          "승률 = 카메라 그룹 중 앞 방법이 나은 비율):", "",
                      "| 목표 면 | 비교 | ΔPSNR dB [95% CI] | 승률 | ΔLPIPS(감소량) [95% CI] | 승률 |",
                      "|---:|---|---:|---:|---:|---:|", *pair_lines, ""]
    return lines


def summary_tables():
    """Headline tables: cost functions (geometry + appearance) and texture post-processing."""
    short = {"wild_ss1": "Wild", "sobel_ss1": "Sobel", "ppq_ss1": "PPQ"}
    # Header built from COST_PAIRS so the column labels cannot drift from the cell order.
    cost = ["| 장면 | 평가 시점 | 목표 면 | PSNR Wild / Sobel / PPQ | "
            + " | ".join(f"ΔPSNR {short[a]}−{short[b]}" for a, b in COST_PAIRS)
            + " | ΔLPIPS×10³ PPQ−Wild | 평균 거리 PPQ/Wild |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    texture = ["| 장면 | 평가 시점 | 목표 면 | SS3 ΔPSNR / ΔLPIPS×10³ | 정제 tex | 정제 full | PPQ+정제 full (PPQ 대비) |",
               "|---|---|---:|---|---|---|---|"]
    for scene in SCENES:
        for protocol, title in PROTOCOLS.items():
            for budget in BUDGETS:
                summary, rows = load(scene, protocol, budget)
                if summary is None:
                    continue
                render, geometry = summary["render_mean"], summary["geometry_mean"]
                deltas = {pair: (paired(rows, *pair, "psnr"), paired(rows, *pair, "lpips")) for pair in COST_PAIRS + TEXTURE_PAIRS
                          if all(x in render for x in pair)}
                psnr = " / ".join(f"{render[x]['psnr']:.2f}" for x in ("wild_ss1", "sobel_ss1", "ppq_ss1"))
                ratio = geometry["ppq_ss1"]["mean_surface_distance_pct_bbox"] / geometry["wild_ss1"]["mean_surface_distance_pct_bbox"]
                cost.append(f"| {scene} | {title} | {budget:,} | {psnr} | " +
                            " | ".join(ci(deltas[p][0][0], deltas[p][0][1], 2) for p in COST_PAIRS) +
                            f" | {deltas[COST_PAIRS[0]][1][0] * 1e3:+.1f} | {ratio:.2f} |")
                cells = [f"{deltas[p][0][0]:+.2f}{'' if deltas[p][0][1][0] <= 0 <= deltas[p][0][1][1] else '*'} / "
                         f"{deltas[p][1][0] * 1e3:+.1f}{'' if deltas[p][1][1][0] <= 0 <= deltas[p][1][1][1] else '*'}"
                         for p in TEXTURE_PAIRS]
                texture.append(f"| {scene} | {title} | {budget:,} | " + " | ".join(cells) + " |")
    return cost, texture


def engine_table():
    """Same pairs on the engine-like renders (evaluate_engine_like.py), if they exist."""
    pairs = COST_PAIRS + TEXTURE_PAIRS
    table = ["| 장면 | 평가 시점 | 목표 면 | " + " | ".join(f"{NAMES[a].strip('*')} − {NAMES[b].strip('*')}" for a, b in pairs) + " |",
             "|---|---|---:|" + "---|" * len(pairs)]
    for scene in SCENES:
        for protocol, title in PROTOCOLS.items():
            for budget in BUDGETS:
                summary, rows = load(scene, protocol, budget, "engine")
                if summary is None:
                    continue
                cells = []
                for pair in pairs:
                    p, l = paired(rows, *pair, "psnr"), paired(rows, *pair, "lpips")
                    cells.append(f"{p[0]:+.2f}{'' if p[1][0] <= 0 <= p[1][1] else '*'} / "
                                 f"{l[0] * 1e3:+.1f}{'' if l[1][0] <= 0 <= l[1][1] else '*'}")
                table.append(f"| {scene} | {title} | {budget:,} | " + " | ".join(cells) + " |")
    return table if len(table) > 2 else []


def main():
    global OUT, EVALUATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default=EVALUATION, help="e.g. evaluation/v1_refine_seams")
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    EVALUATION, OUT = args.evaluation, Path(args.output)
    records = []
    lines = full_tables(records)
    cost, texture = summary_tables()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "simplification_results.md").write_text("\n".join(lines), encoding="utf-8")
    engine = engine_table()
    (OUT / "simplification_summary.md").write_text("\n".join(cost + [""] + texture + [""] + engine) + "\n",
                                                   encoding="utf-8")
    if records:
        with (OUT / "simplification_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # cp949 consoles
    print("\n".join(cost + [""] + texture + [""] + engine))


if __name__ == "__main__":
    main()
