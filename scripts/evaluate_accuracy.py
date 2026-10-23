#!/usr/bin/env python3
"""在带标签的文件夹上评估猫/非猫分类器：准确率、精确率、召回率、F1、混淆矩阵。

目录结构（标签就是子文件夹名，大小写不敏感）：

    test_data/
    ├─ cat/       *.jpg
    └─ notcat/    *.jpg

用法：

    python scripts/evaluate_accuracy.py                       # 默认 ./test_data
    python scripts/evaluate_accuracy.py --data-dir ~/catvision/test_data
    python scripts/evaluate_accuracy.py --threshold 0.5 --threads 4 --json

约定：类别 0 = cat（正类），类别 1 = notcat。p_cat >= 阈值判为 cat。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.catface.cat_classifier import CatClassifier  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABELS = ("cat", "notcat")


def collect(data_dir: Path) -> list[tuple[Path, str]]:
    """返回 [(图片路径, 真实标签)]，标签取自子文件夹名。"""
    items: list[tuple[Path, str]] = []
    for label in LABELS:
        sub = None
        for cand in data_dir.iterdir():
            if cand.is_dir() and cand.name.lower() == label:
                sub = cand
                break
        if sub is None:
            raise SystemExit(f"{data_dir} 下找不到 {label}/ 子文件夹")
        for p in sorted(sub.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                items.append((p, label))
    return items


def rss_mb() -> float:
    try:
        with open("/proc/self/statm", encoding="utf-8") as f:
            return int(f.read().split()[1]) * 4096 / 1024 / 1024
    except Exception:
        return -1.0


def main() -> int:
    ap = argparse.ArgumentParser(description="猫/非猫分类器准确率评估")
    ap.add_argument("--data-dir", default="test_data")
    ap.add_argument("--model", default="models/cat_vs_notcat.onnx")
    ap.add_argument("--threshold", type=float, default=0.5, help="p_cat 判定阈值")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"数据目录不存在：{data_dir}")

    items = collect(data_dir)
    clf = CatClassifier(args.model, intra_op_threads=args.threads)

    rows = []
    latencies = []
    for path, truth in items:
        t0 = time.perf_counter()
        with Image.open(path) as im:
            probs = clf.predict_proba(im)
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)
        pred = "cat" if probs[0] >= args.threshold else "notcat"
        rows.append(
            {
                "file": f"{path.parent.name}/{path.name}",
                "truth": truth,
                "pred": pred,
                "p_cat": float(probs[0]),
                "correct": pred == truth,
                "ms": dt,
            }
        )

    tp = sum(1 for r in rows if r["truth"] == "cat" and r["pred"] == "cat")
    fn = sum(1 for r in rows if r["truth"] == "cat" and r["pred"] != "cat")
    fp = sum(1 for r in rows if r["truth"] != "cat" and r["pred"] == "cat")
    tn = sum(1 for r in rows if r["truth"] != "cat" and r["pred"] != "cat")
    n = len(rows)

    def safe(a: int, b: int) -> float:
        return a / b if b else float("nan")

    acc = safe(tp + tn, n)
    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)          # 猫的召回率（灵敏度）
    specificity = safe(tn, tn + fp)     # 非猫的召回率
    f1 = safe(2 * precision * recall, precision + recall) if precision and recall else 0.0
    balanced = (recall + specificity) / 2
    lat = np.array(latencies)

    summary = {
        "data_dir": str(data_dir),
        "model": str(args.model),
        "threshold": args.threshold,
        "samples": {"cat": tp + fn, "notcat": fp + tn, "total": n},
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "metrics": {
            "accuracy": round(acc, 6),
            "precision_cat": round(precision, 6),
            "recall_cat": round(recall, 6),
            "specificity_notcat": round(specificity, 6),
            "f1_cat": round(f1, 6),
            "balanced_accuracy": round(balanced, 6),
        },
        "latency_ms": {
            "mean": round(float(lat.mean()), 2),
            "median": round(float(np.median(lat)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
        },
        "rss_mb": round(rss_mb(), 1),
        "errors": [r for r in rows if not r["correct"]],
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    w = 62
    print("=" * w)
    print("猫/非猫分类器 · 测试集评估")
    print("=" * w)
    print(f"模型      : {args.model}")
    print(f"数据目录  : {data_dir}")
    print(f"样本      : cat {tp + fn} 张, notcat {fp + tn} 张, 共 {n} 张")
    print(f"判定阈值  : p_cat >= {args.threshold}")
    print("-" * w)
    print("混淆矩阵（行=真实, 列=预测）")
    print(f"{'':<10}{'pred cat':>12}{'pred notcat':>14}")
    print(f"{'true cat':<10}{tp:>12}{fn:>14}")
    print(f"{'true notcat':<10}{fp:>12}{tn:>14}")
    print("-" * w)
    print("指标")
    print(f"  准确率 Accuracy        : {acc * 100:6.2f}%  ({(tp + tn)}/{n})")
    print(f"  精确率 Precision (cat) : {precision * 100:6.2f}%  (猫判对 / 判成猫的总数)")
    print(f"  召回率 Recall    (cat) : {recall * 100:6.2f}%  (猫判对 / 真实猫总数)")
    print(f"  特异度 Specificity     : {specificity * 100:6.2f}%  (非猫判对 / 真实非猫)")
    print(f"  F1 (cat)               : {f1 * 100:6.2f}%")
    print(f"  平衡准确率             : {balanced * 100:6.2f}%")
    print("-" * w)
    print(f"单张耗时  : 均值 {summary['latency_ms']['mean']} ms | "
          f"中位 {summary['latency_ms']['median']} ms | "
          f"p95 {summary['latency_ms']['p95']} ms")
    print(f"进程 RSS  : {summary['rss_mb']} MB")
    print("-" * w)
    errs = summary["errors"]
    if errs:
        print(f"错分 {len(errs)} 张：")
        for r in errs:
            print(f"  {r['file']:<42} 真={r['truth']:<7} 判={r['pred']:<7} p_cat={r['p_cat']:.4f}")
    else:
        print("错分 0 张。")
    print("=" * w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
