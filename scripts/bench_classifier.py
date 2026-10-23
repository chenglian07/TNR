#!/usr/bin/env python3
"""分类器基准测试：延迟 / 吞吐 / 内存占用 / 小样本准确性。

给树莓派用，也能量 PC。在 Pi 上：

    source ~/venvs/catvision/bin/activate
    python scripts/bench_classifier.py --model models/cat_vs_notcat.onnx \
        --images "tests/fixtures/classifier/*.png" --repeat 20

输出里重点看：
- 单帧延迟（fps 反推）：决定摄像头能跑几帧/秒
- 峰值 RSS：1GB 的 Pi4 上要和 picamera2 + 主进程共存
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.catface.cat_classifier import CatClassifier  # noqa: E402


def rss_mb() -> float:
    """当前进程 RSS（MB）。Linux 读 /proc，其它平台返回 -1。"""
    try:
        with open("/proc/self/statm", encoding="utf-8") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024
    except Exception:
        return -1.0


def load_images(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        files.extend(Path(p) for p in glob.glob(pat))
    return sorted(set(files))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/cat_vs_notcat.onnx")
    ap.add_argument("--images", nargs="+", required=True, help="图片 glob，可多个")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--json", action="store_true", help="输出 JSON（方便比对）")
    args = ap.parse_args()

    files = load_images(args.images)
    if not files:
        print(f"没有匹配到图片：{args.images}")
        return 1

    rss_before = rss_mb()
    t0 = time.perf_counter()
    clf = CatClassifier(args.model, intra_op_threads=args.threads)
    load_s = time.perf_counter() - t0
    rss_loaded = rss_mb()

    results = []
    latencies = []
    for f in files:
        with Image.open(f) as im:
            im.load()
            probs = clf.predict_proba(im)  # 预热一次（含首次内存分配）
            t = time.perf_counter()
            for _ in range(args.repeat):
                probs = clf.predict_proba(im)
            dt = (time.perf_counter() - t) / args.repeat
        latencies.append(dt * 1000)
        results.append(
            {
                "file": f.name,
                "p_cat": round(float(probs[0]), 6),
                "label": "cat" if probs[0] >= 0.5 else "notcat",
                "ms": round(dt * 1000, 2),
            }
        )

    lat = np.array(latencies)
    summary = {
        "model": str(args.model),
        "model_mb": round(Path(args.model).stat().st_size / 1024 / 1024, 2),
        "threads": args.threads,
        "images": len(files),
        "repeat": args.repeat,
        "load_s": round(load_s, 3),
        "latency_ms": {
            "mean": round(float(lat.mean()), 2),
            "median": round(float(statistics.median(lat)), 2),
            "min": round(float(lat.min()), 2),
            "max": round(float(lat.max()), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
        },
        "fps": round(1000.0 / float(lat.mean()), 2),
        "rss_mb": {
            "before_load": round(rss_before, 1),
            "after_load": round(rss_loaded, 1),
            "after_infer": round(rss_mb(), 1),
        },
        "results": results,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print(f"模型    : {summary['model']}  ({summary['model_mb']} MB, {summary['threads']} 线程)")
    print(f"加载    : {summary['load_s'] * 1000:.0f} ms")
    print(f"内存    : 加载前 {summary['rss_mb']['before_load']} MB → "
          f"加载后 {summary['rss_mb']['after_load']} MB → "
          f"推理后 {summary['rss_mb']['after_infer']} MB")
    print(f"延迟    : 均值 {summary['latency_ms']['mean']} ms | "
          f"中位 {summary['latency_ms']['median']} ms | "
          f"p95 {summary['latency_ms']['p95']} ms")
    print(f"吞吐    : {summary['fps']} FPS（单帧串行）")
    print(f"{'图片':<28s} {'p_cat':>10s} {'判定':>8s} {'ms':>8s}")
    for r in results:
        print(f"{r['file']:<28s} {r['p_cat']:>10.4f} {r['label']:>8s} {r['ms']:>8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
