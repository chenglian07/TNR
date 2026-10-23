#!/usr/bin/env python3
"""树莓派实时演示：CSI 摄像头取流 → 猫/非猫分类 → 打印结果。

    source ~/venvs/catvision/bin/activate
    python scripts/pi_live_demo.py --frames 30 --fps 5
    python scripts/pi_live_demo.py --frames 10 --save-dir /dev/shm/shots   # 存图看效果

注意：帧存到 /dev/shm（内存盘），别写 SD 卡。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.catface.cat_classifier import CatClassifier  # noqa: E402


def rss_mb() -> float:
    try:
        with open("/proc/self/statm", encoding="utf-8") as f:
            return int(f.read().split()[1]) * 4096 / 1024 / 1024
    except Exception:
        return -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/cat_vs_notcat.onnx")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--fps", type=float, default=5.0, help="取帧目标帧率")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--save-dir", default="", help="把每帧存成 jpg（建议 /dev/shm 下）")
    args = ap.parse_args()

    from picamera2 import Picamera2  # 只在 Pi 上可用

    clf = CatClassifier(args.model, intra_op_threads=args.threads)
    print(f"模型已加载：{args.model}（{args.threads} 线程）RSS={rss_mb():.0f} MB")

    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(1.0)  # 等自动曝光收敛

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    interval = 1.0 / args.fps if args.fps > 0 else 0.0
    cat_count = 0
    grabs, infers = [], []
    try:
        for i in range(args.frames):
            t0 = time.perf_counter()
            frame = picam2.capture_array()  # HWC uint8, RGB
            t1 = time.perf_counter()
            probs = clf.predict_proba(frame)
            t2 = time.perf_counter()

            grabs.append((t1 - t0) * 1000)
            infers.append((t2 - t1) * 1000)
            label = "cat" if probs[0] >= 0.5 else "notcat"
            cat_count += label == "cat"
            print(f"[{i:3d}] {label:6s} p_cat={probs[0]:.4f}  "
                  f"grab={grabs[-1]:6.1f}ms infer={infers[-1]:6.1f}ms")

            if save_dir:
                from PIL import Image

                Image.fromarray(frame).save(save_dir / f"f{i:03d}_{label}.jpg", quality=85)

            slack = interval - (time.perf_counter() - t0)
            if slack > 0:
                time.sleep(slack)
    finally:
        picam2.stop()
        picam2.close()

    print("-" * 48)
    print(f"帧数 {len(grabs)} | 猫 {cat_count} / 非猫 {len(grabs) - cat_count}")
    print(f"平均 取帧 {np.mean(grabs):.1f} ms + 推理 {np.mean(infers):.1f} ms "
          f"= {np.mean(grabs) + np.mean(infers):.1f} ms/帧 "
          f"(≈{1000 / (np.mean(grabs) + np.mean(infers)):.1f} FPS 上限)")
    print(f"峰值 RSS {rss_mb():.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
