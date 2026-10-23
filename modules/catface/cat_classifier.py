"""MobileNetV2「猫 / 非猫」二分类器（ONNX Runtime，纯 CPU）。

设计要点：
- **不依赖 OpenCV**：树莓派上 cv2 的 aarch64 轮子体积大、内存重，这里只用
  numpy + Pillow，和 Pi 上已有的库一致。
- 预处理严格对齐训练时的 torchvision 流程：
  RGB → Resize(224, 224, bilinear) → /255 → Normalize(mean, std) → CHW → NCHW。
- 模型输出是 **logits**（不是 softmax 后的概率），所以这里自己做 softmax。
- 类别顺序固定为 ["cat", "notcat"]，下标 0 = cat。

用法：
    from modules.catface.cat_classifier import CatClassifier

    clf = CatClassifier("models/cat_vs_notcat.onnx")
    is_cat, conf = clf.predict(image_bgr)      # OpenCV 风格的 BGR ndarray
    is_cat, conf = clf.predict_file("x.jpg")   # 直接读文件

命令行自测：
    python -m modules.catface.cat_classifier --image cat.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

CLASSES = ("cat", "notcat")
INPUT_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_MODEL = Path("models/cat_vs_notcat.onnx")


def _to_rgb_array(img) -> np.ndarray:
    """把 PIL.Image / ndarray（RGB 或 BGR）统一成 HWC uint8 RGB。"""
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"))
    if isinstance(img, (str, Path)):
        with Image.open(img) as im:
            return np.asarray(im.convert("RGB"))
    arr = np.asarray(img)
    if arr.ndim == 2:  # 灰度
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


class CatClassifier:
    """猫/非猫二分类。线程数默认 2 —— Pi 4B 是 4 核且只有 1GB 内存。"""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, intra_op_threads: int = 2):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"找不到模型：{model_path}（从开发机 scp 到 ~/catvision/models/ 即可）"
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.output_name = self.session.get_outputs()[0].name
        self.model_path = model_path

    # ---------- 预处理 ----------
    @staticmethod
    def preprocess(img) -> np.ndarray:
        """任意图片 → 形状 (1, 3, 224, 224) 的 float32 张量。"""
        if isinstance(img, Image.Image):
            pil = img.convert("RGB")
        else:
            pil = Image.fromarray(_to_rgb_array(img), mode="RGB")
        pil = ImageOps.exif_transpose(pil)  # 手机照片的旋转标记
        pil = pil.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)

        x = np.asarray(pil, dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(x, dtype=np.float32)

    # ---------- 推理 ----------
    def predict_proba(self, img) -> np.ndarray:
        """返回两类概率 [p_cat, p_notcat]。"""
        x = self.preprocess(img)
        logits = self.session.run([self.output_name], {self.input_name: x})[0][0]
        logits = np.asarray(logits, dtype=np.float64).ravel()
        if logits.size == 1:  # 万一导成了单 logit 的 sigmoid 模型
            p_cat = 1.0 / (1.0 + np.exp(-logits[0]))
            return np.array([p_cat, 1.0 - p_cat], dtype=np.float32)
        e = np.exp(logits - logits.max())
        return (e / e.sum()).astype(np.float32)

    def predict(self, frame_bgr) -> tuple[bool, float]:
        """兼容老接口：输入 BGR ndarray，返回 (是否猫, 猫的置信度)。"""
        arr = np.asarray(frame_bgr)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, ::-1]  # BGR → RGB
        probs = self.predict_proba(arr)
        return bool(probs[0] >= 0.5), float(probs[0])

    def predict_file(self, path: str | Path) -> tuple[bool, float]:
        """读文件直接分类，返回 (是否猫, 猫的置信度)。"""
        with Image.open(path) as im:
            probs = self.predict_proba(im)
        return bool(probs[0] >= 0.5), float(probs[0])


def _cli() -> int:
    ap = argparse.ArgumentParser(description="猫/非猫二分类器自测")
    ap.add_argument("--image", required=True, help="图片路径")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--repeat", type=int, default=5, help="重复推理次数（测延迟）")
    args = ap.parse_args()

    clf = CatClassifier(args.model)
    img = Image.open(args.image)

    probs = clf.predict_proba(img)
    is_cat = bool(probs[0] >= 0.5)
    print(f"图片: {args.image}")
    print(f"结果: {CLASSES[0] if is_cat else CLASSES[1]}  "
          f"(p_cat={probs[0]:.4f}, p_notcat={probs[1]:.4f})")

    import time

    clf.predict_proba(img)  # 预热
    t0 = time.perf_counter()
    for _ in range(args.repeat):
        clf.predict_proba(img)
    dt = (time.perf_counter() - t0) / args.repeat
    print(f"单次推理: {dt * 1000:.1f} ms  ({1 / dt:.1f} FPS, 单线程链路)")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
