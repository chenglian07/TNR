"""猫/非猫分类器的单元测试。

模型文件不进 Git（见 .gitignore），所以没模型时自动跳过，
但**预处理契约**（尺寸/通道/归一化）不需要模型也能测，必须始终通过。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from modules.catface.cat_classifier import (
    MEAN,
    STD,
    CatClassifier,
    _to_rgb_array,
)

MODEL = Path(__file__).resolve().parents[1] / "models" / "cat_vs_notcat.onnx"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "classifier"

needs_model = pytest.mark.skipif(not MODEL.exists(), reason=f"没有模型文件 {MODEL.name}")


def _fake_image(w: int = 800, h: int = 600) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


# ---------- 预处理契约（不需要模型） ----------


def test_rgb_array_from_pil():
    arr = _to_rgb_array(_fake_image())
    assert arr.shape == (600, 800, 3)
    assert arr.dtype == np.uint8


def test_rgb_array_drops_alpha():
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    assert _to_rgb_array(rgba).shape == (10, 10, 3)


def test_rgb_array_expands_grayscale():
    gray = np.zeros((10, 10), dtype=np.uint8)
    assert _to_rgb_array(gray).shape == (10, 10, 3)


def test_rgb_array_rejects_float_overflow():
    """float 图（0-1）也要能进来，不能溢出成垃圾。"""
    arr = _to_rgb_array(np.full((8, 8, 3), 1.0, dtype=np.float32))
    assert arr.dtype == np.uint8
    assert arr.max() == 1


def test_preprocess_shape_and_dtype():
    x = CatClassifier.preprocess(_fake_image())
    assert x.shape == (1, 3, 224, 224)
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]


def test_preprocess_normalization_range():
    """归一化后应大致落在 [-2.12, 2.64]（0 和 1 的极值处）。"""
    white = CatClassifier.preprocess(Image.new("RGB", (64, 64), (255, 255, 255)))
    black = CatClassifier.preprocess(Image.new("RGB", (64, 64), (0, 0, 0)))
    expect_hi = (1.0 - MEAN) / STD
    expect_lo = (0.0 - MEAN) / STD
    np.testing.assert_allclose(white[0, :, 0, 0], expect_hi, rtol=1e-5)
    np.testing.assert_allclose(black[0, :, 0, 0], expect_lo, rtol=1e-5)


def test_preprocess_channels_are_rgb_not_bgr():
    """纯红图：经过预处理后，第 0 通道（R）应比第 2 通道（B）大。"""
    red = Image.new("RGB", (32, 32), (255, 0, 0))
    x = CatClassifier.preprocess(red)[0]
    assert x[0].mean() > x[2].mean()


# ---------- 真模型推理（没有模型就跳过） ----------


@needs_model
def test_model_io_contract():
    clf = CatClassifier(MODEL, intra_op_threads=2)
    probs = clf.predict_proba(_fake_image())
    assert probs.shape == (2,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    assert (probs >= 0).all() and (probs <= 1).all()


@needs_model
def test_predict_returns_bool_and_conf():
    clf = CatClassifier(MODEL, intra_op_threads=2)
    is_cat, conf = clf.predict(np.zeros((64, 64, 3), dtype=np.uint8))
    assert isinstance(is_cat, bool)
    assert 0.0 <= conf <= 1.0
    assert is_cat == (conf >= 0.5)


@pytest.mark.skipif(
    not (FIXTURES / "chelsea_cat.png").exists(), reason="缺少测试图片"
)
@needs_model
def test_known_photos():
    """真实照片回归：猫图判 cat，非猫图判 notcat。"""
    clf = CatClassifier(MODEL, intra_op_threads=2)
    cases = {
        "chelsea_cat.png": "cat",       # skimage 自带的猫
        "coffee_notcat.png": "notcat",
        "astronaut_notcat.png": "notcat",
    }
    for name, expect in cases.items():
        probs = clf.predict_proba(FIXTURES / name)
        got = "cat" if probs[0] >= 0.5 else "notcat"
        assert got == expect, f"{name}: 期望 {expect}，实际 {got} (p_cat={probs[0]:.4f})"
