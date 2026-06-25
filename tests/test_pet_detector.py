"""Smoke tests for pet_detector module.

Run:  python -m pytest tests/test_pet_detector.py -v
"""
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def detector():
    """Load the detector once for all tests."""
    from pipeline.pet_detector import _load_model
    model, device = _load_model()
    return model, device


@pytest.fixture
def dummy_frame():
    """Create a synthetic 640x480 BGR frame with a cat-like blob."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a light blob in the center (simulates a pet)
    cv2.ellipse(frame, (320, 240), (80, 60), 0, 0, 360, (200, 200, 200), -1)
    return frame


@pytest.fixture
def empty_frame():
    """Create a synthetic empty frame (no pet)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_has_pet_fast_returns_bool(dummy_frame):
    """has_pet_fast should return a bool."""
    from pipeline.pet_detector import has_pet_fast
    result = has_pet_fast(dummy_frame)
    assert isinstance(result, bool)


def test_has_pet_fast_empty_frame(empty_frame):
    """has_pet_fast on a blank frame should return False."""
    from pipeline.pet_detector import has_pet_fast
    result = has_pet_fast(empty_frame)
    assert result is False


def test_has_pet_fast_from_path():
    """has_pet_fast should accept a file path."""
    from pipeline.pet_detector import has_pet_fast
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
        cv2.imwrite(path, np.zeros((480, 640, 3), dtype=np.uint8))
    try:
        result = has_pet_fast(path)
        assert isinstance(result, bool)
    finally:
        os.unlink(path)


def test_detect_pet_bbox_returns_none_on_empty(empty_frame):
    """detect_pet_bbox on a blank frame should return None."""
    from pipeline.pet_detector import detect_pet_bbox
    result = detect_pet_bbox(empty_frame)
    assert result is None


def test_detect_pet_bbox_returns_list_or_none(dummy_frame):
    """detect_pet_bbox should return a list [x1,y1,x2,y2] or None."""
    from pipeline.pet_detector import detect_pet_bbox
    result = detect_pet_bbox(dummy_frame)
    if result is not None:
        assert isinstance(result, list)
        assert len(result) == 4
        assert all(isinstance(v, (int, float)) for v in result)
