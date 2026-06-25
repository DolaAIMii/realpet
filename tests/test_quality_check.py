"""Tests for quality_check module — covers each rejection path.

Run:  python -m pytest tests/test_quality_check.py -v
"""
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_VIDEO = os.environ.get("REALPET_TEST_VIDEO", "")


def _make_frame(h=480, w=640, brightness=128):
    """Create a synthetic frame with given brightness."""
    return np.full((h, w, 3), brightness, dtype=np.uint8)


def _make_video(tmpdir, frames_data, fps=30):
    """Create a minimal video from frame data. Returns path."""
    path = os.path.join(tmpdir, "test.mp4")
    h, w = frames_data[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames_data:
        writer.write(frame)
    writer.release()
    return path


class TestQualityResult:
    def test_pass_by_default(self):
        from scripts.track_then_matte import QualityResult
        r = QualityResult()
        assert r.passed is True
        assert r.issues == []

    def test_fail(self):
        from scripts.track_then_matte import QualityResult
        r = QualityResult()
        r.fail("test_code", "test message")
        assert r.passed is False
        assert len(r.issues) == 1
        assert r.issues[0]["code"] == "test_code"

    def test_to_dict(self):
        from scripts.track_then_matte import QualityResult
        r = QualityResult()
        r.fail("code1", "msg1")
        d = r.to_dict()
        assert d["passed"] is False
        assert len(d["issues"]) == 1


class TestCheckQuality:
    """Test check_quality with synthetic frames (no video file needed)."""

    def test_no_frames(self):
        from scripts.track_then_matte import check_quality
        result = check_quality("/nonexistent.mp4", [])
        assert result.passed is False
        assert any(i["code"] == "no_frames" for i in result.issues)

    def test_low_resolution(self):
        from scripts.track_then_matte import check_quality
        with tempfile.TemporaryDirectory() as d:
            # Create a tiny frame (below 480p threshold)
            frame = _make_frame(240, 320)
            path = os.path.join(d, "frame.jpg")
            cv2.imwrite(path, frame)
            result = check_quality("/nonexistent.mp4", [path])
            assert any(i["code"] == "low_resolution" for i in result.issues)

    def test_too_dark(self):
        from scripts.track_then_matte import check_quality
        with tempfile.TemporaryDirectory() as d:
            # Very dark frame (brightness < 20)
            frame = _make_frame(480, 640, brightness=5)
            path = os.path.join(d, "frame.jpg")
            cv2.imwrite(path, frame)
            # Skip pet detection by not having enough frames for blur
            result = check_quality("/nonexistent.mp4", [path])
            assert any(i["code"] == "too_dark" for i in result.issues)

    def test_too_bright(self):
        from scripts.track_then_matte import check_quality
        from scripts.track_then_matte import MAX_OVEREXP_RATIO
        with tempfile.TemporaryDirectory() as d:
            # Fully overexposed frame
            frame = _make_frame(480, 640, brightness=255)
            path = os.path.join(d, "frame.jpg")
            cv2.imwrite(path, frame)
            result = check_quality("/nonexistent.mp4", [path])
            assert any(i["code"] == "too_bright" for i in result.issues)


class TestExtractFrames:
    """Test extract_frames function."""

    @pytest.mark.skipif(not os.path.exists(TEST_VIDEO), reason="Test video not found")
    def test_extract_returns_frames(self):
        from scripts.quality_check import extract_frames
        frames = extract_frames(TEST_VIDEO, num_frames=3)
        assert len(frames) == 3
        for f in frames:
            assert os.path.exists(f)
            assert f.endswith(".jpg")
        # Cleanup
        if frames:
            import shutil
            shutil.rmtree(os.path.dirname(frames[0]), ignore_errors=True)

    @pytest.mark.skipif(not os.path.exists(TEST_VIDEO), reason="Test video not found")
    def test_extract_default_count(self):
        from scripts.quality_check import extract_frames
        frames = extract_frames(TEST_VIDEO)
        assert len(frames) == 6
        # Cleanup
        if frames:
            import shutil
            shutil.rmtree(os.path.dirname(frames[0]), ignore_errors=True)


class TestRunQC:
    """Integration test for run_qc."""

    @pytest.mark.skipif(not os.path.exists(TEST_VIDEO), reason="Test video not found")
    def test_qc_passes_on_good_video(self):
        from scripts.quality_check import run_qc
        result = run_qc(TEST_VIDEO, "/tmp/test_qc")
        assert result["type"] == "qc"
        assert result["passed"] is True

    def test_qc_fails_on_nonexistent(self):
        from scripts.quality_check import run_qc
        result = run_qc("/nonexistent/video.mp4", "/tmp/test_qc")
        assert result["type"] == "qc"
        assert result["passed"] is False
