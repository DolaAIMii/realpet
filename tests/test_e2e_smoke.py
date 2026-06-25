"""End-to-end smoke test — runs the full pipeline on a short video.

Requires: set REALPET_TEST_VIDEO env var to a .mov/.mp4 with a pet.
Run:  REALPET_TEST_VIDEO=path/to/video.mov python -m pytest tests/test_e2e_smoke.py -v
"""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_VIDEO = os.environ.get("REALPET_TEST_VIDEO", "")


@pytest.mark.skipif(not os.path.exists(TEST_VIDEO), reason="Test video not found")
class TestEndToEnd:
    """Smoke tests that verify the pipeline doesn't crash."""

    def test_qc(self):
        """QC gate should pass on a valid pet video."""
        from scripts.quality_check import run_qc
        with tempfile.TemporaryDirectory() as d:
            result = run_qc(TEST_VIDEO, d)
            assert result["type"] == "qc"
            assert result["passed"] is True

    def test_detect(self):
        """Pet detection should find a pet in the test video."""
        from scripts.detect_pet import run_detect
        with tempfile.TemporaryDirectory() as d:
            result = run_detect(TEST_VIDEO, d)
            assert result["type"] == "detected"
            assert "bbox" in result
            assert len(result["bbox"]) == 4

    def test_weight_paths(self):
        """Weight paths should resolve correctly."""
        from scripts.track_then_matte import _sam2_checkpoint, _weights_dir
        assert "weights" in _weights_dir()
        assert _sam2_checkpoint().endswith("sam2.1_hiera_tiny.pt")
