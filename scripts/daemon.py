#!/usr/bin/env python3
"""Resident worker: load detector once, serve QC/detect over stdin/stdout NDJSON.

Phase 1: Only handles qc / detect / ping.  Process (SAM2+BiRefNet) still
runs as a separate subprocess — too risky to share MPS state.

Protocol (NDJSON, one JSON per line):
  Request:  {"id":1, "cmd":"qc", "video":"...", "output_dir":"..."}
  Response: streaming lines each tagged {"id":1, ...}
            final line   {"id":1, "done":true, "result":{...}}
  Ready:    {"type":"ready"} emitted once at startup after model loads.

Exit: close stdin or send SIGTERM.
"""
import json
import os
import sys
import traceback

# Ensure project root is in sys.path for pipeline imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _emit(req_id, msg):
    """Write one NDJSON line to stdout."""
    msg["id"] = req_id
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _handle(req):
    """Dispatch a request and return the terminal result dict (or None)."""
    cmd = req.get("cmd")

    if cmd == "ping":
        return {"ok": True}

    elif cmd == "qc":
        from scripts.quality_check import run_qc
        result = run_qc(req["video"], req["output_dir"])
        return result

    elif cmd == "detect":
        from scripts.detect_pet import run_detect
        result = run_detect(req["video"], req["output_dir"],
                            start=req.get("start", 0))
        return result

    else:
        return {"error": f"unknown cmd: {cmd}"}


def main():
    # Eagerly load the detector so first qc/detect is instant.
    try:
        from pipeline.pet_detector import _load_model
        _load_model()
    except Exception as e:
        _emit(0, {"type": "warning", "message": f"detector preload failed: {e}"})

    # Signal readiness
    _emit(0, {"type": "ready"})

    # Blocking read loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        req = None
        try:
            req = json.loads(line)
            req_id = req.get("id", -1)

            result = _handle(req)
            if result is not None:
                _emit(req_id, {"done": True, "result": result})

        except Exception:
            req_id = req.get("id", -1) if req else -1
            _emit(req_id, {"done": True, "error": traceback.format_exc()})

    # stdin closed — exit cleanly
    sys.exit(0)


if __name__ == "__main__":
    main()
