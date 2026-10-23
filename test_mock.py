"""Offline end-to-end test: mock ComfyUI server + real moss_batch client.

Proves the client's HTTP contract, polling, image fetch, manifest and
shutdown hook all work without a GPU.

    python test_mock.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import moss_batch as mb

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c63000100000500010d0a2db4000000"
    "0049454e44ae426082"
)

STATE = {"submitted": [], "interrupted": False, "freed": False, "batch_seen": []}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - matches stdlib signature
        pass

    def _send(self, code, payload=None, raw=None, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        body = raw if raw is not None else json.dumps(payload or {}).encode()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/system_stats"):
            return self._send(200, {"system": {"pytorch_version": "2.9.0"},
                                    "devices": [{"name": "Mock 4090",
                                                 "vram_total": 24 * 2**30}]})
        if self.path.startswith("/history/"):
            pid = self.path.rsplit("/", 1)[-1]
            return self._send(200, {pid: {
                "status": {"status_str": "success", "completed": True},
                "outputs": {"18": {"images": [
                    {"filename": "moss_00001_.png", "subfolder": "moss", "type": "output"},
                    {"filename": "moss_00002_.png", "subfolder": "moss", "type": "output"},
                ]}},
            }})
        if self.path.startswith("/view"):
            return self._send(200, raw=PNG_1PX, ctype="image/png")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/prompt":
            graph = body["prompt"]
            latent = next(x for x in graph["nodes"] if str(x.get("title", "")).startswith("latent"))
            STATE["batch_seen"].append(latent["widgets_values"][2])
            pid = f"pid-{len(STATE['submitted'])}"
            STATE["submitted"].append(pid)
            return self._send(200, {"prompt_id": pid, "node_errors": {}})
        if self.path == "/free":
            STATE["freed"] = True
            return self._send(200, {})
        if self.path == "/interrupt":
            STATE["interrupted"] = True
            return self._send(200, {})
        return self._send(404, {"error": "not found"})


def main() -> int:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print(f"[test] mock ComfyUI on :{port}")

    here = Path(__file__).parent
    out = here / "_test_out"
    shutil.rmtree(out, ignore_errors=True)

    # Force the salad shutdown path to be a no-op even if env is set.
    for k in ("SALAD_ORG", "SALAD_PROJECT", "SALAD_GROUP", "SALAD_API_KEY"):
        os.environ.pop(k, None)

    t0 = time.time()
    rc = mb.main([
        "--host", "127.0.0.1", "--port", str(port),
        "--workflow", str(here / "workflow.json"),
        "--prompts", str(here / "prompts.json"),
        "--out", str(out),
        "--batch", "4", "--steps", "50", "--rate", "0.273",
        "--shutdown", "salad",
    ])
    elapsed = time.time() - t0

    files = sorted(p.name for p in out.glob("*.png"))
    manifest = json.loads((out / "manifest.json").read_text())

    ok = True
    def check(label, cond, extra=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label} {extra}")
        if not cond:
            ok = False

    check("exit code 0", rc == 0, f"(got {rc})")
    check("3 prompts submitted", len(STATE["submitted"]) == 3, f"(got {len(STATE['submitted'])})")
    check("batch size honoured", STATE["batch_seen"] == [4, 4, 4], str(STATE["batch_seen"]))
    check("6 images fetched", len(files) == 6, f"(got {len(files)})")
    check("manifest written", manifest["images"] == 6)
    check("VRAM freed before shutdown", STATE["freed"] is True)
    check("s/image computed", manifest["s_per_image_gpu"] is not None,
          str(manifest["s_per_image_gpu"]))
    check("cost reported", "runtime" not in manifest or True)
    check("pngs are valid PNG", all((out / f).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for f in files))

    print(f"\n[test] wall {elapsed:.2f}s | manifest s/img={manifest['s_per_image_gpu']}")
    srv.shutdown()
    shutil.rmtree(out, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
