#!/usr/bin/env python3
"""batch_server.py — local relay between the MOSS Image Playground and Salad.

The Playground is a static page; it cannot call Salad's API directly without
shipping a Salad key into the browser. This process sits on localhost, takes the
batch JSON the Playground builds, validates it with the same code the GPU worker
uses, and shells out to ``trigger.sh``.

    export SALAD_API_KEY=... SALAD_ORG=... SALAD_PROJECT=...
    python batch_server.py                    # http://127.0.0.1:8787
    python batch_server.py --port 9000 --playground ../moss-image-playground

Serving the Playground from the same origin also fixes ``file://`` → ``Origin:
null`` rejections and keeps localStorage on one origin.

Endpoints
---------
GET  /                      static files from --playground (if present)
GET  /api/health            {"ok":true,"trigger":"…"}
POST /api/batch             body = prompts.json doc -> {"job_id": "…"}
GET  /api/batch/<job_id>    {"state":"queued|running|done|failed","detail":"…","dir":"…"}
GET  /api/batch             list known jobs

``TRIGGER_CMD`` overrides the shell command (testing without a Salad account),
and ``BATCH_STUB=1`` makes the runner a no-op that just logs.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
sys.path.insert(0, str(HERE))

try:
    import moss_batch
except Exception as e:  # noqa: BLE001
    print(f"[server] cannot import moss_batch: {e}", file=sys.stderr)
    moss_batch = None

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _job_dir(job_id: str) -> Path:
    return RUNS / job_id


def _validate(doc: dict) -> list[str]:
    """Validate with the same loader the worker uses, then report per-job notes."""
    problems: list[str] = []
    if not isinstance(doc, dict):
        return ["body must be a JSON object"]
    prompts = doc.get("prompts") or doc.get("jobs")
    if not isinstance(prompts, list) or not prompts:
        return ["body needs a non-empty 'prompts' array"]
    if moss_batch is None:
        return ["moss_batch.py not importable on the server"]
    tmp = RUNS / f"_validate_{uuid.uuid4().hex[:8]}.json"
    RUNS.mkdir(parents=True, exist_ok=True)
    try:
        tmp.write_text(json.dumps(doc), encoding="utf-8")
        args = argparse.Namespace(negative=None, steps=None, cfg=None, width=None,
                                  height=None, batch=None, seed=None)
        jobs = moss_batch.load_jobs(tmp, args)
        if not jobs:
            problems.append("no jobs after parsing")
        for j in jobs:
            if j.width % 16 or j.height % 16:
                problems.append(f"{j.safe_tag}: size not a multiple of 16")
            px = j.width * j.height
            if not (0.25e6 <= px <= 4.0e6):
                problems.append(f"{j.safe_tag}: {px/1e6:.2f}Mpx outside 0.25M-4M")
            if not j.prompt or not j.prompt.strip():
                problems.append(f"{j.safe_tag}: empty prompt")
    except SystemExit as e:
        problems.append(str(e))
    except Exception as e:  # noqa: BLE001
        problems.append(f"{type(e).__name__}: {e}")
    finally:
        tmp.unlink(missing_ok=True)
    return problems


def _runner_cmd(job_dir: Path) -> list[str]:
    """What to execute for a job. Overridable for tests."""
    override = os.environ.get("TRIGGER_CMD")
    if override:
        return shlex.split(override, posix=True)
    return ["bash", "trigger.sh", str(job_dir / "prompts.json")]


def _watch(job_id: str, proc: subprocess.Popen, log_path: Path) -> None:
    rc = proc.wait()
    with LOCK:
        rec = JOBS.get(job_id)
        if rec:
            rec["state"] = "done" if rc == 0 else "failed"
            rec["detail"] = f"exit {rc}"
            rec["finished"] = time.time()
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n[server] runner exited rc={rc}\n")


def start_job(doc: dict) -> str:
    job_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    prompts = job_dir / "prompts.json"
    prompts.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    log_path = job_dir / "runner.log"
    env = dict(os.environ)
    env["OUT_DIR"] = str(job_dir / "results")
    env["JOB_ID"] = job_id

    with LOCK:
        JOBS[job_id] = {"state": "running", "dir": str(job_dir),
                        "started": time.time(), "detail": "", "log": str(log_path)}

    log = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(_runner_cmd(job_dir), cwd=str(HERE), env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    except Exception as e:  # noqa: BLE001
        log.write(f"[server] failed to spawn runner: {e}\n")
        log.close()
        with LOCK:
            JOBS[job_id]["state"] = "failed"
            JOBS[job_id]["detail"] = f"spawn failed: {e}"
        return job_id

    threading.Thread(target=_watch, args=(job_id, proc, log_path), daemon=True).start()
    return job_id


def _tail(path: str, n: int = 12) -> str:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:  # noqa: BLE001
        return ""


class Handler(SimpleHTTPRequestHandler):
    server_version = "moss-batch-relay/1.0"

    def log_message(self, format, *args):  # noqa: A002 - signature must match base
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), format % args))

    # -- helpers ---------------------------------------------------------
    def _cors(self):
        origin = self.headers.get("Origin", "")
        if origin.startswith("http://127.0.0.1") or origin.startswith("http://localhost"):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    # -- verbs -----------------------------------------------------------
    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json(200, {"ok": True,
                             "trigger": os.environ.get("TRIGGER_CMD") or "bash trigger.sh",
                             "moss_batch": moss_batch is not None})
            return
        if path == "/api/batch":
            with LOCK:
                jobs = {k: {kk: vv for kk, vv in v.items() if kk != "log"}
                        for k, v in JOBS.items()}
            self._json(200, {"jobs": jobs})
            return
        if path.startswith("/api/batch/"):
            job_id = unquote(path[len("/api/batch/"):])
            with LOCK:
                rec = JOBS.get(job_id)
                if rec:
                    rec = dict(rec)
            if not rec:
                self._json(404, {"error": "unknown job_id"})
                return
            self._json(200, {"job_id": job_id, "state": rec["state"],
                             "detail": rec.get("detail", ""),
                             "dir": rec.get("dir", ""),
                             "tail": _tail(rec.get("log", ""))})
            return
        if path.startswith("/api/"):
            self._json(404, {"error": "no such endpoint"})
            return
        super().do_GET()

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/batch":
            self._json(404, {"error": "no such endpoint"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 32 * 1024 * 1024:
            self._json(400, {"error": "missing or oversized body"})
            return
        raw = self.rfile.read(length)
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as e:
            self._json(400, {"error": f"invalid JSON: {e}"})
            return
        problems = _validate(doc)
        if problems:
            self._json(400, {"error": "validation failed", "problems": problems[:20]})
            return
        job_id = start_job(doc)
        self._json(202, {"job_id": job_id, "state": "running"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--playground", default=str(HERE.parent / "moss-image-playground"),
                    help="directory to serve at / (the Playground checkout)")
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    pg = Path(args.playground)
    if pg.is_dir():
        os.chdir(pg)
        print(f"[server] serving {pg} at http://{args.host}:{args.port}/")
    else:
        print(f"[server] {pg} not found — API only, no static files", file=sys.stderr)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[server] POST /api/batch  GET /api/batch/<job_id>")
    print(f"[server] trigger: {os.environ.get('TRIGGER_CMD') or 'bash trigger.sh <prompts.json>'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
