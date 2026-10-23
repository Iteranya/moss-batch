#!/usr/bin/env python3
"""
moss_batch.py — fire-and-forget batch client for ComfyUI running MOSS-Image-SFT (int8).

Reads prompts.json, submits batches of N images to ComfyUI's /prompt API,
polls history, writes PNGs + a manifest, then optionally kills the instance.

Usage:
    python moss_batch.py --prompts prompts.json --out ./out
    python moss_batch.py --prompts prompts.json --out ./out --batch-size 4 --bench
    python moss_batch.py ... --shutdown salad      # or --shutdown none

Exit codes: 0 ok, 1 fatal, 2 some prompts failed (partial success).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ----------------------------------------------------------------------------
# ComfyUI API client
# ----------------------------------------------------------------------------

class ComfyError(RuntimeError):
    pass


class Comfy:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188, timeout: int = 30):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    # -- low level ---------------------------------------------------------
    def _req(self, method: str, path: str, payload=None, timeout=None):
        url = f"{self.base}{path}"
        data = None
        headers: dict = {}
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                body = r.read()
                return json.loads(body) if body else None
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:2000]
            raise ComfyError(f"{method} {path} -> HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise ComfyError(f"{method} {path} -> {e.reason}") from e

    # -- api ---------------------------------------------------------------
    def wait_ready(self, deadline_s: int = 1800, interval_s: int = 3) -> dict:
        """Poll /system_stats until ComfyUI answers. Model load happens on first
        prompt, not here, so this only proves the server itself is up."""
        start = time.time()
        last = None
        while time.time() - start < deadline_s:
            try:
                stats = self._req("GET", "/system_stats", timeout=10)
                dev = (stats or {}).get("devices") or []
                if dev:
                    d = dev[0]
                    print(f"[boot] ready after {time.time()-start:.0f}s | "
                          f"{d.get('name')} | vram_total={d.get('vram_total', 0)/2**30:.1f}GiB | "
                          f"torch={stats.get('system', {}).get('pytorch_version')}")
                else:
                    print(f"[boot] ready after {time.time()-start:.0f}s")
                return stats if isinstance(stats, dict) else {}
            except ComfyError as e:
                last = e
                time.sleep(interval_s)
        raise ComfyError(f"ComfyUI not ready after {deadline_s}s (last: {last})")

    def submit(self, prompt_graph: dict) -> str:
        res = self._req("POST", "/prompt", {"prompt": prompt_graph, "client_id": self.client_id})
        if not res or "prompt_id" not in res:
            raise ComfyError(f"unexpected /prompt response: {res}")
        if res.get("node_errors"):
            raise ComfyError(f"workflow validation failed: {json.dumps(res['node_errors'])[:1500]}")
        return res["prompt_id"]

    def history(self, prompt_id: str):
        return self._req("GET", f"/history/{prompt_id}")

    def wait_done(self, prompt_id: str, deadline_s: int = 7200, interval_s: float = 1.0) -> dict:
        start = time.time()
        while time.time() - start < deadline_s:
            h = self.history(prompt_id)
            if h and prompt_id in h:
                entry = h[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error" or status.get("completed") is False:
                    msgs = status.get("messages", [])
                    raise ComfyError(f"generation failed: {json.dumps(msgs)[:1500]}")
                return entry
            time.sleep(interval_s)
        raise ComfyError(f"prompt {prompt_id} timed out after {deadline_s}s")

    def image_bytes(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        q = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        with urllib.request.urlopen(f"{self.base}/view?{q}", timeout=120) as r:
            return r.read()

    def interrupt(self):
        try:
            self._req("POST", "/interrupt")
        except ComfyError as e:
            print(f"[warn] interrupt failed: {e}", file=sys.stderr)

    def free(self):
        """Unload models + free VRAM. Cheap insurance before shutdown."""
        try:
            self._req("POST", "/free", {"unload_models": True, "free_memory": True}, timeout=60)
        except ComfyError as e:
            print(f"[warn] /free failed: {e}", file=sys.stderr)


# ----------------------------------------------------------------------------
# Workflow manipulation
# ----------------------------------------------------------------------------

def load_workflow(path: Path) -> dict:
    wf = json.loads(path.read_text(encoding="utf-8"))
    if "nodes" not in wf:
        raise SystemExit(f"{path} is not a UI workflow graph (no 'nodes' key). "
                         "Export it with 'Save (API format)' instead if you have an API dump.")
    return wf


def node_by_title_prefix(wf: dict, prefix: str) -> dict:
    for n in wf["nodes"]:
        if str(n.get("title", "")).startswith(prefix):
            return n
    raise SystemExit(f"workflow has no node titled like {prefix!r}")


def set_batch_size(wf: dict, batch: int):
    n = node_by_title_prefix(wf, "latent")
    n["widgets_values"][2] = int(batch)


def apply_job(wf: dict, job: "Job"):
    """Mutate a copy of the workflow for one job."""
    g = json.loads(json.dumps(wf))
    set_batch_size(g, job.batch)

    w, h = job.width, job.height
    if w % 16 or h % 16:
        raise SystemExit(f"width/height must be multiples of 16 (got {w}x{h})")
    if not (0.25e6 <= w * h <= 4.0e6):
        raise SystemExit(f"{w}x{h} is outside MOSS-Image's 0.25M-4M pixel range")

    positive = node_by_title_prefix(g, "positive")["widgets_values"]
    positive[0] = job.prompt
    node_by_title_prefix(g, "negative")["widgets_values"][0] = job.negative

    ks = node_by_title_prefix(g, "KSampler")["widgets_values"]
    ks[0] = job.seed
    ks[2] = job.steps
    ks[3] = job.cfg

    node_by_title_prefix(g, "latent")["widgets_values"][:3] = [w, h, job.batch]
    node_by_title_prefix(g, "SLG")["widgets_values"][1] = str(job.slg_layer)
    node_by_title_prefix(g, "shift")["widgets_values"][0] = job.shift
    node_by_title_prefix(g, "SaveImage")["widgets_values"][0] = f"moss/{job.safe_tag}"
    return g


# ----------------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------------

@dataclass
class Job:
    index: int
    prompt: str
    negative: str = ""
    steps: int = 50
    cfg: float = 3.0
    width: int = 1024
    height: int = 1024
    batch: int = 4
    seed: int = -1
    shift: float = 5.0
    slg_layer: int = 22
    tag: str = ""
    # filled during run
    prompt_id: str = ""
    elapsed: float = 0.0
    images: list = field(default_factory=list)
    error: str = ""

    @property
    def safe_tag(self) -> str:
        base = self.tag or f"{self.index:04d}"
        keep = "".join(c if (c.isalnum() or c in "-_") else "-" for c in base)
        return keep[:60].strip("-") or f"{self.index:04d}"


def load_jobs(path: Path, args) -> list[Job]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        defaults = {k: raw[k] for k in
                    ("negative", "steps", "cfg", "width", "height", "batch", "seed",
                     "shift", "slg_layer") if k in raw}
        prompts = raw.get("prompts") or []
    else:
        defaults, prompts = {}, raw

    if not prompts:
        raise SystemExit(f"{path}: no prompts found")

    # CLI overrides win over file defaults
    for k in ("negative", "steps", "cfg", "width", "height", "batch", "seed"):
        v = getattr(args, k, None)
        if v is not None:
            defaults[k] = v

    jobs = []
    for i, p in enumerate(prompts):
        if isinstance(p, str):
            jobs.append(Job(index=i, prompt=p, **defaults))
        elif isinstance(p, dict):
            merged = {**defaults, **p}
            jobs.append(Job(index=i, prompt=merged.pop("prompt"), **merged))
        else:
            raise SystemExit(f"{path}: prompt {i} is neither a string nor an object")
    return jobs


# ----------------------------------------------------------------------------
# Shutdown hooks
# ----------------------------------------------------------------------------

def shutdown_salad():
    """Salad: stop the container group so billing ends.

    Billing is per-second only while `running`, so stopping here is what makes
    a 150-second job cost cents instead of an hour. Requires the deploy script
    to have exported SALAD_* env vars (see deploy_salad.sh).
    """
    org = os.environ.get("SALAD_ORG")
    project = os.environ.get("SALAD_PROJECT")
    group = os.environ.get("SALAD_GROUP")
    key = os.environ.get("SALAD_API_KEY")
    if not all([org, project, group, key]):
        print("[shutdown] salad: missing SALAD_ORG/SALAD_PROJECT/SALAD_GROUP/SALAD_API_KEY — "
              "leaving container group RUNNING (stop it manually or you keep paying)", file=sys.stderr)
        return
    url = f"https://api.salad.com/api/public/organizations/{org}/projects/{project}/containers/{group}/stop"
    req = urllib.request.Request(url, data=b"", method="POST", headers={"Salad-Api-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"[shutdown] salad stop -> HTTP {r.status}")
    except Exception as e:  # noqa: BLE001
        print(f"[shutdown] salad stop FAILED: {e} — STOP THE GROUP MANUALLY", file=sys.stderr)


def shutdown_runpod():
    """Runpod pods are not Salad: they are stopped via the REST API / console."""
    pod = os.environ.get("RUNPOD_POD_ID")
    key = os.environ.get("RUNPOD_API_KEY")
    if not (pod and key):
        print("[shutdown] runpod: missing RUNPOD_POD_ID/RUNPOD_API_KEY — stop the pod manually",
              file=sys.stderr)
        return
    print(f"[shutdown] runpod: stopping pod {pod}")
    try:
        from urllib.parse import quote
        gql = ('mutation { podStop(input: {podId: "%s"}) { id desiredStatus } }' % pod)
        payload = json.dumps({"query": gql}).encode()
        req = urllib.request.Request(
            "https://api.runpod.io/graphql",
            data=payload, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {quote(key)}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"[shutdown] runpod -> {r.read()[:200]!r}")
    except Exception as e:  # noqa: BLE001
        print(f"[shutdown] runpod stop FAILED: {e} — STOP THE POD MANUALLY", file=sys.stderr)


SHUTDOWNS = {"none": lambda: None, "salad": shutdown_salad, "runpod": shutdown_runpod}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run(args) -> int:
    wf = load_workflow(Path(args.workflow))
    jobs = load_jobs(Path(args.prompts), args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    comfy = Comfy(args.host, args.port)
    print(f"[init] {len(jobs)} job(s) -> {out.resolve()}")
    comfy.wait_ready(deadline_s=args.boot_timeout)

    t_start = time.time()
    t_fetch = 0.0
    failures = 0

    for job in jobs:
        tag = job.safe_tag
        if job.seed < 0:
            job.seed = random.randint(0, 2**31 - 1)
        try:
            graph = apply_job(wf, job)
            t0 = time.time()
            job.prompt_id = comfy.submit(graph)
            entry = comfy.wait_done(job.prompt_id, deadline_s=args.job_timeout)
            job.elapsed = time.time() - t0

            saved = 0
            for node_out in (entry.get("outputs") or {}).values():
                for img in node_out.get("images", []) or []:
                    tf = time.time()
                    blob = comfy.image_bytes(img.get("filename", ""), img.get("subfolder", ""),
                                             img.get("type", "output"))
                    t_fetch += time.time() - tf
                    dest = out / f"{tag}_{img.get('filename', f'img{saved}.png')}"
                    dest.write_bytes(blob)
                    job.images.append(dest.name)
                    saved += 1

            per_img = job.elapsed / max(job.batch, 1)
            print(f"[ok] {tag} seeds={job.seed} {job.batch} imgs in {job.elapsed:.1f}s "
                  f"({per_img:.2f} s/img) -> {saved} saved")
        except ComfyError as e:
            job.error = str(e)
            failures += 1
            print(f"[fail] {tag}: {e}", file=sys.stderr)

    wall = time.time() - t_start
    total_imgs = sum(len(j.images) for j in jobs)
    gen_time = sum(j.elapsed for j in jobs)
    manifest = {
        "started_unix": int(t_start),
        "wall_seconds": round(wall, 2),
        "generation_seconds": round(gen_time, 2),
        "image_fetch_seconds": round(t_fetch, 2),
        "images": total_imgs,
        "s_per_image_wall": round(wall / total_imgs, 3) if total_imgs else None,
        "s_per_image_gpu": round(gen_time / total_imgs, 3) if total_imgs else None,
        "batch_size": args.batch,
        "steps": jobs[0].steps,
        "resolution": f"{jobs[0].width}x{jobs[0].height}",
        "jobs": [asdict(j) for j in jobs],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"images          : {total_imgs} ({failures} job(s) failed)")
    print(f"wall            : {wall:.1f}s")
    print(f"generation      : {gen_time:.1f}s")
    print(f"s/image (gpu)   : {manifest['s_per_image_gpu']}")
    print(f"s/image (wall)  : {manifest['s_per_image_wall']}")
    if args.rate:
        hrs = wall / 3600
        print(f"cost @ ${args.rate:.3f}/hr : ${hrs * args.rate:.4f}")
        if total_imgs:
            print(f"cost per 1k imgs         : ${hrs * args.rate / total_imgs * 1000:.2f}")
    print("=" * 62)

    if args.bench:
        print("\n[bench] to A/B against --use-sage-attention, rebuild is NOT needed:")
        print("        set SAGE=0 then SAGE=1 in the container group env and rerun the same prompts.")
        print("        Compare s/image (gpu) above; sage errors fall back to pytorch silently.")

    comfy.free()
    SHUTDOWNS[args.shutdown]()
    return 0 if failures == 0 else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fire-and-forget MOSS-Image batch client for ComfyUI")
    ap.add_argument("--prompts", default="prompts.json")
    ap.add_argument("--workflow", default="workflow.json")
    ap.add_argument("--out", default="./out")
    ap.add_argument("--host", default=os.environ.get("COMFY_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("COMFY_PORT", "8188")))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("BATCH_SIZE", "4")))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--negative", default=None)
    ap.add_argument("--rate", type=float, default=None,
                    help="USD/hr of the instance, for cost reporting (e.g. 0.273)")
    ap.add_argument("--shutdown", choices=list(SHUTDOWNS), default="none")
    ap.add_argument("--bench", action="store_true", help="print A/B instructions")
    ap.add_argument("--boot-timeout", type=int, default=1800)
    ap.add_argument("--job-timeout", type=int, default=7200)
    args = ap.parse_args(argv)

    def _sigint(signum, frame):
        print("\n[abort] signal received — shutting down", file=sys.stderr)
        SHUTDOWNS[args.shutdown]()
        sys.exit(130)
    signal.signal(signal.SIGINT, _sigint)

    try:
        return run(args)
    except ComfyError as e:
        print(f"[fatal] {e}", file=sys.stderr)
        SHUTDOWNS[args.shutdown]()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
