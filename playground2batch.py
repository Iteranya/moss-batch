#!/usr/bin/env python3
"""playground2batch.py — bridge MOSS Image Playground layouts into a batch run.

The Playground (Iteranya/moss-image-playground) saves a *layout*: a scene brief
plus per-box elements with bboxes in canvas coordinates. MOSS-Image treats that
layout JSON as the prompt text, which is why the Playground posts
``JSON.stringify(toLayout())`` straight into the ``prompt`` field.

This script turns one or more saved layouts into a ``prompts.json`` that
``moss_batch.py`` understands, and validates the pairing between a layout's
coordinate space and the render size — the single most common way to get a
batch of badly-cropped images.

Usage
-----
    # one layout -> one batch job
    python playground2batch.py --layout ddlc-beach.json --size 1776x1168 -o prompts.json

    # several layouts, four images each, seeded
    python playground2batch.py --layout a.json --layout b.json \
        --size 1776x1168 --batch 4 --seed 1337 -o prompts.json

    # validate a prompts.json you assembled by hand (or exported)
    python playground2batch.py --check prompts.json

    # normalise a Playground "export batch" file into moss-batch format
    python playground2batch.py --from-export export.json -o prompts.json

Rules enforced
--------------
* ``bbox`` is ``[y1, x1, y2, x2]`` — y first. Never reorder it to xyxy.
* coordinates are centred on (0,0), spanning +-half the canvas, so every box
  must satisfy |x| <= width/2 and |y| <= height/2.
* width/height must be multiples of 16 and 0.25M..4M pixels (MOSS-Image limits).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from moss_batch import compile_layout, check_layout
except ImportError:  # running from another directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from moss_batch import compile_layout, check_layout

MIN_PIXELS, MAX_PIXELS = 0.25e6, 4.0e6


def parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().replace("*", "x").split("x"))
    except Exception:  # noqa: BLE001
        raise SystemExit(f"bad --size {text!r}, expected WxH like 1776x1168")
    for name, v in (("width", w), ("height", h)):
        if v % 16:
            raise SystemExit(f"{name}={v} is not a multiple of 16 (required by MOSS-Image)")
    if not (MIN_PIXELS <= w * h <= MAX_PIXELS):
        raise SystemExit(
            f"{w}x{h} = {w * h / 1e6:.2f}Mpx is outside MOSS-Image's 0.25M-4M range")
    return w, h


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path}: not valid JSON ({e})")
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object, got {type(data).__name__}")
    return data


def layout_job(layout: dict, width: int, height: int, args) -> dict:
    problems = check_layout(layout, width, height)
    for p in problems:
        print(f"[warn] {args.tag or '<layout>'}: {p}", file=sys.stderr)
    if not layout.get("elements"):
        print(f"[warn] {args.tag or '<layout>'}: no elements, this is a plain "
              f"text prompt", file=sys.stderr)
    job: dict = {"layout": layout, "width": width, "height": height}
    if args.batch is not None:
        job["batch"] = args.batch
    if args.steps is not None:
        job["steps"] = args.steps
    if args.cfg is not None:
        job["cfg"] = args.cfg
    if args.seed is not None:
        job["seed"] = args.seed
    if args.negative:
        job["negative"] = args.negative
    return job


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", action="append", default=[], metavar="FILE",
                    help="Playground layout JSON (repeatable)")
    ap.add_argument("--size", default="1776x1168",
                    help="canvas the layout was drawn at, WxH (default 1776x1168)")
    ap.add_argument("--from-export", metavar="FILE",
                    help="read a Playground 'export batch' file instead of --layout")
    ap.add_argument("--check", metavar="FILE",
                    help="validate an existing prompts.json and exit")
    ap.add_argument("-o", "--out", default="prompts.json")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--negative", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    # ---- validation-only mode -------------------------------------------
    if args.check:
        doc = read_json(Path(args.check))
        jobs = doc.get("prompts") or doc.get("jobs") or []
        if not jobs:
            print("no prompts/jobs found", file=sys.stderr)
            return 1
        bad = 0
        for i, job in enumerate(jobs):
            if not isinstance(job, dict):
                print(f"job {i}: plain string prompt — no layout to check")
                continue
            layout = job.get("layout")
            w = int(job.get("width", doc.get("width", 1024)))
            h = int(job.get("height", doc.get("height", 1024)))
            tag = job.get("tag") or f"job{i}"
            if layout is None:
                if "prompt" in job:
                    n = len(job["prompt"])
                    print(f"{tag:28} text prompt ({n} chars) {w}x{h}")
                else:
                    print(f"{tag:28} MISSING both layout and prompt")
                    bad += 1
                continue
            if isinstance(layout, str):
                try:
                    layout = json.loads(layout)
                except json.JSONDecodeError:
                    print(f"{tag:28} layout is a string but not JSON")
                    bad += 1
                    continue
            problems = check_layout(layout, w, h)
            n_el = len(layout.get("elements") or [])
            status = "ok" if not problems else f"{len(problems)} problem(s)"
            print(f"{tag:28} {n_el} elements  {w}x{h}  {w * h / 1e6:.2f}Mpx  {status}")
            for p in problems:
                print(f"    - {p}")
                bad += 1
        print(f"\n{len(jobs)} job(s), {bad} problem(s)")
        return 1 if bad else 0

    # ---- export -> prompts.json ------------------------------------------
    if args.from_export:
        doc = read_json(Path(args.from_export))
        jobs = doc.get("prompts") or doc.get("jobs")
        if not jobs:
            raise SystemExit(f"{args.from_export}: no 'prompts' or 'jobs' array")
        out = {k: doc[k] for k in ("steps", "cfg", "width", "height", "batch",
                                   "seed", "negative") if k in doc}
        out["prompts"] = jobs
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"wrote {args.out}: {len(jobs)} job(s) from {args.from_export}")
        return 0

    # ---- layouts -> prompts.json -----------------------------------------
    if not args.layout:
        ap.error("give at least one --layout, or use --check / --from-export")

    w, h = parse_size(args.size)
    jobs = []
    for raw in args.layout:
        path = Path(raw)
        layout = read_json(path)
        job_args = argparse.Namespace(**vars(args))
        job_args.tag = args.tag or path.stem
        job = layout_job(layout, w, h, job_args)
        job["tag"] = job_args.tag
        jobs.append(job)

    doc = {"steps": args.steps, "cfg": args.cfg, "seed": args.seed,
           "negative": args.negative, "width": w, "height": h,
           "batch": args.batch, "prompts": jobs}
    Path(args.out).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(f"wrote {args.out}: {len(jobs)} job(s) at {w}x{h} ({w * h / 1e6:.2f}Mpx)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
