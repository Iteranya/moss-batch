#!/usr/bin/env python3
"""Offline test: Playground layout -> prompts.json -> patched ComfyUI workflow.

No GPU, no network, no ComfyUI. Proves the contract holds:
  * the layout compiles to compact JSON (matches JS JSON.stringify)
  * bbox stays in [y1,x1,y2,x2] order
  * load_jobs accepts a layout-only job (no "prompt" key)
  * apply_job writes the layout string into CLIPTextEncode and the matching
    size into the latent node
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PLAYGROUND = REPO.parent / "moss-image-playground"

sys.path.insert(0, str(REPO))
import moss_batch  # noqa: E402

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        failures.append(name)


def load_layout(name: str) -> dict:
    path = PLAYGROUND / name
    if not path.exists():
        raise SystemExit(f"fixture missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    print("== fixture ==")
    layout = load_layout("ddlc-beach.json")
    n_el = len(layout.get("elements") or [])
    check("ddlc-beach.json parsed", n_el == 5, f"{n_el} elements")

    # 1776x1168 => half extents 888 / 584, which is what the file is drawn at
    W, H = 1776, 1168
    check("layout fits 1776x1168",
          not moss_batch.check_layout(layout, W, H),
          str(moss_batch.check_layout(layout, W, H)))

    print("== sizing guard ==")
    wrong = moss_batch.check_layout(layout, 1024, 1024)
    check("wrong size is rejected", bool(wrong), "expected warnings at 1024x1024")

    print("== compile ==")
    text = moss_batch.compile_layout(layout)
    check("compiles to a string", isinstance(text, str))
    check("no pretty-print whitespace", ", \"" not in text and "\": " not in text,
          "found JS-incompatible spacing")
    check("starts with {", text.startswith("{"))
    check("keeps bbox_coord_space",
          json.loads(text).get("bbox_coord_space") == "abs_centralized")

    # bbox order must survive untouched: [y1, x1, y2, x2]
    src0 = layout["elements"][0]["bbox"]
    out0 = json.loads(text)["elements"][0]["bbox"]
    check("bbox order preserved [y1,x1,y2,x2]", src0 == out0, f"{src0} != {out0}")
    check("bbox[0] is the y1 value (|584| <= H/2)",
          abs(out0[0]) <= H / 2 + 1, f"bbox[0]={out0[0]}")

    print("== converter CLI ==")
    out_file = REPO / "prompts.generated.json"
    proc = subprocess.run(
        [sys.executable, str(REPO / "playground2batch.py"),
         "--layout", str(PLAYGROUND / "ddlc-beach.json"),
         "--size", f"{W}x{H}", "--batch", "2", "--seed", "1337",
         "-o", str(out_file)],
        capture_output=True, text=True)
    check("converter exits 0", proc.returncode == 0, proc.stderr.strip()[:300])
    doc = json.loads(out_file.read_text(encoding="utf-8"))
    check("one job emitted", len(doc["prompts"]) == 1)
    check("job carries layout", "layout" in doc["prompts"][0])
    check("job has no 'prompt' key", "prompt" not in doc["prompts"][0])
    check("job size matches canvas",
          (doc["prompts"][0]["width"], doc["prompts"][0]["height"]) == (W, H))

    print("== --check mode ==")
    chk = subprocess.run(
        [sys.executable, str(REPO / "playground2batch.py"), "--check", str(out_file)],
        capture_output=True, text=True)
    check("--check exits 0 on a good file", chk.returncode == 0,
          chk.stdout + chk.stderr)

    print("== load_jobs ==")
    args = argparse.Namespace(negative=None, steps=None, cfg=None, width=None,
                              height=None, batch=None, seed=None)
    jobs = moss_batch.load_jobs(out_file, args)
    check("load_jobs returns 1 job", len(jobs) == 1)
    job = jobs[0]
    check("job.prompt is the compiled layout",
          job.prompt == moss_batch.compile_layout(doc["prompts"][0]["layout"]))
    check("job.batch from CLI/default", job.batch == 2, str(job.batch))
    check("job.seed", job.seed == 1337, str(job.seed))
    check("job size", (job.width, job.height) == (W, H))

    print("== apply_job on the real workflow ==")
    wf = moss_batch.load_workflow(REPO / "workflow.json")
    g = moss_batch.apply_job(wf, job)
    positive = moss_batch.node_by_title_prefix(g, "positive")["widgets_values"][0]
    latent = moss_batch.node_by_title_prefix(g, "latent")["widgets_values"]
    ks = moss_batch.node_by_title_prefix(g, "KSampler")["widgets_values"]
    check("positive prompt IS the layout JSON", positive == job.prompt,
          positive[:80])
    check("latent size applied", latent[:2] == [W, H], str(latent[:2]))
    check("latent batch applied", latent[2] == 2, str(latent[2]))
    check("ksampler seed applied", ks[0] == 1337, str(ks[0]))
    check("saveimage tagged",
          moss_batch.node_by_title_prefix(g, "SaveImage")["widgets_values"][0]
          == "moss/ddlc-beach",
          moss_batch.node_by_title_prefix(g, "SaveImage")["widgets_values"][0])

    print("== plain-text job still works ==")
    plain = REPO / "prompts.plain.json"
    plain.write_text(json.dumps({"prompts": ["a rooftop at golden hour"]}),
                     encoding="utf-8")
    pj = moss_batch.load_jobs(plain, args)
    check("plain string job loads", pj[0].prompt == "a rooftop at golden hour")

    print("== missing prompt+layout is fatal ==")
    bad = REPO / "prompts.bad.json"
    bad.write_text(json.dumps({"prompts": [{"width": 1024}]}), encoding="utf-8")
    try:
        moss_batch.load_jobs(bad, args)
        check("rejects job with neither prompt nor layout", False, "no exception")
    except SystemExit:
        check("rejects job with neither prompt nor layout", True)

    for f in (out_file, plain, bad):
        f.unlink(missing_ok=True)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
