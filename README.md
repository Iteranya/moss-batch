# moss-batch

Fire-and-forget batch image generation for **MOSS-Image-SFT (int8)** on a single
RTX 4090-class GPU. Built for **SaladCloud** (per-second billing, cold start not
billed), also runs on Runpod.

You give it a list of prompts. It boots ComfyUI, generates everything at 50
steps / CFG 3 / shift 5 / SLG layer 22, writes PNGs + a manifest, then exits so
billing stops.

```
prompts.json ──► start.sh ──► ComfyUI (:8188, headless)
                   │              ▲
                   └─► moss_batch.py ── /prompt, /history, /view
                          │
                          └─► out/*.png + out/manifest.json
```

## Why not pure Diffusers?

OpenMOSS also ships a Diffusers pipeline (`inference_runtime.py`), which is
cleaner to wrap in your own API — but on a 24 GB card it is the worse option:

| | ComfyUI + int8 (this repo) | Pure Diffusers |
|---|---|---|
| Weights | 15.4 GB int8-convrot | ~26 GB bf16 (no int8 variant exists) |
| Fits 24 GB | yes | only with CPU offload (`offload="auto"` → `"model"` below 40 GB) |
| Native batch | `batch_size` widget | `num_images_per_prompt` |
| `torch.compile` | ✗ | ✓ block-level, `fullgraph=True` |

The Diffusers runtime auto-selects CPU offload unless VRAM ≥ 40 GB. On a 4090
that means paying PCIe transfer per component. int8 on 24 GB is the better deal,
so this repo drives ComfyUI and the wrapper is a *client*.

## Build

```bash
docker login
docker build --platform linux/amd64 -t <dockerhub-user>/moss-batch:latest .
docker push <dockerhub-user>/moss-batch:latest
```

`--platform linux/amd64` is mandatory — Salad does not support ARM64 images.

The image bakes in ~15.4 GB of weights at build time and asserts byte counts, so
a truncated download fails the build rather than the batch.

## Salad container group

| Setting | Value | Why |
|---|---|---|
| Image | `<dockerhub-user>/moss-batch:latest` | |
| GPU class | RTX 4090 | int8 fits 24 GB |
| vCPU | **8** | VAE decode + PNG encode must keep up with the GPU |
| RAM | **32 GB** | 15.4 GB of mmapped weights count against the cgroup; 16 GB risks `Exited:137` |
| Disk | **40 GB** | ~15.4 GB weights + image layers + PNGs |
| Shared memory | **8 GB** | ComfyUI + torch multiprocessing; 64 MB default causes silent hangs |
| Replicas | **1** | a batch queue, not a service |
| Priority | **Medium** | Low/Lowest are preemptible (30 min min-run). Use Lowest for unattended overnight runs |
| Autoscaling | **off** | your quota counts *maximum* replicas, not current |
| Container Gateway | **off** | nothing needs to reach it from outside |
| Startup probe | HTTP `GET /system_stats` on 8188, `initialDelaySeconds: 150` | weights load lazily on the first prompt; a tight probe kills healthy boots |

GPU rates include vCPU and RAM on Salad, so raising those numbers costs nothing
except a smaller pool of eligible machines.

### Environment variables

| Var | Default | Notes |
|---|---|---|
| `BATCH_SIZE` | 4 | **Start at 4.** Batch 1 peaked at 14.82 GiB on a 5090; batch 8 will exceed 24 GB |
| `STEPS` | 50 | |
| `WIDTH` / `HEIGHT` | 1024 | multiples of 16, total 0.25M–4M pixels |
| `SAGE` | 0 | `1` enables `--use-sage-attention` **if** the import works. Unverified for MOSS — see below |
| `PROMPTS` | `/app/prompts.json` | |
| `OUT_DIR` | `/app/out` | |
| `RATE` | 0 | USD/hr, for cost reporting in the manifest |
| `SHUTDOWN` | none | `salad` / `runpod` calls their API after the batch |
| `SALAD_ORG` / `SALAD_PROJECT` / `SALAD_GROUP` / `SALAD_API_KEY` | — | needed for `SHUTDOWN=salad` |

On Salad you usually **don't need `SHUTDOWN=salad`**: the instance stops billing
when the container exits, and `start.sh` exits when the batch finishes. The hook
is there for Runpod, where a pod keeps billing after the process ends.

## prompts.json

Defaults apply to every job; per-prompt objects override them.

```json
{
  "steps": 50, "cfg": 3.0, "width": 1024, "height": 1024,
  "batch": 4, "seed": -1, "negative": "",
  "prompts": [
    "a rooftop at golden hour, cinematic lighting",
    {"prompt": "a rainy train platform at night", "seed": 1234, "batch": 8}
  ]
}
```

`seed: -1` means random per job; the resolved seed is recorded in
`manifest.json` so every image is reproducible.

## manifest.json

```json
{
  "wall_seconds": 152.3,
  "generation_seconds": 148.9,
  "images": 20,
  "s_per_image_gpu": 7.45,
  "s_per_image_wall": 7.62,
  "batch_size": 4,
  "steps": 50,
  "resolution": "1024x1024",
  "jobs": [{ "prompt": "...", "seed": 1401653621, "images": ["..."] }]
}
```

`s_per_image_gpu` is the number that matters — it is real generation time from
submit to history-complete, excluding boot. Compare it across runs to settle the
optimization questions that are still open.

## Test without a GPU

```bash
python test_mock.py
```

Spins up a mock ComfyUI (`/system_stats`, `/prompt`, `/history`, `/view`, `/free`)
and drives the real client through it. Verifies batch size reaches the graph,
images are fetched and written, the manifest is produced, and VRAM is freed
before shutdown.

## Still unproven

These are the assumptions that only a real run can settle:

- **SageAttention on MOSS.** ComfyUI's `attention_sage` falls back to
  `attention_pytorch` on error, so `SAGE=1` is safe but may be a no-op. MOSS uses
  joint-stream self-attention with RoPE already applied in a custom `_attention()`
  wrapper, which is not what sage is typically used for. **A/B it and compare
  `s_per_image_gpu`.**
- **Batch 8 on 24 GB.** Batch 1 peaked at 14.82 GiB on a 5090; activations at
  batch 8 likely exceed 24 GB. Batch 4 is the safe default.
- **√b throughput scaling.** Real gain from batching is unknown until measured.
- **Baseline speed.** The only measured MOSS number found is 23.79 s/image at 50
  steps / 1024² on a 4090. Your pod will tell you whether that holds.
