# Playground → moss-batch contract

How `moss-image-playground` layouts get rendered on a Salad GPU.

`moss-batch` does not re-implement MOSS's layout logic. It reproduces exactly
what the Playground already does against the hosted API: **the layout JSON is the
prompt string.**

```js
// playground index.html, buildPayload() — already shipped
p.prompt = JSON.stringify(toLayout());
```

`moss_batch.py` puts that same string into the workflow's `CLIPTextEncode` node.
The ComfyUI plugin only ships loaders (`MOSSImageUNETLoader`,
`MOSSImageCLIPLoader`, `MOSSImageHFDiLoader`, …) — **there is no layout node**,
and the official example workflow is a plain `CLIPTextEncode`. So: string in,
image out, same conditioning path as the API.

## Where the two halves meet

```
Playground  ──save layout──►  *.json
                                 │  playground2batch.py --layout a.json --size 1776x1168
                                 ▼
                            prompts.json  ──► trigger.sh ──► Salad S3S ──► GPU ──► S3S ──► results/
```

## Job schema accepted by `moss_batch.py`

Top level (all optional, act as defaults):

```json
{ "steps": 50, "cfg": 3.0, "width": 1776, "height": 1168,
  "batch": 1, "seed": -1, "negative": "", "prompts": [ ... ] }
```

Each entry may be **one of three shapes**:

```jsonc
// 1. plain text
"a rooftop at golden hour"

// 2. plain text with overrides
{ "prompt": "a rooftop at golden hour", "seed": 42, "batch": 4 }

// 3. Playground layout (no "prompt" key!)
{ "layout": { "scene_type": "...", "gen_brief": "...",
              "elements": [ { "type": "obj", "salience": "primary",
                              "identity": "...", "bbox": [-200,-120,200,120] } ],
              "bbox_coord_space": "abs_centralized" },
  "width": 1776, "height": 1168, "batch": 1, "tag": "ddlc-beach" }
```

`"jobs"` is accepted as an alias for `"prompts"`.

## Three rules that will silently ruin output if broken

### 1. `bbox` is `[y1, x1, y2, x2]` — y first

Straight from `toLayout()`:

```js
o.bbox = [y1,x1,y2,x2];   // yxyx, centred on the canvas
```

Do **not** normalise it to `[x1,y1,x2,y2]`. Every box rotates 90° and the model
gets a scene that looks nothing like the canvas. `compile_layout()` passes the
array through untouched on purpose, and the test asserts it.

### 2. Coordinates are coupled to the render size

`abs_centralized` means origin at canvas centre, so a valid box satisfies:

```
|x1|, |x2|  <=  width  / 2
|y1|, |y2|  <=  height / 2
```

`playground2batch.py` warns on every violation, because a layout drawn at
1776×1168 rendered at 1024×1024 produces characters cropped out of frame. Round
numbers on `ddlc-beach.json`: `±888` horizontally, `±584` vertically → 1776×1168.

Render sizes must also be multiples of 16 and stay within **0.25M–4M pixels**
(MOSS-Image's own limit, enforced again in `apply_job`).

### 3. The layout must serialise compactly

`JSON.stringify` emits no spaces. Python's `json.dumps` defaults to `", "` and
`": "`, which changes the token stream even though the JSON is semantically
equal. `compile_layout()` uses `separators=(",", ":")` and
`ensure_ascii=False` to match JavaScript byte-for-byte.

If your Playground export writes the layout as a *string*, that's fine too —
`moss_batch.py` accepts `"layout": "<json string>"` and passes it through.

## Suggested export button for the Playground

Paste next to the existing `buildPayload()`; no dependencies, matches the schema
above.

```js
function exportBatch(){
  const [W,H] = sizeWH();
  const job = {
    tag: (currentName || 'job').replace(/[^a-z0-9_-]/gi,'-').toLowerCase(),
    layout: toLayout(),
    width: W,
    height: H,
    batch: 1
  };
  const doc = {
    steps: parseInt($('steps').value,10) || 50,
    cfg: parseFloat($('cfg').value) || 3,
    negative: $('negative').value.trim(),
    seed: -1,
    width: W, height: H, batch: 1,
    prompts: [job]
  };
  const blob = new Blob([JSON.stringify(doc,null,2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = job.tag + '.batch.json';
  a.click();
  URL.revokeObjectURL(a.href);
}
```

Keep `batch: 1` per job when queuing 5–20 of them: 20 layouts × 4 images × 1024²
is ~80 PNGs, and the results tarball must stay under S3S's **100 MB** object cap
(the container splits at 90 MB, but fewer, bigger files are simpler).

Then merge exports:

```bash
python playground2batch.py --from-export myjob.batch.json -o prompts.json
python playground2batch.py --check prompts.json        # validate before spending GPU
```

## The relay (optional, but it's what "Send to GPU" calls)

`batch_server.py` is a stdlib-only localhost relay. It validates the batch with
the *same* `load_jobs` the GPU worker uses, writes `runs/<job_id>/prompts.json`,
and shells out to `trigger.sh`.

```bash
export SALAD_API_KEY=... SALAD_ORG=... SALAD_PROJECT=...
python batch_server.py                      # http://127.0.0.1:8787
```

| Method | Path | Does |
|---|---|---|
| GET | `/api/health` | `{"ok":true,"trigger":"bash trigger.sh"}` |
| POST | `/api/batch` | body = the doc above → `202 {"job_id":…}`, or `400` with `problems[]` |
| GET | `/api/batch/<job_id>` | `{state, detail, dir, tail}` — `tail` is the last 12 log lines |

It also serves the Playground at `/`, which fixes the `file://` → `Origin: null`
problem and keeps localStorage on one origin. Point the UI's **batch runner**
field at `http://127.0.0.1:8787`, hit **Send to GPU**, and the panel polls status
`running → done|failed` for you.

`TRIGGER_CMD="bash something_else.sh"` swaps the runner — that's how the relay is
tested without a Salad account.

## The Playground UI

`index.html` gained a **Batch queue** card: `+ add current layout` snapshots
`toLayout()` plus the current canvas size, the queue persists in `localStorage`
(`moss.batch`), each row validates its own box extents, and there are
**Export prompts.json** / **Copy JSON** / **Clear** buttons. Warnings are computed
client-side with the same rules as `check_layout()`, so a size mismatch shows up
before a single GPU-second is spent.

## Sanity check without a GPU

```bash
python tests/test_playground_bridge.py
```

24 assertions covering compile fidelity, bbox order, size coupling, `load_jobs`
acceptance, and the workflow mutation (`CLIPTextEncode` + latent + KSampler).
