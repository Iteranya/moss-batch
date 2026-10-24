# MOSS-Image-SFT (int8) batch runner — ComfyUI + official plugin, weights baked in.
#
# Weights are fetched at BUILD time so the container never downloads at runtime.
# On Salad that matters for two reasons:
#   1. `downloading`/`allocating` states are not billed, so a fat image is free;
#   2. a build failure is loud, instead of a 12.5 GB download failing mid-batch.
#
# Build (from the project root):
#   docker build --platform linux/amd64 -t <your-dockerhub-user>/moss-batch:latest .
#   docker push <your-dockerhub-user>/moss-batch:latest
#
# Target GPU: RTX 4090 (24 GB), Salad / Runpod. CUDA 12.8 base so the same image
# also runs on 5090-class nodes (Salad requires CUDA >= 12.8 for 5090).

FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ARG COMFYUI_REF=bef11fcbfa8d772678e2ad00b1294c63eb3e5da7
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------- ComfyUI ---
# Pinned by commit, not branch: an unpinned clone makes builds non-reproducible
# and can silently break the MOSS loaders between runs.
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI \
    && cd /opt/ComfyUI \
    && git checkout ${COMFYUI_REF} \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
    && pip install -r requirements.txt

# --------------------------------------------------- MOSS-Image plugin ------
# Official OpenMOSS plugin: provides MOSSImageUNETLoader / MOSSImageCLIPLoader.
RUN cd /tmp \
    && git clone --depth 1 https://github.com/OpenMOSS/MOSS-Image.git \
    && mkdir -p /opt/ComfyUI/custom_nodes \
    && cp -r /tmp/MOSS-Image/comfyui/moss-image /opt/ComfyUI/custom_nodes/moss-image \
    && rm -rf /tmp/MOSS-Image

# ------------------------------------------------------------- weights ------
# int8-convrot pair + small VAE. ~15.4 GB total (verified against the HF file
# listing: 5.41 + 9.75 + 0.249 GB).
#   moss_image_sft_int8_convrot  5.41 GB  — DiT, int8 with a rotation fix
#   qwen3.5_9b_int8_convrot      9.75 GB  — frozen text encoder
#   flux2-vae_small              0.25 GB  — VAE
# int8 (not bf16) because the bf16 pair is 30.9 GB and would not fit a 24 GB card.
RUN mkdir -p /opt/ComfyUI/models/diffusion_models \
             /opt/ComfyUI/models/text_encoders \
             /opt/ComfyUI/models/vae

# Download each weight in its OWN layer, deliberately.
# Registries cap a single layer at ~10 GB (GHCR enforces 10 GB, Docker Hub is
# similar). Fetching all three in one RUN produced a single 15.4 GB layer, which
# builds fine locally and then FAILS AT PUSH. Three RUNs => 5.41 / 9.75 / 0.25 GB
# layers, every one of them under the cap.
RUN set -eux; \
    f=diffusion_models/moss_image_sft_int8_convrot.safetensors; \
    curl -fL --retry 5 --retry-delay 5 --retry-all-errors \
      -o "/opt/ComfyUI/models/$f" \
      "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/$f"; \
    test -s "/opt/ComfyUI/models/$f"

RUN set -eux; \
    f=text_encoders/qwen3.5_9b_int8_convrot.safetensors; \
    curl -fL --retry 5 --retry-delay 5 --retry-all-errors \
      -o "/opt/ComfyUI/models/$f" \
      "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/$f"; \
    test -s "/opt/ComfyUI/models/$f"

RUN set -eux; \
    f=vae/flux2-vae_small.safetensors; \
    curl -fL --retry 5 --retry-delay 5 --retry-all-errors \
      -o "/opt/ComfyUI/models/$f" \
      "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/$f"; \
    test -s "/opt/ComfyUI/models/$f"

RUN ls -lh /opt/ComfyUI/models/diffusion_models /opt/ComfyUI/models/text_encoders /opt/ComfyUI/models/vae

# Sanity check: sizes are asserted so a truncated download fails the BUILD,
# not the batch. Byte counts come from the HF API (X-Linked-Size).
RUN python - <<'PY'
import os
expect = {
  "/opt/ComfyUI/models/diffusion_models/moss_image_sft_int8_convrot.safetensors": 5_413_698_958,
  "/opt/ComfyUI/models/text_encoders/qwen3.5_9b_int8_convrot.safetensors": 9_748_121_532,
  "/opt/ComfyUI/models/vae/flux2-vae_small.safetensors": 249_521_340,
}
for path, minimum in expect.items():
    size = os.path.getsize(path)
    assert size == minimum, f"{path} is {size} bytes, expected {minimum} (truncated download?)"
    print(f"ok {os.path.basename(path)} {size/2**30:.2f} GiB")
PY

# Optional: sageattention for the A/B test. It is NOT proven to speed up MOSS's
# joint-stream self-attention, so it stays unstaged-by-default; start.sh only
# enables it when SAGE=1 and the import succeeds.
RUN pip install sageattention || echo "sageattention build skipped (optional)"

# ----------------------------------------------------------------- app ------
WORKDIR /app
COPY workflow.json prompts.json moss_batch.py start.sh /app/
RUN chmod +x /app/start.sh

EXPOSE 8188
ENTRYPOINT ["/app/start.sh"]
