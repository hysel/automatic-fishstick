#!/usr/bin/env python3
"""
Stable Diffusion Smart GPU Router v3.0
---------------------------------------
Routes SD generation requests to the optimal GPU instance based on:
  - Requested image resolution (high-res -> high-VRAM GPU)
  - Real-time free VRAM from nvidia-smi
  - Per-instance queue depth (active in-flight requests)
  - GPU capability (xformers, architecture, total VRAM)

Endpoints:
  POST /sdapi/v1/txt2img  -> smart-routed to best GPU
  POST /sdapi/v1/img2img  -> smart-routed to best GPU
  GET  /router/status     -> JSON fleet status for monitoring
  *    /*                 -> pass-through to first available instance

Usage:
  The bash launcher (run_stablediffusion.sh) writes a customised copy of
  this file to ~/sd-router/router.py with GPU_FLEET injected at runtime.
  Do not run this template directly.
"""

import asyncio
import json
import logging
import subprocess
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ROUTER] %(levelname)s %(message)s"
)
log = logging.getLogger("sd-router")

# ---------------------------------------------------------------------------
# GPU fleet — injected by run_stablediffusion.sh at install/update time.
# Each entry describes one WebUI instance and its GPU characteristics.
# Format: {"index": N, "vendor": "nvidia"|"amd"|"intel"|"cpu",
#           "name": "...", "port": 786N, "vram_mb": N,
#           "arch": "...", "xformers": true|false, "nvidia_index": N|null}
# ---------------------------------------------------------------------------
GPU_FLEET = []  # REPLACED_BY_LAUNCHER

# Per-instance queue depth — tracks active in-flight generation requests.
# Used to prefer less-busy GPUs when VRAM availability is otherwise equal.
queue_depth: dict[int, int] = {}

app = FastAPI(title="Stable Diffusion Smart GPU Router", version="3.0")


# ---------------------------------------------------------------------------
# Real-time VRAM query
# ---------------------------------------------------------------------------
def get_free_vram_mb() -> dict[int, int]:
    """
    Query nvidia-smi for free VRAM on each NVIDIA GPU.
    Returns {device_index: free_mb}. Non-NVIDIA GPUs are not queried here;
    they use their static total_vram as a conservative estimate.
    Called per-request so routing decisions always use live data.
    """
    result: dict[int, int] = {}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            timeout=3
        ).decode()
        for line in out.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                result[int(parts[0].strip())] = int(parts[1].strip())
    except Exception as exc:
        log.warning("nvidia-smi query failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# Instance liveness check
# ---------------------------------------------------------------------------
async def is_instance_ready(port: int) -> bool:
    """
    Returns True if the WebUI instance at this port is alive and accepting
    requests. Uses a short 2s timeout so a dead instance doesn't stall routing.
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"http://127.0.0.1:{port}/sdapi/v1/progress")
            return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Core routing logic
# ---------------------------------------------------------------------------
async def pick_best_gpu(width: int, height: int, batch: int) -> Optional[dict]:
    """
    Select the optimal GPU for a generation request.

    Algorithm:
      1. Query real-time free VRAM from nvidia-smi for all NVIDIA GPUs
      2. Check liveness of all instances concurrently to minimise latency
      3. Estimate VRAM needed:
           base_mb  = 4096  (conservative model weight overhead)
           pixel_mb = width x height x batch x 4 bytes / 1MB
           total    = base_mb + pixel_mb
      4. Filter out: offline instances, GPUs with insufficient free VRAM
      5. Score remaining candidates (lower = better):
           (highres_penalty, queue_depth, -free_vram)
           highres_penalty = 0 if GPU has >= 20GB AND request is >= 1024px
                           = 1 otherwise
           This means: for high-res, prefer high-VRAM GPUs; for standard,
           prefer the GPU with the shortest queue and most free VRAM.
      6. If no GPU passes the VRAM filter, fall back to least-busy online GPU
      7. If no GPU is online at all, return None (caller returns HTTP 503)

    Routing examples for a V100x2 + RTX 5000 setup:
      512x512  batch=1  -> RTX 5000 (fastest with xformers, FP16)
      1024x1024 SDXL   -> V100 (32GB preferred for high-res)
      2048x2048 SDXL   -> V100 only (RTX 5000 filtered: 16GB < needed)
      V100s busy       -> RTX 5000 as fallback for standard resolution
    """
    pixel_count = width * height * batch
    base_vram_mb = 4096
    pixel_vram_mb = (pixel_count * 4) // (1024 * 1024)
    estimated_vram_mb = base_vram_mb + pixel_vram_mb
    is_highres = width >= 1024 or height >= 1024

    log.info(
        "Routing: %dx%d batch=%d | est_vram=%dMB | highres=%s",
        width, height, batch, estimated_vram_mb, is_highres
    )

    free_vram = get_free_vram_mb()

    # Check liveness of all instances concurrently
    liveness = await asyncio.gather(
        *[is_instance_ready(g["port"]) for g in GPU_FLEET]
    )

    candidates = []
    for gpu, alive in zip(GPU_FLEET, liveness):
        port = gpu["port"]
        if not alive:
            log.info("  GPU %d (%s): OFFLINE", gpu["index"], gpu["name"])
            continue

        # Use live VRAM for NVIDIA; static total for AMD/Intel (no live query)
        nv_idx = gpu.get("nvidia_index")
        avail_vram = (
            free_vram.get(nv_idx, gpu["vram_mb"])
            if nv_idx is not None
            else gpu["vram_mb"]
        )

        if avail_vram < estimated_vram_mb:
            log.info(
                "  GPU %d (%s): %dMB free < %dMB needed — skipping",
                gpu["index"], gpu["name"], avail_vram, estimated_vram_mb
            )
            continue

        candidates.append({
            **gpu,
            "free_vram": avail_vram,
            "queue": queue_depth.get(port, 0),
        })

    # Fallback: if VRAM filter eliminated everyone, use least-busy online GPU
    if not candidates:
        log.warning(
            "No GPU has %dMB free — falling back to least-busy online GPU",
            estimated_vram_mb
        )
        candidates = [
            {**gpu, "free_vram": 0, "queue": queue_depth.get(gpu["port"], 0)}
            for gpu, alive in zip(GPU_FLEET, liveness)
            if alive
        ]

    if not candidates:
        log.error("All WebUI instances are offline — returning 503")
        return None

    def score(g: dict) -> tuple:
        # Lower tuple = better. Breakdown:
        # [0] highres_penalty: 0 if this GPU is best for high-res, else 1
        # [1] queue: fewer active requests = better
        # [2] -free_vram: more free memory = better (negated for min-sort)
        highres_ok = is_highres and g["vram_mb"] >= 20000
        return (0 if highres_ok else 1, g["queue"], -g["free_vram"])

    candidates.sort(key=score)
    chosen = candidates[0]
    log.info(
        "  -> GPU %d (%s) port=%d queue=%d free_vram=%dMB",
        chosen["index"], chosen["name"], chosen["port"],
        chosen["queue"], chosen["free_vram"]
    )
    return chosen


# ---------------------------------------------------------------------------
# Request forwarding helper
# ---------------------------------------------------------------------------
async def forward_to(request: Request, port: int) -> Response:
    """Forward an incoming request verbatim to a specific WebUI instance."""
    url = f"http://127.0.0.1:{port}{request.url.path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
            params=dict(request.query_params),
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


# ---------------------------------------------------------------------------
# Smart-routed generation endpoints
# ---------------------------------------------------------------------------
async def handle_generation(request: Request) -> Response:
    """
    Common handler for txt2img and img2img.
    Reads width/height/batch_size from the JSON body, picks the best GPU,
    increments queue depth before forwarding, decrements in a finally block
    so stuck counts can't accumulate even on errors.
    """
    body = await request.body()
    try:
        payload = json.loads(body) if body else {}
    except Exception:
        payload = {}

    width  = int(payload.get("width",  512))
    height = int(payload.get("height", 512))
    batch  = int(payload.get("batch_size", 1))

    gpu = await pick_best_gpu(width, height, batch)
    if gpu is None:
        return JSONResponse(
            {
                "error": "No GPU instances available",
                "detail": "All WebUI instances are offline or unreachable"
            },
            status_code=503
        )

    port = gpu["port"]
    queue_depth[port] = queue_depth.get(port, 0) + 1
    try:
        return await forward_to(request, port)
    finally:
        queue_depth[port] = max(0, queue_depth.get(port, 1) - 1)


@app.post("/sdapi/v1/txt2img")
async def txt2img(request: Request) -> Response:
    """Text-to-image — smart GPU routing."""
    return await handle_generation(request)


@app.post("/sdapi/v1/img2img")
async def img2img(request: Request) -> Response:
    """Image-to-image — smart GPU routing."""
    return await handle_generation(request)


# ---------------------------------------------------------------------------
# Pass-through: all other endpoints go to first available instance
# ---------------------------------------------------------------------------
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def passthrough(request: Request, path: str) -> Response:
    """
    Non-generation endpoints (WebUI UI, model management, settings, etc.)
    don't need smart routing — they don't consume GPU VRAM for inference.
    Forward to the first online instance.
    """
    for gpu in GPU_FLEET:
        if await is_instance_ready(gpu["port"]):
            return await forward_to(request, gpu["port"])
    return JSONResponse({"error": "No instances available"}, status_code=503)


# ---------------------------------------------------------------------------
# Fleet status endpoint
# ---------------------------------------------------------------------------
@app.get("/router/status")
async def router_status() -> dict:
    """
    Returns real-time status of the entire GPU fleet.
    Useful for monitoring dashboards, health checks, and debugging routing.

    Example response:
      {
        "router_port": 8080,
        "gpus": [
          {"gpu_index": 0, "name": "Tesla V100", "online": true,
           "queue_depth": 0, "free_vram_mb": 30000, "total_vram_mb": 32768, ...}
        ]
      }
    """
    free_vram = get_free_vram_mb()
    liveness = await asyncio.gather(
        *[is_instance_ready(g["port"]) for g in GPU_FLEET]
    )

    statuses = []
    for gpu, alive in zip(GPU_FLEET, liveness):
        port = gpu["port"]
        nv_idx = gpu.get("nvidia_index")
        fv = (
            free_vram.get(nv_idx, gpu["vram_mb"])
            if nv_idx is not None
            else gpu["vram_mb"]
        )
        statuses.append({
            "gpu_index":     gpu["index"],
            "name":          gpu["name"],
            "vendor":        gpu["vendor"],
            "arch":          gpu["arch"],
            "port":          port,
            "url":           f"http://localhost:{port}",
            "online":        alive,
            "queue_depth":   queue_depth.get(port, 0),
            "free_vram_mb":  fv,
            "total_vram_mb": gpu["vram_mb"],
            "xformers":      gpu["xformers"],
        })

    return {
        "router_port": app.state.router_port if hasattr(app.state, "router_port") else 8080,
        "gpus": statuses,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    app.state.router_port = port
    # Initialise queue_depth from GPU_FLEET (populated by launcher)
    for g in GPU_FLEET:
        queue_depth.setdefault(g["port"], 0)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
