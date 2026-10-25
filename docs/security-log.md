# Security Log

English | [繁體中文](security-log.zh-TW.md)

- Date: 2026-10-25
- Branch: `fix/security-upgrades`
- Scope: `pyproject.toml`, `uv.lock`

## Summary

PyTorch moved to the CUDA 12.6 index so it can reach a patched release, and gradio, inference-gpu and supervision were upgraded. pillow, filelock and wheel still have open advisories because gradio and inference cap their versions.

## Changes

| Item | Before | After | Reason |
|---|---|---|---|
| torch / torchvision | 2.5.1+cu121 / 0.20.1 | 2.14.1+cu126 / 0.29.1 | CVE-2025-32434 (`torch.load(weights_only=True)` remote code execution) and later memory-corruption advisories |
| gradio | 5.1.0 | 6.20.0 | File read, SSRF and path traversal advisories (fixed in 6.16) |
| inference-gpu[yolo-world] | 0.30.0 | 1.8.2 | Pulls in patched dependencies; needed by the newer gradio |
| supervision | 0.22.0 | 0.30.9 | Required by inference 1.8.2 |

## What you need to know

- PyTorch now comes from the CUDA 12.6 wheel index (`pytorch-cu126`) instead of cu121, because the cu121 index stops at torch 2.5.1. You need an NVIDIA driver that supports CUDA 12.6 or newer. Recreate the virtual environment so the new wheels are installed.
- gradio is pinned to 6.20.0: newer 6.x releases conflict with the huggingface-hub version that inference 1.8.2 requires.

## Remaining issues

### Dependency versions held back upstream

| Package | Version | Advisories | Why it is not fixed | Impact and mitigation |
|---|---|---|---|---|
| pillow | 12.3.0 | 41 (fixed in 13.0.0) | gradio >= 6.16 requires `pillow<13`; no gradio release allows Pillow 13 yet. | Crafted image files (for example TIFF or PDF) can crash the process or read out-of-bounds memory when decoded. Run the Gradio app locally only (no `share=True`, no public port) and open only images you trust. Upgrade once gradio allows Pillow 13. |
| filelock | 3.32.7 | 3 (fixed in 4.2.1) | inference 1.8.2 requires `filelock<=4.0.0`. | Lock-handling flaws in `SoftFileLock` on Windows and in the SQLite-based `ReadWriteLock`. Here it only guards the Hugging Face model cache, so the risk is low for single-user local use. |
| wheel | 0.45.1 | 2 (fixed in 0.46.2) | inference 1.8.2 requires `wheel<=0.45.1`. | Path traversal and file-permission changes when a crafted wheel is unpacked with `wheel unpack`. The app never unpacks wheels at runtime. |

## How it was verified

- osv-scanner on `uv.lock`; only the packages listed above remain.
- CPU, torch 2.14.1: EfficientViT-SAM (xl1) builds and runs `set_image()` / `predict()`.
- The YOLOWorld `infer()` / `set_classes()` API used by `app.py` is unchanged in inference 1.8.2.
- The Gradio UI builds and serves with the model modules replaced by stubs.

## Not verified

- Inference on a GPU with the real model weights (the test environment has no GPU and no weights).
