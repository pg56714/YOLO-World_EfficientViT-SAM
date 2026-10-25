# 安全性紀錄

[English](security-log.md) | 繁體中文

- 日期: 2026-10-25
- 分支: `fix/security-upgrades`
- 範圍: `pyproject.toml`、`uv.lock`

## 摘要

PyTorch 改用 CUDA 12.6 的套件來源以升到已修補版本，gradio、inference-gpu、supervision 也一併升級。pillow、filelock、wheel 因為 gradio 與 inference 限制版本，仍有未修正的漏洞。

## 修改內容

| 項目 | 修改前 | 修改後 | 原因 |
|---|---|---|---|
| torch / torchvision | 2.5.1+cu121 / 0.20.1 | 2.14.1+cu126 / 0.29.1 | CVE-2025-32434（`torch.load(weights_only=True)` 可遠端執行程式碼）與之後的記憶體破壞漏洞 |
| gradio | 5.1.0 | 6.20.0 | 任意檔案讀取、SSRF、路徑穿越漏洞（6.16 修正） |
| inference-gpu[yolo-world] | 0.30.0 | 1.8.2 | 帶入已修補的依賴，並與新版 gradio 相容 |
| supervision | 0.22.0 | 0.30.9 | inference 1.8.2 的需求 |

## 使用前須知

- PyTorch 改由 CUDA 12.6 的套件來源（`pytorch-cu126`）安裝，不再使用 cu121，因為 cu121 最高只到 torch 2.5.1。需要支援 CUDA 12.6 以上的 NVIDIA 驅動；請重建虛擬環境以安裝新版套件。
- gradio 固定在 6.20.0：更新的 6.x 版本與 inference 1.8.2 要求的 huggingface-hub 版本衝突。

## 尚未解決的問題

### 被上游限制的套件版本

| 套件 | 版本 | 漏洞 | 無法修正的原因 | 影響與因應 |
|---|---|---|---|---|
| pillow | 12.3.0 | 41 筆（13.0.0 修正） | gradio 6.16 以上要求 `pillow<13`，目前沒有允許 Pillow 13 的 gradio 版本。 | 解碼惡意製作的圖片（例如 TIFF、PDF）可能讓程式當掉或越界讀取記憶體。請只在本機執行 Gradio 介面（不要開 `share=True`、不要對外開放連接埠），只開啟可信任的圖片；等 gradio 允許 Pillow 13 後再升級。 |
| filelock | 3.32.7 | 3 筆（4.2.1 修正） | inference 1.8.2 要求 `filelock<=4.0.0`。 | Windows 上的 `SoftFileLock` 與 SQLite 版 `ReadWriteLock` 的鎖定處理有缺陷。這裡只用於 Hugging Face 模型快取，單人本機使用的風險低。 |
| wheel | 0.45.1 | 2 筆（0.46.2 修正） | inference 1.8.2 要求 `wheel<=0.45.1`。 | 用 `wheel unpack` 解開惡意製作的 wheel 時可能路徑穿越或更改檔案權限。本程式執行時不會解開 wheel。 |

## 驗證方式

- 以 osv-scanner 掃描 `uv.lock`，只剩上表的套件。
- CPU、torch 2.14.1：EfficientViT-SAM（xl1）能建立並執行 `set_image()`、`predict()`。
- `app.py` 使用的 YOLOWorld `infer()`、`set_classes()` 在 inference 1.8.2 中沒有變動。
- 把模型模組換成假模組後，Gradio 介面能正常建立並提供服務。

## 未驗證項目

- 用 GPU 與實際權重執行推論（測試環境沒有 GPU 與權重檔）。
