#!/bin/bash
# setup.sh — Cài đặt môi trường trên NVIDIA Brev
# Chạy một lần sau khi instance khởi động:  bash setup.sh

set -e

echo "=========================================="
echo " DocConverter — NVIDIA Brev Setup"
echo "=========================================="

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/5] Cài system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    libreoffice \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    wget curl git

# ── 2. Python environment ─────────────────────────────────────────────────────
echo "[2/5] Cài Python packages..."
pip install --upgrade pip -q

# PyTorch với CUDA (phù hợp H100/A100/RTX trên Brev)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

# App dependencies
pip install -r requirements.txt -q

# ── 3. Pre-download marker models ─────────────────────────────────────────────
echo "[3/5] Tải marker-pdf models (lần đầu ~2GB)..."
python - <<'PYEOF'
from marker.models import load_all_models
print("Đang tải models vào cache...")
models = load_all_models()
print("Models đã sẵn sàng!")
PYEOF

# ── 4. Kiểm tra GPU ───────────────────────────────────────────────────────────
echo "[4/5] Kiểm tra GPU..."
python - <<'PYEOF'
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    print(f"  GPU: {name} | VRAM: {mem} GB")
else:
    print("  Không tìm thấy GPU — chạy CPU mode")
PYEOF

# ── 5. Kiểm tra LibreOffice ───────────────────────────────────────────────────
echo "[5/5] Kiểm tra LibreOffice..."
libreoffice --version

echo ""
echo "=========================================="
echo " Setup hoàn tất!"
echo " Khởi động app: python app.py"
echo " UI  : http://localhost:7860"
echo " API : http://localhost:8000/docs"
echo "=========================================="
