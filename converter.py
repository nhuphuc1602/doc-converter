"""
converter.py — GPU-accelerated PDF ↔ Word converter
Chạy trên NVIDIA Brev với GPU (H100/A100/RTX).

PDF → Word : dùng marker-pdf (surya OCR + layout detection, GPU)
Word → PDF : dùng LibreOffice headless (chính xác nhất cho .docx)
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import torch

# ── Detect GPU ────────────────────────────────────────────────────────────────

def get_device_info() -> dict:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        return {"device": "cuda", "name": name, "vram_gb": mem}
    return {"device": "cpu", "name": "CPU", "vram_gb": 0}


# ── PDF → Word ────────────────────────────────────────────────────────────────

def pdf_to_word(pdf_path: str, output_dir: str | None = None) -> str:
    """
    Convert PDF → .docx using marker-pdf (GPU-accelerated).
    Returns path to the output .docx file.
    """
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
    from docx import Document

    pdf_path = Path(pdf_path)
    if output_dir is None:
        output_dir = pdf_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models (cached after first load)
    models = load_all_models()

    # Convert PDF → markdown + metadata (GPU-accelerated)
    full_text, images, metadata = convert_single_pdf(
        str(pdf_path),
        models,
        max_pages=None,
        langs=["Vietnamese", "English"],
        batch_multiplier=4,          # tận dụng VRAM lớn của Brev GPU
    )

    # Build .docx from markdown output
    doc = Document()
    doc.add_heading(pdf_path.stem, level=0)

    for line in full_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=1)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=2)
        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=3)
        elif stripped.startswith("- ") or stripped.startswith("* "):
            doc.add_paragraph(stripped[2:], style="List Bullet")
        else:
            doc.add_paragraph(stripped)

    out_path = output_dir / (pdf_path.stem + ".docx")
    doc.save(str(out_path))
    return str(out_path)


# ── Word → PDF ────────────────────────────────────────────────────────────────

def word_to_pdf(docx_path: str, output_dir: str | None = None) -> str:
    """
    Convert .docx → PDF using LibreOffice headless.
    Returns path to the output .pdf file.
    """
    docx_path = Path(docx_path)
    if output_dir is None:
        output_dir = docx_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    libreoffice_bin = _find_libreoffice()
    if libreoffice_bin is None:
        raise RuntimeError(
            "LibreOffice không tìm thấy. Chạy: sudo apt-get install -y libreoffice"
        )

    result = subprocess.run(
        [
            libreoffice_bin,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(output_dir),
            str(docx_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"LibreOffice error:\n{result.stderr}")

    out_path = output_dir / (docx_path.stem + ".pdf")
    if not out_path.exists():
        raise FileNotFoundError(f"LibreOffice không tạo được file: {out_path}")

    return str(out_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_libreoffice() -> str | None:
    for candidate in ["libreoffice", "soffice", "/usr/bin/libreoffice", "/usr/bin/soffice"]:
        if shutil.which(candidate):
            return candidate
    return None


def get_pdf_page_count(pdf_path: str) -> int:
    import fitz  # PyMuPDF
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def get_docx_word_count(docx_path: str) -> int:
    from docx import Document
    doc = Document(docx_path)
    return sum(len(p.text.split()) for p in doc.paragraphs)
