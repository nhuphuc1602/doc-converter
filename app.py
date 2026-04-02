"""
app.py — Gradio UI + FastAPI backend
Chạy: python app.py
UI  : http://localhost:7860
API : http://localhost:8000/docs
"""

import os
import shutil
import tempfile
import threading
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

import converter as conv

# ── FastAPI ───────────────────────────────────────────────────────────────────

api = FastAPI(
    title="DocConverter API",
    description="GPU-accelerated PDF ↔ Word converter — powered by NVIDIA Brev",
    version="1.0.0",
)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "docconverter"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@api.get("/health")
def health():
    info = conv.get_device_info()
    return {"status": "ok", **info}


@api.post("/pdf-to-word", summary="Convert PDF → Word (.docx)")
async def api_pdf_to_word(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Chỉ chấp nhận file .pdf")

    tmp_dir = Path(tempfile.mkdtemp(dir=UPLOAD_DIR))
    in_path = tmp_dir / file.filename

    with open(in_path, "wb") as f:
        f.write(await file.read())

    try:
        out_path = conv.pdf_to_word(str(in_path), str(tmp_dir))
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, str(e))

    return FileResponse(
        out_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=Path(out_path).name,
    )


@api.post("/word-to-pdf", summary="Convert Word (.docx) → PDF")
async def api_word_to_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(400, "Chỉ chấp nhận file .docx")

    tmp_dir = Path(tempfile.mkdtemp(dir=UPLOAD_DIR))
    in_path = tmp_dir / file.filename

    with open(in_path, "wb") as f:
        f.write(await file.read())

    try:
        out_path = conv.word_to_pdf(str(in_path), str(tmp_dir))
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, str(e))

    return FileResponse(
        out_path,
        media_type="application/pdf",
        filename=Path(out_path).name,
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def ui_pdf_to_word(pdf_file):
    if pdf_file is None:
        return None, "Vui lòng upload file PDF."
    try:
        info    = conv.get_device_info()
        pages   = conv.get_pdf_page_count(pdf_file)
        out     = conv.pdf_to_word(pdf_file)
        words   = conv.get_docx_word_count(out)
        msg = (
            f"Chuyển đổi thành công!\n"
            f"GPU: {info['name']} ({info['vram_gb']} GB VRAM)\n"
            f"Trang PDF: {pages} | Từ trong Word: {words:,}"
        )
        return out, msg
    except Exception as e:
        return None, f"Lỗi: {e}"


def ui_word_to_pdf(docx_file):
    if docx_file is None:
        return None, "Vui lòng upload file Word (.docx)."
    try:
        words = conv.get_docx_word_count(docx_file)
        out   = conv.word_to_pdf(docx_file)
        msg   = f"Chuyển đổi thành công!\nSố từ: {words:,}"
        return out, msg
    except Exception as e:
        return None, f"Lỗi: {e}"


def build_ui():
    device = conv.get_device_info()
    gpu_badge = (
        f"GPU: {device['name']} — {device['vram_gb']} GB VRAM"
        if device["device"] == "cuda"
        else "CPU mode (không có GPU)"
    )

    with gr.Blocks(
        title="DocConverter — NVIDIA Brev",
        theme=gr.themes.Soft(primary_hue="green"),
    ) as demo:
        gr.Markdown(
            f"""
# DocConverter — PDF ↔ Word
**Powered by NVIDIA Brev GPU Infrastructure**
`{gpu_badge}`

Sử dụng [marker-pdf](https://github.com/VikParuchuri/marker) với GPU-accelerated OCR và layout detection.
"""
        )

        with gr.Tab("PDF → Word"):
            with gr.Row():
                with gr.Column():
                    pdf_input  = gr.File(label="Upload PDF", file_types=[".pdf"])
                    btn_p2w    = gr.Button("Chuyển đổi", variant="primary")
                with gr.Column():
                    word_out   = gr.File(label="Tải xuống Word (.docx)")
                    status_p2w = gr.Textbox(label="Trạng thái", lines=4, interactive=False)
            btn_p2w.click(ui_pdf_to_word, inputs=pdf_input, outputs=[word_out, status_p2w])

        with gr.Tab("Word → PDF"):
            with gr.Row():
                with gr.Column():
                    docx_input = gr.File(label="Upload Word (.docx)", file_types=[".docx"])
                    btn_w2p    = gr.Button("Chuyển đổi", variant="primary")
                with gr.Column():
                    pdf_out    = gr.File(label="Tải xuống PDF")
                    status_w2p = gr.Textbox(label="Trạng thái", lines=4, interactive=False)
            btn_w2p.click(ui_word_to_pdf, inputs=docx_input, outputs=[pdf_out, status_w2p])

        gr.Markdown(
            """
---
**REST API** có tại `http://localhost:8000/docs`
| Endpoint | Mô tả |
|----------|-------|
| `POST /pdf-to-word` | Upload PDF, nhận .docx |
| `POST /word-to-pdf` | Upload .docx, nhận PDF |
| `GET /health` | Kiểm tra GPU status |
"""
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def start_api():
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    # Chạy FastAPI trong background thread
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()

    # Chạy Gradio UI (blocking)
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,   # đặt True nếu muốn public URL
        show_error=True,
    )
