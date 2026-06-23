"""ModelScope online demo for Unlimited-OCR-ROCm.

Deploy this on modelscope.cn to give users a zero-barrier OCR experience
on free AMD GPU hardware.
"""

import base64
import json
import os
import tempfile
from pathlib import Path

import fitz
import gradio as gr
import requests

SERVER_URL = os.environ.get("SGLANG_SERVER_URL", "http://127.0.0.1:10000")
DEFAULT_PROMPT = "document parsing."


def pdf_page_to_image(pdf_bytes: bytes, page_num: int, dpi: int = 150) -> str:
    """Convert a single PDF page to a PNG file, return the path."""
    tmp_dir = tempfile.mkdtemp(prefix="ocr_demo_")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_num >= len(doc):
        doc.close()
        return ""
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    out_path = os.path.join(tmp_dir, f"page_{page_num + 1:04d}.png")
    page.get_pixmap(matrix=mat).save(out_path)
    doc.close()
    return out_path


def run_ocr(file, dpi: int, image_mode: str) -> tuple[str, str]:
    """Run OCR on uploaded file, return (markdown_output, status_message)."""
    if file is None:
        return "", "Please upload a PDF or image file first."

    file_path = Path(file.name) if hasattr(file, "name") else None
    if file_path is None:
        return "", "Could not read uploaded file."

    ext = file_path.suffix.lower()

    if ext == ".pdf":
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        image_path = pdf_page_to_image(pdf_bytes, 0, dpi=dpi)
        if not image_path:
            return "", "Failed to extract page from PDF."
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        image_path = str(file_path)
    else:
        return "", f"Unsupported file type: {ext}. Please upload PDF, PNG, or JPG."

    # Encode image
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
    mime_type = f"image/{ext.lstrip('.')}"
    if ext in (".jpg", ".jpeg"):
        mime_type = "image/jpeg"

    # Build request
    payload = {
        "model": "Unlimited-OCR",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": DEFAULT_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_data}"}},
            ]
        }],
        "temperature": 0,
        "skip_special_tokens": False,
        "stream": True,
        "images_config": {"image_mode": image_mode},
    }

    try:
        resp = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=300,
            stream=True,
        )
        resp.raise_for_status()

        chunks = []
        token_count = 0
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {}).get("content", "")
            except (json.JSONDecodeError, KeyError):
                continue
            if delta:
                token_count += 1
                chunks.append(delta)

        markdown_output = "".join(chunks)
        status = f"Done — {token_count} tokens generated"

    except requests.ConnectionError:
        return "", "Cannot reach OCR server. The demo may be starting up — try again in 30 seconds."
    except Exception as e:
        return "", f"Error: {e}"

    return markdown_output, status


def build_demo():
    with gr.Blocks(
        title="Unlimited-OCR-ROCm — OCR on AMD GPU",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # Unlimited-OCR-ROCm — OCR on AMD GPU

        Upload a PDF or image and get structured Markdown output in seconds.
        **Powered by AMD ROCm — running on real AMD GPU hardware.**

        Want to process your own files in bulk?
        → [Register on AMD Radeon Cloud](https://radeon.anruicloud.com/) for dedicated GPU access.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload PDF or Image",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"],
                )
                dpi_slider = gr.Slider(
                    minimum=100, maximum=300, value=150, step=50,
                    label="DPI (150 recommended — same quality as 300, faster)",
                )
                mode_radio = gr.Radio(
                    choices=["gundam", "base"],
                    value="gundam",
                    label="Image Mode",
                )
                run_btn = gr.Button("Run OCR", variant="primary", size="lg")

            with gr.Column(scale=2):
                output_text = gr.Markdown(label="OCR Result", value="*Output will appear here...*")
                status_text = gr.Textbox(label="Status", interactive=False)

        run_btn.click(
            fn=run_ocr,
            inputs=[file_input, dpi_slider, mode_radio],
            outputs=[output_text, status_text],
        )

        gr.Markdown("""
        ---
        ### More Options

        - **Batch processing:** [AMD Radeon Cloud](https://radeon.anruicloud.com/)
          gives you a dedicated GPU instance for bulk OCR.
        - **Local install:** `pip install unlimited-ocr-rocm` if you have your own AMD GPU.
        - **Source code:** [GitHub](https://github.com/AIwork4me/Unlimited-OCR-ROCm)
        - **Powered by:** Baidu Unlimited-OCR · SGLang · AMD ROCm
        """)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(max_size=10).launch(server_name="0.0.0.0", server_port=7860)
