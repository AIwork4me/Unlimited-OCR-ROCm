"""Page-id sets + image resolver for the R-SWA spike (Phase 0 / Phase 2)."""
from __future__ import annotations
from pathlib import Path

IMG_DIR = Path("/workspace/OmniDocBench_data/images")
VLLM_SAMPLE_DIR = Path("/root/ocr-eval/predictions/vllm-sample-150")

# 15 pages where vLLM 0.20.2rc1 returned <50B (first-token EOS) on the 150-page
# sample while PyTorch (R-SWA) produced real OCR (312-628 B).
# Source: find /root/ocr-eval/predictions/vllm-sample-150/ -name '*.md' -size -50c
EOS_PAGES = [
    "PPT_1001115_eng_page_005",
    "PPT_CalculusReview_page_033",
    "PPT_Keuk Chan Narith_page_009",
    "PPT_LEP power point presentation-English-FINAL-10-31-07_page_011",
    "PPT_MMAT5390Lecture1_page_023",
    "PPT_all655920_page_001",
    "PPT_sociolinguistics_page_015",
    "book_en_搬书匠-3299-Swift Data Structure and Algorithms-2016-英文版_page_142",
    "color_textbook_zhonggaokao_小学_13.人教新起点英语（4-5年级）_人教新起点五年级英语下册_课本_人教新起点英语5B电子课本_page_034",
    "docstructbench_llm-raw-scihub-o.O-chin.201025015.pdf_1",
    "jiaocaineedrop_jiaocai_needrop_en_3718",
    "magazine_TheEconomist.2023.11.25_page_069",
    "eastmoney_ea3eda50a04cf431d7412a567497c91e8cc52f72b4c5ccb554776c5c57b13e29.pdf_4",
    "page-29ccb4ce-9266-4938-8f2d-b2b69ceb43cd",
    "yanbaopptmerge_yanbaoPPT_620",
]


def resolve_image(page_id: str) -> Path | None:
    """Find the image file for a page-id (any extension)."""
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        p = IMG_DIR / f"{page_id}{ext}"
        if p.exists():
            return p
    hits = sorted(IMG_DIR.glob(f"{page_id}.*")) if IMG_DIR.is_dir() else []
    return hits[0] if hits else None


def control_pages(n: int = 3) -> list[str]:
    """Top-n page-ids by vLLM output size in the 150-sample (clearly-succeeded pages)."""
    if not VLLM_SAMPLE_DIR.is_dir():
        return []
    files = [p for p in VLLM_SAMPLE_DIR.glob("*.md") if p.stem not in EOS_PAGES]
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return [p.stem for p in files[:n]]
