# Thai License Plate OCR & Post-Processing

Final stage of the pipeline: Extracts, sanitizes, and validates license plate text from cropped images.

| Item | Description |
|------|-------------|
| `ocr_pipeline.py` | The core engine: includes image upscaling, spatial routing, regex-based grammar enforcement, and fuzzy province matching. |
| `evaluation.py` | Benchmarking script that calculates Character Accuracy Rate (CAR) and Word Accuracy Rate (WAR). |
| `requirements.txt` | Python dependencies including `easyocr`, `opencv-python`, and `numpy`. |

## Pipeline Overview

| Step | Technique | What it does |
|------|-----------|--------------|
| 1 | **Image Upscaling** | Resizes input by 200% and applies sharpening to fix blurry/low-res characters. |
| 2 | **Detection** | Runs EasyOCR with a strict Thai/Numeric `ALLOWLIST` to prevent hallucinations. |
| 3 | **Spatial Routing** | Uses Y-axis geometry (60/40 split) to separate the **Plate Number** from the **Province**. |
| 4 | **Grammar Sanitization** | Uses Regex to enforce `[Prefix][Consonants][Suffix]` rules and a dictionary to fix visual errors (e.g., `ด` → `1`). |
| 5 | **Fuzzy Matching** | Uses Levenshtein distance to snap misspelled OCR text to the official list of 77 Thai provinces. |

## Results (Evaluation on 41 Images)

| Metric | Number Accuracy | Province Accuracy | Full Plate (Combined) |
|-------|--------|-----------|-----------|
| **CAR (Character)** | 76.51% | 67.04% | 68.40% |
| **WAR (Word/Full)** | 43.90% | 63.41% | **26.83%** |

*Note: WAR represents a perfect "Ground Truth" match. 26.83% represents plates where both the Number and Province were 100% correct.*

## Installation

1. Install PyTorch with CUDA support: [pytorch.org](https://pytorch.org/get-started/locally/)
2. Install dependencies:
```bash
pip install -r requirements.txt
