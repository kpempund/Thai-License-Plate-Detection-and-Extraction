# Thai License Plate OCR & Post-Processing

Final stage of the pipeline: Extracts, sanitizes, and validates license plate text from cropped images[cite: 1].

| Item | Description |
|------|-------------|
| `ocr_pipeline.py` | The core engine: includes image upscaling, spatial routing, regex-based grammar enforcement, and fuzzy province matching[cite: 1]. |
| `evaluation.py` | Benchmarking script that calculates Character Accuracy Rate (CAR) and Word Accuracy Rate (WAR) across the test dataset[cite: 1]. |
| `requirements.txt` | Python dependencies including `easyocr`, `opencv-python`, and `numpy`[cite: 1]. |

## Pipeline Overview

| Step | Technique | What it does |
|------|-----------|--------------|
| 1 | **Image Upscaling** | Resizes input by 200% and applies sharpening to fix blurry/low-res characters[cite: 1]. |
| 2 | **Detection** | Runs EasyOCR with a strict Thai/Numeric `ALLOWLIST` to prevent hallucinations[cite: 1]. |
| 3 | **Spatial Routing** | Uses Y-axis geometry (60/40 split) to separate the **Plate Number** from the **Province**[cite: 1]. |
| 4 | **Grammar Sanitization** | Uses Regex to enforce `[Prefix][Consonants][Suffix]` rules and a dictionary to fix visual errors (e.g., `เ,ไ,โ` → `1`)[cite: 1]. |
| 5 | **Fuzzy Matching** | Uses Levenshtein distance to snap misspelled OCR text to the official list of 77 Thai provinces[cite: 1]. |

## Results (Evaluation on 41 Images)

| Metric | Number Accuracy | Province Accuracy | Full Plate (Combined) |
|-------|--------|-----------|-----------|
| **CAR (Character)** | 76.51% | 67.04% | 68.40%[cite: 1] |
| **WAR (Word/Full)** | 43.90% | 63.41% | **26.83%**[cite: 1] |

*Note: WAR represents a perfect "Ground Truth" match. 26.83% represents plates where both the Number and Province were 100% correct[cite: 1].*

## Installation

1. Install PyTorch with CUDA support: [pytorch.org](https://pytorch.org/get-started/locally/)[cite: 1]
2. Install dependencies:
```bash
pip install -r requirements.txt
```[cite: 1]

## Model & Logic Specs

- **Core Engine:** EasyOCR (CRAFT detector + CRNN recognizer)[cite: 1]
- **Languages:** Thai (`th`), English (`en`)[cite: 1]
- **Spatial Threshold:** 60% Y-axis split for Number vs. Province[cite: 1]
- **Fuzzy Logic:** `difflib.get_close_matches` with a `0.3` cutoff for Province validation[cite: 1]

### Post-Processing Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `upscale_factor` | 2.0 | Bicubic interpolation multiplier[cite: 1] |
| `y_split_ratio` | 0.6 | Separates top row from bottom row[cite: 1] |
| `confidence_min`| 0.1 | Ignores low-confidence OCR "ghost" text[cite: 1] |
| `fuzzy_cutoff` | 0.3 | Forgiveness level for province typos[cite: 1] |
