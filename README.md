# Thai License Plate Detection and Extraction

A two-stage computer vision pipeline for detecting and reading Thai license plates. The system first locates the license plate in a car image using a YOLO detector, then extracts the plate number and province using a YOLO-based character recogniser. A Streamlit web app ties the two stages together for easy interactive use.

---

## Project Structure

```
Thai-License-Plate-Detection-and-Extraction/
│
├── app.py                          # Streamlit web application
├── inference.py                    # CLI inference script + helper functions
├── requirements.txt                # Full Python dependencies
├── car1.jpg                        # Sample car image 1
├── car2.jpg                        # Sample car image 2
│
├── models/
│   ├── lp_detect.pt                # Trained YOLO26s license plate detector
│   └── lp_recog.pt                 # Trained YOLO11n character recogniser
│
├── LP_Detector/
│   ├── train_lp_detector.ipynb     # Notebook: train & evaluate the plate detector
│   ├── CarDemo.mp4                 # Raw demo video
│   ├── CarDemo_Inferenced.mp4      # Demo video with detection overlay
│   └── runs/yolo26s/               # Training artefacts
│
└── LP_Recognizer/
    ├── OCR/                        # Alternative EasyOCR-based recognition approach
    │   ├── ocr_pipeline.py         # EasyOCR pipeline with preprocessing & grammar rules
    │   ├── evaluate.py             # CAR / WAR evaluation script
    │   ├── format_manual_datasets.py  # Dataset formatting utility
    │   └── requirements.txt        # OCR-specific dependencies
    │
    └── YOLO11/
        └── license_plate_extractor_yolo11.ipynb  # Notebook: train & evaluate YOLO11n recogniser
```

---

## File & Folder Descriptions

### Root-level files

| File | Description |
|------|-------------|
| `app.py` | Streamlit web app. Accepts an uploaded car image, calls `crop_license_plate` to detect the plate, then calls `extract_plate_text` to read it, and displays the cropped plate, predicted number, and province. |
| `inference.py` | Core inference logic shared by the app and CLI. Contains `crop_license_plate` (runs YOLO26s at 960 px resolution, returns the highest-confidence bounding-box crop) and `extract_plate_text` (runs YOLO11n, decodes Thai consonant class IDs via `CHAR_DECODER`, maps province codes via `PROVINCE_MAP`, and sorts characters left-to-right). Also includes a `__main__` block for quick command-line testing on `car2.jpg`. |
| `requirements.txt` | Pinned environment for the full project (Streamlit, Ultralytics, OpenCV, etc.). |
| `car1.jpg` / `car2.jpg` | Sample car photos for smoke-testing inference. |

### `models/`

| File | Description |
|------|-------------|
| `lp_detect.pt` | YOLO26s weights for detecting the license plate bounding box from a full car image. Look for more detail in LP_detector directory .|
| `lp_recog.pt` | YOLO11n weights trained for classifying individual Thai characters and province names directly on the cropped plate. Look for more detail in LP_Recognizer/YOLO11 directory |

---

## Setup & Reproduction

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (CPU works but is slow)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Thai-License-Plate-Detection-and-Extraction.git
cd Thai-License-Plate-Detection-and-Extraction
```

### 2. Install PyTorch (CUDA)

Install the correct build for your CUDA version from the official selector before installing other packages:

```bash
# Example for CUDA 11.8 — adjust the index URL for your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

See https://pytorch.org/get-started/locally/ for the exact command and version.

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify model weights

The two pretrained model weights must be present in `models/`:

```
models/
├── lp_detect.pt
└── lp_recog.pt
```

---

### Running the Streamlit App

```bash
streamlit run app.py
```

Open the URL shown in the terminal, upload a car image (JPG or PNG), and the app will display the detected plate crop along with the predicted plate number and province.

---

### Running CLI Inference

Edit the `INPUT_IMAGE` variable at the bottom of `inference.py` if needed, then run:

```bash
python inference.py
```

Output is printed to the terminal, e.g. `กข 1234 กรุงเทพมหานคร`.

---

## Model Performance Summary

**License Plate Detector (YOLO26s)**

| Split | mAP@50 | mAP@50-95 | Precision | Recall |
|-------|--------|-----------|-----------|--------|
| Val   | 0.952  | 0.706     | 0.971     | 0.882  |
| Test  | 0.949  | 0.699     | 0.984     | 0.885  |

**Character Recogniser (YOLO11n)**

| Split | mAP@50 | mAP@50-95 | Precision | Recall |
|-------|--------|-----------|-----------|--------|
| Val   | 0.9370  | 0.8083     | 0.9016     | 0.8993  |
| Test  | 0.9644  | 0.8914     | 0.8881     | 0.9457  |

|             | CAR    | WAR    |
|-------------|--------|--------|
| Number      | 92.27% | 70.73% |
| Province    | 85.77% | 85.37% |
| Full Plate  | 84.32% | 58.54% |
