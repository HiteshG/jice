# Unified Ice Hockey Player Tracking Pipeline

A production-ready pipeline integrating detection, tracking, mask propagation, jersey recognition, and team classification for ice hockey video analysis.

## Features

- **Multi-Object Tracking**: McByte tracker with mask-conditioned association
- **Mask Propagation**: SAM2 (real-time) + CUTIE for occlusion handling
- **Jersey Recognition**: PARSeq STR with temporal locking (no jitter)
- **Team Classification**: SigLIP + UMAP + KMeans clustering
- **Camera Motion Compensation**: Hybrid ORB/ECC for shake handling
- **Multi-Pass Processing**: Forward + backward tracking with interpolation

---

## Complete Model Setup Guide

### Models Overview

| Model | Size | Source | Download |
|-------|------|--------|----------|
| `hockey_yolo.pt` | ~140 MB | Your custom | Manual upload |
| `sam2.1_hiera_large.pt` | ~898 MB | Meta AI | Auto-download |
| `cutie-base-mega.pth` | ~507 MB | GitHub | Auto-download |
| `vitpose-h.pth` | ~1.1 GB | OpenMMLab | Auto-download |
| `parseq_hockey.ckpt` | ~100 MB | Your custom | Manual upload |
| `legibility_resnet34_hockey.pth` | ~85 MB | Your custom | Manual upload |
| SigLIP | ~400 MB | HuggingFace | Auto-download |

---

## Local Setup (Your Machine)

### Step 1: Clone SAM2 Real-Time Fork

```bash
cd /path/to/cv
git clone https://github.com/Gy920/segment-anything-2-real-time.git

cd segment-anything-2-real-time
pip install -e .
python setup.py build_ext --inplace

# Download SAM2 checkpoint
cd checkpoints
bash download_ckpts.sh
# Or manually:
# wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

**Verify:** `ls -lh segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt`

### Step 2: Clone and Setup CUTIE

```bash
cd /path/to/cv
git clone https://github.com/hkchengrex/Cutie.git

cd Cutie
pip install -e .

# Download CUTIE weights
mkdir -p weights
wget -O weights/cutie-base-mega.pth \
    https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth
```

**Verify:** `ls -lh Cutie/weights/cutie-base-mega.pth`

### Step 3: Download ViTPose (Optional - for jersey recognition)

```bash
# Install mmpose dependencies first
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmpose>=1.0.0"

# Create directory and download
mkdir -p unified_pipeline/models/vitpose
wget -O unified_pipeline/models/vitpose/vitpose-h.pth \
    https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth
```

**Verify:** `ls -lh unified_pipeline/models/vitpose/vitpose-h.pth`

### Step 4: Setup SigLIP (Auto-downloads)

SigLIP downloads automatically from HuggingFace on first use. If you have a token:

```bash
# Set HuggingFace token (optional, for private models)
export HF_TOKEN=your_huggingface_token

# Or login via CLI
huggingface-cli login
```

**Test SigLIP download:**
```python
from transformers import AutoProcessor, SiglipVisionModel

model = SiglipVisionModel.from_pretrained('google/siglip-base-patch16-224')
processor = AutoProcessor.from_pretrained('google/siglip-base-patch16-224')
print("SigLIP loaded successfully!")
```

### Step 5: Place Your Custom Models

```bash
# Create directories
mkdir -p unified_pipeline/models/parseq
mkdir -p unified_pipeline/models/legibility

# Copy your models
cp /path/to/hockey_yolo.pt unified_pipeline/models/
cp /path/to/parseq_hockey.ckpt unified_pipeline/models/parseq/
cp /path/to/legibility_resnet34_hockey.pth unified_pipeline/models/legibility/
```

### Step 6: Verify All Models

Run this script to verify everything is in place:

```bash
python -c "
import os

models = {
    'YOLO Detection': 'unified_pipeline/models/hockey_yolo.pt',
    'SAM2': 'segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt',
    'CUTIE': 'Cutie/weights/cutie-base-mega.pth',
    'ViTPose': 'unified_pipeline/models/vitpose/vitpose-h.pth',
    'PARSeq': 'unified_pipeline/models/parseq/parseq_hockey.ckpt',
    'Legibility': 'unified_pipeline/models/legibility/legibility_resnet34_hockey.pth',
}

print('Model Verification:')
print('='*70)
all_ok = True
for name, path in models.items():
    exists = os.path.exists(path)
    size = f'{os.path.getsize(path)/1024**2:.1f} MB' if exists else 'MISSING'
    status = 'OK' if exists else 'MISSING'
    print(f'{name:20s}: {status:8s} {size if exists else \"\"}')
    if not exists:
        all_ok = False
print('='*70)
print('All models ready!' if all_ok else 'Some models are missing!')
"
```

---

## Google Colab Setup (L4 GPU)

### Directory Structure in Colab

```
/content/
├── unified_pipeline/                # Your GitHub repo (cloned)
│   └── models/
│       ├── hockey_yolo.pt           # UPLOAD THIS
│       ├── parseq/
│       │   └── parseq_hockey.ckpt   # UPLOAD THIS
│       ├── legibility/
│       │   └── legibility_resnet34_hockey.pth  # UPLOAD THIS
│       └── vitpose/
│           └── vitpose-h.pth        # Auto-downloaded
├── segment-anything-2-real-time/    # Cloned separately
│   ├── checkpoints/
│   │   └── sam2.1_hiera_large.pt    # Auto-downloaded
│   └── configs/sam2.1/
│       └── sam2.1_hiera_l.yaml
├── Cutie/                           # Cloned separately
│   └── weights/
│       └── cutie-base-mega.pth      # Auto-downloaded
└── config.yaml                      # Generated by notebook
```

### Models You Must Upload Manually

| Model | Colab Upload Path |
|-------|-------------------|
| `hockey_yolo.pt` | `/content/unified_pipeline/models/hockey_yolo.pt` |
| `parseq_hockey.ckpt` | `/content/unified_pipeline/models/parseq/parseq_hockey.ckpt` |
| `legibility_resnet34_hockey.pth` | `/content/unified_pipeline/models/legibility/legibility_resnet34_hockey.pth` |

### Models Auto-Downloaded in Colab

| Model | Download URL | Colab Path |
|-------|--------------|------------|
| SAM2 | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt` | `/content/segment-anything-2-real-time/checkpoints/` |
| CUTIE | `https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth` | `/content/Cutie/weights/` |
| ViTPose | `https://download.openmmlab.com/mmpose/v1/...` | `/content/unified_pipeline/models/vitpose/` |
| SigLIP | HuggingFace `google/siglip-base-patch16-224` | `~/.cache/huggingface/` |

---

## Configuration File

Create `config.yaml`:

```yaml
# For Local Setup
detection:
  model_path: "unified_pipeline/models/hockey_yolo.pt"
  imgsz: 1280
  player_confidence: 0.4
  puck_confidence: 0.2

tracking:
  track_thresh: 0.6
  track_buffer: 30

mask:
  sam2_checkpoint: "segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
  sam2_config: "segment-anything-2-real-time/configs/sam2.1/sam2.1_hiera_l.yaml"
  cutie_checkpoint: "Cutie/weights/cutie-base-mega.pth"

jersey:
  str_model: "unified_pipeline/models/parseq/parseq_hockey.ckpt"
  legibility_model: "unified_pipeline/models/legibility/legibility_resnet34_hockey.pth"
  vitpose_checkpoint: "unified_pipeline/models/vitpose/vitpose-h.pth"
  lock_threshold: 3.0

team:
  classifier_type: "siglip"  # Uses SigLIP + UMAP + KMeans

device: "cuda"
verbose: true
```

---

## Usage

### Command Line

```bash
# Basic usage
python -m unified_pipeline.cli --video input.mp4 --output output.mp4

# With config file
python -m unified_pipeline.cli --video input.mp4 --config config.yaml

# Fast mode (no jersey/team classification)
python -m unified_pipeline.cli --video input.mp4 --no-jersey --no-team
```

### Python API

```python
from unified_pipeline import PipelineConfig, UnifiedPipeline, load_config

# Load configuration
config = load_config("config.yaml")

# Create pipeline
pipeline = UnifiedPipeline(config)

# Process video
tracks = pipeline.process_video("input.mp4", "output.mp4")

# Access results
for track_id, track in tracks.items():
    print(f"Track {track_id}: Jersey={track.jersey_number}, Team={track.team_id}")
```

---

## Class Mapping

| YOLO Class | Internal ID | Name |
|------------|-------------|------|
| 0 | - | Center Ice (ignored) |
| 1 | - | Faceoff (ignored) |
| 2 | - | Goalpost (ignored) |
| 3 | 1 | Goaltender |
| 4 | 0 | Player |
| 5 | 3 | Puck |
| 6 | 2 | Referee |

---

## Output Formats

### Annotated Video
- Bounding boxes colored by team
- Jersey numbers displayed
- Clean overlay

### MOT Format
```
frame,id,x,y,w,h,conf,class,team,jersey
1,1,100.5,200.3,50.0,120.0,0.95,0,1,23
1,2,300.2,150.7,48.0,115.0,0.92,0,0,17
```

---

## Troubleshooting

### CUDA Out of Memory
```yaml
mask:
  max_internal_size: 540  # Reduce from default
```

### SAM2 Import Error
```bash
cd segment-anything-2-real-time
pip install -e . --force-reinstall
python setup.py build_ext --inplace
```

### SigLIP Download Issues
```python
from huggingface_hub import login
login(token="your_hf_token")
```

### ViTPose/mmpose Issues
```bash
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmpose>=1.0.0"
```

---

## Performance Targets

- MOTA >= 70%
- IDF1 >= 65%
- Jersey accuracy >= 95%
- Team classification >= 90%
- Speed: ~5-10 FPS on L4 GPU

---

## License

Components used:
- McByte (tracking)
- SAM2 (Meta AI - Apache 2.0)
- CUTIE (video segmentation)
- PARSeq (text recognition)
- SigLIP (Google - Apache 2.0)
- ViTPose (OpenMMLab)
