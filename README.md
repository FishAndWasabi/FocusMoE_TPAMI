# FocusMoE: Mixture-of-Focused-Experts for Multimodal Remote Sensing Object Detection

Inference code and checkpoints for FocusMoE and the comparison methods.

## Models

Download a checkpoint to `checkpoints/` and use its matching config.
`k` is the number of active experts per token.

| Scale | k | Method | Config | Checkpoint |
|---|---|---|---|---|
| 50M | 3 | SM3Det | [configs/sm3det_50m.py](configs/sm3det_50m.py) | [sm3det_50m.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/sm3det_50m.pth) |
| 50M | 3 | Multi-Branch | [configs/multibranch_50m.py](configs/multibranch_50m.py) | [multibranch_50m.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/multibranch_50m.pth) |
| 50M | 3 | Multi-Stage | [configs/multistage_50m.py](configs/multistage_50m.py) | [multistage_50m.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/multistage_50m.pth) |
| 50M | 3 | FocusMoE | [configs/focusmoe_50m.py](configs/focusmoe_50m.py) | [focusmoe_50m.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/focusmoe_50m.pth) |
| 150M | 2 | SM3Det | [configs/sm3det_150m_top2.py](configs/sm3det_150m_top2.py) | [sm3det_150m_top2.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/sm3det_150m_top2.pth) |
| 150M | 2 | FocusMoE | [configs/focusmoe_150m_top2.py](configs/focusmoe_150m_top2.py) | [focusmoe_150m_top2.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/focusmoe_150m_top2.pth) |
| 150M | 3 | SM3Det | [configs/sm3det_150m.py](configs/sm3det_150m.py) | [sm3det_150m.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/sm3det_150m.pth) |
| 150M | 3 | Multi-Branch | [configs/multibranch_150m.py](configs/multibranch_150m.py) | [multibranch_150m.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/multibranch_150m.pth) |
| 150M | 3 | Multi-Stage | [configs/multistage_150m.py](configs/multistage_150m.py) | [multistage_150m.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/multistage_150m.pth) |
| 150M | 3 | FocusMoE | [configs/focusmoe_150m.py](configs/focusmoe_150m.py) | [focusmoe_150m.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/focusmoe_150m.pth) |
| 250M | 2 | SM3Det | [configs/sm3det_250m_top2.py](configs/sm3det_250m_top2.py) | [sm3det_250m_top2.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/sm3det_250m_top2.pth) |
| 250M | 2 | FocusMoE | [configs/focusmoe_250m_top2.py](configs/focusmoe_250m_top2.py) | [focusmoe_250m_top2.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/focusmoe_250m_top2.pth) |
| 250M | 3 | SM3Det | [configs/sm3det_250m.py](configs/sm3det_250m.py) | [sm3det_250m.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/sm3det_250m.pth) |
| 250M | 3 | Multi-Branch | [configs/multibranch_250m.py](configs/multibranch_250m.py) | [multibranch_250m.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/multibranch_250m.pth) |
| 250M | 3 | Multi-Stage | [configs/multistage_250m.py](configs/multistage_250m.py) | [multistage_250m.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/multistage_250m.pth) |
| 250M | 3 | FocusMoE | [configs/focusmoe_250m.py](configs/focusmoe_250m.py) | [focusmoe_250m.pth](https://github.com/FishAndWasabi/FocusMoE_TPAMI/releases/download/v1.2.0/focusmoe_250m.pth) |

SM3Det top-2 uses the official checkpoints; the other SM3Det variants are from our experiments.
[SHA256 checksums](checkpoints.sha256) are provided for all weights.

## Installation

Requires Linux and an NVIDIA CUDA GPU.

```bash
conda create -n focusmoe python=3.8 -y
conda activate focusmoe
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

## Inference

Run from the repository directory:

```bash
python infer.py configs/focusmoe_50m.py checkpoints/focusmoe_50m.pth \
  /path/to/image.png --modality rgb --out outputs/detections.json
```

Use `--modality sar`, `rgb`, or `ir` and select a config/checkpoint pair from the table.
Detections are saved as JSON in the original image coordinates:

- SAR: `[x1, y1, x2, y2, score]`.
- RGB/IR: `[cx, cy, width, height, angle_radians, score]` (`le90`).
