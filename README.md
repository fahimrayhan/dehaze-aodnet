# Real-Time RGB-Thermal Enhancement + Fusion Pipeline (v2)

Upgraded pipeline addressing two problems found in testing: the previous
CLAHE-based enhancement couldn't recover genuinely dark/night detail, and
bicubic upscaling from the working resolution left output soft. This
version adds **learned low-light enhancement** and **learned
super-resolution**, and makes the whole pipeline GPU-resident (no
CPU↔GPU round-trips between stages) so it's both better-looking and
faster on a laptop RTX 3050/3060/3070.

## Pipeline

```
Capture (RGB + optional Thermal)
        │
        ▼
Downscale to working resolution (e.g. 256px) -- cheap, GPU-side
        │
        ▼
Condition scoring (haze / darkness) -- on a 64x64 GPU downsample, ~free
        │
        ▼
┌─────────────────────────────────────────┐
│ Zero-DCE++  (low-light, gated on darkness)    │
│ AOD-Net     (dehazing, gated on haze score)   │
│ DenseFuse   (RGB-thermal fusion, gated)       │
└─────────────────────────────────────────┘
        │
        ▼
ESPCN super-resolution upscale back to full output res (learned, not bicubic)
        │
        ▼
(optional) YOLO detection overlay
```

## Why these four models

| Stage | Model | Paper | Why |
|---|---|---|---|
| Low-light | **Zero-DCE++** | Li et al., IEEE TPAMI 2021 | Learns a per-pixel brightening curve via 4 *zero-reference* losses (no paired ground truth needed at all — trains on raw dark images only). Fixes the "barely visible enhancement" problem CLAHE had, since CLAHE can only redistribute existing contrast, not recover detail below the noise floor. |
| Dehazing | **AOD-Net** | Li et al., ICCV 2017 | <10KB, reformulates the haze equation into one learned variable — real-time even on CPU, used as the gated haze branch. |
| Fusion | **DenseFuse** | Li & Wu, IEEE TIP 2019 | Dense-block autoencoder, unsupervised pretraining + self-supervised fine-tune on your RGBT pairs; L1-norm feature fusion rule. |
| Super-resolution | **ESPCN** | Shi et al., CVPR 2016 | Does all compute at LOW resolution, expands to high-res only in the final sub-pixel-convolution layer — much cheaper and sharper than bicubic resize, which is what made the previous pipeline's output look soft. |

## Project layout

```
rgbt_pipeline_v2/
├── models/
│   ├── aod_net.py
│   ├── densefuse.py
│   ├── zero_dce.py      # Zero-DCE++ architecture + 4 zero-reference losses
│   └── espcn.py          # sub-pixel convolution super-resolution
├── dataset.py              # Dataset classes for all 4 training stages
├── utils.py                 # SSIM loss, synthetic haze/low-light generation,
│                              GPU-resident condition scoring
├── pipeline_gpu.py          # AdaptivePipelineGPU -- the core GPU-resident,
│                              FP16, condition-gated inference pipeline
├── train_aod.py
├── train_zerodce.py
├── train_densefuse.py
├── train_espcn.py
├── benchmark.py             # per-stage latency profiling -- RUN THIS FIRST
├── inference_live.py        # local webcam/video, cv2.imshow, FPS overlay
├── inference_colab.py       # Colab-compatible (video file in/out, no GUI)
├── export_models.py         # ONNX export for all 4 models (-> TensorRT)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Step 1: Benchmark BEFORE training anything

Even with random (untrained) weights, run the benchmark first to confirm
your actual hardware hits a reasonable FPS before investing time in
training -- the architecture choice is what determines speed, not the
trained weights:

```bash
python -c "
import torch, os
from models.aod_net import AODNet
from models.densefuse import DenseFuseAutoencoder
from models.zero_dce import ZeroDCEPP
from models.espcn import ESPCN
os.makedirs('checkpoints', exist_ok=True)
torch.save(AODNet().state_dict(), 'checkpoints/aod_net_best.pth')
torch.save(DenseFuseAutoencoder().state_dict(), 'checkpoints/densefuse_finetuned.pth')
torch.save(ZeroDCEPP().state_dict(), 'checkpoints/zerodce_best.pth')
torch.save(ESPCN(scale_factor=4).state_dict(), 'checkpoints/espcn_x4_best.pth')
"

python benchmark.py \
    --zerodce_ckpt checkpoints/zerodce_best.pth \
    --aod_ckpt checkpoints/aod_net_best.pth \
    --fuse_ckpt checkpoints/densefuse_finetuned.pth \
    --espcn_ckpt checkpoints/espcn_x4_best.pth \
    --n_frames 100 --with_thermal --dark
```

This prints a per-stage ms breakdown (zerodce / aod / fusion / sr / total)
so you know exactly where time goes before training. On an RTX 3060
laptop GPU with FP16, expect roughly: Zero-DCE++ ~2-4ms, AOD-Net ~1-2ms,
DenseFuse ~3-5ms, ESPCN ~2-3ms at 256px working resolution — the gating
means not every frame pays for every stage.

Try `--no_fp16` vs default, and `--compile` vs default, to see how much
each buys you on your specific GPU.

## Step 2: Train each model

### Zero-DCE++ (low-light) — needs only dark images, no pairs

```bash
python -c "
import glob
from utils import make_synthetic_lowlight_dataset
clear_paths = glob.glob('/content/train/visible/*')[:2000]
make_synthetic_lowlight_dataset(clear_paths, '/content/synthetic_lowlight')
"
python train_zerodce.py --image_dir /content/synthetic_lowlight/dark --epochs 30 --batch_size 16
```

If you have REAL night footage (extract frames from any night driving/
surveillance video), point `--image_dir` directly at those frames instead
— real dark images train a meaningfully better model than synthetic
darkening.

### AOD-Net (dehazing)

```bash
python -c "
import glob
from utils import make_synthetic_haze_dataset
clear_paths = glob.glob('/content/train/visible/*')[:2000]
make_synthetic_haze_dataset(clear_paths, '/content/synthetic_haze')
"
python train_aod.py --data_root /content/synthetic_haze --epochs 20 --batch_size 16
```

Prefer a real hazy/clear dataset (RESIDE ITS/OTS) over synthetic haze if
you can get it — generalizes better to real fog.

### DenseFuse (fusion) — two stages

```bash
python train_densefuse.py --stage autoencoder \
    --image_dir /content/train/visible /content/train/thermal \
    --epochs 15 --batch_size 16

python train_densefuse.py --stage finetune \
    --rgbt_root /content/train \
    --pretrained checkpoints/densefuse_ae_best.pth \
    --epochs 10 --batch_size 8
```

### ESPCN (super-resolution)

```bash
python train_espcn.py --image_dir /content/train/visible \
    --scale_factor 4 --epochs 30 --batch_size 16
```

Trains on any folder of reasonably high-res clear images — it downsamples
each one on the fly to create the (low-res, high-res) training pair, so a
separate low-res dataset isn't needed. `--scale_factor` should roughly
match `working_size -> output_size` for your target display/detector
resolution (e.g. working_size=256, scale_factor=4 -> 1024px output).

## Step 3: Run live on your laptop

```bash
python inference_live.py \
    --rgb_source 0 --thermal_source 1 \
    --zerodce_ckpt checkpoints/zerodce_best.pth \
    --aod_ckpt checkpoints/aod_net_best.pth \
    --fuse_ckpt checkpoints/densefuse_finetuned.pth \
    --espcn_ckpt checkpoints/espcn_x4_best.pth \
    --yolo_ckpt yolov8n.pt
```

- Any checkpoint flag can be omitted — that stage is skipped entirely
  (e.g. no thermal camera? Just drop `--thermal_source` and `--fuse_ckpt`;
  dehazing/low-light/SR still run.)
- Press `q` to quit, `s` to save a snapshot, `t` to print the next frame's
  full timing breakdown (handy for live-tuning during a demo).
- `--no_fp16` disables mixed precision if you hit numerical issues.
- `--compile` wraps each model in `torch.compile` — first frame is slow
  (JIT warmup, several seconds), every frame after is faster. Good for a
  sustained demo, bad for a quick one-off test.

### Tuning the gates

`--haze_thresh` (default 0.35) and `--dark_thresh` (default 0.35, now on
a 0-1 scale since everything is GPU tensors) control when each stage
fires. Recalibrate against your own camera/scene using the `haze_score`/
`brightness_score` values printed in the on-screen overlay.

## Step 4: Run in Colab

Use `inference_colab.py` the same way as before (video file in, `.mp4`
out, inline preview) — see the previous conversation's Colab section for
the exact cell-by-cell walkthrough; swap in the new checkpoint flags
(`--zerodce_ckpt`, `--espcn_ckpt`) alongside the existing ones.

## Step 5: Deploy to edge hardware (optional)

```bash
python export_models.py \
    --zerodce_ckpt checkpoints/zerodce_best.pth \
    --aod_ckpt checkpoints/aod_net_best.pth \
    --fuse_ckpt checkpoints/densefuse_finetuned.pth \
    --espcn_ckpt checkpoints/espcn_x4_best.pth \
    --img_size 256 --scale_factor 4
```

Produces `zerodce.onnx`, `aod_net.onnx`, `densefuse_encoder.onnx`,
`densefuse_decoder.onnx`, `espcn.onnx`. Convert each with `trtexec --fp16`
on the target device (see the full commands printed in the script's
docstring).

## What changed vs. the previous version, and why

| Problem reported | Root cause | Fix in this version |
|---|---|---|
| "Soo slow" (8.83 FPS, 113ms/frame) | Haze/brightness scoring ran on full-res CPU arrays every frame; every stage round-tripped through `.cpu().numpy()` + `cv2.resize` between models | Condition scoring now runs on a 64×64 GPU tensor; the entire pipeline (`pipeline_gpu.py`) stays in GPU tensors from capture to final output, FP16 autocast throughout |
| "Not that much good" (barely visible enhancement) | WB+CLAHE can only redistribute existing contrast — it cannot recover detail below the sensor noise floor in a truly dark scene | Zero-DCE++ learns an actual brightening transform via zero-reference losses, trained for exactly this |
| Soft/blotchy output at full resolution | Working at 256px then `cv2.resize` (bicubic) back to 1080p | ESPCN sub-pixel-convolution upscaling — learned, not interpolated, and cheaper than running the full pipeline at 1080p directly |

## Realistic expectations

- On an RTX 3060-class laptop GPU with FP16 and `working_size=256`,
  `scale_factor=4`, expect roughly 25-40+ FPS depending on how many gated
  stages fire on a given frame (a bright, clear frame that skips
  Zero-DCE/AOD-Net/fusion is nearly free beyond the SR step).
- `--compile` typically adds another 1.3-1.8x after the first few frames'
  warmup cost — worth it for a sustained live demo, not for a single
  quick test.
- Always run `benchmark.py` on your actual machine before quoting an FPS
  number in a presentation — hardware, working resolution, and which
  gated branches fire on your specific test footage all move the number
  meaningfully.
