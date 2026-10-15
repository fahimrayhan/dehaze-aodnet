"""
Per-stage latency benchmark. Run this FIRST on your actual laptop GPU
before trusting any FPS number -- confirms which stage is your real
bottleneck and whether FP16/compile are helping, rather than guessing.

Usage:
    python benchmark.py --zerodce_ckpt checkpoints/zerodce_best.pth \
        --aod_ckpt checkpoints/aod_net_best.pth \
        --fuse_ckpt checkpoints/densefuse_finetuned.pth \
        --espcn_ckpt checkpoints/espcn_x4_best.pth \
        --n_frames 100
"""

import argparse
import time
import numpy as np
import torch

from pipeline_gpu import AdaptivePipelineGPU


def main(args):
    pipeline = AdaptivePipelineGPU(
        zerodce_ckpt=args.zerodce_ckpt,
        aod_ckpt=args.aod_ckpt,
        fuse_ckpt=args.fuse_ckpt,
        espcn_ckpt=args.espcn_ckpt,
        working_size=args.working_size,
        scale_factor=args.scale_factor,
        use_fp16=not args.no_fp16,
        use_compile=args.compile,
    )

    rgb = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)
    thm = (np.random.rand(1080, 1920) * 255).astype(np.uint8) if args.with_thermal else None
    if args.dark:
        rgb = (rgb.astype(np.float32) * 0.1).astype(np.uint8)  # force low-light/fusion branches

    # warmup (JIT/compile/cudnn autotune -- always exclude from timing)
    for _ in range(5):
        pipeline.process(rgb, thm)
    if pipeline.device.type == "cuda":
        torch.cuda.synchronize()

    all_timings = []
    t0 = time.time()
    for _ in range(args.n_frames):
        _, mode, timings = pipeline.process(rgb, thm)
        all_timings.append(timings)
    if pipeline.device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - t0

    print(f"\n--- Benchmark: {args.n_frames} frames, mode={mode} ---")
    print(f"Overall: {args.n_frames/total_time:.2f} FPS ({total_time/args.n_frames*1000:.1f} ms/frame avg)\n")

    stage_keys = [k for k in all_timings[0] if k not in ("haze_score", "brightness_score", "total")]
    print("Per-stage breakdown (ms, mean over frames where that stage ran):")
    for key in stage_keys + ["total"]:
        vals = [t[key] * 1000 for t in all_timings if key in t]
        if vals:
            print(f"  {key:12s}: {np.mean(vals):6.2f} ms  (ran on {len(vals)}/{args.n_frames} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zerodce_ckpt", default=None)
    parser.add_argument("--aod_ckpt", default=None)
    parser.add_argument("--fuse_ckpt", default=None)
    parser.add_argument("--espcn_ckpt", default=None)
    parser.add_argument("--working_size", type=int, default=256)
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument("--n_frames", type=int, default=100)
    parser.add_argument("--with_thermal", action="store_true", help="Include a thermal frame (tests fusion)")
    parser.add_argument("--dark", action="store_true", help="Darken the test frame to force all gated stages on")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    main(args)
