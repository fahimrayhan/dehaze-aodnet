"""
Colab-compatible inference: reads RGB (+ optional thermal) from VIDEO
FILES (no live camera index, no cv2.imshow -- Colab has neither), writes
the processed result to an .mp4, and shows a throttled inline preview.

Usage (inside a Colab cell):
    !python inference_colab.py \
        --rgb_source /content/data/rgb_video.mp4 \
        --thermal_source /content/data/thermal_video.mp4 \
        --zerodce_ckpt checkpoints/zerodce_best.pth \
        --aod_ckpt checkpoints/aod_net_best.pth \
        --fuse_ckpt checkpoints/densefuse_finetuned.pth \
        --espcn_ckpt checkpoints/espcn_x4_best.pth \
        --output /content/outputs/result.mp4 \
        --preview_every 15
"""

import argparse
import os
import time

import cv2
import numpy as np

from pipeline_gpu import AdaptivePipelineGPU


def main(args):
    pipeline = AdaptivePipelineGPU(
        zerodce_ckpt=args.zerodce_ckpt,
        aod_ckpt=args.aod_ckpt,
        fuse_ckpt=args.fuse_ckpt,
        espcn_ckpt=args.espcn_ckpt,
        working_size=args.working_size,
        scale_factor=args.scale_factor,
        haze_thresh=args.haze_thresh,
        dark_thresh=args.dark_thresh,
        use_fp16=not args.no_fp16,
        use_compile=args.compile,
    )

    detector = None
    if args.yolo_ckpt:
        from ultralytics import YOLO
        detector = YOLO(args.yolo_ckpt)

    rgb_cap = cv2.VideoCapture(args.rgb_source)
    thm_cap = cv2.VideoCapture(args.thermal_source) if args.thermal_source else None

    fps_in = rgb_cap.get(cv2.CAP_PROP_FPS) or 25
    out_size = args.working_size * args.scale_factor

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps_in, (out_size, out_size))

    frame_idx = 0
    t_start = time.time()

    try:
        from IPython.display import display, Image as IPImage, clear_output
        in_notebook = True
    except ImportError:
        in_notebook = False

    while True:
        ret1, rgb_frame = rgb_cap.read()
        if not ret1:
            break
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        thm_frame = None
        if thm_cap is not None:
            ret2, raw_thm = thm_cap.read()
            if ret2:
                thm_frame = cv2.cvtColor(raw_thm, cv2.COLOR_BGR2GRAY)

        result, mode, timings = pipeline.process(rgb_frame, thm_frame)

        if detector is not None:
            det_results = detector(result, verbose=False)[0]
            for box in det_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{detector.names[cls]} {conf:.2f}"
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result, label, (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        display_frame = result.copy()
        overlay = f"frame {frame_idx} | {mode} | haze={timings['haze_score']:.2f} bright={timings['brightness_score']:.2f}"
        cv2.putText(display_frame, overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2)

        writer.write(cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))

        if in_notebook and args.preview_every > 0 and frame_idx % args.preview_every == 0:
            clear_output(wait=True)
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
            display(IPImage(data=buf.tobytes()))
            elapsed = time.time() - t_start
            print(f"Frame {frame_idx} | avg FPS so far: {frame_idx / max(elapsed,1e-6):.1f} "
                  f"| last frame timing: {{k: round(v*1000,1) for k,v in timings.items() if isinstance(v,float)}}")

        frame_idx += 1

    rgb_cap.release()
    if thm_cap:
        thm_cap.release()
    writer.release()

    elapsed = time.time() - t_start
    print(f"Done. Processed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx/max(elapsed,1e-6):.1f} FPS average). Output: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_source", required=True)
    parser.add_argument("--thermal_source", default=None)
    parser.add_argument("--zerodce_ckpt", default=None)
    parser.add_argument("--aod_ckpt", default=None)
    parser.add_argument("--fuse_ckpt", default=None)
    parser.add_argument("--espcn_ckpt", default=None)
    parser.add_argument("--yolo_ckpt", default=None)
    parser.add_argument("--output", default="/content/outputs/result.mp4")
    parser.add_argument("--working_size", type=int, default=256)
    parser.add_argument("--scale_factor", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--haze_thresh", type=float, default=0.35)
    parser.add_argument("--dark_thresh", type=float, default=0.35)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--preview_every", type=int, default=15)
    args = parser.parse_args()
    main(args)
