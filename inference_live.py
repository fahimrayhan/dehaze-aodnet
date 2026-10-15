"""
Real-time inference on a local machine (laptop with RTX 3050/3060/3070,
etc.): RGB (+ optional thermal) camera -> AdaptivePipelineGPU -> (optional)
YOLO detection -> display with FPS counter.

Threaded capture: a separate thread continuously reads frames and keeps
only the latest one (dropping stale frames rather than queuing them), so
the display loop is never stuck processing an old frame -- this trades a
dropped frame for lower latency, which is the right tradeoff for a live
demo.

Usage:
    python inference_live.py --rgb_source 0 --thermal_source 1 \
        --zerodce_ckpt checkpoints/zerodce_best.pth \
        --aod_ckpt checkpoints/aod_net_best.pth \
        --fuse_ckpt checkpoints/densefuse_finetuned.pth \
        --espcn_ckpt checkpoints/espcn_x4_best.pth

    # RGB only (no thermal camera), dehazing + low-light + SR still active:
    python inference_live.py --rgb_source 0 \
        --zerodce_ckpt checkpoints/zerodce_best.pth \
        --aod_ckpt checkpoints/aod_net_best.pth \
        --espcn_ckpt checkpoints/espcn_x4_best.pth

    # Video files instead of live cameras:
    python inference_live.py --rgb_source video.mp4 --thermal_source thermal.mp4 ...

Any checkpoint flag can be omitted -- that stage is then skipped entirely
(e.g. no --fuse_ckpt means thermal fusion never runs, even if a thermal
source is given).
"""

import argparse
import time
import threading
import queue

import cv2
import numpy as np

from pipeline_gpu import AdaptivePipelineGPU


class ThreadedCapture:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return None

    def release(self):
        self.running = False
        self.cap.release()


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
        print(f"Loaded detector from {args.yolo_ckpt}")

    rgb_cap = ThreadedCapture(args.rgb_source)
    thm_cap = ThreadedCapture(args.thermal_source) if args.thermal_source is not None else None

    fps_smooth = 0.0
    prev_time = time.time()
    frame_count = 0

    print("Running. Press 'q' to quit, 's' to save a snapshot, 't' to print timing breakdown.")
    print_timing_next = False

    while True:
        rgb_frame = rgb_cap.read()
        if rgb_frame is None:
            print("RGB stream ended.")
            break
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        thm_frame = None
        if thm_cap is not None:
            raw_thm = thm_cap.read()
            if raw_thm is not None:
                thm_frame = raw_thm if raw_thm.ndim == 2 else cv2.cvtColor(raw_thm, cv2.COLOR_BGR2GRAY)

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

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth else fps
        prev_time = now
        frame_count += 1

        display = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        overlay = f"FPS: {fps_smooth:.1f} | {mode} | haze={timings['haze_score']:.2f} bright={timings['brightness_score']:.2f}"
        cv2.putText(display, overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow("Adaptive RGB-Thermal Pipeline (GPU)", display)

        if print_timing_next:
            per_stage = {k: f"{v*1000:.1f}ms" for k, v in timings.items()
                         if k not in ("haze_score", "brightness_score")}
            print(f"[frame {frame_count}] {per_stage}")
            print_timing_next = False

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, display)
            print(f"Saved {fname}")
        elif key == ord("t"):
            print_timing_next = True

    rgb_cap.release()
    if thm_cap:
        thm_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_source", required=True,
                         help="Camera index (e.g. 0) or video file path for RGB")
    parser.add_argument("--thermal_source", default=None,
                         help="Camera index or video file path for thermal (optional)")
    parser.add_argument("--zerodce_ckpt", default=None, help="Zero-DCE++ checkpoint (low-light)")
    parser.add_argument("--aod_ckpt", default=None, help="AOD-Net checkpoint (dehazing)")
    parser.add_argument("--fuse_ckpt", default=None, help="DenseFuse checkpoint (RGB-thermal fusion)")
    parser.add_argument("--espcn_ckpt", default=None, help="ESPCN checkpoint (super-resolution)")
    parser.add_argument("--yolo_ckpt", default=None, help="Optional YOLOv8 checkpoint for detection overlay")
    parser.add_argument("--working_size", type=int, default=256,
                         help="Internal processing resolution (speed/quality tradeoff)")
    parser.add_argument("--scale_factor", type=int, default=4, choices=[2, 3, 4],
                         help="ESPCN upscale factor; output = working_size * scale_factor")
    parser.add_argument("--haze_thresh", type=float, default=0.35)
    parser.add_argument("--dark_thresh", type=float, default=0.35,
                         help="0-1 scale brightness threshold below which low-light/fusion trigger")
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 autocast")
    parser.add_argument("--compile", action="store_true",
                         help="torch.compile each model (slower first frame, faster after warmup)")
    args = parser.parse_args()

    for attr in ["rgb_source", "thermal_source"]:
        val = getattr(args, attr)
        if val is not None and val.isdigit():
            setattr(args, attr, int(val))

    main(args)
