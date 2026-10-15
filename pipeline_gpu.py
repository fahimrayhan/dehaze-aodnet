"""
GPU-resident adaptive pipeline: Zero-DCE++ (low-light) -> AOD-Net (haze,
gated) -> DenseFuse (RGB-thermal fusion, gated) -> ESPCN (super-resolution
upscale back to full camera resolution).

Design goals driving every choice here:
  1. Stay on GPU tensors end-to-end. The previous version converted to
     numpy and back between every stage (`.cpu().numpy()` + `cv2.resize`),
     which was a large chunk of its 113ms/frame cost. This version only
     touches the CPU once, right before display/encode.
  2. Process at a SMALL working resolution, then use a learned
     super-resolution network (not bicubic) to get back to full camera
     resolution -- cheaper AND sharper than running every stage at full
     1080p.
  3. Condition-gate every stage so a clear, bright frame skips everything
     except the cheap pass-through + SR step -- don't pay for enhancement
     a frame doesn't need.
  4. FP16 autocast + optional torch.compile for the remaining real
     speedup on RTX 30-series tensor cores.

This module exposes ONE class, AdaptivePipelineGPU, used identically by
inference_live.py (local webcam/video, cv2.imshow) and
inference_colab.py (video file in, video file out, no GUI).
"""

import time
import torch
import torch.nn.functional as F

from models.aod_net import AODNet
from models.densefuse import DenseFuseAutoencoder, DenseFuseInference
from models.zero_dce import ZeroDCEPP
from models.espcn import ESPCN
from utils import gpu_haze_score, gpu_brightness_score


class AdaptivePipelineGPU:
    def __init__(self, zerodce_ckpt=None, aod_ckpt=None, fuse_ckpt=None, espcn_ckpt=None,
                 device=None, working_size=256, scale_factor=4,
                 haze_thresh=0.35, dark_thresh=0.35, use_fp16=True, use_compile=False):
        """
        working_size: resolution enhancement/fusion run at internally.
        scale_factor: ESPCN upscale factor; final output = working_size * scale_factor
                      (should roughly match your camera's native resolution / working_size).
        dark_thresh: on a 0-1 brightness scale (unlike the old 0-255 CPU version).
        use_fp16: autocast all model forward passes in FP16 on CUDA.
        use_compile: wrap each loaded model in torch.compile (first call is
                      slow -- a few seconds of JIT warmup -- then faster).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.working_size = working_size
        self.scale_factor = scale_factor
        self.haze_thresh = haze_thresh
        self.dark_thresh = dark_thresh
        self.use_fp16 = use_fp16 and self.device.type == "cuda"

        self.zerodce = self._load(ZeroDCEPP(), zerodce_ckpt, use_compile)
        self.aod_net = self._load(AODNet(), aod_ckpt, use_compile)

        self.fuser = None
        if fuse_ckpt:
            ae = DenseFuseAutoencoder().to(self.device)
            ae.load_state_dict(torch.load(fuse_ckpt, map_location=self.device))
            ae.eval()
            fuser = DenseFuseInference(ae).to(self.device)
            self.fuser = torch.compile(fuser) if use_compile else fuser

        self.espcn = self._load(ESPCN(scale_factor=scale_factor), espcn_ckpt, use_compile)

        print(f"Pipeline ready on {self.device} | working_size={working_size} "
              f"| scale_factor={scale_factor} -> output {working_size*scale_factor}px "
              f"| fp16={self.use_fp16}")

    def _load(self, model, ckpt_path, use_compile):
        if ckpt_path is None:
            return None
        model = model.to(self.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        model.eval()
        return torch.compile(model) if use_compile else model

    @torch.no_grad()
    def frame_to_tensor(self, frame_rgb_uint8):
        """numpy HxWx3 uint8 RGB -> 1x3xHxW float32 tensor in [0,1] on device."""
        t = torch.from_numpy(frame_rgb_uint8).to(self.device, non_blocking=True)
        t = t.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return t

    @torch.no_grad()
    def tensor_to_frame(self, tensor):
        """1x3xHxW float32 [0,1] on device -> numpy HxWx3 uint8 RGB (CPU)."""
        t = torch.clamp(tensor.squeeze(0), 0, 1) * 255.0
        return t.permute(1, 2, 0).byte().cpu().numpy()

    @torch.no_grad()
    def process(self, rgb_uint8, thermal_uint8=None):
        """
        rgb_uint8: numpy HxWx3 uint8 RGB frame at ANY input resolution.
        thermal_uint8: optional numpy HxW (or HxWx3) uint8 thermal frame.
        Returns (output_uint8_HxWx3, mode_string, timing_dict).
        """
        timings = {}
        t_all = time.time()

        rgb = self.frame_to_tensor(rgb_uint8)
        small = F.interpolate(rgb, size=(self.working_size, self.working_size),
                               mode="bilinear", align_corners=False)

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_fp16):
            h_score = gpu_haze_score(small).item()
            b_score = gpu_brightness_score(small).item()

            mode_parts = []
            result = small

            if self.zerodce is not None and b_score < self.dark_thresh:
                t0 = time.time()
                result, _ = self.zerodce(result)
                timings["zerodce"] = time.time() - t0
                mode_parts.append("lowlight")

            if self.aod_net is not None and h_score > self.haze_thresh:
                t0 = time.time()
                result = self.aod_net(result)
                timings["aod"] = time.time() - t0
                mode_parts.append("dehaze")

            if self.fuser is not None and thermal_uint8 is not None and \
               (b_score < self.dark_thresh or h_score > self.haze_thresh):
                t0 = time.time()
                thm = torch.from_numpy(thermal_uint8).to(self.device).float() / 255.0
                if thm.ndim == 3:
                    thm = thm.mean(dim=2)  # collapse to grayscale if given 3-channel
                thm = thm.unsqueeze(0).unsqueeze(0)
                thm = F.interpolate(thm, size=(self.working_size, self.working_size),
                                     mode="bilinear", align_corners=False)

                ycrcb_y = (0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3])
                fused_y = self.fuser(ycrcb_y, thm)

                # recombine fused luminance with original chroma to keep color
                cr = result[:, 0:1] - ycrcb_y + 0.5
                cb = result[:, 2:3] - ycrcb_y + 0.5
                delta = fused_y - ycrcb_y
                result = torch.cat([
                    torch.clamp(result[:, 0:1] + delta, 0, 1),
                    torch.clamp(result[:, 1:2] + delta, 0, 1),
                    torch.clamp(result[:, 2:3] + delta, 0, 1),
                ], dim=1)
                timings["fusion"] = time.time() - t0
                mode_parts.append("fused")

            if self.espcn is not None:
                t0 = time.time()
                result = self.espcn(result)
                timings["sr"] = time.time() - t0
                mode_parts.append(f"sr_x{self.scale_factor}")
            else:
                result = F.interpolate(result, scale_factor=self.scale_factor,
                                        mode="bicubic", align_corners=False)
                mode_parts.append("bicubic")

        mode = "+".join(mode_parts) if mode_parts else "passthrough"
        timings["total"] = time.time() - t_all
        timings["haze_score"] = h_score
        timings["brightness_score"] = b_score

        return self.tensor_to_frame(result), mode, timings
