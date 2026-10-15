"""
Export trained models to ONNX for further TensorRT conversion (Jetson or
desktop GPU deployment) -- a further speedup on top of what this pipeline
already gets from FP16 + working-resolution processing.

Usage:
    python export_models.py \
        --zerodce_ckpt checkpoints/zerodce_best.pth \
        --aod_ckpt checkpoints/aod_net_best.pth \
        --fuse_ckpt checkpoints/densefuse_finetuned.pth \
        --espcn_ckpt checkpoints/espcn_x4_best.pth \
        --img_size 256 --scale_factor 4

Then, on the target machine (desktop GPU or Jetson, wherever TensorRT/
trtexec is installed):
    trtexec --onnx=zerodce.onnx --saveEngine=zerodce.engine --fp16
    trtexec --onnx=aod_net.onnx --saveEngine=aod_net.engine --fp16
    trtexec --onnx=densefuse_encoder.onnx --saveEngine=densefuse_encoder.engine --fp16
    trtexec --onnx=densefuse_decoder.onnx --saveEngine=densefuse_decoder.engine --fp16
    trtexec --onnx=espcn.onnx --saveEngine=espcn.engine --fp16
"""

import argparse
import torch

from models.aod_net import AODNet
from models.densefuse import DenseFuseAutoencoder
from models.zero_dce import ZeroDCEPP
from models.espcn import ESPCN


def export_simple(model, dummy_input, out_path, input_names, output_names):
    model.eval()
    torch.onnx.export(
        model, dummy_input, out_path,
        input_names=input_names, output_names=output_names,
        dynamic_axes={input_names[0]: {0: "batch"}, output_names[0]: {0: "batch"}},
        opset_version=12,
        dynamo=False,  # use the legacy TorchScript-based exporter (no onnxscript dependency)
    )
    print(f"Exported -> {out_path}")


def export_zerodce(ckpt_path, img_size):
    model = ZeroDCEPP()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    dummy = torch.randn(1, 3, img_size, img_size)
    export_simple(model, dummy, "zerodce.onnx", ["input"], ["enhanced", "curve"])


def export_aod(ckpt_path, img_size):
    model = AODNet()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    dummy = torch.randn(1, 3, img_size, img_size)
    export_simple(model, dummy, "aod_net.onnx", ["input"], ["output"])


def export_densefuse(ckpt_path, img_size):
    ae = DenseFuseAutoencoder()
    ae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    ae.eval()

    dummy_in = torch.randn(1, 1, img_size, img_size)
    export_simple(ae.encoder, dummy_in, "densefuse_encoder.onnx", ["input"], ["features"])

    feat_channels = ae.encoder.out_channels
    dummy_feat = torch.randn(1, feat_channels, img_size, img_size)
    export_simple(ae.decoder, dummy_feat, "densefuse_decoder.onnx", ["features"], ["output"])


def export_espcn(ckpt_path, img_size, scale_factor):
    model = ESPCN(scale_factor=scale_factor)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    dummy = torch.randn(1, 3, img_size, img_size)
    export_simple(model, dummy, "espcn.onnx", ["input"], ["output"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zerodce_ckpt", default=None)
    parser.add_argument("--aod_ckpt", default=None)
    parser.add_argument("--fuse_ckpt", default=None)
    parser.add_argument("--espcn_ckpt", default=None)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--scale_factor", type=int, default=4)
    args = parser.parse_args()

    did_something = False
    if args.zerodce_ckpt:
        export_zerodce(args.zerodce_ckpt, args.img_size); did_something = True
    if args.aod_ckpt:
        export_aod(args.aod_ckpt, args.img_size); did_something = True
    if args.fuse_ckpt:
        export_densefuse(args.fuse_ckpt, args.img_size); did_something = True
    if args.espcn_ckpt:
        export_espcn(args.espcn_ckpt, args.img_size, args.scale_factor); did_something = True

    if not did_something:
        print("Nothing to export -- pass at least one checkpoint flag")
