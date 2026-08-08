import sys
import os
import torch
import argparse
from train_glm5 import GLM5Config, GLM5ForCausalLM

# Allow unpickling checkpoints saved when train_glm5 was run as __main__
sys.modules["__main__"].GLM5Config = GLM5Config

def export_inference_model(out_dir="out_glm5", input_ckpt="ckpt_best.pt", output_ckpt="model_inference.pt"):
    """
    Extracts only model weights and configuration from a training checkpoint,
    stripping away optimizer states and iteration metadata to create a lightweight
    file for production deployment.
    """
    in_path = os.path.join(out_dir, input_ckpt)
    if not os.path.exists(in_path):
        in_path = os.path.join(out_dir, "ckpt.pt")
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"No checkpoint found in {out_dir} (checked {input_ckpt} and ckpt.pt)")
    
    out_path = os.path.join(out_dir, output_ckpt)
    print(f"Loading checkpoint from: {in_path}...")
    checkpoint = torch.load(in_path, map_location="cpu", weights_only=False)

    inference_dict = {
        "model": checkpoint["model"],
        "config": checkpoint["config"],
    }

    torch.save(inference_dict, out_path)
    
    in_size_mb = os.path.getsize(in_path) / (1024 * 1024)
    out_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    
    print(f"Success! Stripped checkpoint exported to: {out_path}")
    print(f"Original size: {in_size_mb:.2f} MB")
    print(f"Exported size: {out_size_mb:.2f} MB (Saved {in_size_mb - out_size_mb:.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export inference-only model checkpoint")
    parser.add_argument("--out_dir", type=str, default="out_glm5", help="Output directory")
    parser.add_argument("--input_ckpt", type=str, default="ckpt_best.pt", help="Input checkpoint filename")
    parser.add_argument("--output_ckpt", type=str, default="model_inference.pt", help="Output lightweight checkpoint filename")
    args = parser.parse_args()

    export_inference_model(args.out_dir, args.input_ckpt, args.output_ckpt)
