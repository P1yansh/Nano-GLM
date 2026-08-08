import sys
import os
import time
import contextlib
import random
import numpy as np
import torch
import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download

from train_glm5 import GLM5ForCausalLM, GLM5Config

# Ensure PyTorch unpickler can resolve GLM5Config if saved under __main__
sys.modules["__main__"].GLM5Config = GLM5Config

# Global model state
MODEL_STATE = {
    "model": None,
    "tokenizer": None,
    "device": "cpu",
    "checkpoint_path": None,
    "loaded": False,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager.
    Loads the trained Nano-GLM checkpoint into CPU memory once when the server starts.
    """
    device = "cpu"
    out_dir = "out_glm5"
    
    # Priority order for checkpoint loading
    candidate_ckpts = ["model_inference.pt", "ckpt_best.pt", "ckpt.pt"]
    selected_ckpt = None
    for ckpt_name in candidate_ckpts:
        path = os.path.join(out_dir, ckpt_name)
        if os.path.exists(path):
            selected_ckpt = path
            break

    if selected_ckpt is None:
        print("  [WARNING] No model checkpoint found locally. Attempting to download from Hugging Face...")
        try:
            # You can set the repo_id via environment variable or hardcode it
            repo_id = os.environ.get("HF_REPO_ID", "UugaaaBugaaa/nano-glm-120m")
            selected_ckpt = hf_hub_download(repo_id=repo_id, filename="model_inference.pt", local_dir=out_dir)
            print(f"  [SUCCESS] Downloaded model from {repo_id}")
        except Exception as e:
            print(f"  [ERROR] Failed to download model: {e}")
            print("  [WARNING] Serving dummy state.")
            yield
            return

    print(f"  [STARTUP] Loading Nano-GLM model from {selected_ckpt}...")
    checkpoint = torch.load(selected_ckpt, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = GLM5ForCausalLM(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    MODEL_STATE["model"] = model
    MODEL_STATE["tokenizer"] = tokenizer
    MODEL_STATE["device"] = device
    MODEL_STATE["checkpoint_path"] = selected_ckpt
    MODEL_STATE["loaded"] = True

    print(f"  [STARTUP] Model successfully loaded on {device}!")
    yield
    print("  [SHUTDOWN] Cleaning up model state...")
    MODEL_STATE.clear()

app = FastAPI(
    title="Nano-GLM (GLM-5.2 Baby 120M) API",
    description="FastAPI Backend for serving the lightweight from-scratch GLM-5.2 MoE model.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str = Field(default="In conclusion,", description="Input prompt for text generation")
    max_new_tokens: int = Field(default=100, ge=1, le=500, description="Maximum new tokens to generate")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Sampling temperature (0.0 = greedy)")
    top_k: int = Field(default=50, ge=0, le=200, description="Top-k sampling threshold (0 = disable)")
    seed: int = Field(default=42, description="Random seed for reproducibility (-1 for random)")

class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    num_tokens: int
    inference_time_sec: float

@app.get("/info")
def read_root():
    return {
        "title": "Nano-GLM API",
        "status": "online" if MODEL_STATE["loaded"] else "model_not_loaded",
        "docs_url": "/docs",
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if MODEL_STATE["loaded"] else "degraded",
        "model_loaded": MODEL_STATE["loaded"],
        "checkpoint": MODEL_STATE["checkpoint_path"],
        "device": MODEL_STATE["device"],
    }

@app.post("/generate", response_model=GenerateResponse)
def generate_text(req: GenerateRequest):
    if not MODEL_STATE["loaded"]:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first or export a checkpoint.")

    model = MODEL_STATE["model"]
    enc = MODEL_STATE["tokenizer"]
    device = MODEL_STATE["device"]

    # Enforce seed reproducibility if specified
    if req.seed != -1:
        torch.manual_seed(req.seed)
        np.random.seed(req.seed)
        random.seed(req.seed)

    t0 = time.time()
    
    # Tokenize input prompt
    tokens = enc.encode(req.prompt)
    if len(tokens) == 0:
        tokens = [enc.eot_token]
    
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    # Generation context
    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k if req.top_k > 0 else None,
        )

    t1 = time.time()
    inference_time = t1 - t0

    generated_tokens = output[0].tolist()
    generated_text = enc.decode(generated_tokens)

    return GenerateResponse(
        prompt=req.prompt,
        generated_text=generated_text,
        num_tokens=len(generated_tokens),
        inference_time_sec=round(inference_time, 4),
    )

# Mount static web frontend at root (if static directory exists)
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
