import os
import argparse
from huggingface_hub import HfApi, create_repo

def upload_model(repo_id: str, model_path: str = "out_glm5/model_inference.pt"):
    """
    Uploads the inference model weight to a Hugging Face Model Repository.
    Creates the repository automatically if it does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} not found. Did you run export_model.py first?")

    api = HfApi()

    print(f"Checking if repository '{repo_id}' exists...")
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except Exception:
        print(f"Creating new model repository '{repo_id}'...")
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    print(f"Uploading {model_path} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model_inference.pt",
        repo_id=repo_id,
        repo_type="model",
    )
    print("Upload complete! ✅")
    print(f"Your model is available at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Nano-GLM weights to Hugging Face Model Hub")
    parser.add_argument("--repo_id", type=str, default="UugaaaBugaaa/nano-glm-120m", help="Target HF repo ID (e.g. username/repo-name)")
    parser.add_argument("--model_path", type=str, default="out_glm5/model_inference.pt", help="Path to the lightweight model checkpoint")
    
    args = parser.parse_args()
    upload_model(args.repo_id, args.model_path)
