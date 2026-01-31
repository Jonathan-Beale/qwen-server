from huggingface_hub import snapshot_download
import os

MODEL_REPO = "Qwen/Qwen1.5-7B-Chat-AWQ"
TARGET_DIR = "models/Qwen-7B-4bit"

os.makedirs(TARGET_DIR, exist_ok=True)
snapshot_download(
    repo_id=MODEL_REPO,
    local_dir=TARGET_DIR,
    repo_type="model",
    allow_patterns=["*.bin", "*.safetensors", "*.json", "*.txt", "*.model", "*.py", "*.pt", "tokenizer*", "config*", "generation*", "*.gguf", "*.h5"],
    ignore_patterns=["*.tar", "*.zip", "*.md", "*.pdf"]
)
print(f"Model downloaded to {TARGET_DIR}")
