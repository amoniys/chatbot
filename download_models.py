from huggingface_hub import snapshot_download

def download_models():
    print("Downloading deepseek model...")
    snapshot_download(
        repo_id="deepseek-ai/deepseek-coder-1.5b-base",
        local_dir="local_models/deepseek_r1_distill_qwen_1.5b",
        local_dir_use_symlinks=False,
    )
    print("Downloading gpt2 model...")
    snapshot_download(
        repo_id="gpt2",
        local_dir="local_models/gpt2",
        local_dir_use_symlinks=False,
    )
    print("Download complete.")

if __name__ == "__main__":
    download_models()
