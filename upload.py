import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/data/shanhaikang.shk/Block-Attention/qwen_save_dir/checkpoint-1201",
    repo_id="githubear/Qwen-2.5-7B-BLOCK-FT",
    repo_type="model",
)