import os
 
from datasets import load_dataset
  
# Create directories
os.makedirs('/storage/hugging_face/cache', exist_ok=True)
os.makedirs('/storage/hugging_face/tmp', exist_ok=True)
os.makedirs('/storage/hugging_face/datasets', exist_ok=True)
 
# Download the complete dataset
realnewslike = load_dataset(
    "allenai/c4",
    "en",
    cache_dir="/storage/hugging_face/cache"
)
# Optionally, save to disk for faster future loading
# realnewslike.save_to_disk('/storage/hugging_face/datasets/c4_en_full')