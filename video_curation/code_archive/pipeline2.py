import csv
from channel_finder import generate_channels_raw

csv_path = r"E:\Om\mllm\brands_list(in).csv"

brand_names = []
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        brand_name = row.get("brand_name")
        if brand_name:
            brand_names.append(brand_name)

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")

output_dir = "data/channels"
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

def process_brand(brand):
    try:
        output_filename = f"{brand}.json"
        output_path = Path(output_dir) / output_filename
        if output_path.exists():
            return
        generate_channels_raw(
            api_key,
            featured_seed=brand,
            output_dir=output_dir,
            output_filename=output_filename
        )
    except Exception as e:
        print(f"Error processing brand '{brand}': {e}")

if __name__ == "__main__":
    with Pool(max(1, cpu_count() - 1)) as pool:
        list(tqdm(pool.imap_unordered(process_brand, brand_names), total=len(brand_names), desc="Processing brands"))
