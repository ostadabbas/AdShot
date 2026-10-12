from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

model = SentenceTransformer("BAAI/bge-m3")

excel_path = r"E:\Om\mllm\videos\@hieneken_filtered.xlsx"
df = pd.read_excel(excel_path)

# Function to convert a DataFrame row to text, similar to notebook
def row_to_text(row):
    return (
        f"Title: {row['title']}\n"
        f"Channel: {row['channelTitle']} ({row['handle']})\n"
        f"Description: {row['description']}"
    )

# Apply to all rows
texts = df.apply(row_to_text, axis=1).tolist()

# Compute embeddings
embeddings = model.encode(
    texts,
    batch_size=8,
    show_progress_bar=True,
    normalize_embeddings=True
)

# Convert embeddings array to list format if needed by df.insert
df["embedding"] = list(embeddings)

print(embeddings)