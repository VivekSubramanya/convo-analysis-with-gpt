# Monolithic MVP: Steps 1 → 3 with Diagnostics + Auto-K Selection
# Dataset: thoughtvector/customer-support-on-twitter

import os
import kaggle
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

# -------------------
# Config
# -------------------
dataset = "thoughtvector/customer-support-on-twitter"
row_limit = 10000  # keep up to 10k rows for embeddings + clustering

# -------------------
# Setup directories
# -------------------
raw_dir = "data"
staging_dir = os.path.join(raw_dir, "staging")
os.makedirs(staging_dir, exist_ok=True)

# -------------------
# Step 1: Download Kaggle Dataset
# -------------------
print(f"⬇️ Downloading dataset: {dataset}")
kaggle.api.dataset_download_files(dataset, path=raw_dir, unzip=True)
print(f"✅ Dataset downloaded and unzipped into {raw_dir}")

# -------------------
# Step 2: Load Raw CSV
# -------------------
csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {raw_dir}")
file_path = os.path.join(raw_dir, csv_files[0])
print(f"\n📂 Using file: {file_path}")

df = pd.read_csv(file_path)

# -------------------
# Step 1.5: Diagnostics (Optional)
# -------------------
run_diag = input("\nDo you want to run diagnostics on this dataset? (y/n): ").strip().lower()

if run_diag == "y":
    print("\n📊 Running dataset diagnostics...\n")
    print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    for col in df.columns:
        col_dtype = df[col].dtype
        num_missing = df[col].isna().sum()
        num_unique = df[col].nunique()
        total = len(df)
        
        if num_unique == total:
            guess = "🆔 Likely an ID column (all values unique)"
        elif col_dtype == "object" and df[col].str.len().mean() > 30:
            guess = "✍️ Likely running text (long average length)"
        elif num_missing / total > 0.3:
            guess = f"⚠️ High nulls ({num_missing}/{total})"
        elif df.duplicated(subset=[col]).sum() > 0.3 * total:
            guess = "⚠️ Lots of duplicates"
        elif col_dtype == "object":
            guess = "📂 Categorical text (short labels)"
        else:
            guess = "🔢 Numeric/Date-like"
        
        print(f"{col} [{col_dtype}] → {guess}")
    
    print("\n✨ Example row (first non-null):")
    sample_row = df.dropna().iloc[0]
    print(sample_row.to_dict())

# -------------------
# Step 2: Column Selection
# -------------------
print("\nAvailable columns in dataset:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

user_input = input(
    "\nEnter columns to keep (comma-separated). "
    "Use names OR numbers (e.g., '2,3,5' or 'text,created_at'): "
)

columns_to_keep = []
for val in [x.strip() for x in user_input.split(",")]:
    if val.isdigit():
        idx = int(val)
        if 0 <= idx < len(df.columns):
            columns_to_keep.append(df.columns[idx])
    else:
        if val in df.columns:
            columns_to_keep.append(val)

# Check for at least one text-like column
text_like_columns = [
    col for col in columns_to_keep
    if (df[col].dtype == "object") or ("text" in col.lower())
]
if not text_like_columns:
    raise ValueError("❌ Must include at least one column with text data")

print(f"\n✅ Keeping columns: {columns_to_keep}")
print(f"✅ Found text column(s): {text_like_columns}")

# -------------------
# Step 2: Clean & Stage
# -------------------
df = df[columns_to_keep]
df = df.dropna(how="all", subset=columns_to_keep)
df = df.drop_duplicates()
df = df.head(row_limit)

staged_file = os.path.join(staging_dir, "cleaned_conversations.csv")
df.to_csv(staged_file, index=False)
print(f"✅ Staged dataset saved to {staged_file}")

# -------------------
# Step 3: Embeddings
# -------------------
text_col = text_like_columns[0]
print(f"\n📌 Using text column: {text_col}")

print("🔨 Generating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df[text_col].tolist(), show_progress_bar=True)

n = len(df)
print(f"\n📊 Dataset size for clustering: {n}")

# -------------------
# Step 3+: Auto-K Selection
# -------------------
run_silhouette = False
silhouette_sample_size = None

if n * 0.1 > 10000:
    print("⚡ Large dataset → Using Elbow (inertia) only, skipping silhouette.")
else:
    run_silhouette = True
    if n * 0.05 >= 5000:
        silhouette_sample_size = int(n * 0.05)
    else:
        silhouette_sample_size = min(n, 5000)
    print(f"✅ Running silhouette on {silhouette_sample_size} sampled rows")

k_values = [5, 10, 15, 20, 30, 50]
results = []

print("\n📊 Evaluating different K values...")
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    inertia = kmeans.inertia_
    sil_score = None
    
    if run_silhouette:
        sample_idx = random.sample(range(n), silhouette_sample_size)
        sample_embeddings = embeddings[sample_idx]
        sample_labels = labels[sample_idx]
        sil_score = silhouette_score(sample_embeddings, sample_labels)
    
    results.append((k, inertia, sil_score))
    print(f"K={k}: Inertia={inertia:.2f}, Silhouette={sil_score if sil_score else 'N/A'}")

# Pick best K
if run_silhouette:
    best_k = max(results, key=lambda x: x[2])[0]
    print(f"\n🏆 Best K (by silhouette): {best_k}")
else:
    best_k = 10
    print(f"\n⚡ No silhouette run. Defaulting to K={best_k} (tune manually if needed).")

# -------------------
# Step 3: Final Clustering
# -------------------
print(f"\n🔗 Running final KMeans with k={best_k}...")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster_id"] = kmeans.fit_predict(embeddings)

clustered_file = os.path.join(staging_dir, "clustered_autoK.csv")
df.to_csv(clustered_file, index=False)
print(f"✅ Clustered dataset saved to {clustered_file}")

# -------------------
# Save evaluation plot
# -------------------
results_df = pd.DataFrame(results, columns=["k", "inertia", "silhouette_score"])
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(results_df["k"], results_df["inertia"], marker="o")
plt.title("Elbow Method (Inertia)")
plt.xlabel("k")
plt.ylabel("Inertia")

plt.subplot(1,2,2)
plt.plot(results_df["k"], results_df["silhouette_score"], marker="o", color="orange")
plt.title("Silhouette Score (sampled)" if run_silhouette else "Silhouette Skipped")
plt.xlabel("k")
plt.ylabel("Score")

plt.tight_layout()
plot_file = os.path.join(staging_dir, "kmeans_autoK_eval.png")
plt.savefig(plot_file)
print(f"📊 Evaluation plot saved to {plot_file}")
