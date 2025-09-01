# Monolithic MVP: Steps 1 → 3 with Diagnostics + Auto-K (Silhouette + Gini)
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
row_limit = 10000  # cap rows for clustering

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
        elif col_dtype == "object" and df[col].astype(str).str.len().mean() > 30:
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

df = df[columns_to_keep]
df = df.dropna(how="all", subset=columns_to_keep)
df = df.drop_duplicates()

# Random sample instead of head
if len(df) > row_limit:
    df = df.sample(n=row_limit, random_state=42)
    print(f"🎲 Randomly sampled {row_limit} rows from dataset")
else:
    print(f"📉 Dataset smaller than {row_limit}, using all rows")

staged_file = os.path.join(staging_dir, "cleaned_conversations.csv")
df.to_csv(staged_file, index=False)
print(f"✅ Staged dataset saved to {staged_file}")

# -------------------
# Step 3: Choose Text Column for Clustering
# -------------------
print("\nWhich column do you want to use for clustering (must be text-like)?")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

text_choice = input("Enter column name or number: ").strip()
if text_choice.isdigit():
    text_col = df.columns[int(text_choice)]
else:
    text_col = text_choice

if text_col not in df.columns:
    raise ValueError("❌ Invalid column choice.")

print(f"\n📌 Using text column: {text_col}")

# -------------------
# Step 3: Embeddings
# -------------------
print("🔨 Generating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df[text_col].astype(str).tolist(), show_progress_bar=True)

n = len(df)
print(f"\n📊 Dataset size for clustering: {n}")

# -------------------
# Step 3+: Auto-K Selection (Inertia + Silhouette + Gini)
# -------------------
def gini(array):
    array = np.array(array)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = array + 1e-10
    array = np.sort(array)
    n = array.shape[0]
    cumx = np.cumsum(array)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

k_range = range(5, 26)
inertias, silhouettes, ginis = [], [], []

print("\n📊 Evaluating k values...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    inertia = kmeans.inertia_
    inertias.append(inertia)

    sil = silhouette_score(embeddings, labels, sample_size=2000, random_state=42)
    silhouettes.append(sil)

    g = gini(pd.Series(labels).value_counts().values)
    ginis.append(g)

    print(f"k={k:2d}: inertia={inertia:.2f}, silhouette={sil:.3f}, gini={g:.3f}")

results = pd.DataFrame({
    "k": list(k_range),
    "inertia": inertias,
    "silhouette": silhouettes,
    "gini": ginis
})

valid = results[results["gini"] < 0.35]
if not valid.empty:
    best_k = int(valid.sort_values(by="silhouette", ascending=False).iloc[0]["k"])
    print(f"\n🏆 Suggested best k: {best_k} (good silhouette + balanced clusters)")
else:
    best_k = int(results.sort_values(by="silhouette", ascending=False).iloc[0]["k"])
    print(f"\n⚠️ No k had gini < 0.35. Suggested best k by silhouette only: {best_k}")

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
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.plot(results["k"], results["inertia"], marker="o")
plt.title("Elbow Method (Inertia)")
plt.xlabel("k"); plt.ylabel("Inertia")

plt.subplot(1,3,2)
plt.plot(results["k"], results["silhouette"], marker="o", color="orange")
plt.title("Silhouette Score")
plt.xlabel("k"); plt.ylabel("Score")

plt.subplot(1,3,3)
plt.plot(results["k"], results["gini"], marker="o", color="red")
plt.title("Cluster Balance (Gini Index)")
plt.xlabel("k"); plt.ylabel("Gini (lower is better)")

plt.tight_layout()
plot_file = os.path.join(staging_dir, "kmeans_autoK_eval.png")
plt.savefig(plot_file)
print(f"📊 Evaluation plot saved to {plot_file}")
