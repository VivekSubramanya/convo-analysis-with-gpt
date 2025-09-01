import pandas as pd
import numpy as np

# -------------------
# Load clustered dataset
# -------------------
clustered_file = "data/staging/clustered_autoK.csv"
df = pd.read_csv(clustered_file)

# -------------------
# Gini function
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

# -------------------
# Cluster stats
# -------------------
num_clusters = df["cluster_id"].nunique()
print(f"📊 Total clusters: {num_clusters}")

sizes = df["cluster_id"].value_counts().sort_index()
print("\n📊 Cluster sizes:")
print(sizes)

gini_index = gini(sizes.values)
print(f"\n📊 Cluster balance Gini index: {gini_index:.3f}")
if gini_index < 0.2:
    print("✅ Very balanced clusters")
elif gini_index < 0.4:
    print("⚠️ Some imbalance, but acceptable")
else:
    print("🚨 Very lopsided clusters, might need to retry with different k")

# -------------------
# Pick text column (exclude cluster_id)
# -------------------
text_cols = [col for col in df.columns if df[col].dtype == "object" and col != "cluster_id"]

if not text_cols:
    raise ValueError("❌ No text-like columns found in dataset!")
elif len(text_cols) == 1:
    text_col = text_cols[0]
else:
    print("\nAvailable text-like columns:")
    for i, col in enumerate(text_cols):
        print(f"{i}: {col}")
    choice = input("Pick column for cluster sample display (name or number): ").strip()
    if choice.isdigit():
        text_col = text_cols[int(choice)]
    else:
        text_col = choice

print(f"\n📌 Using text column for samples: {text_col}")

# -------------------
# Samples per cluster
# -------------------
print("\n🔍 Sample texts per cluster:")
for cluster in sorted(df["cluster_id"].unique()):
    cluster_df = df[df["cluster_id"] == cluster]
    size = len(cluster_df)
    
    if size < 10:
        print(f"\n=== Cluster {cluster} ({size} rows) ⚠️ Very small cluster! ===")
    else:
        print(f"\n=== Cluster {cluster} ({size} rows) ===")
    
    sample_n = min(5, size)
    for text in cluster_df.sample(sample_n, random_state=42)[text_col]:
        print(f"- {str(text)[:200]}...")
