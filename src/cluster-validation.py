import pandas as pd
import numpy as np
import os

# -------------------
# Load clustered dataset with subtopics
# -------------------
file_path = "data/intermediate/clustered_with_subtopics.csv"
df = pd.read_csv(file_path)

report_path = "data/intermediate/sanity_report_subclusters.txt"
os.makedirs(os.path.dirname(report_path), exist_ok=True)

# -------------------
# Helper: Gini Index
# -------------------
def gini(array):
    array = np.array(array)
    array = np.sort(array)
    n = array.shape[0]
    cumx = np.cumsum(array)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

# -------------------
# Build Report
# -------------------
lines = []

# Main cluster distribution
cluster_counts = df["cluster_id"].value_counts().sort_index()
lines.append("\n📊 Main Cluster Distribution:")
lines.append(str(cluster_counts))

# Gini index on main clusters
gini_index = gini(cluster_counts.values)
lines.append(f"\n⚖️  Gini index (main clusters): {gini_index:.3f}")

# Subcluster analysis
lines.append("\n🪓 Subcluster Breakdown:")
for cid in sorted(df["cluster_id"].unique()):
    cluster_df = df[df["cluster_id"] == cid]
    if "subcluster_id" in cluster_df.columns and cluster_df["subcluster_id"].nunique() > 1:
        sub_counts = cluster_df["subcluster_id"].value_counts().sort_index()
        lines.append(f"\n--- Cluster {cid} → subdivided into {len(sub_counts)} subclusters:")
        lines.append(str(sub_counts))
        
        # preview per subcluster
        for sid in sub_counts.index:
            lines.append(f"\n   Subcluster {cid}.{sid} samples:")
            sample_rows = cluster_df[cluster_df["subcluster_id"] == sid].sample(
                min(2, len(cluster_df[cluster_df["subcluster_id"] == sid])),
                random_state=42
            )
            for _, row in sample_rows.iterrows():
                lines.append(f"   • {row['raw_text']}")
    else:
        # no subdivision
        lines.append(f"\n--- Cluster {cid} → no subclustering applied")
        sample_rows = cluster_df.sample(min(3, len(cluster_df)), random_state=42)
        for _, row in sample_rows.iterrows():
            lines.append(f"   • {row['raw_text']}")

# -------------------
# Save + Print
# -------------------
report = "\n".join(lines)
print(report)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n✅ Subcluster sanity report saved to {report_path}")
