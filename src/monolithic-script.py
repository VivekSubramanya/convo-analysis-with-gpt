# Monolith: Kaggle ingest → pick file → diagnostics (opt) → select columns → clean (raw+clean)
# → VADER sentiment → SBERT embeddings → auto-K → optional subclustering
# → Topic gen via Ollama (mistral) → Save fact + 3 dimension/summary files
# Folders: data/raw, data/intermediate (handles nested CSVs in raw)

import os, re, json, subprocess, random
import kaggle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT

# -------------------
# Config
# -------------------
dataset = "thoughtvector/customer-support-on-twitter"
row_limit = 10000                    # cap rows for clustering pass
raw_dir = os.path.join("data","raw")
intermediate_dir = os.path.join("data","intermediate")
ollama_model = "mistral"
random.seed(42)
np.random.seed(42)

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(intermediate_dir, exist_ok=True)

# -------------------
# Step 1: Download Kaggle Dataset
# -------------------
print(f"⬇️ Downloading dataset: {dataset}")
kaggle.api.dataset_download_files(dataset, path=raw_dir, unzip=True)
print(f"✅ Dataset downloaded and unzipped into {raw_dir}")

# -------------------
# Step 2: Let User Pick CSV (recursive)
# -------------------
csv_files = []
for root, _, files in os.walk(raw_dir):
    for f in files:
        if f.endswith(".csv"):
            rel = os.path.relpath(os.path.join(root, f), raw_dir)
            csv_files.append(rel)
if not csv_files:
    raise FileNotFoundError(f"No CSV files found under {raw_dir}")

print("\n📂 CSV files available (including subfolders):")
for i, f in enumerate(csv_files):
    size_mb = os.path.getsize(os.path.join(raw_dir, f)) / (1024 * 1024)
    print(f"{i}: {f} ({size_mb:.2f} MB)")
choice = input("\nEnter the number of the file you want to use: ").strip()
if not choice.isdigit() or int(choice) >= len(csv_files):
    raise ValueError("❌ Invalid choice.")
file_path = os.path.join(raw_dir, csv_files[int(choice)])
print(f"\n✅ Using file: {file_path}")

# -------------------
# Step 3: Load CSV
# -------------------
df = pd.read_csv(file_path)

# -------------------
# Step 1.5: Diagnostics (Optional)
# -------------------
run_diag = input("\nRun diagnostics on this dataset? (y/n): ").strip().lower()
if run_diag == "y":
    print("\n📊 Diagnostics\n")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols\n")
    for col in df.columns:
        col_dtype = df[col].dtype
        num_missing = df[col].isna().sum()
        num_unique = df[col].nunique()
        total = len(df)
        if num_unique == total:
            guess = "🆔 Likely an ID column"
        elif col_dtype == "object" and df[col].astype(str).str.len().mean() > 30:
            guess = "✍️ Likely running text"
        elif num_missing / total > 0.3:
            guess = f"⚠️ High nulls ({num_missing}/{total})"
        elif df.duplicated(subset=[col]).sum() > 0.3 * total:
            guess = "⚠️ Many duplicates"
        elif col_dtype == "object":
            guess = "📂 Categorical text"
        else:
            guess = "🔢 Numeric/Date-like"
        print(f"{col} [{col_dtype}] → {guess}")
    print("\n✨ Example row:")
    print(df.dropna().iloc[0].to_dict())

# -------------------
# Step 4: Column Selection
# -------------------
print("\nAvailable columns:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")
user_input = input(
    "\nEnter columns to keep (comma-separated). "
    "Use names OR numbers (e.g., '2,3,5' or 'text,created_at'): "
)
columns_to_keep = []
for val in [x.strip() for x in user_input.split(",")]:
    if val.isdigit():
        idx = int(val); 
        if 0 <= idx < len(df.columns): columns_to_keep.append(df.columns[idx])
    elif val in df.columns:
        columns_to_keep.append(val)
df = df[columns_to_keep].dropna(how="all", subset=columns_to_keep).drop_duplicates()

if len(df) > row_limit:
    df = df.sample(n=row_limit, random_state=42)
    print(f"🎲 Sampled {row_limit} rows")
else:
    print(f"📉 Using all {len(df)} rows")

# -------------------
# Step 5: Choose Text Column
# -------------------
print("\nWhich column is the conversation text (for clustering)?")
for i, col in enumerate(df.columns): print(f"{i}: {col}")
text_choice = input("Enter column name or number: ").strip()
text_col = df.columns[int(text_choice)] if text_choice.isdigit() else text_choice
if text_col not in df.columns:
    raise ValueError("❌ Invalid text column choice.")
print(f"📌 Using text column: {text_col}")

# Keep raw + cleaned
df["raw_text"] = df[text_col]

def clean_text(txt: str) -> str:
    txt = str(txt).lower()
    txt = re.sub(r"@\w+","#USER", txt)   # keep a placeholder but not identity
    txt = re.sub(r"#\w+","", txt)
    txt = re.sub(r"http\S+","", txt)
    txt = re.sub(r"\s+"," ", txt).strip()
    return txt
df["clean_text"] = df[text_col].apply(clean_text)

# Save stage
staged_file = os.path.join(intermediate_dir, "cleaned_conversations.csv")
df.to_csv(staged_file, index=False)
print(f"✅ Staged → {staged_file}")

# -------------------
# Step S: VADER Sentiment (fast)
# -------------------
print("\n🧠 VADER sentiment...")
analyzer = SentimentIntensityAnalyzer()
scores = df["clean_text"].fillna("").astype(str).apply(analyzer.polarity_scores)
df["sentiment_compound"] = scores.apply(lambda s: s["compound"])
def _lab(c): 
    return "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
df["sentiment"] = df["sentiment_compound"].apply(_lab)
df["sentiment_num"] = df["sentiment"].map({"negative":-1, "neutral":0, "positive":1})
print("✅ Sentiment columns added")

# -------------------
# Step 6: Embeddings
# -------------------
print("🔨 Embeddings (SBERT all-MiniLM-L6-v2)...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sbert.encode(df["clean_text"].astype(str).tolist(), show_progress_bar=True)

# -------------------
# Step 7: Auto-K (Elbow + Silhouette + Gini)
# -------------------
def gini(arr):
    arr = np.sort(np.array(arr))
    n = arr.shape[0]; cumx = np.cumsum(arr)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

n = len(df)
k_range = range(5, 26)
inertias, silhouettes, ginis = [], [], []
print("\n📊 Evaluating k...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(embeddings, labels, sample_size=min(2000, n), random_state=42))
    ginis.append(gini(pd.Series(labels).value_counts().values))
    print(f"k={k:2d}: inertia={inertias[-1]:.1f}, silhouette={silhouettes[-1]:.3f}, gini={ginis[-1]:.3f}")

results = pd.DataFrame({"k": list(k_range), "inertia": inertias, "silhouette": silhouettes, "gini": ginis})
valid = results[results["gini"] < 0.35]
best_k = int(valid.sort_values(by="silhouette", ascending=False).iloc[0]["k"]) if not valid.empty else int(results.sort_values(by="silhouette", ascending=False).iloc[0]["k"])
print(f"\n🏆 Suggested k = {best_k}")

# -------------------
# Step 8: Final KMeans
# -------------------
print(f"\n🔗 Final KMeans (k={best_k})...")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster_id"] = kmeans.fit_predict(embeddings)

# -------------------
# Step 9: Subclustering (only for vague clusters)
# -------------------
print("\n🪓 Checking vague clusters for subclustering...")
def keyword_diversity(texts, top_n=10):
    vec = TfidfVectorizer(stop_words="english", max_features=50, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()
    return len(terms[:top_n])

df["subcluster_id"] = -1
vague = []
for cid in sorted(df["cluster_id"].unique()):
    texts = df.loc[df["cluster_id"]==cid, "clean_text"].tolist()
    if len(texts) > 500 or keyword_diversity(texts) > 7:
        vague.append(cid)
print(f"⚠️ Vague clusters: {vague}")

for cid in vague:
    sub_df = df[df["cluster_id"]==cid]
    emb = sbert.encode(sub_df["clean_text"].tolist(), show_progress_bar=False)
    best_sub_k, best_score = 2, -1
    for kk in range(2,6):
        km = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(emb)
        sc = silhouette_score(emb, km.labels_)
        if sc > best_score: best_sub_k, best_score = kk, sc
    km = KMeans(n_clusters=best_sub_k, random_state=42, n_init=10).fit(emb)
    df.loc[sub_df.index, "subcluster_id"] = km.labels_

# -------------------
# Step 10: Save FACT (no topic names, only sentiment + IDs)
# -------------------
fact_path = os.path.join(intermediate_dir, "clustered_with_subtopics.csv")
df.to_csv(fact_path, index=False)
print(f"✅ FACT saved → {fact_path}")

# -------------------
# Step 11: Sentiment Rollup per (cluster, subcluster)
# -------------------
rollup = df.groupby(["cluster_id","subcluster_id","sentiment"]).size().unstack(fill_value=0)
rollup = rollup.rename(columns={"negative":"neg_count","neutral":"neu_count","positive":"pos_count"}).reset_index()
rollup["total"] = rollup[["neg_count","neu_count","pos_count"]].sum(axis=1)
for c in ["neg_count","neu_count","pos_count"]:
    rollup[c.replace("_count","_pct")] = (rollup[c]/rollup["total"]).replace([np.inf,np.nan],0).round(4)
avg_sent = df.groupby(["cluster_id","subcluster_id"])["sentiment_num"].mean().reset_index().rename(columns={"sentiment_num":"sentiment_avg_num"})
rollup = rollup.merge(avg_sent, on=["cluster_id","subcluster_id"], how="left")
rollup_path = os.path.join(intermediate_dir, "cluster_sentiment_summary.csv")
rollup.to_csv(rollup_path, index=False)
print(f"✅ Sentiment summary saved → {rollup_path}")

# -------------------
# Step 12: Topic Generation (KeyBERT + Ollama for labels)
# -------------------
print("\n🧾 Generating labels (subclusters + overarching)...")

kw_model = KeyBERT("all-MiniLM-L6-v2")

def strip_boilerplate(text):
    text = text.lower()
    text = re.sub(r"\bdm\b|\bcontact\b|\bemail\b|\bsupport\b", "", text)
    text = re.sub(r"\s+"," ", text).strip()
    return text

def ollama_summarize(texts, model=ollama_model, max_samples=50, overarching=False):
    if not texts: return None
    sample = texts[:max_samples]
    joined = "\n".join(strip_boilerplate(t) for t in sample)
    if overarching:
        prompt = f"""
You will be given subcluster labels. Return ONE short overarching category (max 5 words).
Do NOT use boilerplate like DM, contact, email, support.

Subclusters:
{joined}
""".strip()
    else:
        prompt = f"""
Summarize the following customer support messages into ONE short topic label (max 10 words).
Be specific and concise. Avoid boilerplate like DM, contact, email, support.

Messages:
{joined}
""".strip()

    try:
        res = subprocess.run(["ollama","run",model], input=prompt.encode("utf-8"),
                             capture_output=True, check=True)
        return res.stdout.decode("utf-8").strip()
    except Exception as e:
        print(f"⚠️ Ollama summarization failed: {e}")
        return None

# group by cluster+subcluster for labels
topic_rows = []
sub_summaries = {}
for (cid, sid), grp in df.groupby(["cluster_id","subcluster_id"]):
    texts = grp["clean_text"].dropna().astype(str).tolist()
    raws  = grp["raw_text"].dropna().astype(str).tolist()

    # keywords (for Tableau tags/search)
    try:
        kws = kw_model.extract_keywords(" ".join(texts[:200]),
                                        keyphrase_ngram_range=(1,2),
                                        stop_words="english", top_n=5)
        keywords = ", ".join([k for k,_ in kws])
    except Exception as e:
        print(f"⚠️ KeyBERT failed on {cid}.{sid}: {e}")
        keywords = ""

    # subcluster label via Ollama (fallback to keywords)
    sub_label = ollama_summarize(raws, model=ollama_model) if len(raws) >= 10 else None
    if not sub_label:
        sub_label = keywords if keywords else "Unlabeled"

    topic_rows.append({
        "cluster_id": cid,
        "cluster_label": "",   # fill later
        "subcluster_id": sid,
        "subcluster_label": sub_label,
        "keywords": keywords,
        "size": len(grp)
    })
    sub_summaries.setdefault(cid, []).append(sub_label)

# overarching cluster label per cluster
cluster_labels = {}
for cid, subs in sub_summaries.items():
    if len(subs) > 1:
        cluster_labels[cid] = ollama_summarize(subs, model=ollama_model, overarching=True) or "Unlabeled Cluster"
    else:
        cluster_labels[cid] = subs[0] if subs else "Unlabeled Cluster"

# inject cluster labels into topic rows
for r in topic_rows:
    r["cluster_label"] = cluster_labels.get(r["cluster_id"], "Unlabeled Cluster")

# save topics (detail)
topics_df = pd.DataFrame(topic_rows)
topics_path = os.path.join(intermediate_dir, "cluster_topics.csv")
topics_df.to_csv(topics_path, index=False)
print(f"✅ Cluster topics saved → {topics_path}")

# save overarching (parents only)
overarching_df = pd.DataFrame([{"cluster_id": cid, "cluster_label": lab} for cid, lab in cluster_labels.items()])
over_path = os.path.join(intermediate_dir, "cluster_overarching.csv")
overarching_df.to_csv(over_path, index=False)
print(f"✅ Overarching labels saved → {over_path}")

# -------------------
# Step 13: Save K-eval Plot
# -------------------
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.plot(results["k"], results["inertia"], marker="o"); plt.title("Elbow (Inertia)"); plt.xlabel("k"); plt.ylabel("Inertia")
plt.subplot(1,3,2); plt.plot(results["k"], results["silhouette"], marker="o", color="orange"); plt.title("Silhouette"); plt.xlabel("k"); plt.ylabel("Score")
plt.subplot(1,3,3); plt.plot(results["k"], results["gini"], marker="o", color="red"); plt.title("Cluster Balance (Gini)"); plt.xlabel("k"); plt.ylabel("Gini")
plt.tight_layout()
plot_file = os.path.join(intermediate_dir, "kmeans_autoK_eval.png")
plt.savefig(plot_file)
print(f"📊 Evaluation plot saved → {plot_file}")
