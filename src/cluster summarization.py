import os
import re
import pandas as pd
from keybert import KeyBERT
import subprocess

# -------------------
# Load clustered data
# -------------------
file_path = "data/intermediate/clustered_with_subtopics.csv"
df = pd.read_csv(file_path)

# -------------------
# Setup output paths
# -------------------
topics_file = "data/intermediate/cluster_topics.csv"
overarching_file = "data/intermediate/cluster_overarching.csv"
os.makedirs(os.path.dirname(topics_file), exist_ok=True)

# -------------------
# Initialize KeyBERT
# -------------------
print("🔑 Loading KeyBERT model...")
kw_model = KeyBERT("all-MiniLM-L6-v2")

# -------------------
# Helper: Boilerplate filter
# -------------------
def strip_boilerplate(text):
    text = text.lower()
    text = re.sub(r"\bdm\b|\bcontact\b|\bemail\b|\bsupport\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------
# Helper: Ollama summarizer
# -------------------
def ollama_summarize(texts, model="mistral", max_samples=50, overarching=False):
    sample_texts = texts[:max_samples]
    joined = "\n".join(strip_boilerplate(t) for t in sample_texts)

    if overarching:
        prompt = f"""
        You are an AI assistant. You will be given subcluster labels. 
        Summarize them into ONE short overarching category (max 5 words).
        Do not include boilerplate terms like DM, contact, email, or support.

        Subclusters:
        {joined}
        """
    else:
        prompt = f"""
        You are an AI assistant analyzing customer support conversations.
        Summarize the following messages into ONE short topic label (max 10 words).
        Be specific and concise, avoid boilerplate like DM, contact, email, or support.

        Conversations:
        {joined}
        """

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        print(f"⚠️ Ollama summarization failed: {e}")
        return None

# -------------------
# Process clusters + subclusters
# -------------------
topic_rows = []
groups = df.groupby(["cluster_id", "subcluster_id"])
subcluster_summaries = {}

for (cid, sid), group in groups:
    print(f"\n📦 Processing cluster {cid}, subcluster {sid} (size={len(group)})")

    texts = group["clean_text"].dropna().astype(str).tolist()
    raw_texts = group["raw_text"].dropna().astype(str).tolist()

    # --- Keywords with KeyBERT
    try:
        keywords = kw_model.extract_keywords(
            " ".join(texts[:200]),
            keyphrase_ngram_range=(1,2),
            stop_words="english",
            top_n=5
        )
        keywords = [kw for kw, score in keywords]
    except Exception as e:
        print(f"⚠️ Keyword extraction failed: {e}")
        keywords = []

    # --- Subcluster summary with Ollama
    summary = None
    if len(raw_texts) >= 10:
        summary = ollama_summarize(raw_texts, model="mistral")

    if not summary:
        summary = ", ".join(keywords) if keywords else "No summary available"

    # Save subcluster row
    topic_rows.append({
        "cluster_id": cid,
        "cluster_label": "",  # placeholder, filled later
        "subcluster_id": sid,
        "subcluster_label": summary,
        "keywords": ", ".join(keywords),
        "size": len(group)
    })

    subcluster_summaries.setdefault(cid, []).append(summary)

# -------------------
# Generate overarching cluster labels
# -------------------
cluster_labels = {}
for cid, summaries in subcluster_summaries.items():
    print(f"\n🌐 Generating overarching summary for cluster {cid}")
    if len(summaries) > 1:  # multiple subclusters → generalize
        cluster_labels[cid] = ollama_summarize(summaries, model="mistral", overarching=True)
    else:  # no subclusters → use the only summary
        cluster_labels[cid] = summaries[0]

# -------------------
# Inject cluster labels into rows
# -------------------
for row in topic_rows:
    row["cluster_label"] = cluster_labels.get(row["cluster_id"], "Unlabeled Cluster")

# -------------------
# Save results
# -------------------
# Detailed file (clusters + subclusters)
topics_df = pd.DataFrame(topic_rows)
topics_df.to_csv(topics_file, index=False)
print(f"\n✅ Detailed cluster topics saved to {topics_file}")

# Parent-level file (overarching only)
overarching_df = pd.DataFrame([
    {"cluster_id": cid, "cluster_label": label}
    for cid, label in cluster_labels.items()
])
overarching_df.to_csv(overarching_file, index=False)
print(f"✅ Overarching cluster labels saved to {overarching_file}")
