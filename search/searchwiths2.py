!pip install singlestoredb openai
# ==========================================================
# 1. Setup
# ==========================================================
import time
import numpy as np
import singlestoredb as s2
from openai import OpenAI

# 🔑 Hardcode OpenAI API key here (project key)
OPENAI_API_KEY = <OPEN_AI_Key>

client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Connect to SingleStore
conn = s2.connect("<Helios connection string>")
print("✅ Connected to SingleStore Helios")
cur = conn.cursor()

# ==========================================================
# 2. Update amazon_reviews table with embeddings (100000 docs)
# ==========================================================

# Add embedding column if not exists
try:
    cur.execute("ALTER TABLE amazon_reviews ADD COLUMN embedding VECTOR(1536)")
    print("✅ Added embedding column")
except Exception:
    print("ℹ️ Embedding column already exists, skipping")

# Add FULLTEXT index if not exists
try:
    cur.execute("ALTER TABLE amazon_reviews ADD FULLTEXT INDEX ft_text (Text)")
    print("✅ FULLTEXT index created on Text")
except Exception:
    print("ℹ️ FULLTEXT index already exists, skipping")

# Function to create embeddings (fallback to random if API fails)
def embed(text: str):
    if text is None or str(text).strip() == "":
        return None
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ OpenAI error: {e} → Using mock embeddings instead")
        return np.random.rand(1536).tolist()

# Fetch 100 rows without embeddings
cur.execute("SELECT Id, Text FROM amazon_reviews WHERE embedding IS NULL LIMIT 100")
rows = cur.fetchall()

# Update table with embeddings
for rid, text in rows:
    vec = embed(text)
    if vec:
        vec_json = str(vec).replace("'", '"')  # JSON format
        cur.execute("""
            UPDATE amazon_reviews
            SET embedding = %s
            WHERE Id = %s
        """, (vec_json, rid))

conn.commit()
print(f"✅ Updated {len(rows)} rows with embeddings")


# ==========================================================
# 3. Full-Text Search
# ==========================================================
def full_text_search(query, topk=5):
    start = time.time()
    cur.execute("""
        SELECT Id, Summary, MATCH(Text) AGAINST (%s) AS score
        FROM amazon_reviews
        WHERE MATCH(Text) AGAINST (%s)
        ORDER BY score DESC
        LIMIT %s
    """, (query, query, topk))
    results = cur.fetchall()
    print(f"🔎 Full-text search in {time.time() - start:.4f} sec")
    return results

# Example queries
print(full_text_search("dog food"))
print(full_text_search("cough medicine"))


# ==========================================================
# 4. Vector Search
# ==========================================================
def vector_search(query, topk=5):
    qvec = embed(query)
    vec_json = str(qvec).replace("'", '"')  # JSON array string
    start = time.time()
    cur.execute("""
        SELECT Id, Summary,
               DOT_PRODUCT(embedding, %s) AS score
        FROM amazon_reviews
        WHERE embedding IS NOT NULL
        ORDER BY score DESC
        LIMIT %s
    """, (vec_json, topk))
    results = cur.fetchall()
    print(f"🔎 Vector search in {time.time() - start:.4f} sec")
    return results

# Example queries
print(vector_search("healthy pet food"))
print(vector_search("candy"))


def hybrid_search(query, topk=5, text_weight=0.5, vector_weight=0.5):
    qvec = embed(query)
    vec_json = str(qvec).replace("'", '"')
    start = time.time()
    cur.execute(f"""
        SELECT Id, Summary,
               COALESCE(MATCH(Text) AGAINST (%s), 0) AS text_score,
               COALESCE(DOT_PRODUCT(embedding, %s), 0) AS vector_score,
               (%s * COALESCE(MATCH(Text) AGAINST (%s), 0) +
                %s * COALESCE(DOT_PRODUCT(embedding, %s), 0)) AS hybrid_score
        FROM amazon_reviews
        WHERE embedding IS NOT NULL
        ORDER BY hybrid_score DESC
        LIMIT %s
    """, (query, vec_json, text_weight, query, vector_weight, vec_json, topk))
    results = cur.fetchall()
    print(f"⚡ Hybrid search (fast) in {time.time() - start:.4f} sec")
    return results

# Example
print(hybrid_search("dog food"))
print(hybrid_search("cough medicine"))

# ==========================================================
# 6. Summary Comparison (Full-text vs Vector vs Hybrid)
# ==========================================================
import pandas as pd

def run_comparison(query, topk=3):
    results = []

    # Full-text
    start = time.time()
    ft = full_text_search(query, topk)
    ft_time = time.time() - start
    for r in ft:
        results.append(("Full-Text", query, ft_time, r[0], r[1], r[2]))

    # Vector
    start = time.time()
    vs = vector_search(query, topk)
    vs_time = time.time() - start
    for r in vs:
        results.append(("Vector", query, vs_time, r[0], r[1], r[2]))

    # Hybrid
    start = time.time()
    hs = hybrid_search(query, topk)
    hs_time = time.time() - start
    for r in hs:
        results.append(("Hybrid", query, hs_time, r[0], r[1], r[2]))

    # Build DataFrame
    df = pd.DataFrame(results, columns=[
        "Search_Type", "Query", "Execution_Time_sec", "Id", "Summary", "Score"
    ])
    return df

# Example: compare all three for "dog food"
comparison_df = run_comparison("dog food", topk=3)
comparison_df

