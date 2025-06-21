import openai
import json
import faiss
import numpy as np
from tqdm import tqdm

# 🔑 SET YOUR API KEY
openai.api_key = "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"  # Replace with your actual API key

# 🧠 Embed a single query or text
def embed_text(text):
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None

# 📥 Load pre-embedded .jsonl
with open("embedded_citation_templates.jsonl", "r") as f:
    records = [json.loads(line) for line in f if "embedding" in line]

if not records:
    raise ValueError("No records with embeddings found. Check your JSONL file.")
print(f"✅ Loaded {len(records)} citation templates with embeddings.")

# 🔎 Create FAISS index
dimension = len(records[0]["embedding"])
index = faiss.IndexFlatL2(dimension)
vectors = np.array([r["embedding"] for r in records]).astype("float32")
index.add(vectors)

# 🔍 Search top k matches
def query_rag(query, k=3):
    query_embedding = embed_text(query)
    if query_embedding is None:
        return []
    D, I = index.search(np.array([query_embedding], dtype="float32"), k)
    return [records[i] for i in I[0]]

# 💬 Prompt GPT with context
def generate_response(question, top_matches):
    context = "\n\n".join([entry["template"] for entry in top_matches])
    prompt = f"""
You are a genealogy citation assistant. Based on the following context, provide the most appropriate citation template for the user's question.

Context:
{context}

Question: {question}

Answer:"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in genealogical citation formats."},
            {"role": "user", "content": prompt.strip()}
        ]
    )
    return response.choices[0].message.content.strip()

# 🚀 Example usage
if __name__ == "__main__":
    question = "What is the citation format for a newspaper obituary in Florida?"
    top_matches = query_rag(question)
    if not top_matches:
        print("❌ No matching templates found.")
    else:
        print("\n🔍 Top Matching Templates:")
        for i, m in enumerate(top_matches, 1):
            print(f"\n[{i}] {m['template']}")
        print("\n🧠 GPT-4 Suggested Response:\n")
        print(generate_response(question, top_matches))