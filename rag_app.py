import streamlit as st
import openai
import json
import faiss
import numpy as np

# 🔑 Load API key
openai.api_key = "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"  # Replace this!

# 📥 Load embedded citation records
with open("embedded_citation_templates.jsonl", "r") as f:
    records = [json.loads(line) for line in f if "embedding" in line]

if not records:
    st.error("No records with embeddings found.")
    st.stop()

# 🔎 FAISS index
dimension = len(records[0]["embedding"])
index = faiss.IndexFlatL2(dimension)
vectors = np.array([r["embedding"] for r in records]).astype("float32")
index.add(vectors)

# 🔍 Embed query
def embed_text(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# 🔍 Search similar records
def query_rag(query, k=3):
    query_embedding = embed_text(query)
    D, I = index.search(np.array([query_embedding], dtype="float32"), k)
    return [records[i] for i in I[0]]

# 💬 GPT-4 completion
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

# 🚀 Streamlit app UI
st.title("🧾 Citation Style Guide Assistant")

query = st.text_input("Ask your question:", "What is the citation format for a newspaper obituary in Florida?")

if st.button("Get Citation Template"):
    with st.spinner("Searching templates and generating answer..."):
        top_matches = query_rag(query)
        if not top_matches:
            st.error("No matching templates found.")
        else:
            st.subheader("🧠 Suggested Citation Format")
            st.write(generate_response(query, top_matches))

            st.subheader("🔍 Top Matching Templates")
            for i, match in enumerate(top_matches, 1):
                st.markdown(f"**[{i}]** {match['template']}")