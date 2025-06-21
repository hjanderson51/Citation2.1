import openai
import json
import streamlit as st
from tqdm import tqdm

openai.api_key = st.secrets["MY_API_KEY"]

# Load your existing data
with open("2.1cleaned_citation_templates.jsonl", "r") as f:
    records = [json.loads(line) for line in f]

# Add embeddings
for r in tqdm(records):
    response = openai.embeddings.create(
        input=r["search_text"],
        model="text-embedding-3-small"
    )
    r["embedding"] = response.data[0].embedding

# Save to a new file
with open("embedded_citation_templates.jsonl", "w") as f:
    for r in records:
        json.dump(r, f)
        f.write("\n")

print(f"✅ Embedded {len(records)} records and saved to embedded_citation_templates.jsonl")