import openai
import json
from tqdm import tqdm

openai.api_key = "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"  # Replace this

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