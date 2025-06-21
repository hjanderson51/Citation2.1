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