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