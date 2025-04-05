import json
import pandas as pd

def format_as_chat(prompt, response):
    return {
        "messages": [
            {"from": "user", "value": prompt},
            {"from": "assistant", "value": response}
        ]
    }

# === Load the base JSONL file (already in correct format) ===
with open("./medQuad_BioASQ_prompt_response_chat.jsonl", 'r', encoding='utf-8') as f:
    base_data = [json.loads(line) for line in f]

# === Process medredqa_test.csv ===
csv_data = pd.read_csv("./MedRedQA-QEzDvqEq-/data/medredqa/medredqa_test.csv")
formatted_csv_data = []
for _, row in csv_data.iterrows():
    prompt = str(row.get("Title", "")).strip() + str(row.get("Body", "")).strip()
    response = str(row.get("Response", "")).strip()
    if prompt and response:
        formatted_csv_data.append(format_as_chat(prompt, response))

# === Process medredqa+pubmed_test.json ===
with open("./MedRedQA-QEzDvqEq-/data/medredqa+pubmed/medredqa+pubmed_test.json", 'r', encoding='utf-8') as f:
    json_data = json.load(f)

formatted_json_data = []
seen_prompts = set()
for entry in json_data:
    prompt = str(entry.get("question", "")).strip()
    response = str(entry.get("response", "")).strip()

    normalized_prompt = prompt.lower()
    if prompt and response and normalized_prompt not in seen_prompts:
        formatted_json_data.append(format_as_chat(prompt, response))
        seen_prompts.add(normalized_prompt)

# === Process Med/train.json (multiple choice) ===
with open("./Med/train.json", 'r', encoding='utf-8') as f:
    test_data = [json.loads(line) for line in f]

formatted_test_data = []
for entry in test_data:
    prompt = str(entry.get("question", "")).strip()
    response = str(entry.get("exp", "")).strip()
    if not prompt or not response or response.lower() == "none":
        continue
    
    formatted_test_data.append(format_as_chat(prompt, response))

# === Combine everything ===
final_data = base_data + formatted_csv_data + formatted_json_data + formatted_test_data #working, needs cleaning

# formatted_test_data #removed none values



# === Save final combined file ===
output_path = "./final_dataset_all_sources.jsonl"
with open(output_path, 'w', encoding='utf-8') as f:
    for item in final_data:
        json.dump(item, f)
        f.write('\n')

print(f"Saved {len(final_data)} examples to {output_path}")
