import json
import pandas as pd
import re

# === Regex to match Unicode escape sequences ===
unicode_escape_pattern = re.compile(r'\\u[0-9a-fA-F]{4}')

# === Helper function to clean and convert to chat format ===
def clean_and_format(question, answer):
    if not question or not answer:
        return None

    # Remove Unicode escape sequences
    question = unicode_escape_pattern.sub('', str(question).strip())
    answer = unicode_escape_pattern.sub('', str(answer).strip())

    # Remove various "answer lead-in" patterns
    patterns = [
        re.compile(r"(?i)(?:ans(?:wer)?(?:\s+is)?\s*[-:.\)]?\s*'?\(?[a-d]?\)?'?\s*[:,\-]?)\s*(?:i\.e\.)?\s*"),
        re.compile(r"(?i)\b(?:is[.:]?\s+)?'?([a-d])'?\s*i\.e[.,]?\s*"),
        re.compile(r"(?i)\bis[.:]?\s+'?[a-d][.,:]?'?[.,:]?\s"),
        re.compile(r"(?i)\(?\s*option\s+'?[A-D][.,:]?'?[.,:]?\s*\)?")
    ]


    for pattern in patterns:
        answer = re.sub(pattern, '', answer)

    # Normalize whitespace
    question = re.sub(r'\s+', ' ', question)
    answer = re.sub(r'\s+', ' ', answer)

    return {
        "messages": [
            {"from": "user", "value": question},
            {"from": "assistant", "value": answer}
        ]
    }

# === Step 1: Convert training13b JSON dataset ===
with open("/home/liam23/team5-capstone/data/medical-fine-tune/BioASQ/training13b.json", "r", encoding="utf-8") as infile:
    original_json_data = json.load(infile)

converted_json_data = []
for q in original_json_data.get("questions", []):
    question_text = q.get("body", "")
    answers = q.get("ideal_answer", [])
    answer_text = answers[0] if answers else ""
    formatted = clean_and_format(question_text, answer_text)
    if formatted:
        converted_json_data.append(formatted)

# === Step 2: Convert medquad CSV dataset ===
medquad_df = pd.read_csv("/home/liam23/team5-capstone/data/medical-fine-tune/MedQuAD/medquad.csv")
converted_medquad_data = []
for _, row in medquad_df.iterrows():
    formatted = clean_and_format(row.get("question", ""), row.get("answer", ""))
    if formatted:
        converted_medquad_data.append(formatted)

# === Step 3: Convert medredqa CSV datasets ===
medredqa_df = pd.read_csv("/home/liam23/team5-capstone/data/medical-fine-tune/MedRedQA/medredqa/medredqa_train.csv")
converted_medredqa_data = []
for _, row in medredqa_df.iterrows():
    formatted = clean_and_format(row.get("Title", "")+' '+row.get("Body", ""), row.get("Response", ""))
    if formatted:
        converted_medredqa_data.append(formatted)

# === Step 4: Combine and Save All Datasets ===
with open("/home/liam23/team5-capstone/data/medical-fine-tune/MedRedQA/medredqa+pubmed/medredqa+pubmed_train.json", "r", encoding="utf-8") as infile:
    medredqa_pubmed_data = json.load(infile)

converted_medredqa_pubmed_data = []
for entry in medredqa_pubmed_data:
    question_text = entry.get("question", "").strip()
    answer_text = entry.get("response", "").strip()

    if not question_text or not answer_text:
        continue

    question_text = unicode_escape_pattern.sub('', question_text)
    answer_text = unicode_escape_pattern.sub('', answer_text)
    question_text = re.sub(r'\s+', ' ', question_text)
    answer_text = re.sub(r'\s+', ' ', answer_text)

    chat_entry = {
        "messages": [
            {"from": "user", "value": question_text},
            {"from": "assistant", "value": answer_text}
        ]
    }
    converted_medredqa_pubmed_data.append(chat_entry)

# === Step 5: Load train.json ===
with open("/home/liam23/team5-capstone/data/medical-fine-tune/MedMCQA/train.json", "r", encoding="utf-8") as infile:
    data_json3 = [json.loads(line) for line in infile]

converted_med_mcqa_data = []
for entry in data_json3:
    formatted = clean_and_format(entry.get("question", ""), entry.get("exp", ""))
    if formatted:
        converted_med_mcqa_data.append(formatted)

# === Step 6: Combine and Save All Datasets ===
all_data = converted_json_data + converted_medquad_data + converted_medredqa_data + converted_medredqa_pubmed_data + converted_med_mcqa_data
with open("combined_chat_dataset_all7.jsonl", "w", encoding="utf-8") as outfile:
    for entry in all_data:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write("\n")

print(f"Combined dataset saved with {len(all_data)} total entries.")
