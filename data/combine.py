import json
import pandas as pd
import re

# Regex pattern to identify and remove unicode escape sequences like \u1234
unicode_escape_pattern = re.compile(r'\\u[0-9a-fA-F]{4}')

# Function to clean and format a question-answer pair
def clean_and_format(question, answer):
    # Skip empty or missing questions/answers
    if not question or not answer:
        return None

    # Remove unicode escape characters and strip whitespace
    question = unicode_escape_pattern.sub('', str(question).strip())
    answer = unicode_escape_pattern.sub('', str(answer).strip())

    # Patterns to clean specific answer structures like "Answer is (a): ..." etc.
    patterns = [
        re.compile(r"(?i)(?:ans(?:wer)?(?:\s+is)?\s*[-:.\)]?\s*'?\(?[a-d]?\)?'?\s*[:,\-]?)\s*(?:i\.e\.)?\s*"),
        re.compile(r"(?i)\b(?:is[.:]?\s+)?'?([a-d])['.]?\s*[.]?\s*i\.e[.,]?\s*"),
        re.compile(r"(?i)\bis[.:]?\s+'?[a-d][.,:]?'?[.,:]?\s"),
        re.compile(r"(?i)\(?\s*option\s+'?[a-d][.,:]?'?[.,:]?\s*\)?"),
        re.compile(r"(?i)'?[a-d]'?,?\s+and\s+'?[a-d]'?,?\s*")
    ]

    # Additional early patterns to clean from the beginning of the answer
    early_patterns = [
        re.compile(r"(?i)\([a-d]\)[.,]?\s*"),          # (a), (b), etc.
        re.compile(r"(?i)[.,]\s"),                    # Just punctuation and space
        re.compile(r"(?i)\bnone of the above\b")      # Remove generic responses
    ]

    # Apply all main cleaning regex patterns to the answer
    for pattern in patterns:
        answer = re.sub(pattern, '', answer)

    # Apply early cleaning patterns to the first 20 characters of the answer
    answer_copy = answer
    head = answer_copy[:20]
    tail = answer_copy[20:]

    for early_pattern in early_patterns:
        head = re.sub(early_pattern, '', head)

    answer = head + tail

    # Normalize whitespace in question and answer
    question = re.sub(r'\s+', ' ', question)
    answer = re.sub(r'\s+', ' ', answer)

    # Skip if the final answer is empty or generic
    if answer.lower() in ["", " ", "none"]:
        return None

    # Return formatted message structure suitable for fine-tuning/chat training
    return {
        "messages": [
            {"from": "user", "value": question},
            {"from": "assistant", "value": answer}
        ]
    }

# === Load and process BioASQ dataset ===
with open("./medical-fine-tune/BioASQ/training13b.json", "r", encoding="utf-8") as infile:
    original_json_data = json.load(infile)

converted_json_data = []
for q in original_json_data.get("questions", []):
    question_text = q.get("body", "")
    answers = q.get("ideal_answer", [])
    answer_text = answers[0] if answers else ""
    formatted = clean_and_format(question_text, answer_text)
    if formatted:
        converted_json_data.append(formatted)

# === Load and process MedQuAD dataset ===
medquad_df = pd.read_csv("./medical-fine-tune/MedQuAD/medquad.csv")
converted_medquad_data = []
for _, row in medquad_df.iterrows():
    formatted = clean_and_format(row.get("question", ""), row.get("answer", ""))
    if formatted:
        converted_medquad_data.append(formatted)

# === Load and process MedRedQA dataset (train/test/val) ===
medredqa_train_df = pd.read_csv("./medical-fine-tune/MedRedQA/medredqa/medredqa_train.csv")
medredqa_test_df = pd.read_csv("./medical-fine-tune/MedRedQA/medredqa/medredqa_test.csv")
medredqa_val_df = pd.read_csv("./medical-fine-tune/MedRedQA/medredqa/medredqa_val.csv")

# Combine all MedRedQA splits into one DataFrame
medredqa_df = pd.concat([medredqa_train_df, medredqa_test_df, medredqa_val_df], ignore_index=True)
converted_medredqa_data = []
for _, row in medredqa_df.iterrows():
    combined_question = row.get("Title", "") + ' ' + row.get("Body", "")
    formatted = clean_and_format(combined_question, row.get("Response", ""))
    if formatted:
        converted_medredqa_data.append(formatted)

# === Load and process MedRedQA+PubMed dataset ===
with open("./medical-fine-tune/MedRedQA/medredqa+pubmed/medredqa+pubmed_train.json", "r", encoding="utf-8") as infile:
    medredqa_pubmed_train_data = json.load(infile)
with open("./medical-fine-tune/MedRedQA/medredqa+pubmed/medredqa+pubmed_test.json", "r", encoding="utf-8") as infile:
    medredqa_pubmed_test_data = json.load(infile)
with open("./medical-fine-tune/MedRedQA/medredqa+pubmed/medredqa+pubmed_val.json", "r", encoding="utf-8") as infile:
    medredqa_pubmed_val_data = json.load(infile)

# Merge all pubmed QA data
medredqa_pubmed_full_data = (
    medredqa_pubmed_train_data +
    medredqa_pubmed_test_data +
    medredqa_pubmed_val_data
)

converted_medredqa_pubmed_data = []
for entry in medredqa_pubmed_full_data:
    question_text = entry.get("question", "").strip()
    answer_text = entry.get("response", "").strip()

    # Skip entries missing question or answer
    if not question_text or not answer_text:
        continue

    # Clean unicode and normalize whitespace
    question_text = unicode_escape_pattern.sub('', question_text)
    answer_text = unicode_escape_pattern.sub('', answer_text)
    question_text = re.sub(r'\s+', ' ', question_text)
    answer_text = re.sub(r'\s+', ' ', answer_text)

    # Add formatted entry
    chat_entry = {
        "messages": [
            {"from": "user", "value": question_text},
            {"from": "assistant", "value": answer_text}
        ]
    }
    converted_medredqa_pubmed_data.append(chat_entry)

# === Load and process MedMCQA dataset (train + dev) ===
with open("./medical-fine-tune/MedMCQA/train.json", "r", encoding="utf-8") as infile:
    med_mcqa_train_data = [json.loads(line) for line in infile]
with open("./medical-fine-tune/MedMCQA/dev.json", "r", encoding="utf-8") as infile:
    med_mcqa_dev_data = [json.loads(line) for line in infile]

med_mcqa_full_data = med_mcqa_train_data + med_mcqa_dev_data

converted_med_mcqa_data = []
for entry in med_mcqa_full_data:
    formatted = clean_and_format(entry.get("question", ""), entry.get("exp", ""))
    if formatted:
        converted_med_mcqa_data.append(formatted)

# === Combine all datasets into one ===
all_data = (
    converted_json_data +
    converted_medquad_data +
    converted_medredqa_data +
    converted_medredqa_pubmed_data +
    converted_med_mcqa_data
)

# Remove duplicate Q-A pairs
unique_entries = []
seen_pairs = set()
for entry in all_data:
    q = entry["messages"][0]["value"]
    a = entry["messages"][1]["value"]
    pair = (q, a)
    if pair not in seen_pairs:
        seen_pairs.add(pair)
        unique_entries.append(entry)

# Save all unique Q-A entries to a JSONL file for model training
with open("combined_medical_prompt_response.jsonl", "w", encoding="utf-8") as outfile:
    for entry in unique_entries:
        json.dump(entry, outfile, ensure_ascii=False)
        outfile.write("\n")

# Final output count
print(f"Combined dataset saved with {len(unique_entries)} unique entries (from {len(all_data)} total).")
