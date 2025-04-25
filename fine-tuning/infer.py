from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# define prompts
prompts = [
    "What are the common symptoms of diabetes?",
    "How is high blood pressure diagnosed and treated?",
    "What are the side effects of taking ibuprofen regularly?",
    "How can someone tell the difference between a cold and the flu?",
    "Explain how insulin works in the body.",
    "What is the recommended treatment for strep throat?",
    "Can anxiety cause chest pain?",
    "What is the difference between a heart attack and cardiac arrest?",
    "How is asthma managed in children?",
    "What are the signs of dehydration in elderly patients?",
    "What vaccines are recommended for adults over 50?",
    "How does intermittent fasting affect metabolism?",
    "What are early warning signs of Alzheimer’s disease?",
    "What is the purpose of a colonoscopy?",
    "How do antidepressants work in the brain?",
    "What are the long-term effects of smoking on the lungs?",
    "When should someone get tested for STDs?",
    "What is the difference between type 1 and type 2 diabetes?",
    "How does high cholesterol contribute to heart disease?",
    "What should a person do after experiencing a minor concussion?",
    "What causes chronic fatigue syndrome?",
    "What are common side effects of chemotherapy?",
    "How is Lyme disease diagnosed and treated?",
    "What dietary changes help manage acid reflux?",
    "What does it mean when a mole changes shape or color?",
    "What is the difference between a CT scan and an MRI?",
    "What are some non-opioid options for managing chronic pain?",
    "How does sleep deprivation affect the immune system?",
    "When should a child be taken to the ER for a fever?",
    "What are the most effective ways to prevent the spread of COVID-19?",
    "How can someone manage type 2 diabetes with diet alone?",
    "What does an abnormal EKG result mean?",
    "What are the symptoms and treatments for hypothyroidism?",
    "What’s the safest way to taper off antidepressants?",
    "What are some early signs of skin cancer?",
    "How is a UTI diagnosed and treated in men vs. women?",
    "What are the risks of untreated sleep apnea?",
    "What lifestyle changes help manage hypertension?",
    "What is the role of the liver in detoxification?",
    "What causes low white blood cell count?"
]

# model paths
fine_tuned_path = "Liamayyy/gemma-2-2b-medical-v2"
base_model_path = "google/gemma-2-2b"

# load models and tokenizers
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_path, torch_dtype=torch.float16, device_map="auto")

base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto")

# output path
output_file_path = "model_comparison_output.txt"

with open(output_file_path, "w", encoding="utf-8") as f:
    for i, prompt in enumerate(prompts, 1):
        ft_inputs = fine_tuned_tokenizer(prompt, return_tensors="pt").to(fine_tuned_model.device)
        base_inputs = base_tokenizer(prompt, return_tensors="pt").to(base_model.device)

        with torch.no_grad():
            ft_output = fine_tuned_model.generate(
                **ft_inputs,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.95,
                temperature=0.8
            )
            base_output = base_model.generate(
                **base_inputs,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.95,
                temperature=0.8
            )

        ft_response = fine_tuned_tokenizer.decode(ft_output[0], skip_special_tokens=True)
        base_response = base_tokenizer.decode(base_output[0], skip_special_tokens=True)

        f.write(f"=== Prompt {i} ===\n")
        f.write(f"Prompt: {prompt}\n\n")
        f.write("Fine-tuned Model Response:\n")
        f.write(ft_response + "\n\n")
        f.write("Base Gemma-2b Model Response:\n")
        f.write(base_response + "\n")
        f.write("=" * 40 + "\n\n")

print(f"Output saved to {output_file_path}")
