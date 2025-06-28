
!pip install transformers datasets
!pip install bert-score
!pip install nltk
!pip install rouge

!pip install -U datasets huggingface_hub fsspec

from huggingface_hub import notebook_login

notebook_login()

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, DatasetDict
import re
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import nltk

nltk.download('punkt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_data():
    dataset = load_dataset("MattBastar/Medicine_Details", split="train")

    dataset = dataset.filter(
        lambda x: x["Medicine Name"].strip() and x["Side_effects"].strip()
    ).shuffle(seed=42)

    def preprocess_effects(example):
        effects = example["Side_effects"]

        phrases = re.findall(r'[A-Z][a-z]+(?:\s+[a-z]+)*', effects)
        processed = ", ".join(phrases).strip()
        if processed and not processed.endswith('.'):
            processed += '.'
        return {"Side_effects_processed": processed}

    dataset = dataset.map(preprocess_effects)

    split = dataset.train_test_split(test_size=0.2, seed=42)
    val_test = split['test'].train_test_split(test_size=0.5, seed=42)
    return DatasetDict({
        'train': split['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    })

class PharmaModel:
    SPECIAL_TOKENS = {
        "drug_start": "[DRUG_START]",
        "drug_end": "[DRUG_END]",
        "side_effects": "[SIDE_EFFECTS]"
    }

    def __init__(self, model_name="distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.tokenizer.add_special_tokens({
            "additional_special_tokens": list(self.SPECIAL_TOKENS.values())
        })
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    def format_sample(self, example):
        return {
            "text": (
                f"{self.SPECIAL_TOKENS['drug_start']}{example['Medicine Name']}"
                f"{self.SPECIAL_TOKENS['drug_end']} {self.SPECIAL_TOKENS['side_effects']}"
                f"{example['Side_effects_processed']}{self.tokenizer.eos_token}"
            )
        }
def tokenize_dataset(dataset, tokenizer):
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
    return dataset.map(tokenize_fn, batched=True)

class PharmaEvaluator:
    def __init__(self, model, tokenizer, special_tokens):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.rouge = Rouge()
        self.smoother = SmoothingFunction().method4
    def generate_effects(self, drug_name, max_length=256):
        prompt = (
            f"{self.special_tokens['drug_start']}{drug_name}"
            f"{self.special_tokens['drug_end']} {self.special_tokens['side_effects']}"
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="max_length"
        ).to(device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.2,
            num_beams=3,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        start_tag = self.special_tokens['side_effects']
        start_idx = full_text.find(start_tag) + len(start_tag)
        return full_text[start_idx:].split(self.tokenizer.eos_token)[0].strip()
    def calculate_metrics(self, predictions, references):
        bleu_scores = [
            sentence_bleu([ref.split()], pred.split(), smoothing_function=self.smoother)
            for pred, ref in zip(predictions, references)
        ]
        rouge_scores = self.rouge.get_scores(predictions, references, avg=True)
        jaccard_scores = []
        for pred, ref in zip(predictions, references):
            pred_set = set(p.strip().lower() for p in pred.split(","))
            ref_set = set(r.strip().lower() for r in ref.split(","))
            intersection = pred_set & ref_set
            union = pred_set | ref_set
            jaccard_scores.append(len(intersection) / len(union) if union else 0.0)
        return {
            "BLEU": np.mean(bleu_scores),
            "ROUGE-L": rouge_scores["rouge-l"]["f"],
            "Jaccard": np.mean(jaccard_scores)
        }

datasets = load_and_preprocess_data()
print("Dataset splits and sizes:")
print(f"  Full filtered set: {len(datasets)} samples")

pharma_model = PharmaModel()



for split in datasets:
    datasets[split] = datasets[split].map(pharma_model.format_sample)
    datasets[split] = tokenize_dataset(datasets[split], pharma_model.tokenizer)

    print(f"--- {split} split örnek ---")
    example = datasets[split][0]
    print("Input text:", example['text'])
    print("Tokenized input_ids:", example['input_ids'])
    print("Decoded input:", pharma_model.tokenizer.decode(example['input_ids']))
    print()

training_args = TrainingArguments(
    output_dir="./pharma_model",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none",
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True
)

trainer = Trainer(
    model=pharma_model.model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=pharma_model.tokenizer,
        mlm=False
    )
)

trainer.train()

evaluator = PharmaEvaluator(
    trainer.model,
    pharma_model.tokenizer,
    pharma_model.SPECIAL_TOKENS
)


test_samples = datasets["test"].select(range(50))
predictions = [
    evaluator.generate_effects(sample["Medicine Name"])
    for sample in test_samples
]
references = test_samples["Side_effects_processed"]



print("\nFirst 5 examples:")
for i in range(5):
    print(f"Example {i+1}:")
    print(f"  Real side effects:    {references[i]}")
    print(f"  Predicted side effects: {predictions[i]}")
    print("-" * 40)

metrics = evaluator.calculate_metrics(predictions, references)
print("\nEvaluation Results:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

history = trainer.state.log_history


train_loss = [(entry["epoch"], entry["loss"]) for entry in history if "loss" in entry and "epoch" in entry]
eval_loss = [(entry["epoch"], entry["eval_loss"]) for entry in history if "eval_loss" in entry]

train_epochs, train_losses = zip(*train_loss)
eval_epochs, eval_losses = zip(*eval_loss)


plt.figure(figsize=(10, 6))
plt.plot(train_epochs, train_losses, label="Training Loss")
plt.plot(eval_epochs, eval_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.grid(True)
plt.savefig("training_curve.png")
plt.show()

trainer.save_model("./saved_model")
pharma_model.tokenizer.save_pretrained("./saved_model")
print("Model saved successfully.")

from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

special_tokens = {
    "drug_start": "[DRUG_START]",
    "drug_end": "[DRUG_END]",
    "Side_effects_virgullu": "[SIDE_EFFECTS]"
}

model = AutoModelForCausalLM.from_pretrained("./saved_model")
tokenizer = AutoTokenizer.from_pretrained("./saved_model")


tokenizer.add_special_tokens({"additional_special_tokens": list(special_tokens.values())})
model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def clean_side_effects(text):
    text = text.strip(" ,.\n\t")
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r",+", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    if not text.endswith("."):
        text += "."
    return text.capitalize()

def generate_side_effects(drug_name):
    prompt = f"{special_tokens['drug_start']}{drug_name}" \
             f"{special_tokens['drug_end']} {special_tokens['Side_effects_virgullu']}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=64,
        truncation=True,
        padding="max_length"
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        pad_token_id=tokenizer.eos_token_id,
        num_beams=3,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    start_idx = full_text.find("[SIDE_EFFECTS]") + len("[SIDE_EFFECTS]")
    end_idx = full_text.find(tokenizer.eos_token, start_idx)
    side_effects = full_text[start_idx:end_idx].strip() if end_idx != -1 else full_text[start_idx:].strip()

    return clean_side_effects(side_effects)


drug_name = "Avastin 400mg Injection"
effects = generate_side_effects(drug_name)
print(f"Drug: {drug_name}")
print(f"Predicted side effects: {effects}")