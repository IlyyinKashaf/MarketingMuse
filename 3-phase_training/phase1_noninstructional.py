"""
PHASE 1 — Non-Instructional Domain Pretraining
===============================================
Model: TinyLlama (Decoder-Only, CAUSAL_LM)
Task:  Next-token prediction on raw marketing text
Data:  marketing_pretraining_dataset.jsonl
LoRA:  get_peft_model() — fresh adapters, no manual freezing
       PEFT auto-freezes base weights, only LoRA A+B train

Teaches model: WHAT marketing domain is (vocabulary, concepts, facts)
"""

import os
import shutil
import torch

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ─────────────────────────────────────────────────────────────────
BASE_MODEL     = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
JSON_DATA_PATH = "/content/marketing_pretraining_dataset.jsonl"
OUTPUT_DIR     = "./phase1_noninstructional"
ZIP_FILE_NAME  = "phase1_noninstructional"
MAX_LENGTH     = 512

# ── Load dataset ────────────────────────────────────────────────────────────
data = load_dataset("json", data_files=JSON_DATA_PATH, split="train")
print(f"Dataset loaded with {len(data)} examples.")
print("First example (truncated):\n", data[0]["text"][:200])

# ── Extract raw text ────────────────────────────────────────────────────────
# Non-instructional = raw knowledge text, no user/assistant structure
text_data = [{"text": item["text"]} for item in data]

print("\nExtracted samples. First 2:")
for i, item in enumerate(text_data[:2]):
    print(f"--- Sample {i+1} ---")
    print(f"Text (truncated): {item['text'][:150]}...\n")

# ── Convert list to Dataset ─────────────────────────────────────────────────
text_dataset  = Dataset.from_list(text_data)
print(f"Converted to Dataset: {text_dataset}")

split         = text_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]
print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

# ── Tokenizer ───────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set tokenizer.pad_token to: {tokenizer.pad_token}")

tokenizer.padding_side = "right"   # right padding for training

# ── Tokenize function ───────────────────────────────────────────────────────
# Raw text tokenization — labels = input_ids, no masking
# Model predicts EVERY next token (full causal LM objective)
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["text"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )
    # Labels = input_ids — no -100 masking
    # Every token position contributes to loss
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_train = train_dataset.map(
    tokenize_function, batched=True,
    remove_columns=train_dataset.column_names,
)
tokenized_eval = eval_dataset.map(
    tokenize_function, batched=True,
    remove_columns=eval_dataset.column_names,
)

print("\nFirst tokenized example keys:", tokenized_train[0].keys())
print("Decoded input_ids (first 80 tokens):")
print(tokenizer.decode(
    [t for t in tokenized_train[0]["input_ids"] if t != tokenizer.pad_token_id][:80],
    skip_special_tokens=True,
))

# ── Load model ──────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto",
)
print(f"\nLoaded {BASE_MODEL}.")

# ── Apply LoRA — no manual freezing needed ──────────────────────────────────
# Phase 1: First training run on base model
# get_peft_model() adds fresh LoRA A+B matrices to q_proj and v_proj
# PEFT automatically freezes ALL base model weights
# Only the new LoRA matrices are trainable
# No previous LoRA to worry about — clean start
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
print(f"\nLoRA applied.")
model.print_trainable_parameters()

# ── Data Collator ───────────────────────────────────────────────────────────
# DataCollatorForLanguageModeling with mlm=False for causal LM
# NOT DataCollatorForSeq2Seq (that is for T5/BART encoder-decoder)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,    # False = causal LM (next token prediction)
)
print("Data collator initialized.")

# ── Training Arguments ──────────────────────────────────────────────────────
# Plain TrainingArguments — NOT Seq2SeqTrainingArguments
# No predict_with_generate — causal LM eval uses loss not generation
training_args = TrainingArguments(
    output_dir="./results_phase1",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=20,
    weight_decay=0.01,
    fp16=True,
    logging_dir="./logs_phase1",
    logging_steps=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
)
print("Training arguments defined.")

# ── Trainer ─────────────────────────────────────────────────────────────────
# Plain Trainer — NOT Seq2SeqTrainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
print("Trainer initialized. Starting Phase 1 training...\n")

trainer.train()

# ── Save ────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nPhase 1 model saved to: {OUTPUT_DIR}")
print(f"Files: {os.listdir(OUTPUT_DIR)}")

zip_output = shutil.make_archive(
    base_name=ZIP_FILE_NAME, format="zip",
    root_dir=".", base_dir=OUTPUT_DIR,
)
print(f"Phase 1 zipped: {zip_output}  ({os.path.getsize(zip_output)/1e6:.1f} MB)")

# ── Inference Test ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1 INFERENCE TEST — Domain Continuation Check")
print("="*60)
model.eval()

test_prompt = "Customer Lifetime Value (CLV) is a metric that"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs, max_new_tokens=80,
        temperature=0.7, top_p=0.9, do_sample=True,
        repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id,
    )

input_len = inputs["input_ids"].shape[1]
response  = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
print(f"\nPrompt:       {test_prompt}")
print(f"Continuation: {response}")
print(f"\nPhase 1 Complete. Pass '{OUTPUT_DIR}' as PHASE1_LORA_PATH in Phase 2.")
