"""
PHASE 2 — Instructional Fine-Tuning
=====================================
Model: TinyLlama + Phase 1 LoRA (Decoder-Only, CAUSAL_LM)
Task:  Instruction following with response masking
Data:  marketing_summarization_dataset.jsonl
LoRA:  Load Phase 1 LoRA with is_trainable=True — SAME adapters continue training
       No freezing, no new LoRA — Phase 1 weights keep updating

Why same LoRA (not new):
  Small dataset (12 examples) — new LoRA has too few params to learn from
  Phase 1 adapters already know domain — we extend that knowledge
  is_trainable=True lets them keep updating

Teaches model: HOW to follow instructions, prompt format, response style
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
from peft import PeftModel

# ── Config ─────────────────────────────────────────────────────────────────
BASE_MODEL        = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
PHASE1_LORA_PATH  = "./phase1_noninstructional"   # output from Phase 1
JSON_DATA_PATH    = "/content/marketing_summarization_dataset.jsonl"
OUTPUT_DIR        = "./phase2_instructional"
ZIP_FILE_NAME     = "phase2_instructional"
MAX_LENGTH        = 512
RESPONSE_MARKER   = "### Response:\n"

# ── Load dataset ────────────────────────────────────────────────────────────
data = load_dataset("json", data_files=JSON_DATA_PATH, split="train")
print(f"Dataset loaded with {len(data)} examples.")
print("First example:\n", data[0])

# ── Extract and format as Alpaca-style prompt ───────────────────────────────
# Instructional format: ### Instruction / ### Input / ### Response
# This is the standard prompt template for TinyLlama instruction tuning
def extract_instruction_pair(conversation):
    system_prompt = ""
    user_prompt   = ""
    assistant_response = ""
    for message in conversation:
        if message["role"] == "system":
            system_prompt = message["content"]
        elif message["role"] == "user":
            user_prompt = message["content"]
        elif message["role"] == "assistant":
            assistant_response = message["content"]
            break

    # Embed system prompt into instruction if present
    if system_prompt:
        instruction = f"[System]: {system_prompt}\n\n{user_prompt}"
    else:
        instruction = user_prompt

    # Alpaca-style format — decoder sees this as ONE sequence
    prompt = (
        f"### Instruction:\n{instruction}\n"
        f"### Input:\n\n"
        f"### Response:\n{assistant_response}"
    )
    return {"text": prompt}

instruction_data = [
    extract_instruction_pair(conv)
    for conv in data["messages"]
]

print("\nExtracted instruction pairs. First 2:")
for i, item in enumerate(instruction_data[:2]):
    print(f"--- Pair {i+1} ---")
    print(f"Prompt (truncated): {item['text'][:200]}...\n")

# ── Convert list to Dataset ─────────────────────────────────────────────────
instruction_dataset = Dataset.from_list(instruction_data)
print(f"Converted to Dataset: {instruction_dataset}")

split         = instruction_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]
print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

# ── Tokenizer ───────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set tokenizer.pad_token to: {tokenizer.pad_token}")

tokenizer.padding_side = "right"

# ── Tokenize with response masking ─────────────────────────────────────────
# KEY DIFFERENCE from Phase 1:
# Phase 1: labels = input_ids (train on ALL tokens)
# Phase 2: labels = input_ids with PROMPT tokens masked to -100
# Only response tokens contribute to loss
# Model learns to generate responses, not memorize prompts
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["text"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )

    all_labels = []
    for i, text in enumerate(examples["text"]):
        input_ids = model_inputs["input_ids"][i]
        labels    = input_ids.copy()

        # Find where response starts in the text
        response_start_char = text.find(RESPONSE_MARKER)
        if response_start_char != -1:
            # Tokenize prefix (everything up to and including marker)
            prefix     = text[:response_start_char + len(RESPONSE_MARKER)]
            prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        else:
            prefix_len = 0  # fallback: train on all tokens

        # Mask everything before response with -100
        # Loss computed ONLY on response tokens
        for j in range(min(prefix_len, MAX_LENGTH)):
            labels[j] = -100

        all_labels.append(labels)

    model_inputs["labels"] = all_labels
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

# Verify masking
sample_labels = tokenized_train[0]["labels"]
masked  = sum(1 for l in sample_labels if l == -100)
trained = sum(1 for l in sample_labels if l != -100 and l != tokenizer.pad_token_id)
print(f"Masked (prompt) tokens:   {masked}")
print(f"Trained (response) tokens: {trained}")
print("Decoded response portion:")
print(tokenizer.decode(
    [t for t in tokenized_train[0]["input_ids"]
     if tokenized_train[0]["labels"][tokenized_train[0]["input_ids"].index(t)] != -100
     and t != tokenizer.pad_token_id][:50],
    skip_special_tokens=True,
))

# ── Load base model ─────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto",
)
print(f"\nLoaded base model {BASE_MODEL}.")

# ── Load Phase 1 LoRA — is_trainable=True, no freezing ─────────────────────
# Phase 2 approach: RESUME training Phase 1 adapters
# is_trainable=True = Phase 1 weights remain unfrozen and keep updating
# No get_peft_model() call — no new LoRA added
# No manual freezing — existing adapters just continue learning
# This works well for small datasets (12 examples)
model = PeftModel.from_pretrained(
    base_model,
    PHASE1_LORA_PATH,
    is_trainable=True,     # Phase 1 LoRA stays trainable — keeps updating
)
model.train()
print(f"\nPhase 1 LoRA loaded with is_trainable=True.")
model.print_trainable_parameters()

# ── Data Collator ───────────────────────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
print("Data collator initialized.")

# ── Training Arguments ──────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./results_phase2",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=20,
    weight_decay=0.01,
    fp16=True,
    logging_dir="./logs_phase2",
    logging_steps=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
)
print("Training arguments defined.")

# ── Trainer ─────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
print("Trainer initialized. Starting Phase 2 training...\n")

trainer.train()

# ── Save ────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nPhase 2 model saved to: {OUTPUT_DIR}")
print(f"Files: {os.listdir(OUTPUT_DIR)}")

zip_output = shutil.make_archive(
    base_name=ZIP_FILE_NAME, format="zip",
    root_dir=".", base_dir=OUTPUT_DIR,
)
print(f"Phase 2 zipped: {zip_output}  ({os.path.getsize(zip_output)/1e6:.1f} MB)")

# ── Inference Test ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2 INFERENCE TEST — Instruction Following Check")
print("="*60)
model.eval()

def generate(instruction):
    prompt = (
        f"### Instruction:\n{instruction}\n"
        f"### Input:\n\n"
        f"### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=120,
            temperature=0.7, top_p=0.9, do_sample=True,
            repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

test_questions = [
    "What is Account-Based Marketing (ABM)?",
    "Explain the difference between MQLs and SQLs.",
]
for q in test_questions:
    print(f"\nQ: {q}")
    print(f"A: {generate(q)}")

print(f"\nPhase 2 Complete. Pass '{OUTPUT_DIR}' as PHASE2_LORA_PATH in Phase 3.")
