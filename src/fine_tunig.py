"""
Fixed T5 Summarization Fine-Tuning Script
==========================================
Bugs Fixed:
  1. Dataset.from_list() capitalization and import
  2. label_pad_token_id = -100 (not pad_token_id)
  3. Seq2SeqTrainingArguments instead of TrainingArguments
  4. Seq2SeqTrainer instead of Trainer
  5. Added padding to tokenize_function
  6. Fixed shutil.make_archive arguments
  7. LoRA applied before data collator (correct order)
  8. Added predict_with_generate=True (required for Seq2Seq eval)
"""

import json
import shutil
import os

from datasets import load_dataset, Dataset          # ✅ Fix 1: Dataset not Datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,                                 # ✅ Fix 4: Seq2SeqTrainer
    Seq2SeqTrainingArguments,                       # ✅ Fix 3: Seq2SeqTrainingArguments
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL      = "t5-small"
JSON_DATA_PATH  = "/content/marketing_summarization_dataset.jsonl"
OUTPUT_DIR      = "./fine_tuned_t5_summarizer"
ZIP_FILE_NAME   = "fine_tuned_t5_summarizer"       # shutil adds .zip automatically

# ── Load dataset ──────────────────────────────────────────────────────────────
data = load_dataset("json", data_files=JSON_DATA_PATH, split="train")
print(f"Dataset loaded with {len(data)} examples.")
print("First example:\n", data[0])

# ── Extract user prompt and assistant response ────────────────────────────────
def extract_summarization_pair(conversation):
    user_prompt        = None
    assistant_response = None
    for message in conversation:
        if message["role"] == "user":
            user_prompt = message["content"]
        elif message["role"] == "assistant":
            assistant_response = message["content"]
    return {
        "user_prompt":        user_prompt,
        "assistant_response": assistant_response,
    }

summarization_data = [
    extract_summarization_pair(conv)
    for conv in data["messages"]
]

print("\nExtracted summarization pairs. First 2 pairs:")
for i, item in enumerate(summarization_data[:2]):
    print(f"--- Pair {i+1} ---")
    print(f"User Prompt (truncated): {item['user_prompt'][:100]}...")
    print(f"Assistant Response (truncated): {item['assistant_response'][:100]}...\n")

# ── Convert list to Dataset ───────────────────────────────────────────────────
# ✅ Fix 1: Dataset.from_list() not Datasets.from_list()
summarization_dataset = Dataset.from_list(summarization_data)
print(f"Converted to Dataset: {summarization_dataset}")

# Train / eval split (80/20)
split        = summarization_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]
print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set tokenizer.pad_token to: {tokenizer.pad_token}")

# ── Tokenize function ─────────────────────────────────────────────────────────
# ✅ Fix 5: Added padding="max_length" for consistent tensor shapes
def tokenize_function(examples):
    # INPUT — goes to Encoder (with task prefix for T5)
    inputs = [f"summarize: {prompt}" for prompt in examples["user_prompt"]]

    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length",       # ✅ Fix 5: consistent shape for collator
    )

    # TARGET — goes to Decoder as labels
    # Tokenize targets separately
    labels = tokenizer(
        examples["assistant_response"],
        max_length=128,
        truncation=True,
        padding="max_length",       # ✅ Fix 5: consistent shape
    )

    label_ids = labels["input_ids"]

    # ✅ Fix 2: Replace pad token id with -100
    # -100 tells loss function to IGNORE padding positions
    # Using pad_token_id (not -100) means loss is computed on padding → wrong
    label_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in label_ids
    ]

    model_inputs["labels"] = label_ids
    return model_inputs

# Apply tokenization
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)
tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
)

print("\nFirst tokenized example keys:", tokenized_train[0].keys())
print("Decoded input_ids:")
print(tokenizer.decode(
    [t for t in tokenized_train[0]["input_ids"] if t != tokenizer.pad_token_id],
    skip_special_tokens=True
))
print("\nDecoded labels (ignoring -100):")
print(tokenizer.decode(
    [t for t in tokenized_train[0]["labels"] if t != -100],
    skip_special_tokens=True
))

# ── Load model ────────────────────────────────────────────────────────────────
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
print(f"\nLoaded {BASE_MODEL} model.")

# ── Apply LoRA BEFORE data collator ──────────────────────────────────────────
# ✅ Fix 7: LoRA must be applied before data collator references the model
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,                  # ✅ Adjusted: 2×r is more stable (was 32)
    target_modules=["q", "v"],      # T5 attention module names
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(model, lora_config)
print(f"\nLoRA applied to {BASE_MODEL}.")
model.print_trainable_parameters()

# ── Data Collator ─────────────────────────────────────────────────────────────
# ✅ Fix 2: label_pad_token_id=-100 (not tokenizer.pad_token_id)
# -100 ensures loss is not computed on padded label positions
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,        # ✅ Fix 2: was tokenizer.pad_token_id
    pad_to_multiple_of=8,           # efficient GPU memory usage
)
print("Data collator initialized.")

# ── Training Arguments ────────────────────────────────────────────────────────
# ✅ Fix 3: Seq2SeqTrainingArguments not TrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=10,            # increased from 3 (small dataset needs more)
    per_device_train_batch_size=2,  # reduced from 4 (small dataset)
    per_device_eval_batch_size=2,
    warmup_steps=50,                # reduced from 500 (too high for small dataset)
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=5,
    evaluation_strategy="epoch",    # evaluate after each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    predict_with_generate=True,     # ✅ Fix 3: required for Seq2Seq evaluation
    generation_max_length=128,      # max tokens to generate during eval
    report_to="none",
)
print("Training arguments defined.")

# ── Trainer ───────────────────────────────────────────────────────────────────
# ✅ Fix 4: Seq2SeqTrainer not Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,            # ✅ keep tokenizer here for Seq2SeqTrainer
    data_collator=data_collator,
)
print("Trainer initialized. Starting fine-tuning...\n")

# ── Train ─────────────────────────────────────────────────────────────────────
trainer.train()

# ── Save model ────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✅ Model saved to: {OUTPUT_DIR}")
print(f"   Files: {os.listdir(OUTPUT_DIR)}")

# ── Zip with shutil ───────────────────────────────────────────────────────────
# ✅ Fix 6: Correct shutil.make_archive arguments
# make_archive(base_name, format, root_dir, base_dir)
# base_name = output zip path WITHOUT .zip
# format    = 'zip'
# root_dir  = parent directory of the folder to zip
# base_dir  = folder name to zip (relative to root_dir)

zip_output = shutil.make_archive(
    base_name=ZIP_FILE_NAME,        # output: fine_tuned_t5_summarizer.zip
    format="zip",
    root_dir=".",                   # ✅ Fix 6: root is current directory
    base_dir=OUTPUT_DIR,            # folder to zip
)

zip_size = os.path.getsize(zip_output) / 1e6
print(f"✅ Model zipped: {zip_output}  ({zip_size:.1f} MB)")

# ── Test inference ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("INFERENCE TEST")
print("="*60)

model.eval()

test_text = (
    "Campaign Name: Spring Sale 2024. Sent: 85,000 emails. "
    "Open Rate: 24.3%. Click-Through Rate: 6.8%. "
    "Conversion Rate: 2.1%. Revenue Generated: $142,500. "
    "Top Performing Subject Line: Last Chance 40% Off Ends Tonight. "
    "Best Performing Segment: Loyal customers with 3+ purchases."
)

import torch
inputs = tokenizer(
    f"summarize: {test_text}",
    return_tensors="pt",
    max_length=512,
    truncation=True,
)

with torch.no_grad():
    outputs = model.generate(
        input_ids      = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_new_tokens = 100,
        num_beams      = 4,             # beam search for better quality
        early_stopping = True,
        no_repeat_ngram_size=3,         # avoid repetition
    )

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nInput (truncated):\n{test_text[:200]}...")
print(f"\nGenerated Summary:\n{summary}")
