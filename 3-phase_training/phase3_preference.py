"""
TINYLLAMA 3-PHASE FINE-TUNING
==============================
Phase 3 — Preference Training (DPO)
--------------------------------------
Dataset : marketing_preference_dataset.jsonl
Goal    : Teach model HOW to respond well (quality + style)
Method  : Direct Preference Optimization (DPO)
          Chosen response rewarded over rejected response
          No reward model needed — DPO computes loss directly
LoRA    : Loads Phase 2 as REFERENCE (frozen)
          Loads Phase 2 as POLICY base → continues training same adapters
Freeze  : Reference model fully frozen
          Policy model: Phase 2 adapters keep training (is_trainable=True)
          No new LoRA added (same adapter approach for small dataset)
Output  : ./tinyllama-phase3/

Dataset format:
  {
    "prompt":   "marketing question...",
    "chosen":   "high quality response...",
    "rejected": "low quality response...",
    "category": "branding",
    "preference_reason": "..."
  }

"""

import os
import json
import shutil
import torch

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BASE_MODEL        = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
PHASE2_DIR        = "./tinyllama-phase2"            # Phase 2 output
DATA_FILE         = "/content/marketing_preference_dataset.jsonl"
OUTPUT_DIR        = "./tinyllama-phase3"
ZIP_NAME          = "tinyllama-phase3"
MAX_LENGTH        = 512
MAX_PROMPT_LENGTH = 256

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD AND FORMAT DATASET
# DPOTrainer requires exactly 3 columns: prompt, chosen, rejected
# ══════════════════════════════════════════════════════════════════════════════
raw_data = []
with open(DATA_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            raw_data.append(json.loads(line))

print(f"Loaded {len(raw_data)} preference pairs")
print("Keys in first record:", list(raw_data[0].keys()))

def format_dpo_example(item):
    """
    Format into DPO structure required by DPOTrainer:
      prompt   = Alpaca-style instruction WITHOUT response
      chosen   = preferred (high quality) response
      rejected = dispreferred (low quality) response
    """
    prompt = (
        f"### Instruction:\n{item['prompt']}\n"
        f"### Input:\n\n"
        f"### Response:\n"
    )
    return {
        "prompt"  : prompt,
        "chosen"  : item["chosen"],
        "rejected": item["rejected"],
    }

formatted = [format_dpo_example(item) for item in raw_data]
dataset   = Dataset.from_list(formatted)

# Train / eval split (80/20)
split         = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

print(f"\nTrain: {len(train_dataset)} | Eval: {len(eval_dataset)}")
print("\nSample prompt (truncated):")
print(train_dataset[0]["prompt"][:150])
print("\nChosen (truncated):")
print(train_dataset[0]["chosen"][:150])
print("\nRejected (truncated):")
print(train_dataset[0]["rejected"][:150])

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TOKENIZER
# DPO requires left padding for correct batch generation
# ══════════════════════════════════════════════════════════════════════════════
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"\npad_token set to: {tokenizer.pad_token}")

tokenizer.padding_side = "left"    # DPO requires left padding

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — REFERENCE MODEL (frozen Phase 2)
# This is the FIXED baseline DPO compares the policy against
# Must never update — represents the instruction-tuned behavior
# DPO measures how much policy DEVIATES from reference
# ══════════════════════════════════════════════════════════════════════════════
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

print("\nLoading reference model (Phase 2 — fully frozen)...")
ref_base  = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
ref_model = PeftModel.from_pretrained(ref_base, PHASE2_DIR)

# Freeze ALL parameters in reference model
for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()

frozen_ref = sum(1 for p in ref_model.parameters() if not p.requires_grad)
print(f"Reference model frozen. Frozen tensors: {frozen_ref}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — POLICY MODEL
# Loads Phase 2 adapters with is_trainable=True
# Same adapters continue training during DPO
# No new LoRA added — consistent with Phase 1→2 approach
# Reference and policy start IDENTICAL — DPO gradually separates them
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading policy model (Phase 2 — trainable)...")
policy_base  = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
policy_model = PeftModel.from_pretrained(
    policy_base,
    PHASE2_DIR,
    is_trainable=True,   # Phase 2 adapters keep training during DPO
)
policy_model.train()

print("Policy model loaded with is_trainable=True")
print("Phase 2 adapters will continue training during DPO")
policy_model.print_trainable_parameters()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — DPO CONFIG
# ══════════════════════════════════════════════════════════════════════════════
dpo_config = DPOConfig(
    output_dir="./results-phase3",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,            # lower than SFT — preference shifts are subtle
    warmup_steps=10,
    weight_decay=0.01,
    fp16=True,
    logging_steps=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",

    # DPO specific
    beta=0.1,                      # controls deviation from reference
                                   # low  β = stay close to Phase 2 behavior
                                   # high β = stronger preference enforcement
    max_length=MAX_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
    remove_unused_columns=False,
)
print("\nDPO config defined.")
print(f"  beta = {dpo_config.beta}  (deviation from reference)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — DPO TRAINER
# DPOTrainer handles the DPO loss computation internally
# It manages both policy and reference model forward passes
# Computes log probabilities for chosen and rejected responses
# ══════════════════════════════════════════════════════════════════════════════
trainer = DPOTrainer(
    model=policy_model,         # trains
    ref_model=ref_model,        # frozen reference
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
print("DPO Trainer initialized. Starting Phase 3 training...\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — TRAIN
# ══════════════════════════════════════════════════════════════════════════════
trainer.train()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — SAVE + ZIP
# ══════════════════════════════════════════════════════════════════════════════
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✅ Phase 3 model saved to: {OUTPUT_DIR}")
print(f"   Files: {os.listdir(OUTPUT_DIR)}")

zip_output = shutil.make_archive(
    base_name=ZIP_NAME,
    format="zip",
    root_dir=".",
    base_dir=OUTPUT_DIR,
)
print(f"✅ Zipped: {zip_output}  ({os.path.getsize(zip_output)/1e6:.1f} MB)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — COMPARE REFERENCE VS POLICY OUTPUT
# Shows the effect of DPO training
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("INFERENCE TEST — Phase 3 (Reference vs Policy)")
print("="*60)

tokenizer.padding_side = "left"

test_prompt = (
    "### Instruction:\nWhat is Account-Based Marketing and how does it "
    "differ from traditional inbound marketing?\n"
    "### Input:\n\n"
    "### Response:\n"
)
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

gen_kwargs = dict(
    max_new_tokens=150,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
)

# Reference model output (Phase 2 behavior)
ref_model.eval()
with torch.no_grad():
    ref_out  = ref_model.generate(**inputs, **gen_kwargs)
n         = inputs["input_ids"].shape[1]
ref_resp  = tokenizer.decode(ref_out[0][n:], skip_special_tokens=True).strip()

# Policy model output (Phase 3 — DPO aligned)
policy_model.eval()
with torch.no_grad():
    pol_out  = policy_model.generate(**inputs, **gen_kwargs)
pol_resp  = tokenizer.decode(pol_out[0][n:], skip_special_tokens=True).strip()

print(f"\nPrompt: What is ABM and how does it differ from inbound marketing?")
print(f"\n[Reference — Phase 2]:\n{ref_resp}")
print(f"\n[Policy — Phase 3 DPO]:\n{pol_resp}")
print("\n✅ Phase 3 complete. Full 3-phase pipeline done.")
print("="*60)
print("PIPELINE SUMMARY")
print("="*60)
print("  Phase 1: ./tinyllama-phase1/  — domain knowledge")
print("  Phase 2: ./tinyllama-phase2/  — instruction following")
print("  Phase 3: ./tinyllama-phase3/  — preference aligned")
print("  Zips   : tinyllama-phase1.zip, phase2.zip, phase3.zip")
