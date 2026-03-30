import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model


########################################
# To use Apple GPU
########################################
torch.set_float32_matmul_precision("high")


########################################
# CONFIG
########################################

MODEL_NAME = "gpt2-large"
DATASET_FILE = "dataset/indai_dataset.jsonl"
#DATASET_FILE = "dataset/gk_dataset.jsonl"
OUTPUT_DIR = "./gpt2_large_lora"

MAX_LEN = 32
BATCH_SIZE = 4
EPOCHS = 3

########################################
# DEVICE
########################################

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)

########################################
# LOAD TOKENIZER (LOCAL CACHE)
########################################

print("Loading tokenizer from local cache...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)

tokenizer.pad_token = tokenizer.eos_token

########################################
# LOAD MODEL
########################################

print("Loading model from local cache...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)

########################################
# APPLY LoRA
########################################

print("Applying LoRA adapters...")

lora_config = LoraConfig(

    r=8,
    lora_alpha=16,

    target_modules=["c_attn"],

    lora_dropout=0.1,

    bias="none",

    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

model.to(device)

########################################
# LOAD DATASET
########################################

print("Loading dataset...")

texts = []

with open(DATASET_FILE) as f:

    for line in f:

        data = json.loads(line)

        text = f"User: {data['input']}\nAssistant: {data['output']}"

        texts.append(text)

dataset = Dataset.from_dict({"text": texts})

########################################
# TOKENIZE
########################################

def tokenize(example):

    tokens = tokenizer(

        example["text"],

        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

dataset = dataset.map(tokenize)

dataset.set_format(
    type="torch",
    columns=["input_ids","attention_mask","labels"]
)

########################################
# TRAINING SETTINGS
########################################

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    logging_steps=100,
    save_steps=200,
    learning_rate=2e-4,
    save_total_limit=2,
    dataloader_pin_memory=False,
    report_to="none"
)

########################################
# TRAINER
########################################

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset
)

########################################
# TRAIN
########################################

print("Starting training...")

trainer.train()

########################################
# SAVE MODEL
########################################

print("Saving model...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete!")