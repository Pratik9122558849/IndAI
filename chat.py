import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "gpt2-large"
LORA_MODEL = "gpt2_large_lora"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    local_files_only=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL
)

model.to(device)
model.eval()

print("\nChatbot ready\n")


while True:

    user = input("You: ")

    prompt = f"User: {user}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.encode("User:")[0],
            pad_token_id=tokenizer.eos_token_id
        )



    text = tokenizer.decode(output[0], skip_special_tokens=True)

    response = text.split("Assistant:")[-1]

    stop_words = ["User:", "Assistant:", "\n\n"]

    for w in stop_words:
        if w in response:
            response = response.split(w)[0]

    print("IndAI:", response.strip())