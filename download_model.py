from transformers import AutoTokenizer, AutoModelForCausalLM

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained("gpt2-large")

print("Download finished")
