from datasets import load_dataset

# ✅ Load the dataset from Hugging Face
print("🔹 Downloading Anthropic HH-RLHF dataset...")
dataset = load_dataset("Anthropic/hh-rlhf")

# ✅ Save the dataset locally
save_path = "hh-rlhf-only"
dataset.save_to_disk(save_path)

print(f"✅ Dataset saved to: {save_path}")
print(f"✅ Dataset splits: {dataset.keys()}")
print(f"✅ Example sample: {dataset['train'][0]}")

