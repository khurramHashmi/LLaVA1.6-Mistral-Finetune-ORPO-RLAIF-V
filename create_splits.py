from datasets import load_dataset
import os

# Define save paths
train_path = "rlaif-v-train-only"
val_path = "rlaif-v-validation-only"

# ✅ Check if datasets already exist to avoid re-splitting
if os.path.exists(train_path) and os.path.exists(val_path):
    print(f"✅ Train & Validation splits already exist. Skipping dataset split.")
else:
    print("🔹 Loading full dataset...")
    dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train")

    # ✅ Step 2: Shuffle dataset
    dataset = dataset.shuffle(seed=42)

    # ✅ Step 3: Create 90/10 split
    train_size = int(0.9 * len(dataset))
    train_dataset = dataset.select(range(train_size))  # First 90%
    val_dataset = dataset.select(range(train_size, len(dataset)))  # Last 10%

    # ✅ Step 4: Save both splits
    print(f"✅ Train Set Size: {len(train_dataset)} | Validation Set Size: {len(val_dataset)}")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    print("🔹 Saving train & validation datasets to disk...")
    train_dataset.save_to_disk(train_path)
    val_dataset.save_to_disk(val_path)
    print("✅ Splits saved successfully!")

print(f"✅ Train set location: {train_path}")
print(f"✅ Validation set location: {val_path}")
