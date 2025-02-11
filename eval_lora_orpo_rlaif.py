import torch
from peft import PeftModel
from llava.model.builder import load_pretrained_model
from datasets import load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm
import argparse
import os

def collate_fn(batch, tokenizer, image_processor, device):
    """Collates a batch of samples for efficient inference."""
    questions = [example["question"] for example in batch]
    ground_truths = [example["chosen"] for example in batch]
    images = [example["image"].convert("RGB") for example in batch]

    # Tokenize all at once
    tokenized = tokenizer.batch_encode_plus(
        questions,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )

    # Process images
    image_tensors = image_processor(images=images, return_tensors="pt")["pixel_values"]

    return {
        "input_ids": tokenized["input_ids"].to(device),
        "attention_mask": tokenized["attention_mask"].to(device),
        "images": image_tensors.to(device),
        "ground_truths": ground_truths,
        "questions": questions
    }

def main(args):
    # ✅ Load Model
    print("🔹 Loading model...")
    model_path = args.model_path
    tokenizer, base_model, image_processor, context_len = load_pretrained_model(
        model_path=model_path, model_base=None, model_name="llava-v1.6-mistral-7b"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # If evaluate_original flag is set, use the base model; otherwise load the fine-tuned checkpoint.
    if args.evaluate_original:
        print("🔹 Evaluating original (base) model.")
        model = base_model
    else:
        print("🔹 Loading fine-tuned model checkpoint...")
        checkpoint_path = args.checkpoint_path
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!\n")

    # ✅ Optionally initialize Wandb for reporting
    if args.report_wandb:
        import wandb
        # Normalize checkpoint_path (or base_model run if evaluating original)
        run_source = args.checkpoint_path if not args.evaluate_original else model_path
        run_name = os.path.normpath(run_source).lstrip("./")
        wandb.init(project="huggingface", name=run_name)
        print(f"🔹 Wandb reporting enabled (run name: {run_name}).\n")
    else:
        print("🔹 Wandb reporting disabled.\n")

    # ✅ Load Validation Dataset
    print("🔹 Loading validation dataset...")
    dataset = load_from_disk(args.dataset_path)
    print(f"✅ Loaded {len(dataset)} validation samples.\n")

    # ✅ Create DataLoader
    batch_size = args.batch_size
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=lambda batch: collate_fn(batch, tokenizer, image_processor, device), 
        shuffle=False
    )

    # ✅ Run Batched Inference
    print("🔹 Running inference on validation set...")

    results = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            model_dtype = next(model.parameters()).dtype  # Get model's dtype

            input_ids = batch["input_ids"].to(dtype=torch.long)
            attention_mask = batch["attention_mask"].to(dtype=model_dtype)
            images = batch["images"].to(dtype=model_dtype)

            # Generate responses for the batch
            output = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=images,
                max_length=512
            )

            # Decode predictions
            predictions = tokenizer.batch_decode(output, skip_special_tokens=True)

            # Store results
            for question, pred, gt in zip(batch["questions"], predictions, batch["ground_truths"]):
                results.append({
                    "question": question,
                    "predicted_answer": pred,
                    "ground_truth": gt
                })

    print("✅ Inference complete!\n")

    # ✅ Compute Evaluation Metrics: Only ROUGE-L and F1
    print("🔹 Calculating evaluation metrics...")

    rouge_l_scores = []
    f1_scores = []
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for result in results:
        # Compute ROUGE-L Score
        rouge_l = scorer.score(result["ground_truth"], result["predicted_answer"])["rougeL"].fmeasure
        rouge_l_scores.append(rouge_l)

        # Compute F1 Score (Word-level)
        reference_tokens = set(result["ground_truth"].split())
        hypothesis_tokens = set(result["predicted_answer"].split())
        common_tokens = reference_tokens.intersection(hypothesis_tokens)

        precision = len(common_tokens) / len(hypothesis_tokens) if hypothesis_tokens else 0
        recall = len(common_tokens) / len(reference_tokens) if reference_tokens else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    # Compute Averages
    avg_rouge_l = np.mean(rouge_l_scores)
    avg_f1 = np.mean(f1_scores)

    # ✅ Print Results
    print("\n📊 **Final Evaluation Results:**")
    print(f"🔹 **ROUGE-L Score:** {avg_rouge_l:.4f}")
    print(f"🔹 **F1 Score:** {avg_f1:.4f}\n")
    print("✅ Evaluation complete!")

    # ✅ Log metrics to Wandb if enabled
    if args.report_wandb:
        wandb.log({
            "rouge_l": avg_rouge_l,
            "f1": avg_f1,
            "num_samples": len(dataset)
        })
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLAVA model")
    parser.add_argument("--model_path", type=str, default="../../llava-v1.6-mistral-7b",
                        help="Path to the base LLAVA model directory")
    parser.add_argument("--checkpoint_path", type=str, default="./llava-v1.6-mistral-7b-RLAIF-V-ORPO/checkpoint-99000",
                        help="Path to the fine-tuned checkpoint")
    parser.add_argument("--dataset_path", type=str, default="../rlaif-v-validation-only",
                        help="Path to the validation dataset (saved with load_from_disk)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference")
    parser.add_argument("--report_wandb", action="store_true", default=False,
                        help="If set, report evaluation metrics to Wandb")
    parser.add_argument("--evaluate_original", action="store_true", default=False,
                        help="If set, evaluate the original model instead of loading a checkpoint")
    args = parser.parse_args()
    main(args)
