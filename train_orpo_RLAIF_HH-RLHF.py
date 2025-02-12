import argparse
import os
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_from_disk, concatenate_datasets
from llava.model.builder import load_pretrained_model
from PIL import Image

# ✅ Define collate function
def collate_fn(batch, tokenizer, image_processor, device, dtype, max_length):
    input_ids, attention_masks, labels, images = [], [], [], []

    for example in batch:
        chosen = example.get("chosen")
        rejected = example.get("rejected")
        question = example.get("question")  # Might be missing in HH-RLHF

        # ✅ Handling RLAIF-V dataset (Image + Text)
        if question is not None:
            question = question.strip()
            chosen_response = chosen.strip() if chosen else ""
            rejected_response = rejected.strip() if rejected else ""

        # ✅ Handling HH-RLHF dataset (Text-Only)
        elif chosen and rejected:
            if "Human:" in chosen and "Assistant:" in chosen:
                question = chosen.split("Assistant:")[0].strip()
                chosen_response = chosen.split("Assistant:")[-1].strip()
                rejected_response = rejected.split("Assistant:")[-1].strip()
            else:
                print(f"⚠️ Warning: Unable to extract question from HH-RLHF! Skipping Example: {example}")
                continue  # Skip malformed HH-RLHF samples

        else:
            print(f"⚠️ Warning: Missing required fields! Skipping Example: {example}")
            continue  # Skip invalid samples

        # ✅ Tokenize question & responses
        tokenized_prompt = tokenizer(
            text=question,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized_chosen = tokenizer(
            text=chosen_response,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized_rejected = tokenizer(
            text=rejected_response,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        input_ids.append(tokenized_prompt["input_ids"].squeeze(0).to(dtype=torch.long))
        attention_masks.append(tokenized_prompt["attention_mask"].squeeze(0))
        labels.append(tokenized_chosen["input_ids"].squeeze(0).to(dtype=torch.long))

        # ✅ Handle Image Processing for RLAIF-V
        if question is not None and example.get("image") is not None:
            image = example["image"].convert("RGB")
            image_tensor = image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        else:
            image_tensor = torch.zeros((3, 336, 336))  # Dummy image for HH-RLHF

        images.append(image_tensor.to(dtype=dtype))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks).to(dtype=dtype),
        "labels": torch.stack(labels),
        "images": torch.stack(images),
    }





# ✅ Define ORPO loss function
def orpo_loss(preferred_logits, rejected_logits, labels):
    labels = labels[:, -1].contiguous()
    loss_pref = F.cross_entropy(preferred_logits, labels)
    loss_rej = F.cross_entropy(rejected_logits, labels)
    return loss_pref - 0.5 * loss_rej

# ✅ Custom Trainer with ORPO Loss
class ORPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        inputs["input_ids"] = inputs["input_ids"].to(dtype=torch.long)
        inputs["labels"] = inputs["labels"].to(dtype=torch.long)

        dtype = torch.bfloat16 if self.args.bf16 else torch.float32
        for key in ["attention_mask", "images"]:
            inputs[key] = inputs[key].to(dtype=dtype)

        outputs = model(**inputs)
        logits = outputs.logits
        preferred_logits = logits[:, 0, :]
        rejected_logits = logits[:, 1, :]
        loss = orpo_loss(preferred_logits, rejected_logits, inputs["labels"])

        return (loss, outputs) if return_outputs else loss

def main(args):
    # ✅ Load datasets
    rlaif_train = load_from_disk(args.rlaif_train_data_path)
    hh_rlhf_train = load_from_disk(args.hh_rlhf_data_path)["train"]   
    val_dataset = load_from_disk(args.val_data_path)

    # ✅ Merge datasets
    print(f"🔹 Merging datasets: RLAIF-V ({len(rlaif_train)}) + HH-RLHF ({len(hh_rlhf_train)})...")
    train_dataset = concatenate_datasets([rlaif_train, hh_rlhf_train])
    print(f"✅ Final Training Dataset Size: {len(train_dataset)}\n")
    
    # ✅ Load Model with QLoRA or LoRA
    quantization_config = None
    if args.use_qlora:
        print("🔹 Using QLoRA for fine-tuning...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float32
        )
        gradient_checkpointing = False
    else:
        gradient_checkpointing = True
        print("🔹 Using LoRA for fine-tuning...")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_name,
        model_base=None,
        model_name="llava-v1.6-mistral-7b",
        **({"quantization_config": quantization_config} if args.use_qlora else {"load_8bit": False, "load_4bit": False})
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # ✅ Configure LoRA/QLoRA
    if args.use_qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
    
    target_modules = args.lora_target_modules.split(",")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=target_modules,
        task_type=args.lora_task_type
    )
    
    model = get_peft_model(model, lora_config)
    
    if not args.use_qlora:  # ✅ Keep this only for LoRA
        for name, param in model.named_parameters():
            if "lora" in name or "lm_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    model.print_trainable_parameters()
    print("✅ LoRA/QLoRA configuration complete. Model is ready for training.")
    print(f" Current gradient_checkpointing :  {gradient_checkpointing}")

    
    # ✅ Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=(not args.use_qlora),
        remove_unused_columns=args.remove_unused_columns,
        bf16=args.bf16
    )
    
    if args.bf16:
        model.to(torch.bfloat16)
    else:
        model.to(torch.float32)
    
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda batch: collate_fn(batch, tokenizer, image_processor, device, torch.bfloat16 if args.bf16 else torch.float32, args.max_length)
    )
    
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLAVA with LoRA or QLoRA and ORPO")
    
    # ✅ Dataset and Model paths
    parser.add_argument("--rlaif_train_data_path", type=str, default="../rlaif-v-train-only")
    parser.add_argument("--hh_rlhf_data_path", type=str, default="../hh-rlhf-only")
    parser.add_argument("--val_data_path", type=str, default="../rlaif-v-validation-only")
    parser.add_argument("--model_name", type=str, default="../../llava-v1.6-mistral-7b")
    
    # ✅ LoRA/QLoRA configuration
    parser.add_argument("--use_qlora", action="store_true", help="Enable QLoRA instead of LoRA")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj,down_proj,up_proj,gate_proj")
    parser.add_argument("--lora_task_type", type=str, default="CAUSAL_LM")
    
    # ✅ Training hyperparameters
    parser.add_argument("--output_dir", type=str, default="./llava-output")
    parser.add_argument("--per_device_train_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=625)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--remove_unused_columns", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--max_length", type=int, default=2048)

    args = parser.parse_args()
    main(args)
