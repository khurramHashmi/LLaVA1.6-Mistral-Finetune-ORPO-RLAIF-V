# 🌋 LLaVA 1.6 - Mistral 7B Fine-Tuning with ORPO on RLAIF-V

This repository is a **fork** of the original [LLaVA](https://github.com/haotian-liu/LLaVA) project, modified to **fine-tune LLaVA 1.6 Mistral-7B** using **ORPO** on the **RLAIF-V dataset** with **LoRA**. This approach enhances multimodal understanding by leveraging reinforcement learning with AI feedback (RLAIF) while keeping training efficient with LoRA.

---

## 🚀 Features

- **Fine-Tune LLaVA 1.6 Mistral-7B** on the **RLAIF-V dataset**
- **Optimized Rank Preference Optimization (ORPO)** for alignment
- **LoRA fine-tuning** for efficient model adaptation
- **Support for xFormers & FlashAttention** to reduce memory overhead
- **Evaluation scripts for LoRA fine-tuned models**
- **Dataset preparation based on `create_splits.py`**

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/LLaVA1.6-Mistral-Finetune-ORPO-RLAIF-V.git
cd LLaVA1.6-Mistral-Finetune-ORPO-RLAIF-V
```

### 2️⃣ Set Up Python Environment

```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e ".[train]"
```

### 3️⃣ Install Additional Dependencies

```bash
pip install -r additional_requirements.txt
pip install flash-attn --no-build-isolation
```

### 4️⃣ Download Base Model
The base **LLaVA 1.6 - Mistral 7B** model can be downloaded from **Hugging Face**:
```bash
git lfs install
git clone https://huggingface.co/liuhaotian/LLaVA-1.6-Mistral-7B checkpoints/llava1.6-mistral-7b
```

---

## 📊 Dataset Preparation

This fine-tuning pipeline uses the **RLAIF-V dataset**, which is processed using `create_splits.py`.

1. **Run dataset split creation:**
   ```bash
   python create_splits.py
   ```
2. The processed dataset should be structured and will be saved locally as:
   ```
   ./rlaif-v-train-only/
   ./rlaif-v-validation-only/
   ```
3. Ensure dataset paths are correctly referenced in `finetune_orpo_lora.py`.

---

## 🎯 Fine-Tuning LLaVA 1.6 Mistral-7B

### 1️⃣ LoRA Fine-Tuning with ORPO

Run the fine-tuning script with the required arguments:

```bash
python finetune_orpo_lora.py \
    --train_data_path ./rlaif-v-train-only \
    --val_data_path ./rlaif-v-validation-only \
    --model_name ../../llava1.6-mistral-7b \
    --output_dir ./llava-output \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 32 \
    --max_steps 500 \
    --logging_steps 10 \
    --report_to wandb \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 20 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --remove_unused_columns False \
    --bf16 False \
    --max_length 2048
```

#### 🛠 Fine-Tuning Configurations

- **Model Checkpoints**: `checkpoints/llava1.6-mistral-7b-finetune`
- **LoRA Adapters**: `checkpoints/llava1.6-mistral-7b-finetune_lora`
- **Training Script**: `finetune_orpo_lora.py`
- **Hyperparameters**: Configurable inside `finetune_orpo_lora.py`

---

## 🔬 Model Evaluation

Once training is complete, you can evaluate the fine-tuned model.

### 1️⃣ Run Evaluation Script

```bash
python eval_lora_orpo_rlaif.py \
    --model_path ../../llava1.6-mistral-7b \
    --checkpoint_path ./llava1.6-mistral-7b-RLAIF-V-ORPO/checkpoint-99000 \
    --dataset_path ./rlaif-v-validation-only \
    --batch_size 512 \
    --report_wandb \
    --evaluate_original False
```

---

## 📡 Deployment

### 1️⃣ Merge LoRA and Base Model Weights
Before deployment, you need to merge the LoRA adapter with the base model:

```bash
python -m peft.merge_adapters \
    --base_model_path ../../llava1.6-mistral-7b \
    --lora_adapter_path checkpoints/llava1.6-mistral-7b-finetune_lora \
    --output_path checkpoints/llava1.6-mistral-7b-merged
```

This will create a fully merged model in `checkpoints/llava1.6-mistral-7b-merged`.


---

## 📜 License

Users must comply with any **dataset/model-specific** licensing agreements.

---

## 🙌 Acknowledgements

This work builds upon:

- **[LLaVA](https://github.com/haotian-liu/LLaVA)** for multimodal instruction tuning.
- **[Mistral 7B](https://huggingface.co/mistralai/Mistral-7B)** as the base LLM.
- **[RLAIF-V Dataset](https://example.com)** for AI feedback-based fine-tuning.

---

## 📣 Citation

If you use like this, please star this repository


---

## 💡 Contact

For questions, open an **issue**.

