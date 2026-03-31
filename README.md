# Fine-Tuning a Pre-Trained Language Model with LoRA

This project demonstrates how to **fine-tune a pre-trained instruction model** using **Supervised Fine-Tuning (SFT)** and **LoRA (Low-Rank Adaptation)** for efficient and scalable domain adaptation.

Built on top of **`HuggingFaceTB/SmolLM2-1.7B-Instruct`**, this repository provides a **practical, end-to-end pipeline** covering data preparation, formatting, training, and inference.

> The goal is to provide a **clean, reusable template** for adapting open-source LLMs to specialized tasks with minimal compute.

---

## 🚀 Key Highlights

* Fine-tuning an instruction-tuned LLM using **LoRA**
* Efficient training with **PEFT**, **Transformers**, and **TRL**
* Structured preprocessing for instruction-response datasets
* Chat-template formatting for conversational fine-tuning
* Lightweight training (updates only ~1–2% of parameters)
* End-to-end workflow: **data → training → inference**

---

## 🧠 Model

**Base Model:**
[`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)

Why this model?

* Lightweight and efficient
* Instruction-tuned
* Ideal for experimentation and domain adaptation
* Compatible with parameter-efficient fine-tuning

---

## ⚙️ Fine-Tuning Approach

This project uses:

* **Supervised Fine-Tuning (SFT)**
* **LoRA via PEFT**
* **`trl.SFTTrainer`**
* **Causal Language Modeling objective**

### LoRA Configuration

```python
LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.01,
    target_modules="all-linear"
)
```

LoRA reduces training cost by learning low-rank updates instead of modifying full model weights.

---

## 🔄 Workflow

1. Load and inspect dataset
2. Clean and normalize instruction-response pairs
3. Add out-of-domain samples (optional)
4. Format data using chat templates
5. Tokenize and prepare labels
6. Fine-tune the model using LoRA
7. Save trained adapters/model
8. Run inference on new prompts

---

## ⚡ Quick Start

### Installation

```bash
pip install torch transformers datasets peft trl accelerate wandb pandas
```

---

### Minimal Training Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# Load model & tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
)

# LoRA config
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

# Training args
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True
)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config
)

trainer.train()

# Save model
trainer.model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")
```

---

## 🏋️ Training Setup

Example configuration used:

```python
TrainingArguments(
    output_dir="./SmolLM2-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    lr_scheduler_type="linear"
)
```

---

## 🔮 Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./fine-tuned-model", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model")

def generate(prompt):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=256, temperature=0.5)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate("Your question here"))
```

---

## 📁 Project Structure

```
├── notebooks/fine_tuning.ipynb   # End-to-end walkthrough
├── src/train.py                  # Training script
├── src/inference.py              # Inference utilities
├── requirements.txt
└── README.md
```

---

## 🧩 Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* TRL
* Pandas
* Weights & Biases

---

## 💡 Why This Project Is Useful

This repository is a strong reference for:

* Adapting open-source LLMs to custom domains
* Learning **parameter-efficient fine-tuning (PEFT)**
* Building scalable and cost-effective LLM pipelines
* Understanding modern LLM training workflows

---

## 📚 Resources

* [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) (LoRA Paper)
* [https://huggingface.co/docs/peft](https://huggingface.co/docs/peft)
* [https://huggingface.co/docs/trl](https://huggingface.co/docs/trl)

---

## 🙌 Acknowledgements

* Hugging Face
* SmolLM2 team

---

## 📄 License

Apache 2.0

---

⭐ If you find this useful, consider starring the repo!
