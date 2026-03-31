# Fine-Tuning Pre-Trained Language Models with LoRA

<p align="center">
  <img src="https://www.mygreatlearning.com/blog/wp-content/uploads/2025/04/fine-tuning-banner.jpg" width="800"/>
</p>

A comprehensive guide to **Parameter-Efficient Fine-Tuning (PEFT)** of large language models using **LoRA (Low-Rank Adaptation)**. This repository demonstrates how to adapt a pre-trained model to a domain-specific task while minimizing computational resources and training time.

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [What is Fine-Tuning?](#-what-is-fine-tuning)
- [Types of Fine-Tuning](#-types-of-fine-tuning)
- [Understanding PEFT and LoRA](#-understanding-peft-and-lora)
- [Project Overview](#-project-overview)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Fine-Tuning Pipeline](#-fine-tuning-pipeline)
- [Model Configuration](#-model-configuration)
- [Training](#-training)
- [Inference](#-inference)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [License](#-license)

---

## 🎯 Introduction

Fine-tuning enables adapting powerful pre-trained language models to specific domains and tasks without training from scratch. This project showcases a complete fine-tuning workflow using modern techniques that make the process efficient and accessible.

### Key Highlights

- **Model**: HuggingFace's SmolLM2-1.7B-Instruct
- **Technique**: LoRA (Low-Rank Adaptation) via PEFT
- **Framework**: Hugging Face Transformers + TRL (Transformer Reinforcement Learning)
- **Training**: Supervised Fine-Tuning (SFT) with efficient memory usage

---

## 📖 What is Fine-Tuning?

**Fine-tuning** is the process of taking a pre-trained model and further training it on a smaller, domain-specific dataset to adapt it for a particular task.

### Why Fine-Tune?

Pre-trained models like GPT, BERT, or LLaMA are trained on massive general-purpose corpora but may:

- Lack domain-specific terminology or context
- Not handle company-specific questions or style
- Produce vague or inaccurate answers for niche queries

**Fine-tuning bridges this gap** by:

✅ Producing more accurate and relevant responses  
✅ Enabling faster and personalized answers  
✅ Reducing hallucinations or off-topic replies

### How Fine-Tuning Works

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Start with Pre-trained Model (e.g., SmolLM2-1.7B-Instruct)  │
│                              ↓                                   │
│  2. Prepare Domain-Specific Dataset (instruction-response pairs)│
│                              ↓                                   │
│  3. Fine-Tune with Small Learning Rate (preserve base knowledge)│
│                              ↓                                   │
│  4. Evaluate and Validate on Held-out Data                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Types of Fine-Tuning

| Method | Description | Resource Usage |
|--------|-------------|----------------|
| **Full Fine-Tuning** | Updates all model parameters | High (expensive) |
| **LoRA / QLoRA** | Updates only low-rank adapter matrices | Low (efficient) |
| **Adapter-Based** | Inserts small bottleneck modules | Low |
| **Prefix Tuning** | Learns continuous prompts prepended to inputs | Low |
| **Instruction Tuning (SFT)** | Learns from instruction-response examples | Medium |
| **RLHF** | Refines responses using human feedback | High |

---

## 🧠 Understanding PEFT and LoRA

### What is PEFT?

**PEFT (Parameter-Efficient Fine-Tuning)** is a category of techniques designed to adapt large pre-trained models by training only a small subset of parameters.

### What is LoRA?

**LoRA (Low-Rank Adaptation)** constrains weight updates to low-rank matrices, drastically reducing trainable parameters.

<p align="center">
  <img src="https://www.dailydoseofds.com/content/images/size/w1000/2024/02/image-283.png" width="600"/>
</p>

### Mathematical Foundation

For a weight matrix $W \in \mathbb{R}^{d \times k}$:

- **Traditional fine-tuning**: $W \leftarrow W + \Delta W$
- **LoRA**: $\Delta W = A \cdot B$ where:
  - $A \in \mathbb{R}^{d \times r}$ (tall, skinny matrix)
  - $B \in \mathbb{R}^{r \times k}$ (short, wide matrix)
  - $r \ll \min(d, k)$ is the **rank** (typically 4, 8, 16, 32)

**Parameter Reduction**: Instead of $d \times k$ parameters, LoRA learns only $r(d + k)$ parameters.

### Forward Pass with LoRA

$$W' = W + \alpha \cdot A \cdot B$$

Where:
- $W$ = Original frozen weights
- $\alpha$ = Scaling factor
- $A, B$ = Learned low-rank matrices

### Benefits of LoRA

| Benefit | Description |
|---------|-------------|
| 🚀 **Parameter Efficient** | Fine-tunes only small low-rank matrices |
| ⚡ **Faster Training** | Less computation, quicker convergence |
| 🔒 **Preserves Base Model** | Original weights remain unchanged |
| 🔄 **Modular** | Easy to switch/combine adapters for different tasks |
| 💾 **Smaller Files** | Saves storage and bandwidth |
| 💰 **Cost Effective** | Enables fine-tuning large models affordably |

---

## 📁 Project Overview

This repository demonstrates fine-tuning with a practical example, adapting a model for domain-specific conversational AI.

### Base Model: SmolLM2-1.7B-Instruct

| Specification | Details |
|---------------|---------|
| **Architecture** | Transformer decoder (LLaMA-based) |
| **Parameters** | 1.7 Billion |
| **Training Data** | ~11 trillion tokens |
| **Precision** | bfloat16 |
| **License** | Apache 2.0 |

### Benchmark Performance

| Task | SmolLM2-1.7B-Instruct | Llama-1B-Instruct |
|------|----------------------|-------------------|
| IFEval | 56.7 | 53.5 |
| MT-Bench | 6.13 | 5.48 |
| HellaSwag | 66.1 | 56.1 |
| GSM8K (5-shot) | 48.2 | 26.8 |
| MMLU-Pro | 19.3 | 12.7 |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fine-tuning-llm.git
cd fine-tuning-llm

# Install dependencies
pip install torch transformers datasets accelerate
pip install peft trl wandb
pip install pandas matplotlib seaborn
```

### Requirements

```
torch>=2.0.0
transformers>=4.56.2
datasets>=3.0.0
peft
trl>=0.29.0
accelerate>=1.4.0
wandb
pandas
matplotlib
seaborn
```

---

## 📊 Dataset Preparation

### Data Format

The training data should consist of instruction-response pairs:

```python
{
    "instruction": "User query or question",
    "response": "Expected model response"
}
```

### Data Preprocessing Steps

1. **Remove Duplicates**: Eliminate redundant samples
2. **Clean Text**: Remove offensive content and normalize text
3. **Handle Placeholders**: Standardize template variables
4. **Add Out-of-Domain Data**: Include samples for graceful rejection of off-topic queries

```python
# Example: Converting to chat format
def format_chat(row):
    messages = [
        {"role": "user", "content": row["instruction"]},
        {"role": "assistant", "content": row["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

df["text"] = df.apply(format_chat, axis=1)
```

---

## 🔄 Fine-Tuning Pipeline

### Pipeline Overview

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Load Base   │───▶│  Apply LoRA  │───▶│  Tokenize    │
│    Model     │    │   Config     │    │   Dataset    │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                                               ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    Save      │◀───│    Train     │◀───│   Configure  │
│   Adapters   │    │  with SFT    │    │   Trainer    │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## ⚡ Model Configuration

### Loading the Base Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### LoRA Configuration

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=32,                          # LoRA rank (low-rank dimension)
    lora_alpha=64,                 # Scaling factor for LoRA weights
    lora_dropout=0.01,             # Dropout for regularization
    bias="none",                   # Don't update bias terms
    task_type="CAUSAL_LM",         # For causal language modeling
    target_modules="all-linear"    # Apply LoRA to all linear layers
)
```

### LoRA Hyperparameters Explained

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `r` | Rank of low-rank matrices | 4, 8, 16, 32 |
| `lora_alpha` | Scaling factor ($\alpha$) | 16, 32, 64 |
| `lora_dropout` | Dropout probability | 0.01 - 0.1 |
| `target_modules` | Layers to apply LoRA | "all-linear", ["q_proj", "v_proj"] |

---

## 🏋️ Training

### Training Arguments

```python
from transformers import TrainingArguments

training_arguments = TrainingArguments(
    output_dir='./model-output',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    lr_scheduler_type="linear"
)
```

### Initialize and Run Training

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_dataset,
    peft_config=peft_config
)

# Start training
trainer.train()
```

### Saving the Fine-Tuned Model

```python
output_path = "./fine-tuned-model"

# Save LoRA adapters
trainer.model.save_pretrained(output_path)

# Save tokenizer
tokenizer.save_pretrained(output_path)
```

---

## 🔮 Inference

### Loading the Fine-Tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./fine-tuned-model"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
```

### Generating Responses

```python
def generate_response(instruction, max_new_tokens=256):
    messages = [
        {"role": "user", "content": instruction},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
response = generate_response("How can I cancel my ticket?")
print(response)
```

---

## 📈 Results

### Training Metrics

| Metric | Value |
|--------|-------|
| Training Loss | 0.0957 |
| Total Steps | 1,781 |
| Training Time | ~2 hours (on T4 GPU) |

### Model Architecture After LoRA

The fine-tuned model maintains the base architecture with LoRA adapters injected:

```
LlamaModel
├── embed_tokens: Embedding(49152, 2048)
├── layers: 24 x LlamaDecoderLayer
│   ├── self_attn
│   │   ├── q_proj: lora.Linear (2048 → 32 → 2048)
│   │   ├── k_proj: lora.Linear (2048 → 32 → 2048)
│   │   ├── v_proj: lora.Linear (2048 → 32 → 2048)
│   │   └── o_proj: lora.Linear (2048 → 32 → 2048)
│   └── mlp
│       ├── gate_proj: lora.Linear (2048 → 32 → 8192)
│       ├── up_proj: lora.Linear (2048 → 32 → 8192)
│       └── down_proj: lora.Linear (8192 → 32 → 2048)
└── lm_head: Linear(2048, 49152)
```

---

## 📂 Repository Structure

```
fine-tuning-llm/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── notebooks/
│   └── fine_tuning.ipynb        # Complete training notebook
├── src/
│   ├── data_preprocessing.py    # Data cleaning utilities
│   ├── train.py                 # Training script
│   └── inference.py             # Inference utilities
├── configs/
│   ├── lora_config.yaml         # LoRA hyperparameters
│   └── training_config.yaml     # Training arguments
└── data/
    └── sample_data.csv          # Example dataset format
```

---

## 🔑 Key Takeaways

1. **LoRA enables efficient fine-tuning** by training only ~1-2% of total parameters
2. **Pre-trained models can be adapted** to specific domains without expensive full fine-tuning
3. **Modern tools** (PEFT, TRL, Transformers) make the process accessible
4. **Out-of-domain handling** improves model reliability by teaching graceful rejection
5. **Chat templates** ensure proper formatting for instruction-tuned models

---

## 📚 References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [SmolLM2 Model Card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---
<p align="center">
  <b>⭐ If you find this repository helpful, please consider giving it a star! ⭐</b>
</p>
