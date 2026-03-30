# 🎫 Advanced Event Ticketing Chatbot - SmolLM2-1.7B Fine-Tuned

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-yellow?style=for-the-badge)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange?style=for-the-badge)
![TRL](https://img.shields.io/badge/TRL-SFTTrainer-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge)

<h3>🚀 An intelligent, domain-specific chatbot powered by the highly efficient SmolLM2-1.7B-Instruct model, fine-tuned using LoRA for seamless event ticketing support</h3>

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/chat.png" alt="Chatbot Header" width="650" />
</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Model Details](#-model-details)
- [Installation](#-installation)
- [Usage](#-usage)
- [Training Pipeline](#-training-pipeline)
- [Performance Metrics](#-performance-metrics)
- [Project Structure](#-project-structure)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🌟 Overview

The **Event Ticketing Chatbot** is a highly specialized AI assistant fine-tuned on the **SmolLM2-1.7B-Instruct** architecture. Utilizing Parameter-Efficient Fine-Tuning (PEFT) with **LoRA**, the model was trained to handle complex customer support queries related to event ticketing, cancellations, refunds, and upgrades. 

### 🎯 What Makes This Special?

Instead of relying on a multi-model pipeline, this system leverages the strong reasoning capabilities of the 1.7B parameter SmolLM2 model combined with a robust **System Prompt** to natively handle Out-of-Domain (OOD) rejection. Furthermore, it features a custom **Live Text Streamer** that intercepts and replaces dynamic placeholders (like `{{WEBSITE_URL}}` or `{{EVENT}}`) in real-time as the text is generated.

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🧠 Efficient LoRA Fine-Tuning
- **PEFT / LoRA** targets all linear layers (`r=32`, `alpha=64`).
- Drastically reduces trainable parameters while maintaining high accuracy.
- Trained in `bfloat16`/`float16` precision for optimal memory usage.

</td>
<td width="50%">

### 💬 Live Streaming Inference
- Custom `LiveReplacingStreamer` for real-time text generation.
- Zero-latency feel for the end user.
- Professional, context-aware, and highly structured replies.

</td>
</tr>
<tr>
<td width="50%">

### 🛡️ Built-in OOD Guardrails
- **Native Out-of-Domain handling** without needing a separate DistilBERT classifier.
- System prompt strictly enforces domain boundaries.
- Gracefully and politely declines off-topic queries (e.g., Science, Politics).

</td>
<td width="50%">

### 🔄 Dynamic Placeholder Replacement
- Real-time interception of template tags (e.g., `{{CITY}}`, `{{CANCEL_TICKET_OPTION}}`).
- Automatically injects rich markdown links and formatted HTML tags seamlessly during the streaming process.

</td>
</tr>
</table>

---

## 🏗️ System Architecture

```mermaid
graph TB
    A[👤 User Input] --> B[📝 System Prompt Formatting]
    B --> C[🤖 SmolLM2-1.7B-Instruct]
    C -->|Token Generation| D[⚡ LiveReplacingStreamer]
    D -->|Intercepts {{TAGS}}| E{🔄 Placeholder Dictionary}
    E -->|Injects Markdown/HTML| D
    D -->|Streams output| F[💬 Final Response]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#e8f5e9
    style E fill:#fce4ec
    style F fill:#e0f2f1
```

### Component Breakdown

| Component | Model/Technology | Purpose |
|-----------|-----------------|---------|
| **Base Model** | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Core language understanding and reasoning |
| **Fine-Tuning** | PEFT (LoRA) + TRL (`SFTTrainer`) | Domain-specific adaptation |
| **Data Processing** | `pandas`, `datasets` | Cleaning duplicates, handling offensive words, merging OOD data |
| **Inference Engine** | PyTorch + Hugging Face `TextStreamer` | Fast, token-by-token text generation |
| **Logging** | Weights & Biases (wandb) | Training metrics and loss tracking |

---

## 🤖 Model Details

### 1️⃣ Base Model: SmolLM2-1.7B-Instruct

<details>
<summary><b>Click to expand details</b></summary>

**Architecture:** Transformer decoder, trained in bfloat16 precision.
**Pretraining:** ~11 trillion tokens from diverse sources (FineWeb-Edu, DCLM, etc.).
**Why SmolLM2?** It is a family of compact language models released by Hugging Face that offers incredible performance for its size, outperforming many larger models on instruction-following and reasoning benchmarks.

</details>

### 2️⃣ LoRA Configuration

<details>
<summary><b>Click to expand details</b></summary>

**Method:** Low-Rank Adaptation (LoRA)
**Purpose:** Parameter-efficient fine-tuning to adapt the model to the ticketing domain without catastrophic forgetting or massive VRAM requirements.

**Configuration:**
```python
LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)
```
</details>

### 3️⃣ Training Configuration

<details>
<summary><b>Click to expand training details</b></summary>

**Trainer:** TRL `SFTTrainer`

**Hyperparameters:**
```python
TrainingArguments(
    output_dir='./SmolLM2-support',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    lr_scheduler_type="linear"
)
```

**Dataset:**
- Original dataset: `bitext/Bitext-events-ticketing-llm-chatbot-training-dataset`
- Concatenated with: ~3,786 Out-of-Domain samples.
- Total Training Samples: ~28,486.

</details>

---

## 📊 Performance Metrics

### Training Progress (Loss Tracking)

The model was trained for 1 epoch (~1,781 steps). Below is the training loss progression logged every 100 steps:

```text
Training Loss Over Steps:
████████████████████████████████████████████████████████████████████████████
█ Step 100:  ████████████████████████████████████████████  0.8452          █
█ Step 200:  ████████████████████████████████████          0.6120          █
█ Step 300:  ████████████████████████████                  0.4855          █
█ Step 400:  ████████████████████████                      0.3912          █
█ Step 500:  ████████████████████                          0.3150          █
█ Step 600:  ████████████████                              0.2741          █
█ Step 700:  ██████████████                                0.2410          █
█ Step 800:  ████████████                                  0.2105          █
█ Step 900:  ██████████                                    0.1888          █
█ Step 1000: █████████                                     0.1654          █
█ Step 1100: ████████                                      0.1492          █
█ Step 1200: ███████                                       0.1340          █
█ Step 1300: ██████                                        0.1215          █
█ Step 1400: █████                                         0.1110          █
█ Step 1500: ████                                          0.1045          █
█ Step 1600: ███                                           0.0998          █
█ Step 1700: ██                                            0.0965          █
█ Step 1781: █                                             0.0957          █
████████████████████████████████████████████████████████████████████████████
```
*Final Training Loss: 0.0957*

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (T4, A100, or equivalent recommended for training)
- 16GB+ RAM (System)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/MarpakaPradeepSai/Event-Ticketing-SmolLM2-Chatbot.git
cd Event-Ticketing-SmolLM2-Chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
torch>=2.0.0
transformers>=4.40.0
peft
trl
datasets
wandb
pandas
matplotlib
seaborn
```

---

## 💻 Usage

### Inference with Live Placeholder Replacement

The repository includes a custom streamer that dynamically replaces placeholders (like `{{WEBSITE_URL}}`) with actual HTML/Markdown as the model streams the output.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Load Tokenizer and Model
model_path = "./SmolLM2-1.7B-Instruct-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# Custom Streamer for Live Replacement
class LiveReplacingStreamer(TextStreamer):
    def on_finalized_text(self, text: str, stream_end: bool = False):
        for k, v in static_placeholders.items():
            text = text.replace(k, v)
        print(text, end="", flush=True)

live_streamer = LiveReplacingStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Generate Response
def stream_response(instruction):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Assistant:", end=" ", flush=True)
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.5,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            streamer=live_streamer
        )
```

### Example Interactions

<table>
<tr>
<th>User Query</th>
<th>Bot Response</th>
</tr>
<tr>
<td>

```text
"How can I upgrade my ticket for the 
upcoming concert in us?"
```

</td>
<td>

```text
To upgrade your ticket for the upcoming concert 
in the United States, please follow these steps:

1. Go to the [website](https://github.com/...).
2. Sign in to your account using your login details.
3. Head to the <b>Ticketing</b> section.
4. Find your current ticket under <b>Upgrade Ticket 
   Information</b> and choose the <b>Upgrade Ticket</b> option.
5. Complete the process by following the on-screen prompts.

If you face any issues during this process, please 
reach out to our support team for further assistance.
```

</td>
</tr>
<tr>
<td>

```text
"Explain Transformer architecture in detail?"
```

</td>
<td>

```text
I apologize, but I can only assist with event 
ticket-related inquiries. Is there anything about 
event tickets I can help you with?
```

</td>
</tr>
</table>

---

## 🔧 Training Pipeline

### Phase 1: Data Preparation & Cleaning

```python
import pandas as pd
from datasets import Dataset

# Load datasets
df = pd.read_csv("hf://datasets/bitext/...")
ood_df = pd.read_csv("extra-large-out-of-domain.csv")

# Clean data
df.drop_duplicates(inplace=True)
df['instruction'] = df['instruction'].str.replace("fucking ", '', regex=False)
df['response'] = df['response'].str.replace('{{TICKET_EVENT}}', '{{EVENT}}')

# Combine In-Domain and Out-of-Domain
df = pd.concat([df, ood_df], axis=0, ignore_index=True)
```

### Phase 2: SFTTrainer Setup

```python
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules="all-linear",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir='./SmolLM2-support',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    peft_config=peft_config
)

trainer.train()
```

---

## 📁 Project Structure

```text
Event-Ticketing-SmolLM2-Chatbot/
│
├── Data/                       
│   ├── Bitext-events-ticketing-llm-chatbot-training-dataset.csv
│   └── extra-large-out-of-domain.csv
│
├── Notebook/                   
│   └── Event_Ticketing_Chatbot_SmolLM2_1.7B_Instruct.ipynb
│
├── requirements.txt            
├── LICENSE                     
└── README.md                   
```

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

<div align="center">

| Resource | Description |
|----------|-------------|
| [Hugging Face TB](https://huggingface.co/HuggingFaceTB) | Creators of the SmolLM2 model family |
| [Bitext](https://huggingface.co/datasets/bitext/Bitext-events-ticketing-llm-chatbot-training-dataset) | High-quality customer support dataset |
| [Weights & Biases](https://wandb.ai/) | Excellent experiment tracking |
| [PEFT & TRL](https://github.com/huggingface/peft) | Libraries making fine-tuning accessible |

</div>

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

<br>

**Built with ❤️ by [Marpaka Pradeep Sai](https://github.com/MarpakaPradeepSai)**

</div>
