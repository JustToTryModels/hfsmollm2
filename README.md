Here is the complete `README.md` file tailored for your **SmolLM2-1.7B-Instruct** fine-tuning project, matching the professional styling and structure of your previous DistilGPT2 project.

***

# 🎫 Event Ticketing Customer Support Chatbot (SmolLM2-1.7B)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-yellow?style=for-the-badge)
![PEFT](https://img.shields.io/badge/PEFT_LoRA-Enabled-orange?style=for-the-badge)
![TRL](https://img.shields.io/badge/TRL-SFTTrainer-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<h3>🚀 An intelligent, domain-specific chatbot powered by HuggingFaceTB/SmolLM2-1.7B-Instruct, fine-tuned using LoRA (PEFT) with dynamic live placeholder replacement for seamless event ticketing support</h3>

<img src="https://github.com/MarpakaPradeepSai/Employee-Churn-Prediction/blob/main/Data/Images%20&%20GIFs/Chatbot_Banner.png?raw=true" alt="Banner" width="650" />
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

The **Event Ticketing Customer Support Chatbot** is an advanced generative AI solution designed to handle customer inquiries related to event ticketing. Moving beyond standard classification, this system leverages the powerful **SmolLM2-1.7B-Instruct** model, fine-tuned efficiently using **Low-Rank Adaptation (LoRA)**. 

### 🎯 What Makes This Special?

Instead of relying on multiple smaller models, this architecture uses a single, highly capable 1.7B parameter instruction-tuned model. It features a custom **Live Replacing Streamer** that dynamically injects real-time data (like website URLs, support links, and UI elements) directly into the generated text stream, ensuring responses are not just accurate, but immediately actionable for the user. It is also explicitly trained to politely refuse out-of-domain (OOD) queries to prevent hallucinations.

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🧠 Parameter-Efficient Fine-Tuning (PEFT)
- Utilizes **LoRA** (Rank=32, Alpha=64) targeting all linear layers
- Drastically reduces trainable parameters while maintaining 1.7B model performance
- Fast, memory-efficient training on consumer GPUs

</td>
<td width="50%">

### 🔄 Dynamic Live Streaming
- Custom `LiveReplacingStreamer` intercepts tokens in real-time
- Seamlessly replaces placeholders (e.g., `{{WEBSITE_URL}}`, `{{CITY}}`) with actual HTML/Markdown links
- Provides a fast, typing-like user experience

</td>
</tr>
<tr>
<td width="50%">

### 🛡️ Out-of-Domain Guardrails
- Trained on a concatenated dataset containing thousands of OOD queries (Science, Politics, etc.)
- Model naturally learns to politely decline irrelevant questions without separate classifier models
- "No hallucinations" enforced via strict System Prompts

</td>
<td width="50%">

### 🧹 Rigorous Data Preprocessing
- Automated removal of duplicate samples
- Filtering and cleaning of offensive language
- Capitalization and phrasing standardization ("Should you" → "If you")
- Instruction formatting using official Chat Templates

</td>
</tr>
</table>

---

## 🏗️ System Architecture

```mermaid
graph TB
    A[👤 User Input] --> B[📝 Apply Chat Template & System Prompt]
    B --> C[🤖 SmolLM2-1.7B-Instruct + LoRA Adapters]
    C --> D{⚡ Text Generation Stream}
    D --> E[🔍 LiveReplacingStreamer]
    E -->|Intercept {{PLACEHOLDERS}}| F[🔄 Inject Static Data/Links]
    F --> G[💬 Final Real-Time Response]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8eaf6
    style D fill:#fff9c4
    style E fill:#ffebee
    style F fill:#e8f5e9
    style G fill:#e0f2f1
```

---

## 🤖 Model Details

### 1️⃣ Base Model: SmolLM2-1.7B-Instruct

<details>
<summary><b>Click to expand details</b></summary>

**Architecture:** Transformer decoder trained in `bfloat16` precision.
**Family:** SmolLM2 (Compact language models by Hugging Face).
**Capabilities:** Strong instruction following, text rewriting, and reasoning for its size.

</details>

### 2️⃣ Fine-Tuning Configuration (LoRA)

<details>
<summary><b>Click to expand training details</b></summary>

**LoRA Parameters:**
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

**Training Arguments:**
```python
TrainingArguments(
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
- **In-Domain:** Bitext events ticketing LLM chatbot training dataset (cleaned)
- **Out-of-Domain:** Extra-large OOD dataset
- **Total Training Samples:** ~28,486 rows

</details>

---

## 📊 Performance Metrics

### Training Progress (Loss Over Steps)

The model was trained using Hugging Face's `SFTTrainer`. Below is the progression of the training loss showcasing smooth convergence over ~1781 steps:

```text
Training Loss Over Steps (Epoch 1):
████████████████████████████████████████████████████████████████████████████
█ Step 100:   ████████████████████████████████████████████  0.4521         █
█ Step 200:   ███████████████████████████████████████       0.3812         █
█ Step 300:   ██████████████████████████████████            0.3105         █
█ Step 400:   ██████████████████████████████                0.2743         █
█ Step 500:   ██████████████████████████                    0.2411         █
█ Step 600:   ████████████████████████                      0.2109         █
█ Step 700:   █████████████████████                         0.1854         █
█ Step 800:   ███████████████████                           0.1632         █
█ Step 900:   █████████████████                             0.1487         █
█ Step 1000:  ████████████████                              0.1352         █
█ Step 1100:  ███████████████                               0.1245         █
█ Step 1200:  ██████████████                                0.1189         █
█ Step 1300:  █████████████                                 0.1123         █
█ Step 1400:  ████████████                                  0.1084         █
█ Step 1500:  ███████████                                   0.1041         █
█ Step 1600:  ██████████                                    0.1012         █
█ Step 1700:  █████████                                     0.0985         █
█ Step 1781:  █████████                                     0.0957         █
████████████████████████████████████████████████████████████████████████████
```
*Final Training Loss at Step 1781:* **0.0957**

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (Highly recommended for 1.7B parameter inference)
- 12GB+ RAM / VRAM

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/MarpakaPradeepSai/Event-Ticketing-SmolLM2-Chatbot.git
cd Event-Ticketing-SmolLM2-Chatbot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required libraries
pip install torch transformers peft trl datasets wandb pandas matplotlib seaborn
```

---

## 💻 Usage

### Example Interactions

<table>
<tr>
<th>User Query</th>
<th>Bot Response (Streamed & Replaced)</th>
</tr>
<tr>
<td>

```text
"How can I upgrade my ticket for 
the upcoming concert in us?"
```

</td>
<td>

```html
To upgrade your ticket for the upcoming concert in the United States, please follow these steps:
1. Go to the [website](https://github.com/MarpakaPradeepSai).
2. Sign in to your account using your login details.
3. Head to the <b>Ticketing</b> section.
4. Find your current ticket under <b>Upgrade Ticket Information</b> and choose the <b>Upgrade Ticket</b> option.
5. Complete the process by following the on-screen prompts to select your desired upgrade and confirm the changes.

If you face any issues during this process, please reach out to our support team for further assistance.
```

</td>
</tr>
<tr>
<td>

```text
"Explain Transformer architecture 
in detail?"
```

</td>
<td>

```text
I apologize, but I can only assist with event ticket-related inquiries. Is there anything about event tickets I can help you with today?
```

</td>
</tr>
</table>

### Programmatic Inference with Live Streamer

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 1. Load Model and Tokenizer
model_path = "./HuggingFaceTB-SmolLM2-1.7B-Instruct-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# 2. Define Static Placeholders
static_placeholders = {
    "{{WEBSITE_URL}}": "[website](https://github.com/MarpakaPradeepSai)",
    "{{CANCEL_TICKET_SECTION}}": "<b>Ticket Cancellation</b>",
    # ... other placeholders
}

# 3. Custom Streamer Class
class LiveReplacingStreamer(TextStreamer):
    def on_finalized_text(self, text: str, stream_end: bool = False):
        for k, v in static_placeholders.items():
            text = text.replace(k, v)
        print(text, end="", flush=True)

# 4. Generate Response
def stream_response(instruction):
    messages = [
        {"role": "system", "content": "You are Eventra, an AI assistant..."},
        {"role": "user", "content": instruction}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    live_streamer = LiveReplacingStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
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

---

## 🔧 Training Pipeline

### Supervised Fine-Tuning (SFT) with TRL

```python
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments

# LoRA Configuration
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

# Training Arguments
training_arguments = TrainingArguments(
    output_dir='./SmolLM2-support',
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

# Initialize Trainer
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_dataset,
    peft_config=peft_config
)

# Start Training
trainer.train()
```

---

## 📁 Project Structure

```text
Event-Ticketing-SmolLM2-Chatbot/
│
├── Data/                       
│   ├── bitext-events-ticketing-llm-chatbot-training-dataset.csv  # Base training data
│   └── extra-large-out-of-domain.csv                             # OOD samples
│
├── Notebooks/                   
│   └── Event_Ticketing_Chatbot_SmolLM2_1.7B_FineTuning.ipynb     # Complete EDA, Training & Inference Pipeline
│
├── requirements.txt            # Project Dependencies
├── LICENSE                     # MIT License
└── README.md                   # Documentation
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

<div align="center">

| Resource | Description |
|----------|-------------|
| [Hugging Face](https://huggingface.co/) | Transformers, PEFT, and TRL libraries |
| [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | Base 1.7B Instruction model by HuggingFaceTB |
| [Bitext](https://www.bitext.com/) | High-quality customer support training datasets |
| [Weights & Biases](https://wandb.ai/) | Experiment tracking and loss visualization |

</div>

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

<br>

**Built with ❤️ by [Marpaka Pradeep Sai](https://github.com/MarpakaPradeepSai)**

</div>
