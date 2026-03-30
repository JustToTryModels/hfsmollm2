# 🎫 Eventra: Fine-Tuning SmolLM2 for Event Ticketing Support

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-4.46+-yellow?style=for-the-badge)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-blue?style=for-the-badge)
![SmolLM2](https://img.shields.io/badge/Model-SmolLM2--1.7B-purple?style=for-the-badge)

<h3>🚀 A high-performance, lightweight ticketing chatbot fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) with LoRA</h3>

<img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F62c3e1e2-632b-478a-86c4-1a3b379e9a63_1456x842.png" alt="LoRA Architecture" width="600" />
</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Model Details](#-model-details)
- [Data Cleaning Pipeline](#-data-cleaning-pipeline)
- [Training Pipeline](#-training-pipeline)
- [Performance Metrics](#-performance-metrics)
- [Inference & Usage](#-inference--usage)
- [Project Structure](#-project-structure)

---

## 🌟 Overview

This project focuses on fine-tuning the **SmolLM2-1.7B-Instruct** model to create **Eventra**, a specialized AI assistant for event ticketing support. By leveraging **Low-Rank Adaptation (LoRA)**, we adapt a compact yet powerful 1.7-billion parameter model to handle domain-specific queries like ticket cancellations, upgrades, and refunds with high precision and low computational overhead.

### 🎯 Objective
To transform a general-purpose small language model into a specialized support agent that:
1. Understands ticketing domain terminology.
2. Politely refuses Out-of-Domain (OOD) queries.
3. Provides structured, actionable steps for customer issues.
4. Operates efficiently on consumer-grade hardware.

---

## ✨ Key Features

| Feature | Description |
|:--- |:--- |
| **LoRA Fine-tuning** | Uses PEFT to update only a small subset of parameters, reducing memory usage while maintaining performance. |
| **Robust Data Cleaning** | Automated removal of offensive language, duplicates, and inconsistent placeholder formatting. |
| **OOD Handling** | Integrated Out-of-Domain dataset to train the model on when to say "I cannot assist with this." |
| **Live Streaming** | Implementation of `TextStreamer` and custom `LiveReplacingStreamer` for real-time, interactive responses. |
| **Dynamic Placeholders** | Real-time replacement of technical placeholders (e.g., `{{WEBSITE_URL}}`) with user-friendly links and HTML formatting. |

---

## 🤖 Model Details

### SmolLM2-1.7B-Instruct
SmolLM2 is a family of compact models by Hugging Face. The 1.7B variant strikes an ideal balance between reasoning capability and on-device efficiency.

*   **Architecture:** Transformer Decoder
*   **Precision:** bfloat16 / float16
*   **Training Basis:** 11 trillion tokens

### Fine-Tuning Technique: LoRA
Instead of updating all 1.7B parameters, we inject low-rank matrices into the linear layers:
*   **Rank (r):** 32
*   **Alpha:** 64
*   **Target Modules:** All linear layers
*   **Trainable Parameters:** ~2-3% of the total model size

---

## 🧹 Data Cleaning Pipeline

Before training, the raw dataset underwent a rigorous multi-stage cleaning process:

1.  **Duplicate Removal:** Identified and removed redundant sample pairs.
2.  **Profanity Filtering:** Cleaned the `instruction` column by removing offensive words while preserving the core intent of the query.
3.  **Placeholder Standardization:** Converted various placeholder formats (like `{{TICKET_EVENT}}`) into a unified `{{EVENT}}` format.
4.  **Phrasing Adjustment:** Refined response phrasing (e.g., changing "Should you" to "If you") to ensure a more professional and helpful tone.
5.  **Dataset Augmentation:** Concatenated the core ticketing data with 3,700+ Out-of-Domain samples to prevent hallucinations on non-ticketing topics.

---

## ⚙️ Training Pipeline

The model was trained using the `SFTTrainer` from the TRL library for one epoch.

### Hyperparameters
```python
training_arguments = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    optim="adamw_torch",
    lr_scheduler_type="linear",
    target_modules="all-linear"
)
```

### Training Progress
| Step | Training Loss |
|:--- |:--- |
| 100 | 0.3842 |
| 200 | 0.2155 |
| 500 | 0.1689 |
| 800 | 0.1422 |
| 1100 | 0.1251 |
| 1400 | 0.1104 |
| 1700 | 0.0988 |
| **1781 (Final)** | **0.0957** |

---

## 🚀 Inference & Usage

The inference pipeline includes a **System Prompt** that defines Eventra's identity and a custom streamer for live UI updates.

### System Prompt Logic
Eventra is instructed to:
1. Handle event-ticket queries specifically.
2. Handle placeholders like `{{CITY}}` and `{{EVENT}}`.
3. Provide a standard "I apologize..." response for non-ticket related queries.

### Example Interaction
**User:** "How can I upgrade my ticket for the upcoming concert in us?"

**Eventra:** 
> "To upgrade your ticket for the upcoming concert in the United States, please follow these steps:
> 1. Go to the **Ticketing** section on our website...
> 2. Select **Upgrade Ticket Information**..."

---

## 📁 Project Structure

```text
SmolLM2-Event-Ticketing/
│
├── Notebook/
│   └── SmolLM2_Fine_Tuning_Ticketing_Bot.ipynb   # Complete training & cleaning logic
│
├── Data/
│   ├── ticketing_dataset.csv                     # Core training data
│   └── out_of_domain_samples.csv                 # OOD query samples
│
├── Models/
│   └── SmolLM2-1.7B-Instruct-finetuned/         # Final LoRA adapters & config
│
├── app/
│   └── inference_engine.py                       # Streaming inference implementation
│
└── README.md
```

---

<div align="center">

### ⭐ Support the Project
If you find this fine-tuning implementation helpful, consider starring the repository!

**Developed by [Pradeep](https://github.com/MarpakaPradeepSai)**

</div>
