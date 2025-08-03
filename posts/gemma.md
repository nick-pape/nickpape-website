---
title: Fine-tuning Gemma with LoRA
nav_order: 1
parent: Posts
---

{: .label .label-purple } Project
{: .label .label-green } LLM


> Train a small Gemma model in your voice, on your own GPU, without needing 8 A100s or a datacenter.
{: .fs-5 .fw-300 .mb-4 }

---

## ❓ Why this works

Large language models are massive. Fine-tuning everything in them is expensive, slow, and usually overkill if all you want is a change in tone or narrow behavior tuning.

Enter **LoRA (Low-Rank Adaptation)**.

{: .note }
Instead of touching all billions of weights, LoRA adds a few small matrices inside attention layers. During training, you only update those.

Benefits:
- **Tiny memory footprint** (works on a single consumer GPU)
- **Fast training** (minutes, not days)
- **Composable** (train multiple adapters and swap them in/out)

On top of that, this repo loads Gemma in **4-bit quantized mode**, cutting memory even further without killing performance.

---

## 🎯 Goal of this repo

{: .label .label-blue } Personal Project  
{: .label .label-green } First LLM Fine-Tune

I wanted to build a model that sounded like me. Not just "fine-tuned on Nick’s data" — I mean something that:
- is direct,
- skips the fluff,
- pushes back when needed,
- doesn’t over-apologize or over-explain.

Think: a local custom GPT, with my tone and attitude.

---

## 🧹 Preparing your data

Gemma expects a very specific chat template. You need to match it:

```json
[
  {
    "text": "<bos><start_of_turn>user What's 2+2?<end_of_turn><start_of_turn>assistant It's 4. If you're overthinking it, don't.<end_of_turn>"
  }
]
```

{: .important }
Each item is a full conversation string. You are not feeding message arrays — just flattened chat with tags.

**Formatting rules:**
- Start each chat with `<bos>`
- Wrap turns in `<start_of_turn>{role}` and `<end_of_turn>`
- Alternate roles consistently
- Avoid markdown, extra newlines, or broken tagging
- Keep length under `SEQUENCE_LENGTH` (defaults to 1024 tokens)

To help with formatting, use `tokenizer.apply_chat_template()` to generate compliant samples. Then save as a list of `{ "text": ... }` objects.

{: .tip }
I wrote my first ~300 examples by hand. If you want personality to come through, quality matters more than quantity.

---

## 🛠 Fine-tuning

Set up `config.py` to point at your dataset and model name, then:

```bash
python train.py
```

This runs a full training loop using HuggingFace’s `SFTTrainer`, LoRA adapters, and 4-bit quantization.

What’s actually happening:
- Loads `google/gemma-2b-it` in 4-bit mode
- Adds LoRA adapters to the main attention projections
- Freezes everything else
- Trains just the adapters for one epoch
- Saves output to `./models/{your_model_name}`

**Training defaults:**
- Batch size: 2
- Gradient accumulation: 1
- Learning rate: 2e-4
- Optimizer: `paged_adamw_32bit`
- Logging: every 100 steps

{: .note }
You can change any of this — including LoRA dropout, target modules, or optimizer — but the defaults were enough to get useful tone shifts.

---

## 💬 Using your model

Once trained:

```bash
python chat.py
```

What it does:
- Loads the base model
- Merges in your LoRA weights
- Starts a multi-turn chat loop
- Uses `ChatHistory` to build a valid prompt
- Calls `model.generate()` with beam search + sampling
- Stops at the first `<end_of_turn>`

This is just a CLI tool right now, but it’s enough to test tone, phrasing, and behavior. If you want a UI, you could wrap this with Gradio in ~10 lines.

---

## 📈 Results

Still testing. But even with a few hundred examples, I’m seeing:
- noticeably different tone
- better structure on common prompts
- fewer excessive apologies or disclaimers

Things I still want to test:
- long vs. short examples
- training assistant-only vs. full chats
- adding examples with contradictions or corrections

---

## 🧭 Notes and future work

Ideas for improvement:
- Support for Gemma 1.5
- Adapter switching at runtime
- Gradio or Web UI
- Auto-formatting tool for raw chat logs
- WandB or TensorBoard tracking

---

## 📦 Setup instructions

{: .label .label-yellow } See: [README.md](./README.md)

Covers:
- Python version (3.11 recommended)
- CUDA and GPU support
- HuggingFace token setup
- `requirements.txt` and install notes

If you can run PyTorch + transformers on your GPU, you can run this.
