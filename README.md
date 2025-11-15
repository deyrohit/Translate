#🌐 Transformer-Based English ↔ French Translation Model (Custom NLP Project)

A fully custom translation system built from scratch using a causal Transformer architecture.

🔥 Overview

This project implements a complete machine translation model using a custom-built Transformer (encoder–decoder) architecture in PyTorch.
The goal was to build the entire workflow manually — without relying on pre-trained translation models — to deeply understand modern NLP sequence-to-sequence systems.

The model supports:

English → French translation

French → English translation

It achieves 93% translation accuracy (BLEU-based) for English → French and generates fluent, context-aware output.

🚀 Key Features
🔹 1. Fully Custom Transformer Architecture

Implemented from scratch using PyTorch, including:

Multi-Head Self-Attention

Encoder & Decoder blocks

Cross-Attention

Positional Encoding

Causal (autoregressive) masking

Residual & LayerNorm connections

Beam Search & Greedy Decoding

🔹 2. End-to-End Translation Pipeline

The project includes the full lifecycle:

✔ Data Preprocessing

Text normalization

Cleaning noisy sentence pairs

Removing inconsistencies

✔ Tokenization

Subword tokenization (BPE / SentencePiece)

Vocabulary building for both English & French

Padding & attention masking

✔ Training

Teacher forcing

AdamW optimizer

Learning rate warmup schedule

Loss tracking + BLEU evaluation

🔹 3. Real-Time Translation Application

Built a lightweight app where users can:

Enter English → receive French

Enter French → receive English

See processing (tokenization → model inference → decoding)

🔹 4. High Translation Accuracy

Achieved 93% BLEU score on English→French test data

Fluent, context-aware, and grammatically consistent translations
