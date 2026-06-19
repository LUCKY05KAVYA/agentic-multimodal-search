# 🚀 Agentic Multimodal Search & Re-ranking System

An intelligent, end-to-end search system for e-commerce that combines **agentic reasoning**, **semantic understanding**, **neural re-ranking**, and **multimodal capabilities**.

---

## 📌 Overview

This project demonstrates a modern retrieval system that goes beyond traditional keyword search. It uses a **multi-stage pipeline** combining:

- LLM-powered query understanding
- Dense vector search
- Neural re-ranking
- Multimodal (text + image) understanding


---

## ✨ Key Features

- **Agentic Query Decomposition** using LangGraph + LLM
- **Semantic Search** using Sentence Transformers + FAISS
- **Neural Re-ranking** with Cross-Encoder
- **Multimodal Fusion** using CLIP (Text + Image embeddings)
- **Comprehensive Evaluation** with NDCG@10, Recall@10, and MRR
- **Interactive Demo** built with Gradio

---

Results and ablation study have been conducted.

> **Note**: The dataset used has high relevance density, leading to strong scores across all methods. The system architecture is designed to scale to more challenging datasets.

---

## 🏗️ System Architecture
<img width="965" height="2329" alt="image" src="https://github.com/user-attachments/assets/e1dda746-2ae9-4bfb-b6a2-a37bbee3dd45" />

