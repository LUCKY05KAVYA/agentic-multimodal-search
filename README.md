# 🚀 Agentic Multimodal Search & Re-ranking System

An intelligent, end-to-end search system for e-commerce that combines **agentic reasoning**, **semantic understanding**, **neural re-ranking**, and **multimodal capabilities**.

---

## 📌 Overview

This project demonstrates a modern retrieval system that goes beyond traditional keyword search. It uses a **multi-stage pipeline** combining:

- LLM-powered query understanding
- Dense vector search
- Neural re-ranking
- Multimodal (text + image) understanding

The system is built to be **production-ready** in concept and showcases skills highly relevant for **Applied Scientist** roles at companies like Amazon, Google, and Meta.

---

## ✨ Key Features

- **Agentic Query Decomposition** using LangGraph + LLM
- **Semantic Search** using Sentence Transformers + FAISS
- **Neural Re-ranking** with Cross-Encoder
- **Multimodal Fusion** using CLIP (Text + Image embeddings)
- **Comprehensive Evaluation** with NDCG@10, Recall@10, and MRR
- **Interactive Demo** built with Gradio

---

## 📊 Results & Ablation Study

We compared three retrieval methods on real e-commerce data:

| Method                  | NDCG@10 | Recall@10 | MRR   | Notes |
|-------------------------|---------|-----------|-------|-------|
| **BM25** (Baseline)     | 0.958   | 1.000     | 1.0   | Traditional keyword search |
| **Dense Retrieval**     | 0.946   | 1.000     | 1.0   | Semantic embeddings only |
| **Dense + Re-ranking**  | 0.946   | 1.000     | 1.0   | Full pipeline |

> **Note**: The dataset used has high relevance density, leading to strong scores across all methods. The system architecture is designed to scale to more challenging datasets.

---

## 🏗️ System Architecture
