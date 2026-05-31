from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0)

df = pd.read_csv("data.csv").head(6000).reset_index(drop=True)

product_col = None
for col in ['product_title', 'product_name', 'title', 'name']:
    if col in df.columns:
        product_col = col
        break

print(f"Using column: {product_col}")

# BM25 Setup
corpus = df[product_col].tolist()
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Dense Embeddings + FAISS
embedder = SentenceTransformer('all-MiniLM-L6-v2')
product_embeddings = embedder.encode(corpus)
index = faiss.IndexFlatIP(product_embeddings.shape[1])
faiss.normalize_L2(product_embeddings)
index.add(product_embeddings.astype('float32'))

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

test_queries = [
    {"query": "noise cancelling wireless headphones", "keywords": ["headphone", "noise cancelling"]},
    {"query": "red nike running shoes", "keywords": ["nike", "running", "shoe"]},
    {"query": "mechanical rgb gaming keyboard", "keywords": ["mechanical", "gaming", "keyboard"]},
]

class AgentState(TypedDict):
    query: str
    results: List[str]

def bm25_search(query, top_k=10):
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return df.iloc[top_indices][product_col].tolist()

def semantic_search(state: AgentState):
    query = state["query"]
    query_embedding = embedder.encode([query])
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding.astype('float32'), 25)
    return {"results": df.iloc[indices[0]][product_col].tolist()}

def rerank(state: AgentState):
    query = state["query"]
    results = state["results"]
    pairs = [[query, r] for r in results]
    scores = cross_encoder.predict(pairs)
    sorted_results = [x for _, x in sorted(zip(scores, results), reverse=True)]
    return {"results": sorted_results[:8]}

workflow = StateGraph(AgentState)
workflow.add_node("search", semantic_search)
workflow.add_node("rerank", rerank)
workflow.set_entry_point("search")
workflow.add_edge("search", "rerank")
workflow.add_edge("rerank", END)
app = workflow.compile()

def evaluate_all_methods():
    results = {}
    
    for method in ["BM25", "Dense", "Dense + Re-rank"]:
        ndcg_list, recall_list, mrr_list = [], [], []
        
        for test in test_queries:
            if method == "BM25":
                retrieved = bm25_search(test["query"], top_k=10)
            else:
                result = app.invoke({"query": test["query"]})
                retrieved = result["results"][:10]
            
            y_true = [1 if any(kw in item.lower() for kw in test["keywords"]) else 0 for item in retrieved]
            y_score = list(range(10, 0, -1))
            
            if sum(y_true) > 0:
                dcg = sum([rel / np.log2(rank + 2) for rank, rel in enumerate(y_true)])
                idcg = sum([1 / np.log2(rank + 2) for rank in range(min(sum(y_true), 10))])
                ndcg = dcg / idcg if idcg > 0 else 0
                
                ndcg_list.append(ndcg)
                recall_list.append(min(sum(y_true) / len(test["keywords"]), 1.0))
                
                for rank, item in enumerate(retrieved):
                    if any(kw in item.lower() for kw in test["keywords"]):
                        mrr_list.append(1 / (rank + 1))
                        break
                else:
                    mrr_list.append(0)
        
        results[method] = {
            "NDCG@10": round(np.mean(ndcg_list), 3),
            "Recall@10": round(np.mean(recall_list), 3),
            "MRR": round(np.mean(mrr_list), 3)
        }
    
    return results

def run_agent(query):
    result = app.invoke({"query": query})
    return "\n".join(result["results"])

demo = gr.Interface(
    fn=run_agent,
    inputs=gr.Textbox(label="Search Query"),
    outputs=gr.Textbox(label="Top Results"),
    title="Agentic Multimodal Search System"
)

if __name__ == "__main__":
    print("\n=== ABLATION STUDY ===")
    results = evaluate_all_methods()
    for method, metrics in results.items():
        print(f"{method}: {metrics}")
    demo.launch()