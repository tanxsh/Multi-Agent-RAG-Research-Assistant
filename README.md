# Multi-Agent RAG Research Assistant

**A fully local multi-agent system for intelligent PDF question answering using LangGraph, Llama 3.2, and ChromaDB**

Course: Generative AI for the Enterprise | Santa Clara University

---

## Overview

A multi-agent research assistant that allows users to upload PDF documents and ask questions — combining RAG (Retrieval-Augmented Generation) with a LangGraph-powered agent pipeline. The entire stack runs **locally** with no data leaving the system and zero API costs.

The system goes beyond a basic RAG pipeline by implementing intelligent routing and self-correction: if retrieved documents are irrelevant to the question, the system automatically falls back to general knowledge rather than hallucinating an answer.

---

## Architecture

```
User Question
      │
      ▼
 [Router Agent]  ──── General question? ────► [Generalist Agent] ──► Answer
      │
      │ Document question?
      ▼
 [Retriever Agent]  (ChromaDB vector search)
      │
      ▼
 [Analyzer Agent]  ──── Docs irrelevant? ────► [Generalist Agent] ──► Answer
      │
      │ Docs relevant?
      ▼
 [Synthesizer Agent] ──► Final Answer
```

### Key Architectural Choices

**1. Local Stack — Zero Cost, Zero Data Leakage**
- LLM: `ChatOllama` running **Llama 3.2** — local inference, low latency
- Embeddings: `HuggingFaceEmbeddings` (all-MiniLM-L6-v2) — runs entirely on-device
- Vector DB: **ChromaDB** — local persistent storage

**2. LangGraph over CrewAI**
- Used `StateGraph` with shared `AgentState` and conditional edges for deterministic routing
- CrewAI's autonomous delegation is non-deterministic — LangGraph gives precise control over agent flow and fallback logic

**3. 5-Agent Design**
Breaking the task into specialized agents reduces hallucination risk compared to a single large prompt:

| Agent | Role |
|---|---|
| Router | Intent classification — document query vs. general knowledge |
| Retriever | Fetches top-5 relevant chunks from ChromaDB |
| Analyzer | Validates whether retrieved docs actually answer the question |
| Synthesizer | Drafts final answer grounded in validated document context |
| Generalist | Answers from general knowledge when documents are irrelevant or absent |

**4. Self-Correction Loop**
The Analyzer acts as a quality gate — if retrieved chunks are flagged as `IRRELEVANT`, the graph re-routes to the Generalist instead of generating a hallucinated answer from bad context.

---

## Features

- Upload one or more PDFs via Streamlit sidebar
- Documents chunked (1000 tokens, 200 overlap) and embedded into local ChromaDB
- Real-time agent status display as the pipeline executes
- Source document chunks surfaced only when Analyzer confirms relevance
- Full chat interface with conversation history
- Graceful error handling for empty uploads, failed indexing, and edge cases

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3.2 via Ollama (local) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local) |
| Vector Database | ChromaDB |
| Agent Framework | LangGraph (StateGraph) |
| Document Loader | LangChain PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| UI | Streamlit |

---

## Setup & Running

### Prerequisites
```bash
# Install Ollama and pull Llama 3.2
ollama pull llama3.2

# Install dependencies
pip install streamlit langchain langgraph langchain-community langchain-chroma langchain-huggingface chromadb sentence-transformers pypdf
```

### Run
```bash
streamlit run midterm_solution.py
```

---

## Repository Structure

```
├── midterm_solution.py     # Full application: agents, graph, vector DB, Streamlit UI
└── README.md
```

---

*Generative AI for the Enterprise — Santa Clara University*
