<p align="center">
  <img src="https://raw.githubusercontent.com/github/explore/main/topics/faiss/faiss.png" alt="FAISS logo" width="120"/>
  <h1 align="center">Mini Semantic Search Graph</h1>
  <p align="center"><b>Micro-scale Semantic Search using OpenAI + Python + FAISS HNSW</b></p>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg"></a>
  <a href="https://openai.com"><img src="https://img.shields.io/badge/OpenAI-Embeddings-orange"></a>
  <a href="https://github.com/facebookresearch/faiss"><img src="https://img.shields.io/badge/FAISS-HNSW-green"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey.svg"></a>
</p>

---

### 🧠 Overview

**Mini Semantic Search Graph** is a small experimental project that builds a **semantic search engine**  
using **OpenAI embeddings** and a **FAISS HNSW index** for quick keyword retrieval.

It allows you to:

- 🔍 Enter a **phrase**, **full article**, or provide a **.txt file**
- ✨ Optionally summarize it with **GPT-4o** or **GPT-4o-mini** before searching
- 🧾 Retrieve the **most relevant keywords** from your dataset

---

### 🧩 Project Structure

| File / Folder | Description |
|----------------|-------------|
| `app.py` | Main interactive app |
| `requirements.txt` | Python dependencies |
| `experiments/` | Colab tests and example experiments *(no company data included)* |

---

### ⚙️ Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py
