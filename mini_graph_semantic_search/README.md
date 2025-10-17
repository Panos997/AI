<p align="center">
  <img src="https://raw.githubusercontent.com/github/explore/main/topics/faiss/faiss.png" alt="FAISS logo" width="110"/>
  <h1 align="center">Mini Semantic Search Graph</h1>
  <p align="center"><b>Experimental Semantic Search App using OpenAI + FAISS HNSW</b></p>
</p>

---

## 🧠 Overview

**Mini Semantic Search Graph** is an **experimental project** built during internal R&D testing.  
We started with a **small base of clustered keywords**, each cluster representing a thematic category.  
Initially, we tried **semantic matching per cluster** — searching within each keyword group separately —  
but this approach didn’t generalize well and produced unstable matches.  

By building a **FAISS HNSW index** across all keywords, the graph-based structure achieved  
**much stronger and more consistent results** for semantic similarity and article-to-keyword mapping.  
In short: *the FAISS-based graph simply worked better*.

---

## ⚙️ What It Does

The app lets you:
- 🔍 Enter a **phrase**, **full article**, or a **.txt file**
- ✨ Optionally summarize it with **GPT-4o** or **GPT-4o-mini** before searching
- 🧾 Retrieve the **most relevant keywords** from your dataset

---

## 📁 Project Structure

| File / Folder | Description |
|----------------|-------------|
| `app.py` | Main interactive app |
| `requirements.txt` | Dependencies |
| `experiments/` | Colab tests and sample experiments *(no company data included)* |

---

## 🚀 Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py
