# 🧠 Mini Semantic Search Graph (OpenAI + Python)

A small experimental app that builds a **semantic search engine** using  
**OpenAI embeddings** and a lightweight **FAISS HNSW** index.

It allows you to:
- 🔍 Enter a **phrase**, **full article**, or a **.txt file**
- ✨ Optionally summarize it with **GPT-4o** or **GPT-4o-mini** before searching
- 🧾 Retrieve the **most relevant keywords** from your dataset

---

## 📁 Project Structure

| File / Folder | Description |
|----------------|-------------|
| `app.py` | Main interactive app |
| `requirements.txt` | Python dependencies |
| `experiments/` | Folder with Colab tests and sample experiments *(no company data included)* |

---

## ⚙️ Setup

# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py
