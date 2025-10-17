# 🧠 AI Auto Tagging Generator

An **AI-powered tagging tool** that automatically generates relevant tags for text or article content using **Google’s Gemini GenAI API**.  
It’s designed for use in **content classification**, **SEO**, and **editorial automation** workflows.

---

## 🚀 Overview

This project reads an article or paragraph of text and produces meaningful tags using the `gemini-2.5-flash` model.  
It includes a Python module for programmatic use and a simple UI for interactive testing.

---

## ⚙️ How It Works

1. The Gemini client is initialized with your API key.  
2. The article text is passed to the model.  
3. The model responds with a JSON list of tags.  
4. Tags are cleaned, deduplicated, and limited to your specified maximum count.

---

## 📂 Project Files

| File | Description |
|------|--------------|
| **Code.ipynb** | Example Jupyter/Colab notebook for testing the generator interactively. |
| **genai_client.py** | Core logic for initializing the Gemini API client, validating keys, and generating tags. |
| **requirements.txt** | Lists all Python dependencies required to run the app. |
| **styles.py** | Contains style definitions (colors, layout, fonts) for the user interface. |
| **ui.py** | Launches a small UI that allows you to input article text and see generated tags visually. |

---

## 🧠 Example Usage (Python)

```python
from genai_client import init_client, generate_tags

client = init_client("YOUR_GEMINI_API_KEY")
tags = generate_tags(client, "The European Union introduced new AI regulations.")
print(tags)
