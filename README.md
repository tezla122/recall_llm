# 🧠 Recall-LLM: Local Multimodal Memory Agent

Recall-LLM is a fully local, private, "photographic memory" AI agent. It acts as a bridge between your visual files and a conversational AI, allowing you to seamlessly search and chat with your saved screenshots and photos entirely offline. 

By dropping an image into a designated folder, the system automatically analyzes it, stores its context, and makes it instantly queryable via a conversational LLM interface.

---

## ✨ Features

*   **Zero-Click Ingestion:** Simply drop an image into the `recall-inbox` folder. A background daemon instantly detects, queues, and processes it without interrupting your workflow.
*   **Deep Visual Understanding:** Uses state-of-the-art Vision-Language Models (VLMs) to generate highly detailed, hundreds-of-words-long textual descriptions of your images.
*   **Semantic Search:** Stores memories in a vector database, allowing you to search your images by *meaning* or *content* (e.g., "Find the picture of the green car") rather than filename.
*   **Conversational UI:** A clean, responsive Streamlit web interface for chatting with your visual memory bank.
*   **100% Local & Private:** No API keys, no cloud processing, and no internet connection required after initial setup.

---

## 🛠️ Architecture & Tech Stack

This project meshes several modern AI frameworks into a single asynchronous pipeline:

*   **Frontend:** [Streamlit](https://streamlit.io/) (for the chat interface and image rendering).
*   **Vision Engine:** [Moondream2](https://huggingface.co/vikhyatk/moondream2) (running via PyTorch and Hugging Face `transformers` on CPU).
*   **Language Brain:** [Phi-3.5](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) (running locally via [Ollama](https://ollama.com/)).
*   **Memory Vault (Vector DB):** [LanceDB](https://lancedb.com/) (for embedding and retrieving semantic text).
*   **Concurrency:** Python `asyncio` & `multiprocessing` (keeps the UI fast while the CPU processes heavy vision tasks in the background).

---

## 🚀 Prerequisites

Before installing, ensure you have the following on your system:
1.  **Python 3.10+**
2.  **Ollama** installed and running on your machine.
3.  The **Phi-3.5** model pulled in Ollama:
    ```bash
    ollama run phi3.5
    ```

---

## ⚙️ Installation & Setup

**1. Clone the repository and navigate to the project directory:**
```bash
git clone [https://github.com/YOUR_USERNAME/recall_llm.git](https://github.com/YOUR_USERNAME/recall_llm.git)
cd recall_llm
