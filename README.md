# 🩺 End-to-End Medical Chatbot using LLaMA2, LangChain, FAISS, and Flask

This project is an end-to-end **Medical Chatbot** designed to intelligently answer user questions based on medical documents. It integrates the **LLaMA2 language model**, **LangChain**, **FAISS vector store**, and a **Flask-based web interface**.

## 🚀 Features

* ✅ Locally hosted **LLaMA2** model via `CTransformers`
* ✅ Document-based **Retrieval-Augmented Generation (RAG)**
* ✅ **Semantic Search** with `FAISS` & `HuggingFace Embeddings`
* ✅ **Flask Web App** for user interaction
* ✅ Custom prompt template to improve answer reliability
* ✅ Source documents shown for transparency

---

## 🧠 Tech Stack

| Component     | Library / Tool                                          |
| ------------- | ------------------------------------------------------- |
| LLM           | [LLaMA 2](https://huggingface.co/models) (GGML version) |
| Retrieval     | [FAISS](https://github.com/facebookresearch/faiss)      |
| Embeddings    | `sentence-transformers/all-MiniLM-L6-v2`                |
| Framework     | [LangChain](https://github.com/langchain-ai/langchain)  |
| Web Framework | Flask (Python)                                          |

---

## 📁 Project Structure

```bash
project/
│
├── app.py                           # Flask application
├── templates/
│   └── index.html                   # UI template
├── models/
│   └── llama-2-7b-chat.ggmlv3.q4_0.bin  # Quantized LLaMA2 model
├── vectorstore/
│   └── db_faiss/                    # FAISS database
└── README.md                        # Project description
```

---

## ⚙️ Setup Instructions

### 1. 📦 Install Dependencies

Make sure you have Python 3.9+ installed. Then:

```bash
pip install flask langchain faiss-cpu sentence-transformers ctransformers
```

> Optional: Create a virtual environment.

### 2. 📂 Prepare the Files

* Place your **FAISS vector store** in `vectorstore/db_faiss/`.
* Place the quantized **LLaMA2 model** (e.g., `.bin` file) in `models/`.

### 3. ▶️ Run the Application

```bash
python app.py
```

Go to: `http://127.0.0.1:5000`

---

## 🧪 Example Workflow

1. Ask a medical question (e.g., *"What are the symptoms of anemia?"*)
2. The system:

   * Embeds your query
   * Retrieves top relevant chunks from your FAISS vectorstore
   * Combines them with the custom prompt
   * Sends them to the LLaMA2 model
   * Returns the answer + sources

---

## 📌 Custom Prompt

```text
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know...

Context: {context}
Question: {question}
Helpful answer:
```

This ensures **factual and reliable answers** only from the available context.

---

## 🛠 Troubleshooting

* **❌ FAISS deserialization error**: Use `allow_dangerous_deserialization=True`.
* **❌ Model loading error**: Ensure your model path and format are correct (`.bin`, GGML).
* **Slow inference**: Try quantized models like `q4_0`, or reduce `max_new_tokens`.

---

## 📚 Acknowledgments
* [LangChain](https://github.com/langchain-ai/langchain)
* [FAISS by Facebook](https://github.com/facebookresearch/faiss)
* [CTransformers](https://github.com/marella/ctransformers)
* [LLaMA2](https://huggingface.co)

---

## 📄 License
This project is for educational and research purposes only. Always consult a medical professional for real-life decisions.
