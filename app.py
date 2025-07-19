from flask import Flask, render_template, request
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import os

app = Flask(__name__)

DB_FAISS_PATH = r'C:\Users\LENOVO\AMIT AI\Amit-1\myenv\Scripts\End-to-end-Medical-Chatbot-using-Llama2-main\End-to-end-Medical-Chatbot-using-Llama2-main\vectorstore\db_faiss'
MODEL_PATH = r'C:\Users\LENOVO\AMIT AI\Amit-1\myenv\Scripts\End-to-end-Medical-Chatbot-using-Llama2-main\End-to-end-Medical-Chatbot-using-Llama2-main\models\llama-2-7b-chat.ggmlv3.q4_0.bin'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Prepare prompt
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# Load the model
def load_llm():
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# Build the QA chain once
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': qa_prompt}
    )
    return qa

# Load bot once globally
qa_chain = qa_bot()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    try:
        result = qa_chain({'query': question})
        answer = result['result']
        sources = result.get("source_documents", [])
        source_list = [doc.metadata.get("source", "Unknown") for doc in sources]
        unique_sources = list(set(source_list))
        source_text = "\n".join(f"- {src}" for src in unique_sources)
        if source_text:
            answer += f"\n\n📚 Sources:\n{source_text}"
        else:
            answer += "\n\n⚠️ No sources found."
    except Exception as e:
        answer = f"❌ Error: {str(e)}"
    return render_template("index.html", question=question, answer=answer, sources=source_list)


if __name__ == '__main__':
    app.run(debug=True)
