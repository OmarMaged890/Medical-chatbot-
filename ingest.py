from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load PDF
loader = PyPDFLoader(r"C:\Users\LENOVO\AMIT AI\Amit-1\myenv\Scripts\End-to-end-Medical-Chatbot-using-Llama2-main\End-to-end-Medical-Chatbot-using-Llama2-main\Medical_book.pdf")
pages = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
docs = text_splitter.split_documents(pages)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

print("✅ Ingestion Done! PDF indexed and vector DB saved.")
