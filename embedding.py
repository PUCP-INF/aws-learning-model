from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

file_paths = [
    "./corpus/CORPUS CARLOS.pdf",
    "./corpus/CORPUS JHOSEPT.pdf",
    "./corpus/CORPUS VICTOR.pdf",
    "./corpus/CORPUS DIANA.pdf"
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_splits = []

for file_path in file_paths:
    print(f"Loading {file_path} to vector store")
    loader = PyMuPDF4LLMLoader(file_path, mode="single")
    docs = loader.load()
    doc_splits = text_splitter.split_documents(docs)
    all_splits.extend(doc_splits)
    print(f"Loaded {len(doc_splits)} documents")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./ec2_chroma_vectorstore",
)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)