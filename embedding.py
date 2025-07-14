from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import CharacterTextSplitter

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

file_paths = [
    "./corpus/CORPUS CARLOS.pdf"
]

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = []

for file_path in file_paths:
    loader = PyMuPDF4LLMLoader(file_path, mode="single")
    docs = loader.load()
    doc_splits = text_splitter.split_documents(docs)
    all_splits.extend(doc_splits)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)