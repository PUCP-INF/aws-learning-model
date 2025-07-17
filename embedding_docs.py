# Setup
#!pip install -q langchain langchain-community  wikipedia chromadb
#!apt-get install -y poppler-utils
#!apt-get install tesseract-ocr
#!pip install "unstructured[pdf]" "pytesseract" "pdf2image" "langchain" "pymupdf" "python-docx"
#!pip install -qU langchain-community unstructured pdf2image
#!pip install unstructured pdfminer.six python-docx pi_heif unstructured_inference

import os
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredPDFLoader

#LECTURA
loader = UnstructuredPDFLoader("./ec2-ug.pdf")
ec2documents = loader.load()
print(f"{len(ec2documents)} documento(s) cargado(s).")

loader = UnstructuredPDFLoader("./rds-ug.pdf")
rdsdocuments = loader.load()
print(f"{len(rdsdocuments)} documento(s) cargado(s).")

loader = UnstructuredPDFLoader("./s3-userguide.pdf")
s3documents = loader.load()
print(f"{len(s3documents)} documento(s) cargado(s).")

#CHUNKEO
print("\n--- Using RecursiveCharacterTextSplitter ---")
rec_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)
ec2_splits = rec_splitter.split_documents(ec2documents)
rds_splits = rec_splitter.split_documents(rdsdocuments)
s3_splits = rec_splitter.split_documents(s3documents)
print(f"ec2 Split into {len(ec2_splits)} chunks")
print(f"rds Split into {len(rds_splits)} chunks")
print(f"s3 Split into {len(s3_splits)} chunks")

#CONVERSION A EMBEDDING
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import CharacterTextSplitter

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

# Función auxiliar para dividir en lotes
def batch_documents(docs, batch_size):
    for i in range(0, len(docs), batch_size):
        yield docs[i:i + batch_size]

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


ec2vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./ec2_chroma_vectorstore"
)
for batch in batch_documents(ec2_splits, batch_size=5000):
    ec2vector_store.add_documents(documents=batch)



rdsvector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./rds_chroma_vectorstore"
)

for batch in batch_documents(rds_splits, batch_size=5000):
    rdsvector_store.add_documents(documents=batch)

s3vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./s3_chroma_vectorstore"
)

for batch in batch_documents(s3_splits, batch_size=5000):
    s3vector_store.add_documents(documents=batch)


#ZIPEO
import shutil
from google.colab import files

# Comprimir la carpeta 
shutil.make_archive('ec2_chroma_vectorstore', 'zip', 'ec2_chroma_vectorstore')

# Descargar el archivo ZIP
files.download('ec2_chroma_vectorstore.zip')

# Comprimir la carpeta
shutil.make_archive('rds_chroma_vectorstore', 'zip', 'rds_chroma_vectorstore')

# Descargar el archivo ZIP
files.download('rds_chroma_vectorstore.zip')

# Comprimir la carpeta 
shutil.make_archive('s3_chroma_vectorstore', 'zip', 's3_chroma_vectorstore')

# Descargar el archivo ZIP
files.download('s3_chroma_vectorstore.zip')


