from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="qwen3:0.6b",
    temperature=0.3
)

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./ec2_chroma_vectorstore", 
)

prompt = ChatPromptTemplate.from_template("""
Answer in spanish briefly

{context}

Question: {input}
Answer:""")

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# "Cómo puedo hacer que mi backend y frontend se comuniquen"
while True:
    query = input("Ingrese consulta: ")
    if query == "EXIT":
        break

    result = retrieval_chain.invoke({
        "input": query
    })

    print(result["answer"])


