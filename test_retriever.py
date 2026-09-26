from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------
# 1. Load PDF
# --------------------------------

PDF_PATH = "sample_resumes/Sumit_Naresh_Ghodke_Sep_2026_Resume.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Pages loaded: {len(documents)}")


# --------------------------------
# 2. Split text
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")


# --------------------------------
# 3. Create embeddings
# --------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# --------------------------------
# 4. Create FAISS vector store
# --------------------------------

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("FAISS vector database created")


# --------------------------------
# 5. Create Retriever
# --------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("Retriever created successfully")


# --------------------------------
# 6. Test Retriever
# --------------------------------

query = """
What technical skills and machine learning
experience does the candidate have?
"""

retrieved_documents = retriever.invoke(query)


# --------------------------------
# 7. Display results
# --------------------------------

print("\n========== RETRIEVED DOCUMENTS ==========")

for i, document in enumerate(retrieved_documents):

    print(f"\n--- DOCUMENT {i + 1} ---")

    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)