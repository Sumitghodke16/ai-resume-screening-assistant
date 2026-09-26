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
# 2. Split PDF into chunks
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")


# --------------------------------
# 3. Load embedding model
# --------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# --------------------------------
# 4. Create FAISS vector database
# --------------------------------

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("FAISS vector database created successfully")


# --------------------------------
# 5. Test similarity search
# --------------------------------

query = "What experience does the candidate have with Python and machine learning?"

results = vectorstore.similarity_search(
    query,
    k=3
)

print("\n========== SEARCH RESULTS ==========")

for i, result in enumerate(results):
    print(f"\n--- RESULT {i + 1} ---")
    print(result.page_content)