from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# -----------------------------
# 1. Load PDF
# -----------------------------

PDF_PATH = "sample_resumes/Sumit_Naresh_Ghodke_Sep_2026_Resume.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Pages loaded: {len(documents)}")


# -----------------------------
# 2. Split text
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")


# -----------------------------
# 3. Create embedding model
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully")


# -----------------------------
# 4. Test one chunk
# -----------------------------

vector = embeddings.embed_query(chunks[0].page_content)

print(f"Embedding vector length: {len(vector)}")

print("First 10 values:")
print(vector[:10])