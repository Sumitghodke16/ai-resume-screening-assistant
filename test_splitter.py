from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Load PDF
PDF_PATH = "sample_resumes/Sumit_Naresh_Ghodke_Sep_2026_Resume.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Original pages: {len(documents)}")


# 2. Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


# 3. Split the document
chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")


# 4. Display chunks
for i, chunk in enumerate(chunks):
    print(f"\n========== CHUNK {i + 1} ==========")
    print(chunk.page_content)