from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "sample_resumes/Sumit_Naresh_Ghodke_Sep_2026_Resume.pdf"

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print(f"Number of pages: {len(documents)}")

for i, document in enumerate(documents):
    print(f"\n--- PAGE {i + 1} ---")
    print(document.page_content[:1000])