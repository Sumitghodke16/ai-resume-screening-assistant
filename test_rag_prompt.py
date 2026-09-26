from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------
# 1. Load resume
# --------------------------------

PDF_PATH = "sample_resumes/Sumit_Naresh_Ghodke_Sep_2026_Resume.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()


# --------------------------------
# 2. Split resume
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)


# --------------------------------
# 3. Embeddings
# --------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------
# 4. FAISS
# --------------------------------

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)


# --------------------------------
# 5. Retriever
# --------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)


# --------------------------------
# 6. Job Description
# --------------------------------

job_description = """
We are looking for a Data Scientist.

Required skills:
- Python
- SQL
- Machine Learning
- Statistics
- Scikit-learn
- Predictive Modeling
- Data Analysis

The candidate should have experience building
machine learning models and analyzing business data.
"""


# --------------------------------
# 7. Retrieve resume information
# --------------------------------

retrieved_documents = retriever.invoke(job_description)


# --------------------------------
# 8. Combine retrieved documents
# --------------------------------

context = "\n\n".join(
    document.page_content
    for document in retrieved_documents
)


# --------------------------------
# 9. Create RAG prompt
# --------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are an AI Resume Screening Assistant.

Your task is to evaluate a candidate against a Job Description.

IMPORTANT RULES:

1. Use ONLY the information provided in the retrieved resume context.
2. Do NOT invent skills, experience, education, or achievements.
3. If information is not present in the resume context, say "Not mentioned".
4. Base the evaluation on evidence from the resume.
5. The Job Description is used as the comparison requirement.
6. Do not use outside knowledge about the candidate.

JOB DESCRIPTION:

{job_description}

RETRIEVED RESUME CONTEXT:

{context}

Evaluate the candidate and provide:

1. Match Score: 0-100
2. Matching Skills
3. Missing Skills
4. Candidate Summary
5. Strengths
6. Weaknesses
7. Hiring Recommendation
8. Evidence from Resume
"""
)


# --------------------------------
# 10. Generate final prompt
# --------------------------------

final_prompt = prompt.format(
    job_description=job_description,
    context=context
)


# --------------------------------
# 11. Display prompt
# --------------------------------

print("\n========== RETRIEVED CONTEXT ==========")
print(context)

print("\n========== FINAL RAG PROMPT ==========")
print(final_prompt)