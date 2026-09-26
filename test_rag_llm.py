# ============================================================
# AI RESUME SCREENING ASSISTANT
# RAG + FAISS + Llama 3.2
# Controlled Job Requirements
# ============================================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# ============================================================
# 1. LOAD RESUME PDF
# ============================================================

pdf_path = "sample_resumes/Sumit_Naresh_Ghodke_Sep_2026_Resume.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("\nPDF loaded successfully.")
print(f"Number of pages: {len(documents)}")


# ============================================================
# 2. SPLIT RESUME INTO CHUNKS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")


# ============================================================
# 3. CREATE EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ============================================================
# 4. CREATE FAISS VECTOR DATABASE
# ============================================================

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("FAISS vector database created successfully.")


# ============================================================
# 5. CREATE RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 12}
)

print("Retriever created successfully.")


# ============================================================
# 6. JOB DESCRIPTION
# ============================================================

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


# ============================================================
# 7. CONTROLLED REQUIREMENTS
# ============================================================
# IMPORTANT:
# Python controls the requirements.
# Llama is NOT allowed to create new requirements.

required_skills = [
    "Python",
    "SQL",
    "Machine Learning",
    "Statistics",
    "Scikit-learn",
    "Predictive Modeling",
    "Data Analysis"
]


print("\n" + "=" * 70)
print("JOB REQUIREMENTS CONTROLLED BY PYTHON")
print("=" * 70)

for skill in required_skills:
    print(f"- {skill}")


# ============================================================
# 8. RETRIEVE RESUME INFORMATION
# ============================================================

query = """
Retrieve resume information relevant to evaluating the candidate
against the Job Description.

Focus on:

Python
SQL
Machine Learning
Statistics
Scikit-learn
Predictive Modeling
Data Analysis

Also retrieve relevant professional experience, projects,
education, certifications, model-building experience,
and business data analysis experience.
"""

retrieved_documents = retriever.invoke(query)


# ============================================================
# 9. DISPLAY RETRIEVED INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("RETRIEVED RESUME INFORMATION")
print("=" * 70)

context = ""

for i, doc in enumerate(retrieved_documents, start=1):

    print(f"\n--- Retrieved Chunk {i} ---")
    print(doc.page_content)

    context += f"\n--- Resume Chunk {i} ---\n"
    context += doc.page_content


# ============================================================
# 10. CONVERT REQUIREMENTS TO TEXT
# ============================================================

requirements_text = "\n".join(
    f"{i}. {skill}"
    for i, skill in enumerate(required_skills, start=1)
)


# ============================================================
# 11. STRICT RAG PROMPT
# ============================================================

prompt = f"""
You are an AI Resume Screening Assistant.

Your task is to evaluate ONE candidate resume against ONE
Job Description.

You must follow the rules exactly.


============================================================
CRITICAL RULES
============================================================

RULE 1:
Use ONLY the information inside RESUME CONTEXT.

RULE 2:
Do NOT use outside knowledge about the candidate.

RULE 3:
Do NOT invent candidate skills, experience, education,
projects, technologies, achievements, or responsibilities.

RULE 4:
You MUST evaluate ONLY the seven requirements listed below.

These are the ONLY requirements:

{requirements_text}

RULE 5:
DO NOT create additional requirements.

DO NOT evaluate:

- Certifications
- Colleges
- Degrees
- Hobbies
- Personal details
- Dates
- Locations
- Additional technologies
- Additional machine learning algorithms
- Any other information

unless it is directly relevant as evidence for one of
the seven required skills.

RULE 6:
The seven requirements above are fixed.

You MUST NOT add requirements such as:

Power BI
Tableau
Excel
Random Forest
KNN
XGBoost
PCA
NLP
Deep Learning
ANN
CNN
RNN
Git
GitHub

unless one of those is explicitly one of the seven requirements.

RULE 7:
A requirement is MATCHING only when the Resume Context
contains evidence supporting that requirement.

RULE 8:
A requirement is MISSING only when there is no evidence
supporting that requirement in the available Resume Context.

RULE 9:
If the available Resume Context is insufficient to determine
the status, use:

NOT VERIFIABLE

RULE 10:
If the resume explicitly mentions a required skill,
NEVER classify that required skill as MISSING.

RULE 11:
Do not confuse "not mentioned in one chunk" with
"missing from the resume."

RULE 12:
Every evidence statement must come from the Resume Context.

RULE 13:
The Job Description describes requirements.
It is NOT evidence that the candidate possesses those skills.


============================================================
JOB DESCRIPTION
============================================================

{job_description}


============================================================
FIXED REQUIREMENTS
============================================================

{requirements_text}


============================================================
RESUME CONTEXT
============================================================

{context}


============================================================
TASK
============================================================

Evaluate ONLY the seven fixed requirements.

For each requirement, provide:

Requirement:
Evidence from Resume:
Status:

Status MUST be exactly one of:

MATCHING
MISSING
NOT VERIFIABLE


============================================================
FINAL OUTPUT
============================================================

After evaluating all seven requirements, provide:

1. Match Score

2. Matching Skills

3. Missing Skills

4. Not Verifiable

5. Candidate Summary

6. Strengths

7. Weaknesses

8. Hiring Recommendation

9. Evidence


============================================================
IMPORTANT OUTPUT RULES
============================================================

MATCHING SKILLS:
Include ONLY skills from the seven fixed requirements.

MISSING SKILLS:
Include ONLY skills from the seven fixed requirements.

NOT VERIFIABLE:
Include ONLY skills from the seven fixed requirements.

NEVER add additional skills to these lists.

The Match Score must be based ONLY on the seven
fixed requirements.

Do NOT calculate the score using certifications,
projects, hobbies, education, or additional technologies.

Hiring Recommendation must be exactly one of:

Strong Match
Moderate Match
Weak Match


============================================================
FINAL SELF-CHECK
============================================================

Before answering, check:

1. Did I evaluate exactly seven requirements?
2. Did I add any requirement that was not provided?
3. Did I mark an explicitly mentioned required skill as missing?
4. Are Matching Skills limited to the seven requirements?
5. Are Missing Skills limited to the seven requirements?
6. Is the score based only on the seven requirements?

If any answer is wrong, correct the response before
returning the final evaluation.
"""


# ============================================================
# 12. LOAD LOCAL LLAMA 3.2
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

print("\n" + "=" * 70)
print("SENDING RAG PROMPT TO LLAMA 3.2")
print("=" * 70)


# ============================================================
# 13. GENERATE EVALUATION
# ============================================================

response = llm.invoke(prompt)


# ============================================================
# 14. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("AI RESUME EVALUATION")
print("=" * 70)

print(response.content)


# ============================================================
# 15. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("RAG TEST COMPLETED")
print("=" * 70)