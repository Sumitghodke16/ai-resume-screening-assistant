# ============================================================
# AI RESUME SCREENING ASSISTANT
# STRUCTURED RAG + PYDANTIC + DETERMINISTIC SCORE
# ============================================================

from typing import List, Literal

from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# ============================================================
# 1. PYDANTIC STRUCTURED OUTPUT
# ============================================================

class RequirementEvaluation(BaseModel):
    requirement: str
    evidence: str
    status: Literal[
        "MATCHING",
        "MISSING",
        "NOT VERIFIABLE"
    ]


class ResumeEvaluation(BaseModel):
    requirements: List[RequirementEvaluation]

    candidate_summary: str

    strengths: List[str]

    weaknesses: List[str]

    hiring_recommendation: Literal[
        "Strong Match",
        "Moderate Match",
        "Weak Match"
    ]


# ============================================================
# 2. LOAD RESUME PDF
# ============================================================

pdf_path = "sample_resumes/Sumit_Naresh_Ghodke_Sep_2026_Resume.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("\nPDF loaded successfully.")
print(f"Number of pages: {len(documents)}")


# ============================================================
# 3. SPLIT RESUME
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")


# ============================================================
# 4. EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ============================================================
# 5. FAISS VECTOR DATABASE
# ============================================================

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("FAISS vector database created successfully.")


# ============================================================
# 6. RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 12}
)

print("Retriever created successfully.")


# ============================================================
# 7. JOB DESCRIPTION
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
# 8. FIXED REQUIREMENTS
# ============================================================

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
print("FIXED JOB REQUIREMENTS")
print("=" * 70)

for skill in required_skills:
    print(f"- {skill}")


# ============================================================
# 9. RETRIEVE RESUME INFORMATION
# ============================================================

query = """
Retrieve resume information relevant to the following requirements:

Python
SQL
Machine Learning
Statistics
Scikit-learn
Predictive Modeling
Data Analysis

Also retrieve relevant evidence from professional experience,
projects, technical skills, and machine learning experience.
"""

retrieved_documents = retriever.invoke(query)


# ============================================================
# 10. BUILD RESUME CONTEXT
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
# 11. REQUIREMENTS TEXT
# ============================================================

requirements_text = "\n".join(
    f"{i}. {skill}"
    for i, skill in enumerate(required_skills, start=1)
)


# ============================================================
# 12. STRUCTURED RAG PROMPT
# ============================================================

prompt = f"""
You are an AI Resume Screening Assistant.

Evaluate ONE candidate resume against ONE Job Description.


============================================================
JOB DESCRIPTION
============================================================

{job_description}


============================================================
FIXED REQUIREMENTS
============================================================

You MUST evaluate exactly these seven requirements:

{requirements_text}


============================================================
RESUME CONTEXT
============================================================

{context}


============================================================
STRICT RULES
============================================================

1. Use ONLY the Resume Context.

2. Do NOT use outside knowledge.

3. Do NOT invent candidate information.

4. Evaluate ONLY the seven fixed requirements.

5. Do NOT create additional requirements.

6. Do NOT evaluate certifications, hobbies, personal details,
   education, or unrelated technologies as requirements.

7. A requirement is MATCHING if the Resume Context provides
   explicit evidence supporting it.

8. A requirement is MISSING if there is no supporting evidence
   for it in the Resume Context.

9. Use NOT VERIFIABLE only when the available Resume Context
   is insufficient to determine whether the requirement is present.

10. If a requirement is explicitly present in the Resume Context,
    it MUST NOT be classified as MISSING.

11. Every evidence statement must come directly from the
    Resume Context.

12. Do not calculate the Match Score.

13. Do not create a numerical score.

14. Python will calculate the final Match Score after your response.

15. The requirements list in your response MUST contain exactly
    seven items.

16. The requirement names MUST exactly match the seven fixed
    requirements.


============================================================
REQUIRED STRUCTURE
============================================================

Return a structured evaluation containing:

requirements:
    Exactly seven requirement evaluations.

Each requirement evaluation must contain:

    requirement
    evidence
    status

Status must be exactly:

MATCHING
MISSING
NOT VERIFIABLE


Also provide:

candidate_summary
strengths
weaknesses
hiring_recommendation


Hiring recommendation must be exactly one of:

Strong Match
Moderate Match
Weak Match


============================================================
IMPORTANT
============================================================

Do NOT provide:

- Match Score
- Percentage
- 7/7
- Additional requirements
- Additional skills as requirements

Python will calculate the score.
"""


# ============================================================
# 13. LOAD LLAMA 3.2
# ============================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ============================================================
# 14. STRUCTURED OUTPUT
# ============================================================

structured_llm = llm.with_structured_output(
    ResumeEvaluation
)


print("\n" + "=" * 70)
print("SENDING STRUCTURED RAG PROMPT TO LLAMA 3.2")
print("=" * 70)


# ============================================================
# 15. INVOKE MODEL
# ============================================================

try:

    result = structured_llm.invoke(prompt)

except Exception as e:

    print("\nERROR: Structured output failed.")
    print(e)

    print("\nTrying normal LLM output instead...")

    response = llm.invoke(prompt)

    print("\nRAW LLM RESPONSE:")
    print(response.content)

    raise SystemExit


# ============================================================
# 16. VALIDATE REQUIREMENTS
# ============================================================

print("\n" + "=" * 70)
print("STRUCTURED OUTPUT RECEIVED")
print("=" * 70)

print(f"\nNumber of evaluated requirements: {len(result.requirements)}")


# ============================================================
# 17. CHECK THAT REQUIREMENTS ARE CORRECT
# ============================================================

expected_requirements = set(required_skills)

actual_requirements = {
    item.requirement
    for item in result.requirements
}


if actual_requirements != expected_requirements:

    print("\nWARNING:")
    print("The LLM did not return exactly the expected requirements.")

    print("\nExpected:")
    print(expected_requirements)

    print("\nReceived:")
    print(actual_requirements)

else:

    print("\nRequirement validation: PASSED")


# ============================================================
# 18. CALCULATE DETERMINISTIC MATCH SCORE
# ============================================================

matching_count = sum(
    1
    for item in result.requirements
    if item.status == "MATCHING"
)

missing_count = sum(
    1
    for item in result.requirements
    if item.status == "MISSING"
)

not_verifiable_count = sum(
    1
    for item in result.requirements
    if item.status == "NOT VERIFIABLE"
)


total_requirements = len(required_skills)


# Only MATCHING requirements contribute to the score.
match_score = round(
    (matching_count / total_requirements) * 100
)


# ============================================================
# 19. BUILD MATCHING / MISSING LISTS
# ============================================================

matching_skills = [
    item.requirement
    for item in result.requirements
    if item.status == "MATCHING"
]

missing_skills = [
    item.requirement
    for item in result.requirements
    if item.status == "MISSING"
]

not_verifiable_skills = [
    item.requirement
    for item in result.requirements
    if item.status == "NOT VERIFIABLE"
]


# ============================================================
# 20. DISPLAY REQUIREMENT EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("REQUIREMENT EVALUATION")
print("=" * 70)

for item in result.requirements:

    print(f"\nRequirement: {item.requirement}")
    print(f"Status: {item.status}")
    print(f"Evidence: {item.evidence}")


# ============================================================
# 21. DISPLAY DETERMINISTIC SCORE
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESUME EVALUATION")
print("=" * 70)

print(f"\nMatch Score: {match_score}/100")

print(
    f"Matching: {matching_count}/{total_requirements}"
)

print(
    f"Missing: {missing_count}/{total_requirements}"
)

print(
    f"Not Verifiable: {not_verifiable_count}/{total_requirements}"
)


print("\nMatching Skills:")

if matching_skills:

    for skill in matching_skills:
        print(f"- {skill}")

else:

    print("- None")


print("\nMissing Skills:")

if missing_skills:

    for skill in missing_skills:
        print(f"- {skill}")

else:

    print("- None")


print("\nNot Verifiable:")

if not_verifiable_skills:

    for skill in not_verifiable_skills:
        print(f"- {skill}")

else:

    print("- None")


# ============================================================
# 22. CANDIDATE SUMMARY
# ============================================================

print("\nCandidate Summary:")
print(result.candidate_summary)


# ============================================================
# 23. STRENGTHS
# ============================================================

print("\nStrengths:")

for strength in result.strengths:
    print(f"- {strength}")


# ============================================================
# 24. WEAKNESSES
# ============================================================

print("\nWeaknesses:")

for weakness in result.weaknesses:
    print(f"- {weakness}")


# ============================================================
# 25. HIRING RECOMMENDATION
# ============================================================

print("\nHiring Recommendation:")
print(result.hiring_recommendation)


# ============================================================
# 26. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("STRUCTURED RAG TEST COMPLETED")
print("=" * 70)