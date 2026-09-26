from typing import List, Literal

from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# ============================================================
# 1. REQUIRED JOB SKILLS
# ============================================================

REQUIRED_SKILLS = [
    "Python",
    "SQL",
    "Machine Learning",
    "Statistics",
    "Scikit-learn",
    "Predictive Modeling",
    "Data Analysis"
]


# ============================================================
# 2. STRUCTURED OUTPUT SCHEMA
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


# ============================================================
# 3. LOAD PDF
# ============================================================

def load_resume(pdf_path: str):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    return documents


# ============================================================
# 4. SPLIT DOCUMENT
# ============================================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    return chunks


# ============================================================
# 5. CREATE EMBEDDINGS
# ============================================================

def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# ============================================================
# 6. CREATE VECTOR DATABASE
# ============================================================

def create_vectorstore(chunks, embeddings):

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


# ============================================================
# 7. CREATE RETRIEVER
# ============================================================

def create_retriever(vectorstore, k=12):

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k
        }
    )

    return retriever


# ============================================================
# 8. CREATE LOCAL LLM
# ============================================================

def create_llm():

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    return llm


# ============================================================
# 9. RAG PROMPT
# ============================================================

def build_prompt(job_description, context):

    requirements_text = "\n".join(
        f"- {skill}"
        for skill in REQUIRED_SKILLS
    )

    prompt = f"""
You are an AI Resume Screening Assistant.

Evaluate the candidate ONLY using the supplied resume context.

JOB DESCRIPTION:

{job_description}


REQUIRED SKILLS:

{requirements_text}


RESUME CONTEXT:

{context}


STRICT RULES:

1. Use ONLY information from the resume context.

2. Do NOT use outside knowledge.

3. Do NOT invent skills, experience, projects, certifications,
   achievements, responsibilities, or job history.

4. Evaluate ONLY the required skills listed above.

5. Mark a requirement as MATCHING when the resume contains
   direct or clearly equivalent evidence for that requirement.

6. For example:

   Requirement: Statistics

   Valid evidence includes:
   - Statistics
   - Statistical Analysis
   - Descriptive Statistics
   - Inferential Statistics
   - Probability and Statistics
   - Statistical Modeling

7. Mark a requirement as MISSING only when the resume context
   provides sufficient evidence that the candidate does not
   mention or demonstrate the requirement.

8. Use NOT VERIFIABLE when the available resume context is
   insufficient to determine whether the candidate has the
   requirement.

9. Do NOT mark a skill as MISSING simply because the exact
   wording of the requirement is not present.

10. Consider closely related professional terminology when
    evaluating requirements.

11. Prefer direct evidence from:
    - Work experience
    - Projects
    - Professional summary
    - Technical skills
    - Responsibilities
    - Machine learning model development

12. Do NOT rely only on a certification title when stronger
    direct experience evidence is available.

13. Evidence must be concise and directly supported by the
    resume context.

14. Do not use:
    - Date of birth
    - Marital status
    - Nationality
    - Phone number
    - Email
    - Address
    - Hobbies
    - Other unrelated personal information

15. Strengths must be based on matching requirements.

16. Weaknesses must be based on genuine gaps or limitations
    found in the resume evidence.

17. Do not create weaknesses simply because information was not
    found in one retrieved chunk.

18. Evaluate exactly these requirements:

{requirements_text}
"""

    return prompt


# ============================================================
# 10. ADDITIONAL EVIDENCE VALIDATION
# ============================================================

def validate_requirement_status(evaluation, context):

    """
    Performs a lightweight deterministic check for obvious
    evidence that the small local LLM may have missed.

    This is NOT replacing RAG or the LLM.
    It acts as a safety check for clear resume evidence.
    """

    context_lower = context.lower()

    # Known equivalent terms for the required skills.
    evidence_terms = {

        "Python": [
            "python"
        ],

        "SQL": [
            "sql",
            "mysql",
            "microsoft sql server"
        ],

        "Machine Learning": [
            "machine learning",
            "supervised learning",
            "unsupervised learning"
        ],

        "Statistics": [
            "statistics",
            "statistical analysis",
            "descriptive statistics",
            "inferential statistics",
            "statistical modeling"
        ],

        "Scikit-learn": [
            "scikit-learn",
            "sklearn"
        ],

        "Predictive Modeling": [
            "predictive modeling",
            "predictive model",
            "regression",
            "classification"
        ],

        "Data Analysis": [
            "data analysis",
            "data analytics",
            "exploratory data analysis",
            "eda"
        ]
    }

    for item in evaluation.requirements:

        requirement = item.requirement

        terms = evidence_terms.get(
            requirement,
            []
        )

        found_term = None

        for term in terms:

            if term in context_lower:

                found_term = term

                break

        # ----------------------------------------------------
        # If strong evidence exists but LLM marked MISSING,
        # correct the false negative.
        # ----------------------------------------------------

        if (
            found_term is not None
            and item.status == "MISSING"
        ):

            item.status = "MATCHING"

            item.evidence = (
                f"Resume contains evidence related to "
                f"{requirement}: '{found_term}'."
            )

    return evaluation


# ============================================================
# 11. CALCULATE RECOMMENDATION
# ============================================================

def calculate_recommendation(score):

    if score >= 80:

        return "Strong Match"

    elif score >= 60:

        return "Moderate Match"

    else:

        return "Weak Match"


# ============================================================
# 12. CALCULATE MATCH SCORE
# ============================================================

def calculate_match_score(evaluation):

    total = len(
        evaluation.requirements
    )

    matching = sum(
        1
        for item in evaluation.requirements
        if item.status == "MATCHING"
    )

    missing = sum(
        1
        for item in evaluation.requirements
        if item.status == "MISSING"
    )

    not_verifiable = sum(
        1
        for item in evaluation.requirements
        if item.status == "NOT VERIFIABLE"
    )

    if total == 0:

        score = 0

    else:

        score = round(
            (matching / total) * 100
        )

    recommendation = calculate_recommendation(
        score
    )

    return {

        "score": score,

        "matching": matching,

        "missing": missing,

        "not_verifiable": not_verifiable,

        "total": total,

        "recommendation": recommendation
    }


# ============================================================
# 13. MAIN RESUME SCREENING PIPELINE
# ============================================================

def screen_resume(
    pdf_path,
    job_description
):

    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    documents = load_resume(
        pdf_path
    )

    # --------------------------------------------------------
    # Split PDF
    # --------------------------------------------------------

    chunks = split_documents(
        documents
    )

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings()

    # --------------------------------------------------------
    # Create FAISS vector database
    # --------------------------------------------------------

    vectorstore = create_vectorstore(
        chunks,
        embeddings
    )

    # --------------------------------------------------------
    # Create retriever
    # --------------------------------------------------------

    retriever = create_retriever(
        vectorstore,
        k=12
    )

    # --------------------------------------------------------
    # Retrieval query
    # --------------------------------------------------------

    query = f"""
Job Description:

{job_description}

Required skills:

{", ".join(REQUIRED_SKILLS)}

Find resume evidence relevant to these requirements.
"""

    retrieved_docs = retriever.invoke(
        query
    )

    # --------------------------------------------------------
    # Build retrieved context
    # --------------------------------------------------------

    context_parts = []

    for doc in retrieved_docs:

        page_number = doc.metadata.get(
            "page",
            "unknown"
        )

        text = doc.page_content

        if page_number != "unknown":

            page_display = page_number + 1

        else:

            page_display = "unknown"

        context_parts.append(
            f"[Resume Page {page_display}]\n{text}"
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = build_prompt(
        job_description,
        context
    )

    # --------------------------------------------------------
    # Create LLM
    # --------------------------------------------------------

    llm = create_llm()

    # --------------------------------------------------------
    # Structured output
    # --------------------------------------------------------

    structured_llm = llm.with_structured_output(
        ResumeEvaluation
    )

    # --------------------------------------------------------
    # Generate evaluation
    # --------------------------------------------------------

    evaluation = structured_llm.invoke(
        prompt
    )

    # --------------------------------------------------------
    # Validate requirement count
    # --------------------------------------------------------

    if len(
        evaluation.requirements
    ) != len(
        REQUIRED_SKILLS
    ):

        raise ValueError(
            "LLM returned an incorrect number "
            "of requirements."
        )

    # --------------------------------------------------------
    # Validate requirement names
    # --------------------------------------------------------

    returned_requirements = [
        item.requirement
        for item in evaluation.requirements
    ]

    if set(
        returned_requirements
    ) != set(
        REQUIRED_SKILLS
    ):

        raise ValueError(
            "LLM returned requirements different "
            "from the required skill list."
        )

    # --------------------------------------------------------
    # Additional evidence validation
    # --------------------------------------------------------

    evaluation = validate_requirement_status(
        evaluation,
        context
    )

    # --------------------------------------------------------
    # Calculate score
    # --------------------------------------------------------

    score_data = calculate_match_score(
        evaluation
    )

    # --------------------------------------------------------
    # Return final result
    # --------------------------------------------------------

    return {

        "evaluation": evaluation,

        "score_data": score_data,

        "retrieved_documents": retrieved_docs,

        "context": context
    }

# ============================================================
# 14. SCREEN MULTIPLE RESUMES
# ============================================================

def screen_multiple_resumes(
    pdf_paths,
    job_description
):

    results = []

    for pdf_path in pdf_paths:

        try:

            result = screen_resume(
                pdf_path,
                job_description
            )

            results.append({
                "pdf_path": pdf_path,
                "success": True,
                "result": result
            })

        except Exception as e:

            results.append({
                "pdf_path": pdf_path,
                "success": False,
                "error": str(e)
            })

    return results